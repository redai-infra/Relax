# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optimizer-step scoped ESS pre-pass for P3O.

Task 40 requires that neither the number of micro-batches nor the DP/CP split
change the adaptive cap or the final loss. The paper's Algorithm 2 and the
reference implementation both compute ESS per micro-batch, which makes the cap a
function of the gradient-accumulation factor. Relax therefore computes ESS over
one whole *optimizer* step:

    stats pass (no grad) over every micro-batch of the window
        -> local S1 / S2 / N
        -> one all-reduce over DP x CP
        -> immutable P3OStepContext
    train pass over the same data, same RNG, one frozen cap
        -> token-sum loss, global-token normalization

The pre-pass replays the same iterator window, so it snapshots and restores both
the iterator offsets and the RNG state. Anything that mutates state during a
no-grad forward (dropout, FP8 amax history) would break that replay and is
rejected in ``arguments.py`` rather than silently tolerated here.
"""

from argparse import Namespace
from collections.abc import Sequence
from contextlib import contextmanager

import torch
from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from relax.utils.logging_utils import get_logger
from relax.utils.training.p3o_replay import preserved_iterator_positions, preserved_rng_state
from relax.utils.training.p3o_utils import (
    P3OStepContext,
    P3OSufficientStats,
    compute_p3o_sufficient_stats,
    finalize_p3o_step_context,
)

from .cp_utils import get_cp_local_valid_mask, maybe_padded_total_lengths
from .data import DataIterator, get_batch


logger = get_logger(__name__)

P3O_STEP_CONTEXT_ATTR = "_p3o_step_context"


def _local_stats_from_batch(args: Namespace, batch: dict, log_probs: list[torch.Tensor]) -> P3OSufficientStats:
    """Accumulate one micro-batch's ESS contribution from computed log-probs."""
    if batch.get("__is_dummy__", False):
        # Dummy micro-batches exist only to align num_microbatches across DP
        # ranks; they must contribute nothing to S1 / S2 / N.
        return P3OSufficientStats.zeros(device=log_probs[0].device if log_probs else "cpu")

    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    padded_total_lengths = batch.get("padded_total_lengths", None)

    current = torch.cat(log_probs, dim=0)
    behavior = torch.cat(batch["rollout_log_probs"], dim=0)
    valid_mask = get_cp_local_valid_mask(
        total_lengths,
        response_lengths,
        batch["loss_masks"],
        args.qkv_format,
        batch.get("max_seq_lens", None),
        padded_total_lengths,
        dynamic_cp_size=batch.get("dynamic_cp_size", None),
        dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
    )
    return compute_p3o_sufficient_stats(current, behavior, valid_mask)


def reduce_p3o_stats(stats: P3OSufficientStats) -> P3OSufficientStats:
    """Sum sufficient statistics across the DP x CP group.

    Only DP and CP are reduced. TP and PP ranks hold *replicas* of the selected
    tokens' log-probs, so including them would multiply N (and S1, S2) by the
    TP/PP degree and silently rescale the cap.
    """
    vector = stats.as_vector()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        group = mpu.get_data_parallel_group(with_context_parallel=True)
        torch.distributed.all_reduce(vector, op=torch.distributed.ReduceOp.SUM, group=group)
    return P3OSufficientStats.from_vector(vector)


def compute_p3o_step_context(
    args: Namespace,
    data_iterator: Sequence[DataIterator],
    model: Sequence[torch.nn.Module],
    num_microbatches: int,
) -> P3OStepContext:
    """Run the no-grad stats pass and return this step's frozen P3O context.

    Args:
        args: Runtime arguments.
        data_iterator: The same iterator(s) the training pass will consume.
        model: DDP-wrapped model chunks.
        num_microbatches: Micro-batch count for this optimizer step.

    Returns:
        The immutable :class:`P3OStepContext` for the step.
    """
    from .loss import get_log_probs_and_entropy

    # Accumulated in a cell rather than a rebound local: the write happens inside
    # the nested loss callback that Megatron's schedule invokes, one level deeper
    # than forward_step.
    stats_acc: list[P3OSufficientStats] = [
        P3OSufficientStats.zeros(device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    ]

    def forward_step(iterator: DataIterator, model_chunk: torch.nn.Module):
        batch = get_batch(
            iterator,
            [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "rollout_log_probs",
                "max_seq_lens",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
            getattr(args, "is_vl_model", False),
        )
        batch["padded_total_lengths"] = maybe_padded_total_lengths(
            batch["total_lengths"],
            args.qkv_format,
            getattr(args, "is_vl_model", False)
            or batch.get("multimodal_train_inputs") is not None
            or getattr(args, "uses_unsplit_forward", False),
        )

        output_tensor = model_chunk(
            input_ids=batch["tokens"],
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=batch["packed_seq_params"],
            loss_mask=batch["full_loss_masks"],
        )

        def collect(logits: torch.Tensor):
            # Only the pipeline last stage sees real logits; earlier stages just
            # participate in the schedule.
            if mpu.is_pipeline_last_stage():
                _, computed = get_log_probs_and_entropy(
                    logits,
                    args=args,
                    unconcat_tokens=batch["unconcat_tokens"],
                    total_lengths=batch["total_lengths"],
                    response_lengths=batch["response_lengths"],
                    with_entropy=False,
                    max_seq_lens=batch.get("max_seq_lens", None),
                    padded_total_lengths=batch.get("padded_total_lengths", None),
                    dynamic_cp_size=batch.get("dynamic_cp_size", None),
                    dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
                )
                stats_acc[0] = stats_acc[0] + _local_stats_from_batch(args, batch, computed["log_probs"])
            zero = torch.zeros((), device=logits.device, dtype=torch.float32)
            return zero, 1, {"keys": [], "values": zero.reshape(1)}

        return output_tensor, collect

    forward_backward_func = get_forward_backward_func()

    with preserved_iterator_positions(data_iterator), preserved_rng_state(), torch.no_grad():
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=True,
        )

    # Accumulate every local micro-batch first, then reduce exactly once.
    reduced = reduce_p3o_stats(stats_acc[0])
    step_context = finalize_p3o_step_context(reduced)

    if step_context.clamp_events:
        logger.warning("P3O: clamped %d out-of-range ESS value(s) this step", step_context.clamp_events)

    return step_context


@contextmanager
def p3o_step_context_published(args: Namespace, step_context: P3OStepContext):
    """Publish the step context on ``args`` for the duration of the train pass.

    The loss function reads the cap from here rather than from the micro-batch
    dict: a per-micro-batch copy could diverge, and the whole point is that all
    micro-batches of the step share one immutable cap. Cleared afterwards so a
    stale cap can never leak into the next step.
    """
    previous = getattr(args, P3O_STEP_CONTEXT_ATTR, None)
    setattr(args, P3O_STEP_CONTEXT_ATTR, step_context)
    try:
        yield
    finally:
        setattr(args, P3O_STEP_CONTEXT_ATTR, previous)
