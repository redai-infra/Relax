# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optimizer-step scoped ESS pre-pass for P3O.

Relax computes ESS over one whole optimizer step to ensure that neither the
number of micro-batches nor the DP/CP split change the adaptive cap or the
final loss. The paper's Algorithm 2 and the reference implementation both
compute ESS per micro-batch, which makes the cap a function of the
gradient-accumulation factor. Relax's approach provides partition invariance:

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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from relax.utils.logging_utils import get_logger
from relax.utils.training.p3o_replay import preserved_iterator_positions, preserved_rng_state
from relax.utils.training.p3o_utils import (
    P3OStepContext,
    P3OSufficientStats,
    compute_p3o_sufficient_stats_unchecked,
    finalize_p3o_step_context,
)

from .cp_utils import get_cp_local_valid_mask, maybe_padded_total_lengths
from .data import DataIterator, get_batch


logger = get_logger(__name__)

P3O_STEP_CONTEXT_ATTR = "_p3o_step_context"
P3O_NONFINITE_RATIO_ERROR = (
    "P3O: non-finite importance ratio at a valid response token on at least one rank; "
    "refusing to silently fall back to ESS=1. Check rollout log-probs and mask alignment."
)


def _local_stats_from_batch(
    args: Namespace, batch: dict, log_probs: list[torch.Tensor]
) -> tuple[P3OSufficientStats, torch.Tensor]:
    """Accumulate one micro-batch's ESS contribution from its log-probs.

    Returns:
        ``(stats, invalid_flag)``, where ``invalid_flag`` is a device-resident
        ``float64`` scalar set to ``1.0`` if this micro-batch produced a
        non-finite ratio. It is reduced with ``S1/S2/N`` rather than checked
        here, so the pre-pass adds no GPU-CPU sync per micro-batch.
    """
    if batch.get("__is_dummy__", False):
        # Dummy micro-batches exist only to align num_microbatches across DP
        # ranks; they must contribute nothing to S1 / S2 / N.
        device = log_probs[0].device if log_probs else "cpu"
        return (
            P3OSufficientStats.zeros(device=device),
            torch.zeros((), dtype=torch.float64, device=device),
        )

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
    return compute_p3o_sufficient_stats_unchecked(current, behavior, valid_mask)


def synchronize_p3o_stats(
    stats: P3OSufficientStats,
    invalid_count: torch.Tensor,
) -> P3OSufficientStats:
    """Reduce last-stage stats over DP x CP, then publish them over PP.

    Pipeline-last is the only stage with logits. It first sums ``S1/S2/N`` and
    the invalid-ratio flag over DP x CP. The already-global vector is then
    broadcast, never summed, over PP so every stage finalizes the same context.
    TP replicas use independent but equivalent groups.
    """
    vector = torch.cat((stats.as_vector(), invalid_count.reshape(1).to(dtype=torch.float64)))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            group = mpu.get_data_parallel_group(with_context_parallel=True)
            torch.distributed.all_reduce(vector, op=torch.distributed.ReduceOp.SUM, group=group)

        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size > 1:
            torch.distributed.broadcast(
                vector,
                group=mpu.get_pipeline_model_parallel_group(),
                group_src=pp_size - 1,
            )

    valid = vector[3] <= 0
    if valid.device.type == "cpu":
        if not bool(valid):
            raise ValueError(P3O_NONFINITE_RATIO_ERROR)
    else:
        torch._assert_async(valid, P3O_NONFINITE_RATIO_ERROR)
    return P3OSufficientStats.from_vector(vector[:3])


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
    invalid_count_acc = [stats_acc[0].valid_token_count.clone()]

    def forward_step(
        iterator: DataIterator,
        model_chunk: torch.nn.Module,
        return_schedule_plan: bool = False,
    ):
        if return_schedule_plan:
            raise ValueError("P3O ESS pre-pass does not support schedule plan generation")
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

        # The forward inputs must be selected exactly as the training pass in
        # model.py::train_one_step does, or the two passes read different token
        # layouts and the frozen cap would be computed from logits the gradient
        # pass never sees. The VL bridge (Qwen3VLModel.forward) does its own
        # CP+SP splitting, so it takes unsplit tokens and no caller-side
        # packed_seq_params.
        mm_kwargs = batch.get("multimodal_train_inputs") or {}
        needs_unsplit = (
            getattr(args, "is_vl_model", False)
            or batch.get("multimodal_train_inputs") is not None
            or getattr(args, "uses_unsplit_forward", False)
        )

        if needs_unsplit and "unsplit_tokens" in batch:
            forward_input_ids = batch["unsplit_tokens"]
            forward_packed_seq_params = None
        else:
            forward_input_ids = batch["tokens"]
            forward_packed_seq_params = batch["packed_seq_params"]

        # thd bridge+CP: the bridge needs the per-sample attention mask and the
        # matching thd packed_seq_params; loss_mask is None there because
        # labels=None means the model runs no internal loss.
        if needs_unsplit and "vlm_packed_seq_params" in batch:
            forward_attention_mask = batch["unsplit_attention_mask"]
            forward_packed_seq_params = batch["vlm_packed_seq_params"]
            forward_loss_mask = None
        else:
            forward_attention_mask = None
            forward_loss_mask = batch["full_loss_masks"]

        # Dynamic CP: the VL bridge reads pg_collection.cp directly, so point it
        # at this micro-batch's sub-group for the forward and restore after.
        orig_cp_group = None
        inner = None
        dynamic_cp_size = batch.get("dynamic_cp_size")
        if dynamic_cp_size is not None and needs_unsplit:
            inner = model_chunk
            while hasattr(inner, "module"):
                inner = inner.module
            orig_cp_group = inner.pg_collection.cp
            inner.pg_collection.cp = mpu.get_dynamic_data_context_parallel_groups(group_size=dynamic_cp_size)

        try:
            output_tensor = model_chunk(
                input_ids=forward_input_ids,
                position_ids=None,
                attention_mask=forward_attention_mask,
                labels=None,
                packed_seq_params=forward_packed_seq_params,
                loss_mask=forward_loss_mask,
                **mm_kwargs,
            )
        finally:
            if orig_cp_group is not None:
                inner.pg_collection.cp = orig_cp_group

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
                # _local_stats_from_batch returns a device-resident invalid_flag
                # instead of raising, so the non-finite detection rides the
                # existing allreduce rather than adding a per-micro-batch
                # GPU-CPU sync via bool() or .item().
                micro_stats, invalid_flag = _local_stats_from_batch(args, batch, computed["log_probs"])
                invalid_count_acc[0] = invalid_count_acc[0] + invalid_flag
                stats_acc[0] = stats_acc[0] + micro_stats
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

    # Accumulate every local micro-batch first, reduce exactly once over DP x CP
    # on pipeline-last, then broadcast that fixed vector over PP.
    reduced = synchronize_p3o_stats(stats_acc[0], invalid_count_acc[0])
    step_context = finalize_p3o_step_context(reduced)

    if step_context.clamp_events:
        logger.warning("P3O: clamped %d out-of-range ESS value(s) this step", step_context.clamp_events)

    return step_context


@contextmanager
def p3o_step_context_published(args: Namespace, step_context: P3OStepContext) -> Iterator[None]:
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
