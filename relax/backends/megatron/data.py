# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training.global_vars import get_args
from torch.nn.utils.rnn import pad_sequence

from relax.utils import device as device_utils
from relax.utils import tracking_utils
from relax.utils.data.data import get_minimum_num_micro_batch_size
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_pass_rate, compute_rollout_step
from relax.utils.timer import Timer
from relax.utils.training import train_metric_utils
from relax.utils.training.flops_counter import FlopsCounter
from relax.utils.types import RolloutBatch

from .cp_utils import get_sum_of_sample_mean, maybe_padded_total_lengths, slice_with_cp


logger = get_logger(__name__)

PAD_RULES = {
    # shape like [1, 128, 1036]
    "input_features": dict(
        transpose=(0, 2),
        padding_value=0.0,
    ),
    # shape like [1, 1036]
    "feature_attention_mask": dict(
        transpose=(0, 1),
        padding_value=0,
    ),
}


def _round_up_to_microbatch_group(num_microbatches: torch.Tensor, microbatch_group_size: int) -> torch.Tensor:
    return torch.clamp(
        (num_microbatches + microbatch_group_size - 1) // microbatch_group_size * microbatch_group_size,
        min=1,
    )


def pad_and_flatten(
    tensor_list,
    transpose=None,
    padding_value=0,
):
    """Pad a list of variable-length tensors to the same length and then
    flatten them.

    This function:
    1. Padding is applied along dim=0 by default, pass 'transpose' to swap dimensions
       before and after padding;
    2. Pad all tensors in the tensor list to match the longest tensor in the list.;
    3. Concatenates the padded tensors along the first dimension;
    4. Returns the concatenated tensor and a list of lengths (size of padding dim) for
       each padded tensor.

    Args:
        tensor_list (List[Tensor]):
            A list of tensors. The first dimension is treated as the variable-length
            dimension.
        transpose (Tuple[int, int] or None, optional):
            If not None, each tensor is transposed with `t.transpose(*transpose)`
            before padding, and each padded tensor is transposed back with the same
            arguments afterward.
        padding_value (float or int, optional):
            Value used for padding shorter tensors up to the maximum length.

    Returns:
        Tuple[Tensor, List[int]]:
            - A single concatenated tensor of all padded tensors along dim 0.
            - A list of integers indicating the size of padding dim for each
              padded tensor.

    Note:
        - When `len(tensor_list) == 1`, no padding is performed. The single tensor
          is returned as-is along with its length in a list.
        - This function assumes that all tensors are compatible for padding and
          concatenation after any optional transpose.
    """
    if len(tensor_list) == 1:
        t = tensor_list[0]
        return t, [t.size(0)]

    if transpose is not None:
        tensor_list = [t.transpose(*transpose) for t in tensor_list]

    padded = pad_sequence(
        tensor_list,
        batch_first=True,
        padding_value=padding_value,
    )

    if transpose is not None:
        padded_list = [t.transpose(*transpose) for t in padded]
    else:
        padded_list = [t for t in padded]  # noqa: C416

    num_items = [t.size(0) for t in padded_list]
    return torch.cat(padded_list, dim=0), num_items


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    qkv_format: str = "thd",
    allgather_cp: bool = False,
    is_vl_model: bool = False,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    """Generate a CP-ready micro-batch with packed sequence parameters.

    Steps:
    - Fetch raw fields via iterator.
    - Save original token tensors under "unconcat_tokens".
    - Slice tokens into two chunks for Context Parallelism (CP), concatenate, and pad to a configurable multiple.
    - Build cu_seqlens and `PackedSeqParams` with T-H-D layout (T: sequence length, H: attention heads, D: head dimension).

    Args:
        data_iterator: Iterator providing micro-batch data.
        keys: List of keys to fetch from the iterator.
        pad_multiplier: Multiplier for padding size calculation (default: 128).

    Returns a dict including:
    - "tokens": torch.LongTensor of shape [1, T_padded] on the current CUDA device
    - "unconcat_tokens": list[torch.LongTensor] for the micro-batch before CP slicing/concat
    - "packed_seq_params": PackedSeqParams with T-H-D settings (cu_seqlens on CUDA, dtype=int)
    Plus any other requested keys forwarded from the iterator.
    """

    assert "tokens" in keys
    if isinstance(data_iterator, DataIterator):
        batch = data_iterator.get_next(keys)

        if "dynamic_global_batch_size" in data_iterator.rollout_data:
            batch["dynamic_global_batch_size"] = data_iterator.rollout_data["dynamic_global_batch_size"]
    else:
        batch, _ = next(data_iterator)
    tokens = batch["tokens"]
    # use 0 as the pad token id should be fine?
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if qkv_format == "bshd":
        max_seqlen = batch["max_seq_lens"][0]
        assert max([t.size(0) for t in tokens]) <= max_seqlen

        # For VL models with CP > 1, Bridge expects UNSPLIT tokens (it handles CP
        # splitting internally after vision embedding).  Save padded-but-unsplit
        # tokens so model.py can pass them to Bridge instead of the CP-split ones.
        if cp_size > 1:
            chunk_size = (max_seqlen + 2 * cp_size - 1) // (2 * cp_size)
            padded_len = 2 * cp_size * chunk_size
            unsplit = [F.pad(t, (0, padded_len - t.size(0)), value=pad_token_id) for t in tokens]
            batch["unsplit_tokens"] = torch.stack(unsplit)

        tokens = [slice_with_cp(t, pad_token_id, qkv_format, max_seqlen) for t in tokens]
        tokens = torch.stack(tokens)
        packed_seq_params = None

    elif qkv_format == "thd":
        # bridge Qwen3VLModel.forward (used for Qwen3-VL and text-only
        # Qwen3.5 / Qwen3.6 sharing the same architecture) expects per-sample
        # BSHD-padded input_ids + attention_mask, and re-derives the THD
        # packing internally with align_size = tp*cp*2.  Provide unsplit
        # inputs and a matching packed_seq_params so the caller-side cu_seqlens
        # agrees with what the bridge derives from attention_mask.
        # Mirrors verl's build_vlm_attn_mask_thd + preprocess_thd_engine.
        # Routed by `is_vl_model` (set from hf_config) rather than presence of
        # multimodal_train_inputs, so VL models with text-only batches still
        # land here — bridge skips vision embedding when image_grid_thw is None.
        has_mm_inputs = batch.get("multimodal_train_inputs") is not None
        needs_unsplit_input = is_vl_model or has_mm_inputs or getattr(get_args(), "uses_unsplit_forward", False)
        if needs_unsplit_input and cp_size > 1:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            align_size = tp_size * cp_size * 2
            device = device_utils.make_current_torch_device()

            seqlens = torch.tensor([t.size(0) for t in tokens], dtype=torch.int32, device=device)
            seqlens_padded = (seqlens + align_size - 1) // align_size * align_size
            cu_seqlens_padded = torch.zeros(len(tokens) + 1, dtype=torch.int32, device=device)
            cu_seqlens_padded[1:] = torch.cumsum(seqlens_padded, dim=0)
            max_seqlen_padded = int(seqlens_padded.max().item())

            unsplit_tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_token_id)
            unsplit_attention_mask = torch.zeros_like(unsplit_tokens, dtype=torch.bool)
            for i, s in enumerate(seqlens.tolist()):
                unsplit_attention_mask[i, :s] = True

            batch["unsplit_tokens"] = unsplit_tokens
            batch["unsplit_attention_mask"] = unsplit_attention_mask
            batch["vlm_packed_seq_params"] = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens_padded,
                cu_seqlens_kv=cu_seqlens_padded,
                max_seqlen_q=max_seqlen_padded,
                max_seqlen_kv=max_seqlen_padded,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv_padded=cu_seqlens_padded,
            )
            # Per-sample tp*cp*2-aligned lengths consumed by loss helpers so
            # their per-sample chunking matches bridge's preprocess_packed_seqs.
            batch["padded_total_lengths"] = seqlens_padded.tolist()

        if allgather_cp:
            # DSA mode: concatenate all sequences first, then slice once with CP.
            # We also pad the *global* concatenated stream to make per-rank chunks equal.
            cu_seqlens_list: list[int] = [0]
            for t in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

            tokens = torch.cat(tokens, dim=0)

            # Pad global stream so (1) divisible by cp_size (equal chunks),
            # (2) divisible by pad_size (reduce fragmentation).
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

            cu_seqlens = torch.tensor(
                cu_seqlens_list, dtype=torch.int, device=device_utils.make_current_torch_device()
            )
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        else:
            tokens = [slice_with_cp(t, pad_token_id, qkv_format) for t in tokens]

            cu_seqlens = [0]
            for t in tokens:
                cu_seqlens.append(cu_seqlens[-1] + t.size(0))

            tokens = torch.cat(tokens)

            # Always pad to reduce memory fragmentation and maybe make the computation faster
            pad = (pad_size - tokens.size(0) % pad_size) % pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens.append(cu_seqlens[-1] + pad)

            # thd requires the cu_seqlens to be of the origin length
            cu_seqlens = (
                torch.tensor(cu_seqlens, dtype=torch.int).to(device_utils.make_current_torch_device()) * cp_size
            )

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

        tokens = tokens.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params

    from relax.utils.sft_utils import align_loss_mask_for_sft

    loss_masks: list[torch.Tensor] = []
    per_sample_loss_masks: list[torch.Tensor] = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"], batch["total_lengths"], batch["response_lengths"], strict=True
    ):
        if response_length == total_length:
            loss_mask = align_loss_mask_for_sft(loss_mask)
            per_sample_loss_masks.append(loss_mask)
        else:
            per_sample_loss_masks.append(loss_mask)
            prompt_length = total_length - response_length
            loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        if not allgather_cp:
            loss_mask = slice_with_cp(loss_mask, 0, qkv_format, max_seqlen)
        loss_masks.append(loss_mask)
    batch["loss_masks"] = per_sample_loss_masks

    if qkv_format == "bshd":
        loss_masks = torch.stack(loss_masks)
    elif qkv_format == "thd" and allgather_cp:
        # DSA: concatenate first (same as tokens), pad globally (same pad as above), then slice once.
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    elif qkv_format == "thd":
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}  # key -> concatenated tensor
        multimodal_num_items = {}  # key -> list of item counts per sequence
        tensor_dict_list = {}

        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is None:
                continue
            for key, mm_tensor in mm_input_dict.items():
                if isinstance(mm_tensor, list):
                    mm_tensor = torch.tensor(mm_tensor)
                tensor_dict_list.setdefault(key, []).append(mm_tensor)

        for key, tensor_list in tensor_dict_list.items():
            if key in PAD_RULES:
                multimodal_data[key], multimodal_num_items[key] = pad_and_flatten(
                    tensor_list,
                    **PAD_RULES[key],
                )
            else:
                if len(tensor_list) == 1:
                    multimodal_data[key] = tensor_list[0]
                else:
                    multimodal_data[key] = torch.cat(tensor_list, dim=0)
                multimodal_num_items[key] = [t.size(0) for t in tensor_list]

        batch["multimodal_train_inputs"] = multimodal_data
        batch["multimodal_num_items"] = multimodal_num_items

    batch = move_tensors_to_device(batch, batch["tokens"].device)
    return batch


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, float],
) -> dict[str, float] | None:
    """Gather per-rank metrics, reduce by mean on the DP source rank, and log.

    Expects `log_dict` to contain plain scalars. The DP source rank prints and
    optionally logs to WandB/TensorBoard with a step derived from `rollout_id`
    and batch sizes. Returns the reduced dict on the DP source rank; returns
    None on others.
    """

    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)

        gathered_log_dict = [None] * dp_size
        # Not sure if this will be a performance bottleneck.
        dist.gather_object(
            log_dict,
            gathered_log_dict,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )

        reduced_log_dict = {
            f"{metric_name}/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
        }
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        tracking_utils.log(args, reduced_log_dict, step_key="rollout/step")

        return reduced_log_dict
    else:
        dist.gather_object(
            log_dict,
            None,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )
        return None


class DataIterator:
    """Micro-batch iterator over rollout dicts.

    Supports either fixed contiguous micro-batches or an explicit per-step
    index schedule (for dynamic batch sizing / sequence-length balancing).
    """

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_size: int | None = None,
        micro_batch_indices: list[list[int]] | None = None,
    ) -> None:
        """Initialize an iterator over `rollout_data`.

        Args:
            rollout_data: Dict of per-sample fields for the local step.
            micro_batch_size: Fixed contiguous slice size when not using dynamic scheduling.
            micro_batch_indices: Explicit indices per micro-batch when using dynamic balancing.
                Must be mutually exclusive with `micro_batch_size`.
        """
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        assert micro_batch_size is None or micro_batch_indices is None
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        - If `micro_batch_indices` is provided, selects rows according to the current
          index list for each requested key.
        - Otherwise, slices a contiguous window of size `micro_batch_size` starting
          at the current offset.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                if self.micro_batch_indices is not None:
                    indices = self.micro_batch_indices[self.offset]
                    batch[key] = [vals[i] for i in indices]
                else:
                    assert self.offset + self.micro_batch_size <= len(vals), (
                        f"offset: {self.offset}, micro_batch_size: {self.micro_batch_size}, len(vals): {len(vals)}"
                    )
                    batch[key] = vals[self.offset : self.offset + self.micro_batch_size]

        if self.micro_batch_indices is not None:
            self.offset += 1
        else:
            self.offset += self.micro_batch_size
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def get_data_iterator(
    args: Namespace,
    model: torch.nn.Module | Sequence[torch.nn.Module],
    rollout_data: RolloutBatch,
) -> tuple[list[DataIterator], list[int]]:
    """Create iterators and a micro-batch schedule for a rollout step.

    - If `use_dynamic_batch_size` is False, splits into fixed-size contiguous
      micro-batches of `micro_batch_size`.
    - If True, computes the number of micro-batches per local step based on
      `max_tokens_per_gpu` and per-sample lengths, all-reduces to a DP-wide
      maximum, optionally enforces divisibility for Virtual Pipeline Parallelism (VPP), and builds a balanced
      index schedule to equalize token counts across micro-batches.

    Returns `(data_iterators, num_microbatches)` where:
    - `data_iterators`: list of `DataIterator`, one per VPP stage (size 1 if VPP disabled)
    - `num_microbatches`: list[int], one per local step in the rollout (length = steps)
    """
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dp_group = mpu.get_data_parallel_group()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1
    if vpp_size > 1:
        from megatron.core.utils import get_model_config

        config = get_model_config(model[0])
        microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
    cp_size = mpu.get_context_parallel_world_size()

    num_local_samples = len(rollout_data["total_lengths"])
    global_batch_size = rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
    num_local_gbs = global_batch_size // dp_size

    if getattr(args, "balance_data", False):
        # When balance_data is enabled, SeqlenBalancedSampler assigns
        # samples with equal_size=False, so each DP rank may receive a
        # different number of samples.  Use num_local_samples directly as
        # the local batch size to process all assigned samples in a single
        # step, avoiding remainder / empty-step issues.
        num_local_gbs = num_local_samples
        num_steps_per_rollout = 1
    elif num_local_samples <= num_local_gbs:
        num_local_gbs = num_local_samples
        num_steps_per_rollout = 1
    else:
        num_steps_per_rollout = num_local_samples // num_local_gbs

    # When balance_data is enabled, each DP rank may receive a different number
    # of samples from the SeqlenBalancedSampler. Synchronise num_steps_per_rollout
    # across DP ranks so that all ranks execute the same number of training steps
    # (required by collective operations in the training loop).
    if getattr(args, "balance_data", False):
        steps_tensor = torch.tensor(
            [num_steps_per_rollout], dtype=torch.int, device=device_utils.make_current_torch_device()
        )
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.MAX, group=dp_group)
        num_steps_per_rollout = steps_tensor.item()

    if global_batch_size != args.global_batch_size:
        logger.info(
            f"Using dynamic global_batch_size={global_batch_size} (original={args.global_batch_size}), "
            f"num_local_samples={num_local_samples}, num_steps_per_rollout={num_steps_per_rollout}"
        )

    def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
        data_iterator = []
        for _ in range(vpp_size):
            data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
        return data_iterator

    if not args.use_dynamic_batch_size:
        num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
        data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    else:
        assert args.max_tokens_per_gpu is not None
        # calculate the number of mirobatches for each step
        samples = rollout_data["total_lengths"]
        assert len(samples) == num_local_samples
        num_microbatches = []
        for i in range(num_steps_per_rollout):
            start = i * num_local_gbs
            end = min((i + 1) * num_local_gbs, num_local_samples)
            num_microbatches.append(
                get_minimum_num_micro_batch_size(samples[start:end], args.max_tokens_per_gpu * cp_size)
            )

        num_microbatches = torch.tensor(
            num_microbatches, dtype=torch.int, device=device_utils.make_current_torch_device()
        )
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)

        if vpp_size > 1:
            # vpp requires the number of microbatches to be divisible by vpp_size
            num_microbatches = _round_up_to_microbatch_group(num_microbatches, microbatch_group_size_per_vp_stage)

        num_microbatches = num_microbatches.tolist()

        # balance the each micro batch
        samples = rollout_data["total_lengths"]
        # balance the number of mirobatches across steps
        micro_batch_indices = []
        for i, num_mbs in enumerate(num_microbatches):
            start = i * num_local_gbs
            end = min((i + 1) * num_local_gbs, num_local_samples)
            samples = rollout_data["total_lengths"][start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)

        assert len(set(sum(micro_batch_indices, []))) == num_local_samples
        logger.info(
            f"After dynamic batching, num_microbatches: {num_microbatches}, micro_batch_indices: {micro_batch_indices}"
        )
        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)

    return (
        data_iterator,
        num_microbatches,
    )


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """Summarize rollout fields and log reduced metrics on PP last stage, TP
    rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        max_seq_lens = rollout_data.get("max_seq_lens", None)
        padded_total_lengths = maybe_padded_total_lengths(
            total_lengths,
            args.qkv_format,
            rollout_data.get("multimodal_train_inputs") is not None or getattr(args, "uses_unsplit_forward", False),
        )

        # OPD dynamic metric: overlap ratio on top-k token sets.
        student_topk_ids = rollout_data.get("topk_token_ids", None)
        teacher_topk_ids = rollout_data.get("teacher_topk_token_ids", None)
        if isinstance(student_topk_ids, list) and isinstance(teacher_topk_ids, list):
            assert len(student_topk_ids) == len(teacher_topk_ids)

            overlap_ratio_per_sample = []
            for s_ids, t_ids in zip(student_topk_ids, teacher_topk_ids, strict=False):
                if not isinstance(s_ids, torch.Tensor) or not isinstance(t_ids, torch.Tensor):
                    continue
                if s_ids.ndim < 2 or t_ids.ndim < 2:
                    continue
                k = min(int(s_ids.size(-1)), int(t_ids.size(-1)))
                if k <= 0:
                    continue
                s_ids = s_ids[:, :k].long()
                t_ids = t_ids[:, :k].long()
                overlap_matrix = s_ids.unsqueeze(-1).eq(t_ids.unsqueeze(-2))
                overlap_ratio = overlap_matrix.any(dim=-1).float().sum(dim=-1) / float(k)
                overlap_ratio_per_sample.append(overlap_ratio)

            if overlap_ratio_per_sample:
                loss_masks_t = loss_masks
                total_lengths_i = total_lengths
                response_lengths_i = response_lengths
                overlap_ratio_flat = torch.cat(overlap_ratio_per_sample).clone().detach().to(loss_masks_t[0].device)
                sum_of_sample_mean = get_sum_of_sample_mean(
                    total_lengths_i,
                    response_lengths_i,
                    loss_masks_t,
                    qkv_format=args.qkv_format,
                    max_seq_lens=max_seq_lens,
                    padded_total_lengths=padded_total_lengths,
                )
                overlap_ratio_value = cp_size * sum_of_sample_mean(overlap_ratio_flat) / len(loss_masks_t)
                log_dict["opd_overlap_ratio"] = overlap_ratio_value.item()

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_routed_experts",
                "max_seq_lens",
                "dynamic_global_batch_size",
                "topk_token_ids",
                "topk_log_probs",
                "teacher_topk_token_ids",
                "teacher_topk_log_probs",
                "teacher_topk_k",
            ]:
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, (list, tuple)):
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    if key in [
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "teacher_log_probs",
                        "opd_reverse_kl",
                    ]:
                        val = torch.cat(val).clone().detach()
                        val = val.to(loss_masks[0].device)
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                            padded_total_lengths=padded_total_lengths,
                        )
                        val = cp_size * sum_of_sample_mean(val) / len(loss_masks)
                    else:
                        val = torch.cat(val).clone().detach()
                        val = val.mean() * cp_size
                else:
                    val = sum(val) / len(val)
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        if args.ci_test and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # TODO: figure out why there is a small numerical difference in log_probs and ref_log_probs in CI test, and whether it's expected or not.
                # assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
            if "rollout/log_probs" in reduced_log_dict:
                assert -0.5 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.5

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data, ignore_num_groups=True)

    if args.log_correct_samples:
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
            cp_size = mpu.get_context_parallel_world_size()
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            def quantile(total_value, n_quantiles, data) -> dict:
                import math

                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                percentile = {
                    f"p{min(math.ceil(q * 100), 100)}": p for q, p in zip(quantiles, percentile, strict=True)
                }
                return percentile

            raw_rewards = rollout_data["raw_reward"]
            # Additional metrics for correct cases are calculated separately below.
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            correct_padded_total_lengths_full = maybe_padded_total_lengths(
                total_lengths,
                args.qkv_format,
                rollout_data.get("multimodal_train_inputs") is not None
                or getattr(args, "uses_unsplit_forward", False),
            )
            correct_padded_total_lengths: list[int] | None = (
                [] if correct_padded_total_lengths_full is not None else None
            )
            # true_on_policy_mode skips actor_fwd so log_probs is unavailable here;
            # fall back to rollout_log_probs (numerically close, used only for logging).
            entropy_source = rollout_data.get("log_probs") or rollout_data.get("rollout_log_probs")
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    if entropy_source is not None:
                        correct_entropy.append(-entropy_source[i])
                    if correct_padded_total_lengths is not None:
                        correct_padded_total_lengths.append(correct_padded_total_lengths_full[i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths,
                    correct_response_lengths,
                    correct_loss_masks,
                    padded_total_lengths=correct_padded_total_lengths,
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """Log multi-turn auxiliary metrics such as raw/observed response lengths
    and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when
    available to compute statistics without host transfers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(
    rollout_id: int, args: Namespace, rollout_data: RolloutBatch, ignore_num_groups: bool = False
) -> None:
    """Compute pass@k metrics from `raw_reward` groups and log the results.

    `raw_reward` is reshaped to `[group_number, group_size]`, then pass@k is
    estimated per problem and averaged.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=None if ignore_num_groups else args.rollout_batch_size,
            )

        gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data_fwd(args, rollout_id):
    from megatron.core import mpu

    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()
    is_primary_rank = (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}
    step = compute_rollout_step(args, rollout_id)
    log_dict["actor_fwd/step"] = step
    tracking_utils.log(args, log_dict, step_key="actor_fwd/step")


def log_perf_data(rollout_id: int, args: Namespace, flops_counter: FlopsCounter | None = None) -> None:
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.is_pipeline_last_stage()
            and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        ),
        flops_counter=flops_counter,
        world_size=dist.get_world_size(),
    )


def sync_actor_critic_data(
    args: Namespace,
    rollout_data: RolloutBatch | None = None,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Broadcast `values` (from critic) and optionally
    `log_probs`/`ref_log_probs` (from actor) across PP ranks to align data
    dependencies.

    - Values are broadcast from src=1.
    - Log-probs and ref-log-probs are broadcast from src=0 when KL is used.
    Updates `rollout_data` in place with the synchronized tensors.
    """
    log_probs_key = "log_probs" if not args.use_rollout_logprobs else "rollout_log_probs"
    values, log_probs, ref_log_probs = map(rollout_data.get, ("values", log_probs_key, "ref_log_probs"))

    # return when not the pp last stage
    if not values and not log_probs:
        return

    handles = []

    if not values:
        values = [torch.empty_like(log_prob) for log_prob in log_probs]
    for value in values:
        handles.append(dist.broadcast(value, src=1, group=group, async_op=True))

    if args.kl_coef != 0 or args.use_kl_loss:
        if not log_probs:
            log_probs = [torch.empty_like(value) for value in values]
        if not ref_log_probs:
            ref_log_probs = [torch.empty_like(value) for value in values]
        for ref_log_prob, log_prob in zip(ref_log_probs, log_probs, strict=False):
            handles.append(dist.broadcast(log_prob, src=0, group=group, async_op=True))
            handles.append(dist.broadcast(ref_log_prob, src=0, group=group, async_op=True))

    for handle in handles:
        handle.wait()

    rollout_data.update(
        {
            k: v
            for k, v in {
                "values": values,
                log_probs_key: log_probs,
                "ref_log_probs": ref_log_probs,
            }.items()
            if v is not None
        }
    )


def move_tensors_to_device(data, device):
    """Recursively move tensors in a (nested) dict/list to the specified
    device.

    Non-tensor values are left unchanged.
    """
    if isinstance(data, dict):
        return {k: move_tensors_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_device(v, device) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data  # e.g., int, str, None, etc.
