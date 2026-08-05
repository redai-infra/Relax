# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class LiveWeightSyncMemoryStatus:
    min_free_bytes: int
    max_required_bytes: int

    @property
    def can_sync(self) -> bool:
        return self.min_free_bytes >= self.max_required_bytes


def estimate_full_model_bytes(
    named_tensors: Mapping[str, torch.Tensor],
    *,
    tensor_parallel_size: int,
    expert_tensor_parallel_size: int,
) -> tuple[int, int]:
    """Estimate full-model bytes conservatively from one PP/EP rank."""
    if tensor_parallel_size <= 0 or expert_tensor_parallel_size <= 0:
        raise ValueError("tensor parallel sizes must be positive")

    total_bytes = 0
    largest_param_bytes = 0
    for name, tensor in named_tensors.items():
        tp_size = expert_tensor_parallel_size if ".experts." in name else tensor_parallel_size
        param_bytes = tensor.numel() * tensor.element_size() * tp_size
        total_bytes += param_bytes
        largest_param_bytes = max(largest_param_bytes, param_bytes)
    return total_bytes, largest_param_bytes


def required_live_weight_sync_bytes(
    *,
    full_model_bytes: int,
    largest_param_bytes: int,
    update_weight_buffer_size: int,
    memory_margin_bytes: int,
) -> int:
    """Estimate extra memory for rollout weights and two in-flight chunks."""
    values = (full_model_bytes, largest_param_bytes, update_weight_buffer_size, memory_margin_bytes)
    if any(value < 0 for value in values):
        raise ValueError("live weight sync memory inputs must be non-negative")

    chunk_bytes = max(largest_param_bytes, update_weight_buffer_size)
    return full_model_bytes + 2 * chunk_bytes + memory_margin_bytes


def get_distributed_memory_status(
    *,
    local_free_bytes: int,
    local_required_bytes: int,
    process_group,
) -> LiveWeightSyncMemoryStatus:
    """Use one CPU collective so every rank makes the same memory decision."""
    if local_free_bytes < 0 or local_required_bytes < 0:
        raise ValueError("live weight sync memory values must be non-negative")

    # MIN(free, -required) gives the global minimum free bytes and the
    # negated global maximum required bytes in one Gloo collective.
    values = torch.tensor([local_free_bytes, -local_required_bytes], dtype=torch.int64)
    dist.all_reduce(values, op=dist.ReduceOp.MIN, group=process_group)
    return LiveWeightSyncMemoryStatus(
        min_free_bytes=int(values[0].item()),
        max_required_bytes=-int(values[1].item()),
    )


def run_live_weight_sync(
    *,
    update_weights: Callable[[], None],
    offload_actor: Callable[[], None],
    synchronize_after_offload: Callable[[], None],
    onload_rollout_kv: Callable[[], None],
) -> None:
    """Offload every Actor rank before restoring rollout KV memory."""
    try:
        update_weights()
    finally:
        try:
            offload_actor()
        finally:
            synchronize_after_offload()
    onload_rollout_kv()
