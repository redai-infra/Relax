# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron model modules and Bridge injection for Mixture-of-LoRA."""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from relax.utils.mixture_lora import (
    MixtureLoraConfig,
    RoutingDecision,
    RoutingStatistics,
    compute_routing_statistics,
    route_topk,
)


@dataclass(frozen=True)
class MixtureLoRARoutingRecord:
    """Detached routing values retained for one microbatch and site."""

    key: tuple[int, int, str]
    statistics: RoutingStatistics
    balance_loss: torch.Tensor
    aux_loss: torch.Tensor
    objective_weight: torch.Tensor


class _AttachAuxLoss(torch.autograd.Function):
    """Attach an auxiliary loss to an activation with an explicit scale."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor, backward_scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss, backward_scale)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        aux_loss, backward_scale = ctx.saved_tensors
        aux_loss_grad = torch.ones_like(aux_loss) * backward_scale.reshape(())
        return grad_output, aux_loss_grad, None


@dataclass
class MixtureLoRARoutingContext:
    """Routing state and aux-loss scaling for one training microbatch."""

    optimizer_step: int
    microbatch_id: int
    response_mask: torch.Tensor
    num_microbatches: int
    num_sites: int
    num_samples: int
    calculate_per_token_loss: bool
    objective_scale: float
    main_loss_backward_scale: torch.Tensor
    context_parallel_group: Any = None
    context_parallel_world_size: int = 1
    is_dummy: bool = False
    records: dict[str, MixtureLoRARoutingRecord] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.optimizer_step < 0 or self.microbatch_id < 0:
            raise ValueError("optimizer_step and microbatch_id must be non-negative")
        if self.num_microbatches <= 0 or self.num_sites <= 0 or self.num_samples <= 0:
            raise ValueError("num_microbatches, num_sites, and num_samples must be positive")
        if not torch.is_tensor(self.response_mask) or self.response_mask.ndim not in (1, 2):
            raise ValueError("response_mask must be a one- or two-dimensional tensor")
        if not torch.is_tensor(self.main_loss_backward_scale) or self.main_loss_backward_scale.numel() != 1:
            raise ValueError("main_loss_backward_scale must be a one-element tensor")
        if not math.isfinite(self.objective_scale) or self.objective_scale < 0:
            raise ValueError("objective_scale must be finite and non-negative")
        if self.context_parallel_world_size <= 0:
            raise ValueError("context_parallel_world_size must be positive")
        if self.context_parallel_world_size > 1 and self.context_parallel_group is None:
            raise ValueError("context_parallel_group is required when context parallelism is enabled")

    def response_mask_for(self, x: torch.Tensor) -> torch.Tensor:
        """Align a batch-first mask with Megatron's activation layout."""

        activation_shape = tuple(x.shape[:-1])
        mask = self.response_mask
        if tuple(mask.shape) == activation_shape:
            aligned = mask
        elif x.ndim == 3 and mask.ndim == 2 and tuple(mask.shape) == (x.shape[1], x.shape[0]):
            aligned = mask.transpose(0, 1)
        else:
            raise ValueError(
                f"response_mask shape {tuple(mask.shape)} does not match activation token layout {activation_shape}"
            )
        if aligned.device != x.device:
            raise ValueError("response_mask and routed activation must be on the same device")
        if self.is_dummy:
            aligned = torch.zeros_like(aligned)
        return aligned.reshape(-1)

    def attach_aux_loss(
        self,
        output: torch.Tensor,
        x: torch.Tensor,
        site_id: str,
        config: MixtureLoraConfig,
        decision: RoutingDecision,
        *,
        record_statistics: bool = True,
        backward_divisor: int = 1,
    ) -> torch.Tensor:
        """Attach one site's balance loss and replace its detached record."""

        if backward_divisor <= 0:
            raise ValueError("backward_divisor must be positive")

        response_mask = self.response_mask_for(x)
        statistics = compute_routing_statistics(decision, response_mask)
        gradient_balance_loss, recorded_balance_loss, global_valid_token_count = _context_parallel_balance_losses(
            statistics,
            self.context_parallel_group,
            self.context_parallel_world_size,
        )
        site_aux_loss = gradient_balance_loss * (config.aux_loss_coef / self.num_sites)
        recorded_site_aux_loss = recorded_balance_loss * (config.aux_loss_coef / self.num_sites)
        sample_or_token_count = (
            global_valid_token_count
            if self.calculate_per_token_loss
            else statistics.valid_token_count.new_tensor(self.num_samples)
        )
        objective_weight = sample_or_token_count * self.objective_scale
        aux_loss_payload = site_aux_loss * sample_or_token_count
        objective_aux_loss = recorded_site_aux_loss * objective_weight / self.context_parallel_world_size
        if record_statistics:
            key = (self.optimizer_step, self.microbatch_id, site_id)
            self.records[site_id] = MixtureLoRARoutingRecord(
                key=key,
                statistics=_detach_routing_statistics(statistics),
                balance_loss=recorded_balance_loss,
                aux_loss=objective_aux_loss.detach(),
                objective_weight=objective_weight.detach(),
            )

        backward_scale = (
            self.main_loss_backward_scale.to(device=output.device) * self.objective_scale / backward_divisor
        )
        return _AttachAuxLoss.apply(output, aux_loss_payload, backward_scale)


_ACTIVE_ROUTING_CONTEXT: ContextVar[MixtureLoRARoutingContext | None] = ContextVar(
    "mixture_lora_routing_context", default=None
)


@contextmanager
def activate_mixture_lora_routing_context(context: MixtureLoRARoutingContext) -> Iterator[None]:
    """Make a microbatch routing context visible to routed adapters."""

    token = _ACTIVE_ROUTING_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_ROUTING_CONTEXT.reset(token)


def get_mixture_lora_routing_context() -> MixtureLoRARoutingContext | None:
    """Return the active microbatch routing context, if one exists."""

    return _ACTIVE_ROUTING_CONTEXT.get()


def install_mixture_lora_checkpoint_context() -> None:
    """Restore the active routing context during Megatron recomputation."""

    from megatron.core import tensor_parallel

    checkpoint = tensor_parallel.checkpoint
    if getattr(checkpoint, "_relax_mixture_lora_context", False):
        return

    @wraps(checkpoint)
    def checkpoint_with_routing_context(function, distribute_saved_activations, *args):
        routing_context = get_mixture_lora_routing_context()
        if routing_context is None:
            return checkpoint(function, distribute_saved_activations, *args)

        @wraps(function)
        def run_with_routing_context(*function_args):
            with activate_mixture_lora_routing_context(routing_context):
                return function(*function_args)

        return checkpoint(run_with_routing_context, distribute_saved_activations, *args)

    checkpoint_with_routing_context._relax_mixture_lora_context = True
    tensor_parallel.checkpoint = checkpoint_with_routing_context


def ensure_mixture_lora_recompute_inputs_grad(model) -> None:
    """Keep full activation recompute reachable when the base is frozen."""

    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.utils import unwrap_model

    unwrapped = unwrap_model(model)
    model_chunks = unwrapped if isinstance(unwrapped, list) else [unwrapped]
    for model_chunk in model_chunks:
        config = getattr(model_chunk, "config", None)
        if config is None or getattr(config, "recompute_method", None) is None:
            continue
        for module in model_chunk.modules():
            if not isinstance(module, TransformerBlock) or getattr(
                module, "_relax_mixture_lora_input_grad_patched", False
            ):
                continue
            original_forward = module.forward

            @wraps(original_forward)
            def forward_with_input_grad(hidden_states, *args, _original_forward=original_forward, **kwargs):
                if (
                    torch.is_tensor(hidden_states)
                    and hidden_states.is_floating_point()
                    and not hidden_states.requires_grad
                ):
                    hidden_states = hidden_states.detach().requires_grad_(True)
                return _original_forward(hidden_states, *args, **kwargs)

            module.forward = forward_with_input_grad
            module._relax_mixture_lora_input_grad_patched = True


def get_microbatch_objective_scale(
    *,
    calculate_per_token_loss: bool,
    is_dummy: bool,
    explicit_loss_scale: float | None,
    num_microbatches: int,
    global_batch_size: int,
    data_parallel_world_size_with_cp: int,
) -> float:
    """Return the final per-microbatch scale applied before parameter grads."""

    if num_microbatches <= 0 or global_batch_size <= 0 or data_parallel_world_size_with_cp <= 0:
        raise ValueError("microbatch, global batch, and data-parallel sizes must be positive")
    if is_dummy:
        return 0.0
    if calculate_per_token_loss:
        return 1.0
    if explicit_loss_scale is not None:
        if not math.isfinite(explicit_loss_scale) or explicit_loss_scale < 0:
            raise ValueError("explicit_loss_scale must be finite and non-negative")
        return explicit_loss_scale / num_microbatches
    return data_parallel_world_size_with_cp / global_batch_size


def _detach_routing_statistics(statistics: RoutingStatistics) -> RoutingStatistics:
    return RoutingStatistics(
        pre_topk_prob_sum=statistics.pre_topk_prob_sum.detach(),
        post_topk_weight_sum=statistics.post_topk_weight_sum.detach(),
        selection_count=statistics.selection_count.detach(),
        top1_count=statistics.top1_count.detach(),
        pre_topk_entropy_sum=statistics.pre_topk_entropy_sum.detach(),
        post_topk_entropy_sum=statistics.post_topk_entropy_sum.detach(),
        valid_token_count=statistics.valid_token_count.detach(),
        top_k=statistics.top_k,
    )


def _context_parallel_balance_losses(
    statistics: RoutingStatistics,
    context_parallel_group: Any,
    context_parallel_world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the local gradient contribution and global CP balance loss."""

    if context_parallel_world_size == 1:
        balance_loss = statistics.balance_loss
        return balance_loss, balance_loss.detach(), statistics.valid_token_count

    num_experts = statistics.pre_topk_prob_sum.numel()
    reduced = torch.cat(
        (
            statistics.pre_topk_prob_sum.detach(),
            statistics.selection_count.detach(),
            statistics.valid_token_count.detach().reshape(1),
        )
    ).clone()
    torch.distributed.all_reduce(reduced, group=context_parallel_group)
    global_pre_topk_prob_sum = reduced[:num_experts]
    global_selection_count = reduced[num_experts : 2 * num_experts]
    global_valid_token_count = reduced[-1]
    denominator = global_valid_token_count.clamp_min(1)
    has_valid_tokens = (global_valid_token_count > 0).to(reduced.dtype)
    selection_share = global_selection_count / (denominator * statistics.top_k)

    # Keep only this rank's probability sum differentiable. CP/DDP gradient
    # reduction adds the rank-local contributions into the full-sequence loss.
    local_mean_prob = statistics.pre_topk_prob_sum / denominator
    local_balance_loss = num_experts * torch.sum(selection_share * local_mean_prob) * has_valid_tokens
    global_mean_prob = global_pre_topk_prob_sum / denominator
    global_balance_loss = num_experts * torch.sum(selection_share * global_mean_prob) * has_valid_tokens
    return local_balance_loss, global_balance_loss, global_valid_token_count


def pack_mixture_lora_routing_records(
    contexts: list[MixtureLoRARoutingContext],
    site_ids: tuple[str, ...],
    *,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack local step statistics into a fixed site-major tensor."""

    if not site_ids or len(set(site_ids)) != len(site_ids):
        raise ValueError("site_ids must be non-empty and unique")
    if num_experts <= 0 or not 1 <= top_k <= num_experts:
        raise ValueError("num_experts and top_k do not describe a valid router")

    scalar_fields = 6
    row_width = 4 * num_experts + scalar_fields
    packed = torch.zeros(len(site_ids), row_width, dtype=torch.float64, device=device)
    site_indices = {site_id: index for index, site_id in enumerate(site_ids)}
    scalar_offset = 4 * num_experts

    for context in contexts:
        for site_id, record in context.records.items():
            if site_id not in site_indices:
                raise ValueError(f"routing record references unknown site_id {site_id!r}")
            statistics = record.statistics
            if statistics.pre_topk_prob_sum.numel() != num_experts or statistics.top_k != top_k:
                raise ValueError(f"routing record for {site_id!r} does not match the configured router")

            row = packed[site_indices[site_id]]
            row[0:num_experts] += statistics.pre_topk_prob_sum.to(device=device, dtype=packed.dtype)
            row[num_experts : 2 * num_experts] += statistics.post_topk_weight_sum.to(device=device, dtype=packed.dtype)
            row[2 * num_experts : 3 * num_experts] += statistics.selection_count.to(device=device, dtype=packed.dtype)
            row[3 * num_experts : 4 * num_experts] += statistics.top1_count.to(device=device, dtype=packed.dtype)
            row[scalar_offset] += statistics.pre_topk_entropy_sum.to(device=device, dtype=packed.dtype)
            row[scalar_offset + 1] += statistics.post_topk_entropy_sum.to(device=device, dtype=packed.dtype)
            row[scalar_offset + 2] += statistics.valid_token_count.to(device=device, dtype=packed.dtype)
            row[scalar_offset + 3] += (record.balance_loss * record.objective_weight).to(
                device=device, dtype=packed.dtype
            )
            row[scalar_offset + 4] += record.objective_weight.to(device=device, dtype=packed.dtype)
            row[scalar_offset + 5] += record.aux_loss.to(device=device, dtype=packed.dtype)

    return packed


def mixture_lora_metrics_from_packed_records(
    packed: torch.Tensor,
    site_ids: tuple[str, ...],
    *,
    num_experts: int,
    top_k: int,
    calculate_per_token_loss: bool,
    data_parallel_world_size_with_cp: int,
) -> dict[str, torch.Tensor]:
    """Compute routed-site and global metrics after distributed reduction."""

    expected_shape = (len(site_ids), 4 * num_experts + 6)
    if tuple(packed.shape) != expected_shape:
        raise ValueError(f"packed routing statistics must have shape {expected_shape}, got {tuple(packed.shape)}")
    if data_parallel_world_size_with_cp <= 0:
        raise ValueError("data_parallel_world_size_with_cp must be positive")

    scalar_offset = 4 * num_experts
    metrics: dict[str, torch.Tensor] = {}

    def divide_or_zero(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        nonzero = denominator > 0
        safe_denominator = torch.where(nonzero, denominator, torch.ones_like(denominator))
        return numerator / safe_denominator * nonzero.to(numerator.dtype)

    def add_statistics(prefix: str, row: torch.Tensor) -> None:
        valid_token_count = row[scalar_offset + 2]
        pre_topk_mean_prob = divide_or_zero(row[0:num_experts], valid_token_count)
        post_topk_mean_weight = divide_or_zero(row[num_experts : 2 * num_experts], valid_token_count)
        selection_share = divide_or_zero(row[2 * num_experts : 3 * num_experts], valid_token_count * top_k)
        top1_fraction = divide_or_zero(row[3 * num_experts : 4 * num_experts], valid_token_count)
        for expert_id in range(num_experts):
            metrics[f"{prefix}/expert_{expert_id}_pre_topk_mean_prob"] = pre_topk_mean_prob[expert_id]
            metrics[f"{prefix}/expert_{expert_id}_post_topk_mean_weight"] = post_topk_mean_weight[expert_id]
            metrics[f"{prefix}/expert_{expert_id}_selection_share"] = selection_share[expert_id]
            metrics[f"{prefix}/expert_{expert_id}_top1_fraction"] = top1_fraction[expert_id]
        metrics[f"{prefix}/pre_topk_normalized_entropy"] = divide_or_zero(row[scalar_offset], valid_token_count)
        metrics[f"{prefix}/post_topk_normalized_entropy"] = divide_or_zero(row[scalar_offset + 1], valid_token_count)

    for site_id, row in zip(site_ids, packed, strict=True):
        prefix = f"molora/{site_id}"
        add_statistics(prefix, row)
        metrics[f"{prefix}/balance_loss"] = divide_or_zero(row[scalar_offset + 3], row[scalar_offset + 4])

    global_row = packed.sum(dim=0)
    add_statistics("molora/global", global_row)
    aux_loss_sum = global_row[scalar_offset + 5]
    if calculate_per_token_loss:
        # Every configured site routes the same response-token stream. Use one
        # site's count so the site sum in the numerator is not divided twice.
        aux_denominator = packed[0, scalar_offset + 2]
    else:
        aux_denominator = packed.new_tensor(data_parallel_world_size_with_cp)
    metrics["molora/aux_loss"] = divide_or_zero(aux_loss_sum, aux_denominator)
    return metrics


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Keep the forward value and sum tensor-parallel input gradients."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_input = grad_output.contiguous().clone()
        torch.distributed.all_reduce(grad_input, group=ctx.group)
        return grad_input, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """Sum tensor-parallel partials while leaving backward rank-local."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, group: Any) -> torch.Tensor:
        del ctx
        output = value.contiguous().clone()
        torch.distributed.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        return grad_output, None


class _GatherLastDimFromTensorParallelRegion(torch.autograd.Function):
    """Gather hidden shards and return the local shard in backward."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, group: Any, rank: int, world_size: int) -> torch.Tensor:
        ctx.rank = rank
        ctx.world_size = world_size
        gathered = [torch.empty_like(value) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, value.contiguous(), group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        return grad_output.chunk(ctx.world_size, dim=-1)[ctx.rank].contiguous(), None, None, None


class _GatherFirstDimFromTensorParallelRegion(torch.autograd.Function):
    """Gather sequence shards and reduce-scatter their input gradients."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, group: Any, rank: int, world_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        gathered = [torch.empty_like(value) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, value.contiguous(), group=group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        grad_input = grad_output.contiguous().clone()
        torch.distributed.all_reduce(grad_input, group=ctx.group)
        return grad_input.chunk(ctx.world_size, dim=0)[ctx.rank].contiguous(), None, None, None


class _ScatterFirstDimToTensorParallelRegion(torch.autograd.Function):
    """Scatter sequence tokens and gather their output gradients."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, group: Any, rank: int, world_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = world_size
        if value.shape[0] % world_size != 0:
            raise ValueError("sequence dimension must be divisible by tensor parallel size")
        return value.chunk(world_size, dim=0)[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        gathered = [torch.empty_like(grad_output) for _ in range(ctx.world_size)]
        torch.distributed.all_gather(gathered, grad_output.contiguous(), group=ctx.group)
        return torch.cat(gathered, dim=0), None, None, None


class MegatronDenseRoutedLoRAExecutor(nn.Module):
    """Dense expert execution with explicit Megatron tensor-parallel
    collectives."""

    def __init__(self, *, input_is_parallel: bool, tp_group: Any, tp_rank: int, tp_world_size: int) -> None:
        super().__init__()
        self.input_is_parallel = input_is_parallel
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size

    def forward(
        self,
        x: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        routing_decision: RoutingDecision,
        scale: float,
    ) -> torch.Tensor:
        input_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        expert_hidden = torch.einsum("ti,nri->tnr", x_flat, lora_a)
        if self.tp_world_size > 1:
            if self.input_is_parallel:
                expert_hidden = _ReduceFromTensorParallelRegion.apply(expert_hidden, self.tp_group)
            else:
                expert_hidden = _GatherLastDimFromTensorParallelRegion.apply(
                    expert_hidden,
                    self.tp_group,
                    self.tp_rank,
                    self.tp_world_size,
                )
            # B is output-sharded, so its input gradient must include every
            # output shard before it reaches A.
            expert_hidden = _CopyToTensorParallelRegion.apply(expert_hidden, self.tp_group)

        expert_outputs = torch.einsum("tnr,nor->tno", expert_hidden, lora_b)
        if self.input_is_parallel and self.tp_world_size > 1:
            expert_outputs = _GatherLastDimFromTensorParallelRegion.apply(
                expert_outputs,
                self.tp_group,
                self.tp_rank,
                self.tp_world_size,
            )
        routing_weights = routing_decision.dense_weights().to(dtype=expert_outputs.dtype)
        delta = torch.sum(expert_outputs * routing_weights.unsqueeze(-1), dim=1)
        return (delta * scale).reshape(*input_shape, expert_outputs.shape[-1])


class MixtureLoRAExperts(nn.Module):
    """LoRA expert parameters stored in one stable logical layout."""

    def __init__(
        self,
        config: MixtureLoraConfig,
        input_size: int,
        output_size: int,
        *,
        local_rank: int | None = None,
        local_input_size: int | None = None,
        local_output_size: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        local_rank = config.rank if local_rank is None else local_rank
        local_input_size = input_size if local_input_size is None else local_input_size
        local_output_size = output_size if local_output_size is None else local_output_size
        self.lora_A = nn.Parameter(
            torch.empty(config.num_experts, local_rank, local_input_size, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(config.num_experts, local_output_size, config.rank, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match the current Bridge LoRA initialization independently for each expert.
        for expert_weight in self.lora_A:
            nn.init.xavier_uniform_(expert_weight)
        nn.init.zeros_(self.lora_B)


class MixtureLoRARouter(nn.Module):
    """Per-token linear router with FP32 logits."""

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, input_size, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight.float())


class MixtureLoRAAdapter(nn.Module):
    """Route tokens across LoRA experts and combine their outputs."""

    def __init__(
        self,
        config: MixtureLoraConfig,
        site_id: str,
        input_size: int,
        output_size: int,
        *,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
        input_is_parallel: bool = False,
        sequence_parallel: bool = False,
        tp_group: Any = None,
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(site_id, str) or not site_id.strip():
            raise ValueError("site_id must be a non-empty string")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point, got {dtype}")
        if tp_world_size <= 0 or not 0 <= tp_rank < tp_world_size:
            raise ValueError("tp_rank and tp_world_size do not describe a valid tensor-parallel group")
        if tp_world_size > 1 and tp_group is None:
            raise ValueError("tp_group is required when tensor parallelism is enabled")
        if input_size % tp_world_size != 0 or output_size % tp_world_size != 0:
            raise ValueError("Mixture-of-LoRA input and output sizes must be divisible by tensor parallel size")
        if not input_is_parallel and config.rank % tp_world_size != 0:
            raise ValueError("Mixture-of-LoRA rank must be divisible by tensor parallel size for column layers")

        self.config = config
        self.site_id = site_id
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        local_rank = config.rank if input_is_parallel else config.rank // tp_world_size
        local_input_size = input_size // tp_world_size if input_is_parallel else input_size
        local_output_size = output_size // tp_world_size
        self.experts = MixtureLoRAExperts(
            config,
            input_size,
            output_size,
            local_rank=local_rank,
            local_input_size=local_input_size,
            local_output_size=local_output_size,
            device=device,
            dtype=dtype,
        )
        self.router = MixtureLoRARouter(config.num_experts, local_input_size, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.executor = MegatronDenseRoutedLoRAExecutor(
            input_is_parallel=input_is_parallel,
            tp_group=tp_group,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        if tp_world_size > 1 and not input_is_parallel:
            self._synchronize_replicated_router()
            self.router.weight.register_hook(self._reduce_replicated_router_gradient)

    def _synchronize_replicated_router(self) -> None:
        source_rank = torch.distributed.get_global_rank(self.tp_group, 0)
        with torch.no_grad():
            torch.distributed.broadcast(self.router.weight, src=source_rank, group=self.tp_group)

    def _reduce_replicated_router_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        reduced_gradient = gradient.contiguous()
        torch.distributed.all_reduce(reduced_gradient, group=self.tp_group)
        return reduced_gradient

    def get_extra_state(self) -> dict[str, Any]:
        """Return the configuration required to validate checkpoint restore."""

        return {
            "schema_version": self.config.schema_version,
            "site_id": self.site_id,
            "num_experts": self.config.num_experts,
            "rank": self.config.rank,
            "top_k": self.config.top_k,
            "temperature": self.config.temperature,
            "aux_loss_coef": self.config.aux_loss_coef,
            "alpha": self.config.alpha,
            "target_modules": list(self.config.target_modules),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "dtype": str(self.router.weight.dtype),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Validate saved Mixture-of-LoRA metadata before loading tensors."""

        if not isinstance(state, dict):
            raise RuntimeError(f"Mixture-of-LoRA checkpoint metadata must be a dict, got {type(state).__name__}")
        expected = self.get_extra_state()
        mismatches = {
            key: (expected_value, state.get(key))
            for key, expected_value in expected.items()
            if state.get(key) != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: expected {expected_value!r}, checkpoint has {saved_value!r}"
                for key, (expected_value, saved_value) in mismatches.items()
            )
            raise RuntimeError(f"Mixture-of-LoRA checkpoint metadata mismatch for {self.site_id}: {details}")

    def route(self, x: torch.Tensor) -> RoutingDecision:
        logits = self.router(x.reshape(-1, x.shape[-1]))
        if self.input_is_parallel and self.tp_world_size > 1:
            logits = _ReduceFromTensorParallelRegion.apply(logits, self.tp_group)
        return route_topk(logits, self.config.top_k, self.config.temperature)

    def forward_with_routing(self, x: torch.Tensor) -> tuple[torch.Tensor, RoutingDecision]:
        routed_input = x
        if not self.input_is_parallel and self.tp_world_size > 1:
            if self.sequence_parallel:
                routed_input = _GatherFirstDimFromTensorParallelRegion.apply(
                    routed_input,
                    self.tp_group,
                    self.tp_rank,
                    self.tp_world_size,
                )
            else:
                routed_input = _CopyToTensorParallelRegion.apply(routed_input, self.tp_group)

        decision = self.route(routed_input)
        delta = self.executor(
            self.dropout(routed_input),
            self.experts.lora_A,
            self.experts.lora_B,
            decision,
            self.config.scale,
        )
        if self.input_is_parallel and self.sequence_parallel and self.tp_world_size > 1:
            delta = _ScatterFirstDimToTensorParallelRegion.apply(
                delta,
                self.tp_group,
                self.tp_rank,
                self.tp_world_size,
            )
        routing_context = get_mixture_lora_routing_context()
        if routing_context is not None:
            delta = routing_context.attach_aux_loss(
                delta,
                routed_input,
                self.site_id,
                self.config,
                decision,
                record_statistics=self.tp_rank == 0,
                backward_divisor=self.tp_world_size if not self.input_is_parallel else 1,
            )
        return delta, decision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta, _ = self.forward_with_routing(x)
        return delta


class MixtureParallelLinearAdapter(nn.Module):
    """Add a routed LoRA delta while preserving Megatron's linear protocol."""

    def __init__(
        self,
        to_wrap: nn.Module,
        config: MixtureLoraConfig,
        site_id: str,
        input_size: int,
        output_size: int,
        *,
        dropout: float,
        input_is_parallel: bool = False,
        sequence_parallel: bool = False,
        tp_group: Any = None,
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> None:
        super().__init__()
        try:
            first_parameter = next(to_wrap.parameters())
        except StopIteration as error:
            raise ValueError(f"Mixture-of-LoRA target {site_id} has no parameters") from error

        self.to_wrap = to_wrap
        self.mixture_lora = MixtureLoRAAdapter(
            config,
            site_id,
            input_size,
            output_size,
            dropout=dropout,
            device=first_parameter.device,
            dtype=first_parameter.dtype,
            input_is_parallel=input_is_parallel,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        self._adapter_enabled = True

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False

    def base_linear_forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Normalize the return forms used by Megatron parallel linear
        layers."""

        result = self.to_wrap(x, *args, **kwargs)
        if not isinstance(result, tuple):
            raise TypeError(f"{type(self.to_wrap).__name__} must return a tuple, got {type(result).__name__}")

        bias = None
        adapter_input = x
        if len(result) == 2:
            output, bias = result
            if isinstance(output, tuple) and len(output) == 2:
                output, adapter_input = output
        elif len(result) == 3:
            output, bias, adapter_input = result
        else:
            raise ValueError(f"{type(self.to_wrap).__name__} returned an unsupported tuple of length {len(result)}")
        return output, bias, adapter_input

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        output, bias, adapter_input = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return output, bias
        adapter_output = self.mixture_lora(adapter_input.contiguous()).reshape(output.shape)
        return output + adapter_output, bias

    def state_dict(
        self,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Keep base keys unchanged and store routed parameters under
        mixture_lora."""

        if destination is None:
            destination = {}
        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.mixture_lora.state_dict(
            destination=destination,
            prefix=f"{prefix}mixture_lora.",
            keep_vars=keep_vars,
        )
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Combine the base layer and routed TP shards for native
        checkpointing."""

        from megatron.core import parallel_state
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        sharded_state = self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata)
        adapter_state = self.mixture_lora.state_dict(prefix="", keep_vars=True)
        axis_map = {
            "experts.lora_A": 2 if self.mixture_lora.input_is_parallel else 1,
            "experts.lora_B": 1,
        }
        if self.mixture_lora.input_is_parallel:
            axis_map["router.weight"] = 1
        dp_cp_group = (
            metadata["dp_cp_group"]
            if metadata is not None and metadata.get("dp_cp_group") is not None
            else parallel_state.get_data_parallel_group(with_context_parallel=True)
        )
        sharded_state.update(
            make_sharded_tensors_for_checkpoint(
                adapter_state,
                f"{prefix}mixture_lora.",
                axis_map,
                sharded_offsets,
                tp_group=self.mixture_lora.tp_group,
                dp_cp_group=dp_cp_group,
            )
        )
        return sharded_state


def build_mixture_lora_peft(config: MixtureLoraConfig, dropout: float):
    """Build a Bridge PEFT object that injects routed adapters at matched
    sites."""

    try:
        from megatron.bridge.peft.base import PEFT
        from megatron.bridge.peft.module_matcher import ModuleMatcher
        from megatron.bridge.peft.utils import get_adapter_attributes_from_linear
        from megatron.core import parallel_state
    except ImportError as error:
        raise RuntimeError(
            "Mixture-of-LoRA training requires a Megatron-Bridge image with PEFT support. "
            "Please upgrade the training image."
        ) from error

    @dataclass
    class MixtureLoRAPEFT(PEFT, ModuleMatcher):
        target_modules: list[str] = field(default_factory=list)
        mixture_config: MixtureLoraConfig | None = None
        dropout: float = 0.0

        def transform(
            self,
            module: nn.Module,
            name: str | None = None,
            prefix: str | None = None,
        ) -> nn.Module:
            if isinstance(module, MixtureParallelLinearAdapter):
                return module
            match = self.match(module, name, prefix)
            if match is None:
                return module
            if self.mixture_config is None:
                raise RuntimeError("Mixture-of-LoRA PEFT is missing its configuration")
            _, full_name = match
            attributes = get_adapter_attributes_from_linear(module)
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_group = getattr(module, "tp_group", None)
            if tp_world_size > 1 and tp_group is None:
                tp_group = parallel_state.get_tensor_model_parallel_group()
            return MixtureParallelLinearAdapter(
                module,
                self.mixture_config,
                full_name,
                attributes.in_features,
                attributes.out_features,
                dropout=self.dropout,
                input_is_parallel=attributes.input_is_parallel,
                sequence_parallel=getattr(getattr(module, "config", None), "sequence_parallel", False)
                and not attributes.disable_sequence_parallel_comm,
                tp_group=tp_group,
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                tp_world_size=tp_world_size,
            )

    return MixtureLoRAPEFT(
        target_modules=list(config.target_modules),
        mixture_config=config,
        dropout=dropout,
    )


__all__ = [
    "MixtureLoRARoutingContext",
    "MixtureLoRARoutingRecord",
    "MixtureLoRAAdapter",
    "MixtureLoRAExperts",
    "MixtureLoRARouter",
    "MixtureParallelLinearAdapter",
    "MegatronDenseRoutedLoRAExecutor",
    "activate_mixture_lora_routing_context",
    "build_mixture_lora_peft",
    "ensure_mixture_lora_recompute_inputs_grad",
    "get_microbatch_objective_scale",
    "get_mixture_lora_routing_context",
    "install_mixture_lora_checkpoint_context",
    "mixture_lora_metrics_from_packed_records",
    "pack_mixture_lora_routing_records",
]
