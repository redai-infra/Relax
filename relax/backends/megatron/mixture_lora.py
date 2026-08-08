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
    DenseRoutedLoRAExecutor,
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
    ) -> torch.Tensor:
        """Attach one site's balance loss and replace its detached record."""

        response_mask = self.response_mask_for(x)
        statistics = compute_routing_statistics(decision, response_mask)
        # This is a microbatch-level F_e * P_e objective. It is not an
        # average of independent per-token losses.
        balance_loss = statistics.balance_loss
        site_aux_loss = balance_loss * (config.aux_loss_coef / self.num_sites)
        if self.calculate_per_token_loss:
            aux_loss_payload = site_aux_loss * statistics.valid_token_count
        else:
            aux_loss_payload = site_aux_loss * self.num_samples

        objective_aux_loss = aux_loss_payload * self.objective_scale
        key = (self.optimizer_step, self.microbatch_id, site_id)
        self.records[site_id] = MixtureLoRARoutingRecord(
            key=key,
            statistics=_detach_routing_statistics(statistics),
            balance_loss=balance_loss.detach(),
            aux_loss=objective_aux_loss.detach(),
        )

        backward_scale = self.main_loss_backward_scale.to(device=output.device) * self.objective_scale
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


class MixtureLoRAExperts(nn.Module):
    """LoRA expert parameters stored in one stable logical layout."""

    def __init__(
        self,
        config: MixtureLoraConfig,
        input_size: int,
        output_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(
            torch.empty(config.num_experts, config.rank, input_size, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(config.num_experts, output_size, config.rank, device=device, dtype=dtype)
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
    ) -> None:
        super().__init__()
        if not isinstance(site_id, str) or not site_id.strip():
            raise ValueError("site_id must be a non-empty string")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point, got {dtype}")

        self.config = config
        self.site_id = site_id
        self.experts = MixtureLoRAExperts(config, input_size, output_size, device=device, dtype=dtype)
        self.router = MixtureLoRARouter(config.num_experts, input_size, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.executor = DenseRoutedLoRAExecutor()

    def route(self, x: torch.Tensor) -> RoutingDecision:
        logits = self.router(x.reshape(-1, x.shape[-1]))
        return route_topk(logits, self.config.top_k, self.config.temperature)

    def forward_with_routing(self, x: torch.Tensor) -> tuple[torch.Tensor, RoutingDecision]:
        decision = self.route(x)
        delta = self.executor(
            self.dropout(x),
            self.experts.lora_A,
            self.experts.lora_B,
            decision,
            self.config.scale,
        )
        routing_context = get_mixture_lora_routing_context()
        if routing_context is not None:
            delta = routing_context.attach_aux_loss(delta, x, self.site_id, self.config, decision)
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
            if parallel_state.get_tensor_model_parallel_world_size() != 1:
                raise NotImplementedError("Mixture-of-LoRA tensor parallel execution is not implemented yet")

            _, full_name = match
            attributes = get_adapter_attributes_from_linear(module)
            return MixtureParallelLinearAdapter(
                module,
                self.mixture_config,
                full_name,
                attributes.in_features,
                attributes.out_features,
                dropout=self.dropout,
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
    "activate_mixture_lora_routing_context",
    "build_mixture_lora_peft",
    "ensure_mixture_lora_recompute_inputs_grad",
    "get_microbatch_objective_scale",
    "get_mixture_lora_routing_context",
    "install_mixture_lora_checkpoint_context",
]
