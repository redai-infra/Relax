# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron model modules and Bridge injection for Mixture-of-LoRA."""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from relax.utils.mixture_lora import DenseRoutedLoRAExecutor, MixtureLoraConfig, RoutingDecision, route_topk


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
        """Normalize the return forms used by Megatron parallel linear layers."""

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
        """Keep base keys unchanged and store routed parameters under mixture_lora."""

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
    """Build a Bridge PEFT object that injects routed adapters at matched sites."""

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
    "MixtureLoRAAdapter",
    "MixtureLoRAExperts",
    "MixtureLoRARouter",
    "MixtureParallelLinearAdapter",
    "build_mixture_lora_peft",
]
