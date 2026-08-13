# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang Qwen3 external model with token-routed LoRA experts."""

from types import MethodType
from typing import Any, Iterable

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.models.qwen3 import Qwen3ForCausalLM as SGLangQwen3ForCausalLM
from torch import nn
from torch.nn import functional as F

from relax.utils.env import Envs
from relax.utils.mixture_lora import (
    DenseRoutedLoRAExecutor,
    MixtureLoraConfig,
    deserialize_mixture_lora_config,
    megatron_mixture_lora_name_to_sglang,
    route_topk,
)


class SGLangMixtureLoRA(nn.Module):
    """Parameter-compatible dense Mixture-of-LoRA execution for rollout."""

    def __init__(
        self,
        config: MixtureLoraConfig,
        site_id: str,
        input_size: int,
        output_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.site_id = site_id
        self.experts = nn.Module()
        self.experts.register_parameter(
            "lora_A",
            nn.Parameter(torch.empty(config.num_experts, config.rank, input_size, device=device, dtype=dtype)),
        )
        self.experts.register_parameter(
            "lora_B",
            nn.Parameter(torch.empty(config.num_experts, output_size, config.rank, device=device, dtype=dtype)),
        )
        self.router = nn.Linear(input_size, config.num_experts, bias=False, device=device, dtype=dtype)
        self.executor = DenseRoutedLoRAExecutor()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert_weight in self.experts.lora_A:
            nn.init.xavier_uniform_(expert_weight)
        nn.init.zeros_(self.experts.lora_B)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def route(self, x: torch.Tensor):
        logits = F.linear(x.reshape(-1, x.shape[-1]).float(), self.router.weight.float())
        return route_topk(logits, self.config.top_k, self.config.temperature)

    def forward_with_routing(self, x: torch.Tensor):
        decision = self.route(x)
        output = self.executor(
            x,
            self.experts.lora_A,
            self.experts.lora_B,
            decision,
            self.config.scale,
        )
        return output, decision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_routing(x)
        return output


def _routed_linear_forward(linear, x: torch.Tensor, *args: Any, **kwargs: Any):
    base_result = linear._relax_mixture_lora_base_forward(x, *args, **kwargs)
    if not isinstance(base_result, tuple) or len(base_result) != 2:
        raise TypeError(f"{type(linear).__name__} must return an (output, bias) tuple")
    output, bias = base_result
    delta = linear.mixture_lora(x).reshape(output.shape)
    return output + delta, bias


def attach_sglang_mixture_lora(
    linear: nn.Module,
    config: MixtureLoraConfig,
    site_id: str,
    input_size: int,
    output_size: int,
) -> None:
    """Add routed parameters while preserving SGLang base parameter names."""

    if hasattr(linear, "mixture_lora"):
        raise RuntimeError(f"SGLang linear {site_id} already has a Mixture-of-LoRA adapter")
    try:
        base_parameter = next(linear.parameters())
    except StopIteration as error:
        raise ValueError(f"SGLang linear {site_id} has no parameters") from error
    linear.add_module(
        "mixture_lora",
        SGLangMixtureLoRA(
            config,
            site_id,
            input_size,
            output_size,
            device=base_parameter.device,
            dtype=base_parameter.dtype,
        ),
    )
    linear._relax_mixture_lora_base_forward = linear.forward
    linear.forward = MethodType(_routed_linear_forward, linear)


def load_sglang_mixture_lora_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Validate and copy one chunk of routed weights into an SGLang model."""

    parameters = dict(model.named_parameters())
    loaded_names = set()
    for source_name, loaded_weight in weights:
        target_name = (
            source_name
            if source_name.startswith("model.layers.")
            else megatron_mixture_lora_name_to_sglang(source_name)
        )
        if target_name in loaded_names:
            raise ValueError(f"Duplicate Mixture-of-LoRA weight: {target_name}")
        if target_name not in parameters:
            layer_id = int(target_name.split(".", maxsplit=3)[2])
            start_layer = getattr(getattr(model, "model", None), "start_layer", None)
            end_layer = getattr(getattr(model, "model", None), "end_layer", None)
            if start_layer is not None and end_layer is not None and not start_layer <= layer_id < end_layer:
                continue
            raise ValueError(f"Unknown Mixture-of-LoRA weight for SGLang: {target_name}")
        parameter = parameters[target_name]
        if tuple(loaded_weight.shape) != tuple(parameter.shape):
            raise ValueError(
                f"Mixture-of-LoRA weight {target_name} has shape {tuple(loaded_weight.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )
        if loaded_weight.dtype != parameter.dtype:
            raise TypeError(
                f"Mixture-of-LoRA weight {target_name} has dtype {loaded_weight.dtype}, expected {parameter.dtype}"
            )
        with torch.no_grad():
            parameter.copy_(loaded_weight.to(device=parameter.device))
        loaded_names.add(target_name)
    return loaded_names


# SGLang registers external models by the checkpoint architecture name. Keeping
# this name replaces its built-in Qwen3 entry while preserving checkpoint metadata.
class Qwen3ForCausalLM(SGLangQwen3ForCausalLM):
    """Qwen3 external model that routes LoRA experts at attention
    projections."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        runtime_config = Envs.RELAX_MIXTURE_LORA_CONFIG
        if runtime_config is None:
            raise RuntimeError("RELAX_MIXTURE_LORA_CONFIG must be set before constructing the external model")
        self.mixture_lora_config = deserialize_mixture_lora_config(runtime_config)
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Qwen3 Mixture-of-LoRA rollout currently requires SGLang TP=1")
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self._install_mixture_lora()

    def _install_mixture_lora(self) -> None:
        targets = set(self.mixture_lora_config.target_modules)
        supported_targets = {"linear_qkv", "linear_proj"}
        if not targets.issubset(supported_targets):
            raise ValueError(f"Unsupported SGLang Mixture-of-LoRA targets: {sorted(targets - supported_targets)}")
        start_layer = getattr(self.model, "start_layer", 0)
        end_layer = getattr(self.model, "end_layer", len(self.model.layers))
        for layer_id in range(start_layer, end_layer):
            layer = self.model.layers[layer_id]
            attention = layer.self_attn
            if "linear_qkv" in targets:
                site_id = f"decoder.layers.{layer_id}.self_attention.linear_qkv"
                attach_sglang_mixture_lora(
                    attention.qkv_proj,
                    self.mixture_lora_config,
                    site_id,
                    attention.qkv_proj.input_size,
                    attention.qkv_proj.output_size,
                )
            if "linear_proj" in targets:
                site_id = f"decoder.layers.{layer_id}.self_attention.linear_proj"
                attach_sglang_mixture_lora(
                    attention.o_proj,
                    self.mixture_lora_config,
                    site_id,
                    attention.o_proj.input_size,
                    attention.o_proj.output_size,
                )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        mixture_weights = []

        def base_weight_iterator():
            for name, weight in weights:
                if ".mixture_lora." in name:
                    mixture_weights.append((name, weight))
                else:
                    yield name, weight

        super().load_weights(base_weight_iterator())
        if not mixture_weights:
            return

        load_sglang_mixture_lora_weights(self, mixture_weights)


EntryClass = Qwen3ForCausalLM
