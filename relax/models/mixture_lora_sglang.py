# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Architecture-independent Mixture-of-LoRA support for SGLang models.

Rollout injection, routed execution and weight loading are identical for every
SGLang architecture: only the mapping from a decoder layer to the linears that
carry routed experts differs. Concrete external models therefore mix in
:class:`MixtureLoraSGLangModelMixin` and implement
:meth:`MixtureLoraSGLangModelMixin.mixture_lora_site_modules`; everything else
lives here so a second architecture costs one small subclass.
"""

from types import MethodType
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from relax.utils.env import Envs
from relax.utils.mixture_lora_common import (
    DenseRoutedLoRAExecutor,
    MixtureLoraConfig,
    deserialize_mixture_lora_config,
    megatron_mixture_lora_name_to_sglang,
    route_topk,
)


__all__ = [
    "MixtureLoraSGLangModelMixin",
    "SGLangMixtureLoRA",
    "attach_sglang_mixture_lora",
    "load_sglang_mixture_lora_weights",
    "mixture_lora_site_id",
    "read_runtime_mixture_lora_config",
]

# Site ids are the Megatron parameter names the training backend publishes, so
# rollout and training agree on one vocabulary for every architecture.
SUPPORTED_MIXTURE_LORA_TARGETS = frozenset({"linear_qkv", "linear_proj"})


def mixture_lora_site_id(layer_id: int, target: str) -> str:
    """Build the Megatron site id for one routed linear."""

    return f"decoder.layers.{layer_id}.self_attention.{target}"


def read_runtime_mixture_lora_config() -> MixtureLoraConfig:
    """Read the routing configuration the trainer handed to this engine."""

    runtime_config = Envs.RELAX_MIXTURE_LORA_CONFIG
    if runtime_config is None:
        raise RuntimeError("RELAX_MIXTURE_LORA_CONFIG must be set before constructing the external model")
    return deserialize_mixture_lora_config(runtime_config)


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
    adapter_dtype = getattr(linear, "params_dtype", base_parameter.dtype)
    if not isinstance(adapter_dtype, torch.dtype) or not adapter_dtype.is_floating_point:
        raise TypeError(
            f"SGLang linear {site_id} must expose a floating-point params_dtype for Mixture-of-LoRA, "
            f"got {adapter_dtype}"
        )
    linear.add_module(
        "mixture_lora",
        SGLangMixtureLoRA(
            config,
            site_id,
            input_size,
            output_size,
            device=base_parameter.device,
            dtype=adapter_dtype,
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


class MixtureLoraSGLangModelMixin:
    """Add token-routed LoRA experts to an SGLang causal-LM model.

    Mix in *before* the SGLang base class so the routed weights are split off
    before its loader sees them::

        class Qwen3ForCausalLM(MixtureLoraSGLangModelMixin, SGLangQwen3ForCausalLM):
            def mixture_lora_site_modules(self, layer_id): ...
    """

    supported_mixture_lora_targets: frozenset[str] = SUPPORTED_MIXTURE_LORA_TARGETS

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.mixture_lora_config = read_runtime_mixture_lora_config()
        # Routed parameters are replicated whole on every rollout worker, so a
        # sharded engine would serve mismatched experts.
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError(f"{type(self).__name__} Mixture-of-LoRA rollout currently requires SGLang TP=1")
        super().__init__(*args, **kwargs)
        self.install_mixture_lora()

    def mixture_lora_site_modules(self, layer_id: int) -> dict[str, nn.Module]:
        """Map each supported target name to the linear it wraps in a layer."""

        raise NotImplementedError

    def install_mixture_lora(self) -> None:
        """Wrap every routed linear this pipeline stage owns."""

        targets = set(self.mixture_lora_config.target_modules)
        unsupported = sorted(targets - set(self.supported_mixture_lora_targets))
        if unsupported:
            raise ValueError(f"Unsupported SGLang Mixture-of-LoRA targets: {unsupported}")
        start_layer = getattr(self.model, "start_layer", 0)
        end_layer = getattr(self.model, "end_layer", len(self.model.layers))
        for layer_id in range(start_layer, end_layer):
            site_modules = self.mixture_lora_site_modules(layer_id)
            missing = sorted(targets - set(site_modules))
            if missing:
                raise ValueError(f"{type(self).__name__} exposes no Mixture-of-LoRA site for {missing}")
            for target in sorted(targets):
                linear = site_modules[target]
                attach_sglang_mixture_lora(
                    linear,
                    self.mixture_lora_config,
                    mixture_lora_site_id(layer_id, target),
                    linear.input_size,
                    linear.output_size,
                )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Route the checkpoint stream: base to SGLang, routed tensors here."""

        mixture_weights = []

        def base_weight_iterator():
            for name, weight in weights:
                if ".mixture_lora." in name:
                    mixture_weights.append((name, weight))
                else:
                    yield name, weight

        result = super().load_weights(base_weight_iterator())
        if mixture_weights:
            load_sglang_mixture_lora_weights(self, mixture_weights)
        return result
