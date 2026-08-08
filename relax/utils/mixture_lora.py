# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Backend-independent routing primitives for Mixture-of-LoRA."""

import math
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

import torch
from torch import nn


MIXTURE_LORA_SCHEMA_VERSION = 1
MixtureLoraParameterKind = Literal["experts.lora_A", "experts.lora_B", "router.weight"]
_PARAMETER_KINDS = {"experts.lora_A", "experts.lora_B", "router.weight"}


@dataclass(frozen=True)
class MixtureLoraConfig:
    """Backend-independent Mixture-of-LoRA configuration."""

    num_experts: int
    rank: int
    top_k: int
    temperature: float
    aux_loss_coef: float
    alpha: float
    target_modules: tuple[str, ...]
    schema_version: int = MIXTURE_LORA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_positive_int("num_experts", self.num_experts)
        if self.num_experts <= 1:
            raise ValueError(f"num_experts must be greater than 1, got {self.num_experts}")
        _validate_positive_int("rank", self.rank)
        _validate_positive_int("top_k", self.top_k)
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k must be no greater than num_experts, got {self.top_k} and {self.num_experts}")
        _validate_finite_number("temperature", self.temperature, minimum=0.0, minimum_inclusive=False)
        _validate_finite_number("aux_loss_coef", self.aux_loss_coef, minimum=0.0, minimum_inclusive=True)
        _validate_finite_number("alpha", self.alpha, minimum=0.0, minimum_inclusive=False)
        _validate_positive_int("schema_version", self.schema_version)

        if isinstance(self.target_modules, str):
            raise TypeError("target_modules must be a sequence of module names, not a string")
        target_modules = tuple(self.target_modules)
        if not target_modules or any(not isinstance(target, str) or not target.strip() for target in target_modules):
            raise ValueError("target_modules must contain non-empty module names")
        if len(set(target_modules)) != len(target_modules):
            raise ValueError("target_modules must not contain duplicates")
        object.__setattr__(self, "target_modules", target_modules)

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


@dataclass(frozen=True)
class MixtureLoraStateSpec:
    """Stable identity and global layout for one Mixture-of-LoRA tensor."""

    schema_version: int
    site_id: str
    parameter_kind: MixtureLoraParameterKind
    global_shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        _validate_positive_int("schema_version", self.schema_version)
        if not isinstance(self.site_id, str) or not self.site_id.strip():
            raise ValueError("site_id must be a non-empty string")
        if self.parameter_kind not in _PARAMETER_KINDS:
            raise ValueError(f"unsupported Mixture-of-LoRA parameter kind: {self.parameter_kind}")

        global_shape = tuple(self.global_shape)
        expected_ndim = 2 if self.parameter_kind == "router.weight" else 3
        if len(global_shape) != expected_ndim or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in global_shape
        ):
            raise ValueError(
                f"global_shape for {self.parameter_kind} must contain {expected_ndim} positive dimensions, "
                f"got {global_shape}"
            )
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise TypeError(f"dtype must be a floating-point torch dtype, got {self.dtype}")
        object.__setattr__(self, "global_shape", global_shape)

    @property
    def parameter_name(self) -> str:
        return f"{self.site_id}.mixture_lora.{self.parameter_kind}"


@dataclass(frozen=True)
class TransportTensorSpec:
    """Tensor identity plus the TP shard that is being transported."""

    state: MixtureLoraStateSpec
    tp_shard_dim: int | None
    tp_rank: int
    tp_world_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, MixtureLoraStateSpec):
            raise TypeError(f"state must be a MixtureLoraStateSpec, got {type(self.state).__name__}")
        _validate_positive_int("tp_world_size", self.tp_world_size)
        if not isinstance(self.tp_rank, int) or isinstance(self.tp_rank, bool):
            raise TypeError(f"tp_rank must be an integer, got {type(self.tp_rank).__name__}")
        if not 0 <= self.tp_rank < self.tp_world_size:
            raise ValueError(f"tp_rank must satisfy 0 <= tp_rank < tp_world_size, got {self.tp_rank}")
        if self.tp_shard_dim is not None:
            if not isinstance(self.tp_shard_dim, int) or isinstance(self.tp_shard_dim, bool):
                raise TypeError(f"tp_shard_dim must be an integer or None, got {type(self.tp_shard_dim).__name__}")
            if not 0 <= self.tp_shard_dim < len(self.state.global_shape):
                raise ValueError(
                    f"tp_shard_dim must index global_shape {self.state.global_shape}, got {self.tp_shard_dim}"
                )

    @property
    def parameter_name(self) -> str:
        return self.state.parameter_name

    @property
    def schema_version(self) -> int:
        return self.state.schema_version

    @property
    def site_id(self) -> str:
        return self.state.site_id

    @property
    def parameter_kind(self) -> MixtureLoraParameterKind:
        return self.state.parameter_kind

    @property
    def global_shape(self) -> tuple[int, ...]:
        return self.state.global_shape

    @property
    def dtype(self) -> torch.dtype:
        return self.state.dtype


@dataclass(frozen=True)
class RoutedLoRAParallelContext:
    """Parallel layout passed explicitly to a routed LoRA executor."""

    target_module: str
    sequence_parallel: bool = False
    tensor_parallel_group: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_module, str) or not self.target_module.strip():
            raise ValueError("target_module must be a non-empty string")
        if not isinstance(self.sequence_parallel, bool):
            raise TypeError(f"sequence_parallel must be a bool, got {type(self.sequence_parallel).__name__}")


@runtime_checkable
class RoutedLoRAExecutor(Protocol):
    """Parameter-free expert execution interface shared by model backends."""

    def execute(
        self,
        x: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        routing_decision: "RoutingDecision",
        scale: float,
        parallel_context: RoutedLoRAParallelContext | None,
    ) -> torch.Tensor: ...


class DenseRoutedLoRAExecutor(nn.Module):
    """Compute every expert with batched einsums and apply sparse weights."""

    def forward(
        self,
        x: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        routing_decision: "RoutingDecision",
        scale: float,
        parallel_context: RoutedLoRAParallelContext | None = None,
    ) -> torch.Tensor:
        return self.execute(x, lora_a, lora_b, routing_decision, scale, parallel_context)

    def execute(
        self,
        x: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        routing_decision: "RoutingDecision",
        scale: float,
        parallel_context: RoutedLoRAParallelContext | None = None,
    ) -> torch.Tensor:
        _validate_dense_executor_inputs(x, lora_a, lora_b, routing_decision, scale, parallel_context)

        input_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        expert_hidden = torch.einsum("ti,nri->tnr", x_flat, lora_a)
        expert_outputs = torch.einsum("tnr,nor->tno", expert_hidden, lora_b)
        routing_weights = routing_decision.dense_weights().to(dtype=expert_outputs.dtype)
        delta = torch.sum(expert_outputs * routing_weights.unsqueeze(-1), dim=1)
        return (delta * scale).reshape(*input_shape, lora_b.shape[1])


def build_mixture_lora_state_specs(
    config: MixtureLoraConfig,
    site_id: str,
    input_size: int,
    output_size: int,
    dtype: torch.dtype,
) -> tuple[MixtureLoraStateSpec, ...]:
    """Build the executor-independent global parameter schema for one site."""

    _validate_positive_int("input_size", input_size)
    _validate_positive_int("output_size", output_size)
    return (
        MixtureLoraStateSpec(
            config.schema_version,
            site_id,
            "experts.lora_A",
            (config.num_experts, config.rank, input_size),
            dtype,
        ),
        MixtureLoraStateSpec(
            config.schema_version,
            site_id,
            "experts.lora_B",
            (config.num_experts, output_size, config.rank),
            dtype,
        ),
        MixtureLoraStateSpec(
            config.schema_version,
            site_id,
            "router.weight",
            (config.num_experts, input_size),
            dtype,
        ),
    )


@dataclass(frozen=True)
class RoutingDecision:
    """Token-level Top-K routing results.

    All tensors use ``[token, expert]`` or ``[token, selected_expert]``
    layouts. Probabilities and selected weights are always FP32.
    """

    pre_topk_probs: torch.Tensor
    topk_indices: torch.Tensor
    post_topk_weights: torch.Tensor

    @property
    def num_experts(self) -> int:
        return self.pre_topk_probs.shape[-1]

    @property
    def top_k(self) -> int:
        return self.topk_indices.shape[-1]

    def dense_weights(self) -> torch.Tensor:
        """Return Top-K weights scattered into the full expert dimension."""

        weights = torch.zeros_like(self.pre_topk_probs)
        return weights.scatter(-1, self.topk_indices, self.post_topk_weights)


@dataclass(frozen=True)
class RoutingStatistics:
    """Unnormalized per-expert values that can be reduced across ranks."""

    pre_topk_prob_sum: torch.Tensor
    post_topk_weight_sum: torch.Tensor
    selection_count: torch.Tensor
    top1_count: torch.Tensor
    pre_topk_entropy_sum: torch.Tensor
    post_topk_entropy_sum: torch.Tensor
    valid_token_count: torch.Tensor
    top_k: int

    def _per_token(self, value: torch.Tensor) -> torch.Tensor:
        denominator = self.valid_token_count.clamp_min(1)
        mean = value / denominator
        return mean * (self.valid_token_count > 0).to(mean.dtype)

    @property
    def pre_topk_mean_prob(self) -> torch.Tensor:
        return self._per_token(self.pre_topk_prob_sum)

    @property
    def post_topk_mean_weight(self) -> torch.Tensor:
        return self._per_token(self.post_topk_weight_sum)

    @property
    def selection_share(self) -> torch.Tensor:
        return self._per_token(self.selection_count) / self.top_k

    @property
    def top1_fraction(self) -> torch.Tensor:
        return self._per_token(self.top1_count)

    @property
    def pre_topk_normalized_entropy(self) -> torch.Tensor:
        return self._per_token(self.pre_topk_entropy_sum)

    @property
    def post_topk_normalized_entropy(self) -> torch.Tensor:
        return self._per_token(self.post_topk_entropy_sum)

    @property
    def balance_loss(self) -> torch.Tensor:
        """Return ``N * sum(F_e * P_e)`` with Top-K-normalized ``F_e``."""

        # Top-K membership is discrete. Keep F_e detached while P_e remains
        # differentiable so the auxiliary loss updates only the router scores.
        selection_share = self.selection_share.detach()
        loss = self.pre_topk_prob_sum.shape[0] * torch.sum(selection_share * self.pre_topk_mean_prob)
        return loss * (self.valid_token_count > 0).to(loss.dtype)


def route_topk(router_logits: torch.Tensor, top_k: int, temperature: float) -> RoutingDecision:
    """Compute FP32 softmax probabilities and normalized Top-K weights."""

    if router_logits.ndim != 2:
        raise ValueError(f"router_logits must have shape [token, expert], got {tuple(router_logits.shape)}")
    if not torch.is_floating_point(router_logits):
        raise TypeError(f"router_logits must be floating point, got {router_logits.dtype}")

    num_experts = router_logits.shape[-1]
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must satisfy 1 <= top_k <= {num_experts}, got {top_k}")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError(f"temperature must be a real number, got {type(temperature).__name__}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and greater than 0, got {temperature}")

    # Keep router normalization in FP32 even when model parameters and
    # activations use a lower-precision dtype.
    pre_topk_probs = torch.softmax(router_logits.float() / temperature, dim=-1)
    selected_probs, topk_indices = torch.topk(pre_topk_probs, k=top_k, dim=-1)
    post_topk_weights = selected_probs / selected_probs.sum(dim=-1, keepdim=True)
    return RoutingDecision(
        pre_topk_probs=pre_topk_probs,
        topk_indices=topk_indices,
        post_topk_weights=post_topk_weights,
    )


def _normalized_entropy(probs: torch.Tensor, num_choices: int) -> torch.Tensor:
    if num_choices <= 1:
        return torch.zeros(probs.shape[0], dtype=probs.dtype, device=probs.device)

    log_probs = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
    return -(probs * log_probs).sum(dim=-1) / math.log(num_choices)


def compute_routing_statistics(
    decision: RoutingDecision,
    response_mask: torch.Tensor,
) -> RoutingStatistics:
    """Collect response-token routing values without converting tensors to
    Python."""

    _validate_routing_decision(decision)
    if response_mask.ndim != 1 or response_mask.shape[0] != decision.pre_topk_probs.shape[0]:
        raise ValueError(
            f"response_mask must have shape [token] matching the routing decision, got {tuple(response_mask.shape)}"
        )
    if response_mask.device != decision.pre_topk_probs.device:
        raise ValueError("response_mask and routing tensors must be on the same device")

    probs = decision.pre_topk_probs
    indices = decision.topk_indices
    weights = decision.post_topk_weights
    num_experts = decision.num_experts
    top_k = decision.top_k

    valid_mask = response_mask.to(dtype=torch.bool)
    token_weights = valid_mask.to(dtype=probs.dtype)
    selected_token_weights = token_weights.unsqueeze(-1).expand(-1, top_k)

    pre_topk_prob_sum = torch.sum(probs * token_weights.unsqueeze(-1), dim=0)

    # Accumulate selected experts directly instead of materializing another
    # token-by-expert tensor for every metric.
    post_topk_weight_sum = torch.zeros(num_experts, dtype=probs.dtype, device=probs.device)
    post_topk_weight_sum.scatter_add_(
        0,
        indices.reshape(-1),
        (weights * selected_token_weights).reshape(-1),
    )

    selection_count = torch.zeros_like(post_topk_weight_sum)
    selection_count.scatter_add_(0, indices.reshape(-1), selected_token_weights.reshape(-1))

    top1_count = torch.zeros_like(post_topk_weight_sum)
    top1_count.scatter_add_(0, indices[:, 0], token_weights)

    pre_topk_entropy_sum = torch.sum(_normalized_entropy(probs, num_experts) * token_weights)
    post_topk_entropy_sum = torch.sum(_normalized_entropy(weights, top_k) * token_weights)

    return RoutingStatistics(
        pre_topk_prob_sum=pre_topk_prob_sum,
        post_topk_weight_sum=post_topk_weight_sum,
        selection_count=selection_count,
        top1_count=top1_count,
        pre_topk_entropy_sum=pre_topk_entropy_sum,
        post_topk_entropy_sum=post_topk_entropy_sum,
        valid_token_count=token_weights.sum(),
        top_k=top_k,
    )


def mean_routing_balance_loss(statistics: Sequence[RoutingStatistics]) -> torch.Tensor:
    """Average independently computed balance losses across routed sites."""

    if not statistics:
        raise ValueError("statistics must contain at least one routed site")
    return torch.stack([site_statistics.balance_loss for site_statistics in statistics]).mean()


def _validate_routing_decision(decision: RoutingDecision) -> None:
    probs = decision.pre_topk_probs
    indices = decision.topk_indices
    weights = decision.post_topk_weights

    if probs.ndim != 2 or indices.ndim != 2 or weights.ndim != 2:
        raise ValueError("routing decision tensors must all be two-dimensional")
    if indices.shape != weights.shape or indices.shape[0] != probs.shape[0]:
        raise ValueError("routing decision tensor shapes do not agree")
    if indices.shape[1] < 1 or indices.shape[1] > probs.shape[1]:
        raise ValueError("routing decision has an invalid Top-K dimension")
    if probs.device != indices.device or probs.device != weights.device:
        raise ValueError("routing decision tensors must be on the same device")
    if probs.dtype != torch.float32 or weights.dtype != torch.float32:
        raise TypeError("routing probabilities and weights must be FP32")
    if indices.dtype != torch.long:
        raise TypeError("topk_indices must use torch.long")


def _validate_dense_executor_inputs(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    decision: RoutingDecision,
    scale: float,
    parallel_context: RoutedLoRAParallelContext | None,
) -> None:
    _validate_routing_decision(decision)
    if x.ndim < 2:
        raise ValueError(f"x must have at least two dimensions, got {tuple(x.shape)}")
    if lora_a.ndim != 3 or lora_b.ndim != 3:
        raise ValueError("lora_a and lora_b must have shapes [expert, rank, input] and [expert, output, rank]")
    if not torch.is_floating_point(x) or not torch.is_floating_point(lora_a) or not torch.is_floating_point(lora_b):
        raise TypeError("x, lora_a, and lora_b must be floating-point tensors")
    if x.dtype != lora_a.dtype or x.dtype != lora_b.dtype:
        raise TypeError(
            f"x, lora_a, and lora_b must use the same dtype, got {x.dtype}, {lora_a.dtype}, {lora_b.dtype}"
        )
    if x.device != lora_a.device or x.device != lora_b.device or x.device != decision.pre_topk_probs.device:
        raise ValueError("x, expert parameters, and routing tensors must be on the same device")

    num_experts, rank, input_size = lora_a.shape
    if lora_b.shape[0] != num_experts or lora_b.shape[2] != rank:
        raise ValueError(f"lora_a and lora_b expert/rank dimensions do not agree: {lora_a.shape} and {lora_b.shape}")
    if x.shape[-1] != input_size:
        raise ValueError(f"x hidden size {x.shape[-1]} does not match lora_a input size {input_size}")
    if decision.num_experts != num_experts:
        raise ValueError(
            f"routing decision has {decision.num_experts} experts but adapter parameters have {num_experts}"
        )
    token_count = math.prod(x.shape[:-1])
    if decision.pre_topk_probs.shape[0] != token_count:
        raise ValueError(
            f"routing decision has {decision.pre_topk_probs.shape[0]} tokens but x contains {token_count}"
        )
    _validate_finite_number("scale", scale)

    # TP collectives depend on the target layer contract and are implemented
    # by backend executors. The shared dense executor handles local tensors.
    if parallel_context is not None and parallel_context.tensor_parallel_group is not None:
        raise NotImplementedError("DenseRoutedLoRAExecutor does not perform tensor-parallel collectives")


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0, got {value}")


def _validate_finite_number(
    name: str,
    value: float,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if minimum is None:
        return
    if minimum_inclusive and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if not minimum_inclusive and value <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}, got {value}")
