# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Backend-independent routing primitives for Mixture-of-LoRA."""

import math
from dataclasses import dataclass

import torch


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
