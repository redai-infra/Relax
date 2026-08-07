# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure helpers shared by DPO and pairwise reward-model training."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def require_tensor_condition(condition: torch.Tensor, message: str) -> None:
    """Raise on CPU immediately and enqueue a device-side assertion on CUDA."""
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _validate_same_shape(name: str, *values: torch.Tensor) -> None:
    if not values:
        raise ValueError(f"{name} requires at least one tensor")
    expected = values[0].shape
    if any(value.shape != expected for value in values[1:]):
        shapes = [tuple(value.shape) for value in values]
        raise ValueError(f"{name} tensors must have identical shapes, got {shapes}")


def build_causal_lm_labels(tokens: torch.Tensor, raw_loss_mask: torch.Tensor) -> torch.Tensor:
    """Build next-token labels from an unshifted completion-token mask."""
    if tokens.ndim != 1 or raw_loss_mask.ndim != 1:
        raise ValueError("tokens and raw_loss_mask must be one-dimensional")
    if tokens.shape != raw_loss_mask.shape:
        raise ValueError(f"tokens/raw_loss_mask shape mismatch: {tuple(tokens.shape)} vs {tuple(raw_loss_mask.shape)}")
    labels = torch.full_like(tokens, -100)
    if tokens.numel() > 1:
        supervised = raw_loss_mask[1:].to(dtype=torch.bool)
        labels[:-1][supervised] = tokens[1:][supervised]
    return labels


def dpo_pair_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    *,
    reference_chosen: torch.Tensor | None = None,
    reference_rejected: torch.Tensor | None = None,
    beta: float = 0.1,
    reference_free: bool = False,
) -> torch.Tensor:
    """Return unreduced sigmoid-DPO loss, one value per preference pair."""
    if beta <= 0:
        raise ValueError(f"DPO beta must be positive, got {beta}")
    _validate_same_shape("policy log-probabilities", policy_chosen, policy_rejected)
    policy_logratio = policy_chosen - policy_rejected
    if reference_free:
        if reference_chosen is not None or reference_rejected is not None:
            raise ValueError("reference-free DPO must not receive reference log-probabilities")
        reference_logratio = torch.zeros_like(policy_logratio)
    else:
        if reference_chosen is None or reference_rejected is None:
            raise ValueError("standard DPO requires chosen and rejected reference log-probabilities")
        _validate_same_shape(
            "reference log-probabilities",
            policy_chosen,
            reference_chosen,
            reference_rejected,
        )
        reference_logratio = reference_chosen - reference_rejected
    logits = beta * (policy_logratio - reference_logratio)
    require_tensor_condition(torch.isfinite(logits).all(), "DPO logits must contain only finite values")
    return -F.logsigmoid(logits)


def build_preference_pair_indices(
    branch_pair_ids: Sequence[int], branch_is_chosen: Sequence[bool]
) -> tuple[list[int], list[int]]:
    """Return chosen/rejected branch indices grouped by stable pair
    identity."""
    if len(branch_pair_ids) != len(branch_is_chosen):
        raise ValueError(
            "preference pair identity fields must be branch aligned: "
            f"{len(branch_pair_ids)} vs {len(branch_is_chosen)}"
        )
    if not branch_pair_ids:
        raise ValueError("DPO micro-batch must contain at least one preference pair")

    pairs: dict[int, dict[bool, int]] = {}
    order: list[int] = []
    for index, (raw_pair_id, raw_is_chosen) in enumerate(zip(branch_pair_ids, branch_is_chosen, strict=True)):
        pair_id = int(raw_pair_id)
        is_chosen = bool(raw_is_chosen)
        if pair_id not in pairs:
            pairs[pair_id] = {}
            order.append(pair_id)
        if is_chosen in pairs[pair_id]:
            branch = "chosen" if is_chosen else "rejected"
            raise ValueError(f"preference pair {pair_id!r} contains duplicate {branch} branches")
        pairs[pair_id][is_chosen] = index

    chosen_indices: list[int] = []
    rejected_indices: list[int] = []
    for pair_id in order:
        pair = pairs[pair_id]
        if set(pair) != {False, True}:
            raise ValueError(f"preference pair {pair_id!r} must contain exactly one chosen and one rejected branch")
        chosen_indices.append(pair[True])
        rejected_indices.append(pair[False])
    return chosen_indices, rejected_indices


def reward_model_pair_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    """Return unreduced Bradley-Terry loss, one value per preference pair."""
    _validate_same_shape("reward-model scores", chosen_scores, rejected_scores)
    margins = chosen_scores - rejected_scores
    if not torch.isfinite(margins).all():
        raise ValueError("reward-model margins must contain only finite values")
    return -F.logsigmoid(margins)


def select_packed_sequence_scores(
    logits: torch.Tensor,
    total_lengths: Sequence[int],
    score_positions: Sequence[int],
) -> torch.Tensor:
    """Select one scalar score per sequence from a CP=1 THD packed output."""
    if len(total_lengths) != len(score_positions):
        raise ValueError(
            f"total_lengths/score_positions length mismatch: {len(total_lengths)} vs {len(score_positions)}"
        )
    if logits.ndim == 3 and logits.shape[0] == 1 and logits.shape[-1] == 1:
        flat_logits = logits[0, :, 0]
    elif logits.ndim == 2 and logits.shape[-1] == 1:
        flat_logits = logits[:, 0]
    elif logits.ndim == 1:
        flat_logits = logits
    else:
        raise ValueError(f"reward-model logits must have shape [1,T,1], [T,1], or [T], got {tuple(logits.shape)}")

    offsets: list[int] = []
    cursor = 0
    for index, (length, position) in enumerate(zip(total_lengths, score_positions, strict=True)):
        length = int(length)
        position = int(position)
        if length <= 0:
            raise ValueError(f"sequence {index} has non-positive total length {length}")
        if not 0 <= position < length:
            raise ValueError(f"sequence {index} score position {position} is outside [0, {length})")
        offsets.append(cursor + position)
        cursor += length
    if cursor > flat_logits.numel():
        raise ValueError(f"packed reward logits contain {flat_logits.numel()} tokens, expected at least {cursor}")
    if not offsets:
        return flat_logits.new_empty((0,))
    return flat_logits[torch.tensor(offsets, device=flat_logits.device, dtype=torch.long)]


def pack_preference_pair_indices(
    costs: Sequence[int],
    pair_ids: Sequence[str],
    *,
    capacity: int,
) -> list[list[int]]:
    """Deterministic capacity-aware first-fit-decreasing pair packing."""
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if len(costs) != len(pair_ids):
        raise ValueError(f"costs/pair_ids length mismatch: {len(costs)} vs {len(pair_ids)}")
    normalized_costs = [int(cost) for cost in costs]
    for pair_id, cost in zip(pair_ids, normalized_costs, strict=True):
        if cost <= 0:
            raise ValueError(f"pair {pair_id!r} has non-positive cost {cost}")
        if cost > capacity:
            raise ValueError(f"oversize preference pair {pair_id!r} has cost {cost}, capacity={capacity}")

    order = sorted(range(len(normalized_costs)), key=lambda index: (-normalized_costs[index], str(pair_ids[index])))
    bins: list[list[int]] = []
    bin_costs: list[int] = []
    for index in order:
        cost = normalized_costs[index]
        for bin_index, bin_cost in enumerate(bin_costs):
            if bin_cost + cost <= capacity:
                bins[bin_index].append(index)
                bin_costs[bin_index] += cost
                break
        else:
            bins.append([index])
            bin_costs.append(cost)

    if sorted(index for group in bins for index in group) != list(range(len(normalized_costs))):
        raise RuntimeError("preference pair packer lost or duplicated pair indices")
    if any(sum(normalized_costs[index] for index in group) > capacity for group in bins):
        raise RuntimeError("preference pair packer produced an over-capacity micro-batch")
    return bins


__all__ = [
    "build_preference_pair_indices",
    "build_causal_lm_labels",
    "dpo_pair_loss",
    "pack_preference_pair_indices",
    "require_tensor_condition",
    "reward_model_pair_loss",
    "select_packed_sequence_scores",
]
