# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor
from typing import Literal


CrossVersionKVPauseMode = Literal["abort", "retract", "in_place"]


@dataclass(frozen=True)
class JointCarryAdmitPlan:
    debt_remaining: int
    debt_admit_groups: int
    current_deficit: int
    carry_work_equivalents: float
    resident_cap: int
    current_reserve: int
    fresh_admit_groups: int
    work_overcommit_equivalents: float
    reason: str


def plan_joint_carry_admission(
    *,
    phase: str | None,
    debt_target: int,
    debt_committed: int,
    current_target: int,
    current_committed: int,
    resident_group_ids: Sequence[str],
    carry_groups: Sequence[tuple[str, int]],
    debt_eligible_group_ids: Sequence[str],
    carry_current_group_ids: Sequence[str],
    fresh_current_group_ids: Sequence[str],
    rollout_batch_size: int,
    max_response_length: int,
    strict_retry_pending: bool = False,
    final_backfill: bool = False,
) -> JointCarryAdmitPlan:
    """Plan debt and current admission from one shared resident/work budget."""
    counts = (
        debt_target,
        debt_committed,
        current_target,
        current_committed,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise ValueError("Joint carry-admit counts must be integers")
    if min(counts) < 0:
        raise ValueError("Joint carry-admit counts must be non-negative")
    if (
        isinstance(rollout_batch_size, bool)
        or not isinstance(rollout_batch_size, int)
        or isinstance(max_response_length, bool)
        or not isinstance(max_response_length, int)
    ):
        raise ValueError("Joint carry-admit size limits must be integers")
    if rollout_batch_size <= 0 or max_response_length <= 0:
        raise ValueError("Joint carry-admit size limits must be positive")
    if debt_committed > debt_target:
        raise ValueError("debt_committed must not exceed debt_target")
    id_sequences = (
        resident_group_ids,
        debt_eligible_group_ids,
        carry_current_group_ids,
        fresh_current_group_ids,
    )
    if any(any(not isinstance(group_id, str) or not group_id for group_id in ids) for ids in id_sequences):
        raise ValueError("Joint carry-admit group IDs must be non-empty strings")
    if any(len(set(ids)) != len(ids) for ids in id_sequences):
        raise ValueError("Joint carry-admit group IDs must be unique within each set")

    carry_ids: list[str] = []
    normalized_remaining: list[int] = []
    for group_id, value in carry_groups:
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("Joint carry-admit carry IDs must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("carry remaining-token estimates must be integers")
        if value < 0 or value > max_response_length:
            raise ValueError("carry remaining-token estimates must be within the response bound")
        carry_ids.append(group_id)
        normalized_remaining.append(value)
    if len(set(carry_ids)) != len(carry_ids):
        raise ValueError("Joint carry-admit carry IDs must be unique")

    resident_set = set(resident_group_ids)
    carry_set = set(carry_ids)
    debt_set = set(debt_eligible_group_ids)
    carry_current_set = set(carry_current_group_ids)
    fresh_current_set = set(fresh_current_group_ids)
    if not carry_set <= resident_set or not fresh_current_set <= resident_set:
        raise ValueError("carry and fresh current groups must be resident")
    if not debt_set <= carry_set or not carry_current_set <= carry_set:
        raise ValueError("classified carry groups must be subsets of the carry set")
    if debt_set & carry_current_set:
        raise ValueError("debt-eligible and current carry groups must be disjoint")
    if carry_set & fresh_current_set:
        raise ValueError("carry and fresh current groups must be disjoint")

    debt_eligible_inflight = len(debt_set)
    carry_current_inflight = len(carry_current_set)
    fresh_current_inflight = len(fresh_current_set)
    resident_groups = len(resident_set)
    useful_current_inflight = carry_current_inflight + fresh_current_inflight
    if current_committed + useful_current_inflight > current_target:
        raise ValueError("current committed and inflight work exceed current_target")

    resident_cap = cross_version_kv_resident_cap(rollout_batch_size)
    if resident_groups > resident_cap:
        raise ValueError("resident_groups exceed the joint planner cap")

    debt_remaining = debt_target - debt_committed
    current_deficit = current_target - current_committed - useful_current_inflight
    carry_work = sum(value / max_response_length for value in normalized_remaining)
    free_slots = resident_cap - resident_groups
    work_slots = floor(max(rollout_batch_size - carry_work - fresh_current_inflight, 0))

    blocked_phase = phase in {"quiesce", "weight_sync"}
    if blocked_phase:
        debt_admit = 0
        fresh_admit = 0
        current_reserve = 0
        reason = f"phase:{phase}"
    else:
        uncovered_debt = max(debt_remaining - debt_eligible_inflight, 0)
        debt_admit = min(uncovered_debt, free_slots, work_slots)
        free_after_debt = free_slots - debt_admit
        work_after_debt = work_slots - debt_admit

        current_reserve = 0
        if (
            not final_backfill
            and not strict_retry_pending
            and uncovered_debt == debt_admit
            and useful_current_inflight == 0
            and current_deficit > 0
        ):
            reserve_cap = max(1, rollout_batch_size // 4)
            current_reserve = min(reserve_cap, current_deficit, free_after_debt)

        if final_backfill:
            fresh_admit = 0
            reason = "final_backfill"
        elif strict_retry_pending:
            fresh_admit = 0
            reason = "strict_retry"
        elif uncovered_debt > debt_admit:
            fresh_admit = 0
            reason = "debt_first"
        else:
            fresh_admit = min(
                current_deficit,
                free_after_debt,
                max(work_after_debt, current_reserve),
            )
            reason = "joint_budget"

    planned_work = carry_work + fresh_current_inflight + debt_admit + fresh_admit
    work_overcommit = max(0.0, planned_work - rollout_batch_size)
    return JointCarryAdmitPlan(
        debt_remaining=debt_remaining,
        debt_admit_groups=debt_admit,
        current_deficit=current_deficit,
        carry_work_equivalents=carry_work,
        resident_cap=resident_cap,
        current_reserve=current_reserve,
        fresh_admit_groups=fresh_admit,
        work_overcommit_equivalents=work_overcommit,
        reason=reason,
    )


def cross_version_kv_enabled(args: object) -> bool:
    return bool(getattr(args, "enable_cross_version_kv_continuation", False))


def validate_cross_version_kv_args(args: object) -> None:
    if not cross_version_kv_enabled(args):
        return
    if not getattr(args, "hybrid", False):
        raise ValueError("--enable-cross-version-kv-continuation requires --hybrid.")
    if not getattr(args, "fully_async", False):
        raise ValueError("--enable-cross-version-kv-continuation requires fully-async rollout.")
    if not getattr(args, "partial_rollout", False):
        raise ValueError("--enable-cross-version-kv-continuation requires --partial-rollout.")
    if not getattr(args, "colocate", False):
        raise ValueError("--enable-cross-version-kv-continuation currently requires colocated weight sync.")
    if getattr(args, "offload_rollout", False):
        raise ValueError("--enable-cross-version-kv-continuation is incompatible with rollout offload.")
    if getattr(args, "update_weights_interval", 1) != 1:
        raise ValueError(
            "--enable-cross-version-kv-continuation requires --update-weights-interval 1 "
            "so every Actor step still publishes weights."
        )
    max_gap = int(getattr(args, "cross_version_kv_max_gap", 0))
    if max_gap < 1:
        raise ValueError("--cross-version-kv-max-gap must be >= 1.")
    max_staleness = int(getattr(args, "max_staleness"))
    if max_gap > max_staleness:
        raise ValueError(f"--cross-version-kv-max-gap must not exceed --max-staleness ({max_gap} > {max_staleness}).")


def cross_version_kv_strict_refresh(weight_version: int, max_gap: int) -> bool:
    """Return whether this publication must rebuild all retained KV state."""
    if weight_version < 1:
        raise ValueError(f"weight_version must be positive, got {weight_version}")
    if max_gap < 1:
        raise ValueError(f"max_gap must be positive, got {max_gap}")
    return (weight_version - 1) % (max_gap + 1) == 0


def cross_version_kv_pause_mode(args: object, weight_version: int) -> CrossVersionKVPauseMode:
    if not cross_version_kv_enabled(args):
        return "abort"
    max_gap = int(getattr(args, "cross_version_kv_max_gap"))
    if cross_version_kv_strict_refresh(weight_version, max_gap):
        return "abort"
    return "in_place"


def cross_version_kv_group_ready_for_finalize(group: Sequence[object]) -> bool:
    """Return whether group-level reward and post-processing are safe."""
    for sample in group:
        status = getattr(sample, "status", None)
        if getattr(status, "value", status) == "aborted":
            return False
    return True


def cross_version_kv_group_requires_strict_retry(group: Sequence[object]) -> bool:
    return any(
        bool(
            getattr(sample, "metadata", {}).get("cross_version_kv_carried")
            or getattr(sample, "metadata", {}).get("targeted_retirement_aborted")
        )
        for sample in group
    )


def cross_version_kv_resident_cap(rollout_batch_size: int) -> int:
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be positive")
    return rollout_batch_size + max(1, rollout_batch_size // 4)


def count_cross_version_kv_progress_hedge_groups(groups: Sequence[Sequence[object]]) -> int:
    return sum(
        any(bool(getattr(sample, "metadata", {}).get("cross_version_kv_progress_hedge")) for sample in group)
        for group in groups
    )


def clear_cross_version_kv_progress_hedge_marker(group: Sequence[object]) -> bool:
    was_hedge = False
    for sample in group:
        metadata = getattr(sample, "metadata", {})
        was_hedge = bool(metadata.pop("cross_version_kv_progress_hedge", False)) or was_hedge
    return was_hedge


def clear_cross_version_kv_task_markers(group: Sequence[object]) -> bool:
    """Clear markers that are valid only while the original task is alive."""
    was_hedge = clear_cross_version_kv_progress_hedge_marker(group)
    for sample in group:
        metadata = getattr(sample, "metadata", {})
        metadata.pop("targeted_retirement_aborted", None)
        metadata.pop("cross_version_kv_carried", None)
    return was_hedge


def estimate_cross_version_kv_group_remaining_tokens(
    group: Sequence[object],
    *,
    recent_completed_response_lengths: Sequence[int],
    max_response_length: int,
) -> int:
    """Estimate group tail from the conditional residual-length
    distribution."""
    if max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    history = sorted(
        min(int(length), max_response_length) for length in recent_completed_response_lengths if int(length) >= 0
    )
    estimated_remaining = 0
    for sample in group:
        status = getattr(sample, "status", None)
        status_value = getattr(status, "value", status)
        if status_value in {"completed", "truncated"}:
            continue
        current_length = min(max(int(getattr(sample, "response_length", 0)), 0), max_response_length)
        residuals = [length - current_length for length in history if length > current_length]
        if len(residuals) >= 4:
            # A bounded upper-middle estimate avoids treating surviving requests
            # as nearly done while remaining robust to one extreme completion.
            residuals.sort()
            estimate = residuals[(3 * (len(residuals) - 1)) // 4]
        else:
            estimate = max_response_length - current_length
        estimated_remaining = max(estimated_remaining, estimate)
    return estimated_remaining


def plan_cross_version_kv_progress_hedge(
    *,
    adopted_groups: int,
    adopted_debt_groups: int,
    resident_groups: int,
    remaining_fresh_groups: int,
    rollout_batch_size: int,
    strict_retry_pending: bool = False,
    estimated_remaining_tokens: int | None = None,
    max_response_length: int | None = None,
) -> int:
    """Reserve bounded current-partition progress behind carried fresh work."""
    values = (
        adopted_groups,
        adopted_debt_groups,
        resident_groups,
        remaining_fresh_groups,
        rollout_batch_size,
    )
    if min(values) < 0:
        raise ValueError("KV continuation progress hedge inputs must be non-negative")
    if rollout_batch_size == 0:
        raise ValueError("rollout_batch_size must be positive")
    if adopted_debt_groups > adopted_groups:
        raise ValueError("adopted_debt_groups must not exceed adopted_groups")
    if estimated_remaining_tokens is not None and estimated_remaining_tokens < 0:
        raise ValueError("estimated_remaining_tokens must be non-negative")
    if estimated_remaining_tokens is not None and (max_response_length is None or max_response_length <= 0):
        raise ValueError("max_response_length must be positive with an estimate")

    adopted_fresh_groups = adopted_groups - adopted_debt_groups
    if strict_retry_pending or adopted_fresh_groups == 0 or remaining_fresh_groups == 0:
        return 0

    max_resident_groups = cross_version_kv_resident_cap(rollout_batch_size)
    hedge_cap = max_resident_groups - rollout_batch_size
    if estimated_remaining_tokens is not None:
        pressure_groups = (estimated_remaining_tokens + max_response_length - 1) // max_response_length
        hedge_cap = min(hedge_cap, pressure_groups)
    free_slots = max(max_resident_groups - resident_groups, 0)
    return min(hedge_cap, adopted_fresh_groups, remaining_fresh_groups, free_slots)


def mark_cross_version_kv_carry(
    groups: Sequence[Sequence[object]],
    *,
    num_old_samples: int,
) -> int:
    if num_old_samples < 0:
        raise ValueError(f"num_old_samples must be non-negative, got {num_old_samples}")

    adopted_debt = min(num_old_samples, len(groups))
    for group_index, group in enumerate(groups):
        origin = "old_debt" if group_index < adopted_debt else "cross_version_fresh"
        for sample in group:
            metadata = getattr(sample, "metadata")
            status = getattr(sample, "status", None)
            status_value = getattr(status, "value", status)
            if status_value == "aborted":
                # A strict publication aborts the retained request before the
                # next physical rollout adopts it. Its retry starts a fresh KV
                # epoch, so the old epoch must not count another boundary.
                carryovers = 0
            else:
                # Physical rollouts may run ahead of Actor publications, so
                # this is observability only. The hard version-gap bound is
                # enforced by the publication-side strict flush cycle.
                carryovers = int(metadata.get("cross_version_kv_carryovers", 0)) + 1
            metadata["work_origin"] = origin
            metadata["cross_version_kv_carried"] = True
            metadata["cross_version_kv_carryovers"] = carryovers
    return adopted_debt
