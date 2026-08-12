# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Literal


CrossVersionKVPauseMode = Literal["abort", "retract", "in_place"]
CROSS_VERSION_KV_ABORT_RETRY_INTERVAL_ENV = "RELAX_CROSS_VERSION_KV_ABORT_RETRY_INTERVAL_SECONDS"
CROSS_VERSION_KV_ABORT_TIMEOUT_ENV = "RELAX_CROSS_VERSION_KV_ABORT_TIMEOUT_SECONDS"
CROSS_VERSION_KV_PROTECTED_DRAIN_TIMEOUT_ENV = "RELAX_CROSS_VERSION_KV_PROTECTED_DRAIN_TIMEOUT_SECONDS"


def _positive_float_env(name: str, default: str) -> float:
    raw_value = os.environ.get(name, default)
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number, got {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def cross_version_kv_abort_retry_interval_seconds() -> float:
    return _positive_float_env(CROSS_VERSION_KV_ABORT_RETRY_INTERVAL_ENV, "0.5")


def cross_version_kv_abort_timeout_seconds() -> float:
    return _positive_float_env(CROSS_VERSION_KV_ABORT_TIMEOUT_ENV, "15")


def cross_version_kv_protected_drain_timeout_seconds() -> float:
    return _positive_float_env(CROSS_VERSION_KV_PROTECTED_DRAIN_TIMEOUT_ENV, "600")


def plan_dp_aligned_extra_groups(
    *,
    current_groups: int,
    available_extra_groups: int,
    dp_size: int,
) -> int:
    if min(current_groups, available_extra_groups) < 0:
        raise ValueError("group counts must be non-negative")
    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    aligned_total = ((current_groups + available_extra_groups) // dp_size) * dp_size
    return max(aligned_total - current_groups, 0)


def validate_disjoint_rollout_groups(
    accepted_groups: list[list[object]],
    buffered_groups: list[list[object]],
) -> None:
    accepted_ids = {id(sample) for group in accepted_groups for sample in group}
    buffered_ids = {id(sample) for group in buffered_groups for sample in group}
    overlap = accepted_ids & buffered_ids
    if overlap:
        raise RuntimeError(f"Rollout samples have dual accepted/buffer ownership: {len(overlap)} samples")


def plan_baseline_window_fetch(
    *,
    resident_groups: int,
    submit_target_groups: int,
    fetch_batch_groups: int,
) -> int:
    """Mirror the ordinary rollout's fixed-batch oversampling behavior."""
    if min(resident_groups, submit_target_groups) < 0:
        raise ValueError("resident and submit target groups must be non-negative")
    if fetch_batch_groups <= 0:
        raise ValueError("fetch batch groups must be positive")
    if submit_target_groups == 0 or resident_groups >= submit_target_groups:
        return 0
    return fetch_batch_groups


def plan_carry_aware_oversampling_seed(
    *,
    oversampling_envelope_groups: int,
    adopted_current_groups: int,
    missing_debt_groups: int,
) -> int:
    """Preserve the baseline oversampling envelope after adopting carry."""
    if min(oversampling_envelope_groups, adopted_current_groups, missing_debt_groups) < 0:
        raise ValueError("oversampling envelope, adopted current, and missing debt groups must be non-negative")
    if oversampling_envelope_groups == 0:
        raise ValueError("oversampling_envelope_groups must be positive")
    return missing_debt_groups + max(oversampling_envelope_groups - adopted_current_groups, 0)


def mark_work_origin(
    samples: list[list[object]],
    old_debt_groups: int,
) -> None:
    for group_index, group in enumerate(samples):
        origin = "old_debt" if group_index < old_debt_groups else "fresh"
        for sample in group:
            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                setattr(sample, "metadata", metadata)
            metadata["work_origin"] = origin


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
    if not (getattr(args, "use_tis", False) or getattr(args, "use_rollout_logprobs", False)):
        raise ValueError(
            "--enable-cross-version-kv-continuation requires --use-tis or --use-rollout-logprobs "
            "to preserve the request-level behavior policy; --keep-old-actor cannot represent "
            "a request that continues decoding across weight publications."
        )
    update_weights_interval = int(getattr(args, "update_weights_interval", 1))
    if update_weights_interval < 1:
        raise ValueError("--update-weights-interval must be >= 1.")
    max_gap = int(getattr(args, "cross_version_kv_max_gap", 0))
    if max_gap < 1:
        raise ValueError("--cross-version-kv-max-gap must be >= 1.")
    max_staleness = int(getattr(args, "max_staleness"))
    effective_actor_update_gap = max_gap * update_weights_interval
    if effective_actor_update_gap > max_staleness:
        raise ValueError(
            "--cross-version-kv-max-gap * --update-weights-interval must not exceed "
            f"--max-staleness ({max_gap} * {update_weights_interval} > {max_staleness})."
        )


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


def clear_cross_version_kv_task_markers(group: Sequence[object]) -> bool:
    """Clear markers that are valid only while the original task is alive."""
    for sample in group:
        metadata = getattr(sample, "metadata", {})
        metadata.pop("targeted_retirement_aborted", None)
        metadata.pop("cross_version_kv_carried", None)
    return False


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
