# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal


CrossVersionKVPauseMode = Literal["abort", "retract", "in_place"]


def cross_version_kv_enabled(args: object) -> bool:
    return bool(getattr(args, "enable_cross_version_kv_continuation", False))


def validate_cross_version_kv_args(args: object, *, sync_intent_enabled: bool) -> None:
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
    if not sync_intent_enabled:
        raise ValueError("--enable-cross-version-kv-continuation currently requires RELAX_SYNC_INTENT_POLICY=1.")


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
