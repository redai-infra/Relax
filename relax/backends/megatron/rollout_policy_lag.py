# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Scheduling helpers for fixed-lag rollout policy snapshots."""

from typing import Protocol


ROLLOUT_POLICY_TAG = "rollout_policy"


class _TensorBackuperLike(Protocol):
    def copy(self, *, src_tag: str, dst_tag: str) -> None:
        """Copy one stored tensor snapshot to another tag."""


def validate_update_weights_interval(update_weights_interval: int) -> int:
    """Validate and return the rollout weight-update interval."""
    if update_weights_interval < 1:
        raise ValueError(f"update_weights_interval must be a positive integer, got {update_weights_interval}")
    return update_weights_interval


def rollout_weights_tag(update_weights_interval: int) -> str:
    """Return the TensorBackuper tag whose weights should be pushed to
    rollout."""
    interval = validate_update_weights_interval(update_weights_interval)
    return ROLLOUT_POLICY_TAG if interval > 1 else "actor"


def should_refresh_rollout_policy(
    rollout_id: int,
    update_weights_interval: int,
    num_rollout: int,
) -> bool:
    """Return whether the fixed rollout snapshot should adopt the trained
    actor.

    The final step always refreshes so end-of-training evaluation sees the
    latest actor even when the step is not an interval boundary.
    """
    interval = validate_update_weights_interval(update_weights_interval)
    completed_steps = rollout_id + 1
    return interval == 1 or completed_steps % interval == 0 or completed_steps == num_rollout


def maybe_refresh_rollout_policy(
    weights_backuper: _TensorBackuperLike,
    rollout_id: int,
    update_weights_interval: int,
    num_rollout: int,
) -> bool:
    """Refresh a fixed rollout snapshot when its schedule reaches a
    boundary."""
    interval = validate_update_weights_interval(update_weights_interval)
    if interval == 1 or not should_refresh_rollout_policy(rollout_id, interval, num_rollout):
        return False

    weights_backuper.copy(src_tag="actor", dst_tag=ROLLOUT_POLICY_TAG)
    return True
