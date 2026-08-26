# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Scheduling helpers for periodic rollout policy snapshots."""

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


def compute_rollout_policy_age_rollouts(
    current_rollout_id: int,
    snapshot_rollout_id: int,
) -> int:
    """Return the age of the behavior snapshot used by a training batch.

    The metric is emitted before the post-batch rollout snapshot refresh. At a
    refresh boundary the just-trained batch therefore still reports the age of
    the snapshot that generated it; the next batch observes the refreshed
    snapshot.
    """
    if current_rollout_id < 0:
        raise ValueError("current_rollout_id must be non-negative")
    if snapshot_rollout_id < 0:
        raise ValueError("snapshot_rollout_id must be non-negative")
    if current_rollout_id < snapshot_rollout_id:
        raise ValueError("current_rollout_id cannot precede snapshot_rollout_id")
    return current_rollout_id - snapshot_rollout_id


def initial_rollout_policy_snapshot_rollout(
    backend_start_rollout_id: int,
    *,
    configured_start_rollout_id: int | None = None,
    is_megatron_resume: bool = False,
) -> int:
    """Return the snapshot version aligned with the rollout service.

    A cold HuggingFace load reports Megatron iteration zero, so the backend's
    next-step value is one while the service correctly starts at its configured
    rollout zero. For a cold load, prefer that configured service value; a
    native Megatron resume keeps its backend-derived value.
    """
    snapshot_rollout_id = (
        backend_start_rollout_id
        if is_megatron_resume or configured_start_rollout_id is None
        else configured_start_rollout_id
    )
    if snapshot_rollout_id < 0:
        raise ValueError("start_rollout_id must be non-negative")
    return snapshot_rollout_id


def validate_p3o_periodic_snapshot_resume(
    *,
    advantage_estimator: str | None,
    update_weights_interval: int,
    is_megatron_resume: bool,
) -> None:
    """Reject P3O resumes that cannot restore the behavior-policy snapshot.

    A periodic rollout-policy snapshot is held only in the in-memory
    ``TensorBackuper``. Resuming currently reconstructs it from the current
    actor, which changes the behavior policy for the first resumed rollout.
    P3O's importance ratio cannot silently absorb that semantic change.
    """
    interval = validate_update_weights_interval(update_weights_interval)
    if advantage_estimator == "p3o" and interval > 1 and is_megatron_resume:
        raise ValueError(
            "P3O cannot resume exactly with update_weights_interval > 1 because the periodic rollout-policy "
            "snapshot is not checkpointed. Start a fresh run, use update_weights_interval=1, or add "
            "snapshot checkpointing."
        )


def build_rollout_policy_age_metrics(
    *,
    current_rollout_id: int,
    rollout_policy_snapshot_rollout: int,
) -> dict[str, int]:
    """Build rollout-unit policy-age metrics for one training batch."""
    return {
        "train/current_rollout_id": current_rollout_id,
        "train/rollout_policy_snapshot_rollout": rollout_policy_snapshot_rollout,
        "train/p3o/rollout_policy_age_rollouts": compute_rollout_policy_age_rollouts(
            current_rollout_id,
            rollout_policy_snapshot_rollout,
        ),
    }


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
