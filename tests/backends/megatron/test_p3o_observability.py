# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Behavior tests for P3O rollout-policy age observability."""

import pytest

from relax.backends.megatron.rollout_policy_lag import (
    ROLLOUT_POLICY_TAG,
    build_rollout_policy_age_metrics,
    compute_rollout_policy_age_rollouts,
    initial_rollout_policy_snapshot_rollout,
    maybe_refresh_rollout_policy,
    rollout_weights_tag,
    should_refresh_rollout_policy,
    validate_update_weights_interval,
)


class _RecordingBackuper:
    def __init__(self) -> None:
        self.copies: list[tuple[str, str]] = []

    def copy(self, *, src_tag: str, dst_tag: str) -> None:
        self.copies.append((src_tag, dst_tag))


@pytest.mark.parametrize("interval", [0, -1, -10])
def test_rollout_policy_interval_rejects_invalid_values(interval):
    with pytest.raises(ValueError, match="positive integer"):
        validate_update_weights_interval(interval)


def test_rollout_policy_age_uses_rollout_units():
    assert compute_rollout_policy_age_rollouts(15, 11) == 4


@pytest.mark.parametrize(
    ("current_rollout_id", "snapshot_rollout_id", "message"),
    [
        (-1, 0, "current_rollout_id"),
        (0, -1, "snapshot_rollout_id"),
        (2, 3, "cannot precede"),
    ],
)
def test_rollout_policy_age_rejects_invalid_versions(current_rollout_id, snapshot_rollout_id, message):
    with pytest.raises(ValueError, match=message):
        compute_rollout_policy_age_rollouts(current_rollout_id, snapshot_rollout_id)


def test_rollout_policy_age_interval_three_sequence():
    snapshot_rollout = 0
    observed = []
    backuper = _RecordingBackuper()

    for rollout_id in range(6):
        observed.append(compute_rollout_policy_age_rollouts(rollout_id, snapshot_rollout))
        if maybe_refresh_rollout_policy(backuper, rollout_id, 3, 6):
            snapshot_rollout = rollout_id + 1

    assert observed == [0, 1, 2, 0, 1, 2]
    assert backuper.copies == [("actor", ROLLOUT_POLICY_TAG), ("actor", ROLLOUT_POLICY_TAG)]


def test_rollout_policy_snapshot_initializes_for_fresh_and_resumed_runs():
    assert initial_rollout_policy_snapshot_rollout(0) == 0
    assert initial_rollout_policy_snapshot_rollout(101) == 101
    assert compute_rollout_policy_age_rollouts(101, initial_rollout_policy_snapshot_rollout(101)) == 0


def test_rollout_policy_snapshot_rejects_invalid_resume_version():
    with pytest.raises(ValueError, match="start_rollout_id"):
        initial_rollout_policy_snapshot_rollout(-1)


def test_rollout_policy_age_metrics_have_exact_keys_and_values():
    assert build_rollout_policy_age_metrics(current_rollout_id=7, rollout_policy_snapshot_rollout=5) == {
        "train/current_rollout_id": 7,
        "train/rollout_policy_snapshot_rollout": 5,
        "train/p3o/rollout_policy_age_rollouts": 2,
    }


@pytest.mark.parametrize("optimizer_steps_per_rollout", [1, 2, 8])
def test_rollout_policy_age_is_independent_of_optimizer_steps_per_rollout(optimizer_steps_per_rollout):
    optimizer_step = 4 * optimizer_steps_per_rollout

    metrics = build_rollout_policy_age_metrics(current_rollout_id=4, rollout_policy_snapshot_rollout=3)

    assert optimizer_step >= 4
    assert metrics["train/p3o/rollout_policy_age_rollouts"] == 1


def test_rollout_policy_refresh_calls_backuper_only_at_boundary():
    backuper = _RecordingBackuper()

    assert not maybe_refresh_rollout_policy(backuper, rollout_id=0, update_weights_interval=3, num_rollout=6)
    assert backuper.copies == []

    assert maybe_refresh_rollout_policy(backuper, rollout_id=2, update_weights_interval=3, num_rollout=6)
    assert backuper.copies == [("actor", ROLLOUT_POLICY_TAG)]


def test_on_policy_mode_uses_actor_and_refreshes_every_rollout():
    assert rollout_weights_tag(1) == "actor"
    assert should_refresh_rollout_policy(5, 1, 10)
