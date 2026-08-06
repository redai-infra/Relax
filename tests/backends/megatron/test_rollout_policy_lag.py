# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for periodic rollout policy snapshot scheduling."""

import pytest

from relax.backends.megatron.rollout_policy_lag import (
    ROLLOUT_POLICY_TAG,
    maybe_refresh_rollout_policy,
    rollout_weights_tag,
    should_refresh_rollout_policy,
    validate_update_weights_interval,
)


class _RecordingBackuper:
    def __init__(self):
        self.copies = []

    def copy(self, *, src_tag: str, dst_tag: str) -> None:
        self.copies.append((src_tag, dst_tag))


def test_rollout_policy_lag_interval_one_preserves_actor_updates():
    assert rollout_weights_tag(1) == "actor"
    assert all(should_refresh_rollout_policy(step, 1, 5) for step in range(5))


def test_rollout_policy_lag_interval_three_refreshes_boundaries_and_final_step():
    refreshes = [should_refresh_rollout_policy(step, 3, 8) for step in range(8)]

    assert rollout_weights_tag(3) == ROLLOUT_POLICY_TAG
    assert refreshes == [False, False, True, False, False, True, False, True]


def test_rollout_policy_lag_copies_only_at_scheduled_boundaries():
    backuper = _RecordingBackuper()

    refreshed = [maybe_refresh_rollout_policy(backuper, step, 3, 8) for step in range(8)]

    assert refreshed == [False, False, True, False, False, True, False, True]
    assert backuper.copies == [("actor", ROLLOUT_POLICY_TAG)] * 3


@pytest.mark.parametrize("interval", [0, -1])
def test_rollout_policy_lag_rejects_non_positive_intervals(interval):
    with pytest.raises(ValueError, match="positive integer"):
        validate_update_weights_interval(interval)
