# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for P3O rollout policy-age observability."""

from pathlib import Path

from relax.backends.megatron.rollout_policy_lag import (
    compute_rollout_policy_age_rollouts,
    rollout_weights_tag,
    should_refresh_rollout_policy,
)


MODEL_PATH = Path(__file__).resolve().parents[3] / "relax" / "backends" / "megatron" / "model.py"


class TestP3OObservability:
    """Test suite for P3O policy-age tracking."""

    def test_snapshot_rollout_refresh_schedule(self):
        """The final rollout in an interval refreshes the snapshot."""
        interval = 11
        num_rollout = 11

        for rollout_id in range(10):
            assert not should_refresh_rollout_policy(rollout_id, interval, num_rollout)
        assert should_refresh_rollout_policy(10, interval, num_rollout)

    def test_age_calculation_uses_rollout_units(self):
        """Age is current rollout minus behavior-snapshot rollout."""
        assert compute_rollout_policy_age_rollouts(15, 11) == 4

    def test_on_policy_mode_uses_actor_and_zero_age(self):
        """Interval one pushes the actor and observes a fresh policy."""
        assert rollout_weights_tag(1) == "actor"
        assert compute_rollout_policy_age_rollouts(5, 5) == 0

    def test_age_boundaries(self):
        """The refresh affects the next batch, not the boundary batch
        metric."""
        observations = [(1, 0), (2, 0), (3, 2)]
        actual = [compute_rollout_policy_age_rollouts(current, snapshot) for current, snapshot in observations]

        assert actual == [1, 2, 1]


def test_p3o_observability_production_logging_uses_shared_age_semantics():
    """Pin production metric keys and shared age calculation without a Ray actor."""
    source = MODEL_PATH.read_text(encoding="utf-8")

    assert "compute_rollout_policy_age_rollouts(current_rollout, snapshot_rollout)" in source
    assert 'log_dict["train/rollout_policy_snapshot_rollout"]' in source
    assert 'log_dict["train/p3o/rollout_policy_age_rollouts"]' in source
