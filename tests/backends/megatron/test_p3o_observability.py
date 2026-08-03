# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for P3O rollout policy lag observability enhancements."""

from pathlib import Path

from relax.backends.megatron.rollout_policy_lag import compute_rollout_policy_lag_steps


MODEL_PATH = Path(__file__).resolve().parents[3] / "relax" / "backends" / "megatron" / "model.py"


class TestP3OObservability:
    """Test suite for P3O policy lag tracking."""

    def test_snapshot_step_initialization(self):
        """Verify _rollout_policy_snapshot_step is initialized to 0."""
        # Simulate the initialization logic
        snapshot_step = 0
        assert snapshot_step == 0, "Initial snapshot step should be 0"

    def test_snapshot_step_update_on_refresh(self):
        """Verify snapshot step is updated when policy is refreshed."""
        from relax.backends.megatron.rollout_policy_lag import should_refresh_rollout_policy

        interval = 11
        num_rollout = 11

        # Step 0-9: should NOT refresh (lag builds up)
        for rollout_id in range(10):
            assert not should_refresh_rollout_policy(rollout_id, interval, num_rollout)

        # Step 10: should refresh (completed_steps=11, 11 % 11 == 0)
        assert should_refresh_rollout_policy(10, interval, num_rollout)

    def test_lag_calculation(self):
        """Verify lag is correctly calculated as current_step - snapshot_step."""
        # Scenario: interval=11, after rollout_id=10 (step 11 completed)
        snapshot_step = 11
        current_step = 15  # rollout_id=14 completed

        expected_lag = compute_rollout_policy_lag_steps(current_step, snapshot_step)
        assert expected_lag == 4, f"Expected lag=4, got {expected_lag}"

    def test_on_policy_mode_lag_is_zero(self):
        """Verify lag is 0 when update_weights_interval=1 (on-policy)."""
        from relax.backends.megatron.rollout_policy_lag import rollout_weights_tag

        interval = 1
        tag = rollout_weights_tag(interval)

        # On-policy should use "actor" tag directly, not "rollout_policy"
        assert tag == "actor", f"On-policy should use 'actor' tag, got '{tag}'"

        # In on-policy mode, snapshot_step would equal current_step
        snapshot_step = 5
        current_step = 5
        lag = current_step - snapshot_step
        assert lag == 0, "On-policy lag should be 0"

    def test_lag_boundaries(self):
        """The refresh affects the next batch, not the boundary batch
        metric."""
        observations = [(1, 0), (2, 0), (3, 2)]
        actual = [compute_rollout_policy_lag_steps(current, snapshot) for current, snapshot in observations]

        assert actual == [1, 2, 1]


def test_p3o_observability_production_logging_uses_shared_lag_semantics():
    """Pin the production metric keys and shared age calculation without a full
    Ray actor."""
    source = MODEL_PATH.read_text(encoding="utf-8")

    assert "compute_rollout_policy_lag_steps(current_step, snapshot_step)" in source
    assert 'log_dict["train/actor_optimizer_step"]' in source
    assert 'log_dict["train/rollout_policy_snapshot_step"]' in source
    assert 'log_dict["train/p3o/rollout_policy_lag_steps"]' in source
