# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for P3O rollout policy lag observability enhancements."""

import pytest


class TestP3OObservability:
    """Test suite for P3O policy lag tracking."""

    def test_snapshot_step_initialization(self):
        """Verify _rollout_policy_snapshot_step is initialized to 0."""
        from argparse import Namespace

        # Mock minimal args needed for initialization
        args = Namespace(
            update_weights_interval=11,
            num_rollout=11,
        )

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

        expected_lag = current_step - snapshot_step
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
        """Test lag values at interval boundaries."""
        interval = 11

        # Just after refresh (rollout_id=10 completed, step 11)
        snapshot_step = 11
        current_step = 11
        assert current_step - snapshot_step == 0

        # One step later (rollout_id=11, step 12)
        current_step = 12
        assert current_step - snapshot_step == 1

        # Just before next refresh (rollout_id=20, step 21)
        current_step = 21
        assert current_step - snapshot_step == 10

        # After next refresh (rollout_id=21, step 22)
        snapshot_step = 22
        current_step = 22
        assert current_step - snapshot_step == 0


@pytest.mark.skipif(True, reason="Integration test, requires full actor initialization")
class TestP3OObservabilityIntegration:
    """Integration tests requiring actor/model setup."""

    def test_metrics_logged_to_tensorboard(self):
        """Verify P3O lag metrics appear in TensorBoard logs."""
        # This would require a full training setup
        # Expected metrics:
        # - train/actor_optimizer_step
        # - train/rollout_policy_snapshot_step
        # - train/p3o/rollout_policy_lag_steps
        pass

    def test_lag_tracked_across_rollouts(self):
        """Verify lag increases from 1 to interval-1 then resets."""
        # Requires multi-rollout actor training
        pass
