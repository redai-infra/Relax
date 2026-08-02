# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest


try:
    from relax.backends.megatron.actor import _should_publish_hybrid_weights
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.megatron.actor unavailable: {_exc}", allow_module_level=True)


def test_hybrid_weight_publication_interval_one_preserves_existing_behavior() -> None:
    assert all(_should_publish_hybrid_weights(step, 1, 5) for step in range(5))


def test_hybrid_weight_publication_interval_two_uses_completed_step_boundary() -> None:
    decisions = [_should_publish_hybrid_weights(step, 2, 6) for step in range(6)]

    assert decisions == [False, True, False, True, False, True]


def test_hybrid_weight_publication_forces_final_step() -> None:
    assert _should_publish_hybrid_weights(4, 3, 5)


def test_hybrid_weight_publication_forces_configured_evaluation_when_epoch_is_unknown() -> None:
    assert _should_publish_hybrid_weights(
        0,
        2,
        20,
        evaluation_configured=True,
        eval_interval=10,
        num_rollout_per_epoch=None,
    )


def test_hybrid_weight_publication_uses_known_evaluation_boundaries() -> None:
    assert not _should_publish_hybrid_weights(
        2,
        5,
        20,
        evaluation_configured=True,
        eval_interval=10,
        num_rollout_per_epoch=4,
    )
    assert _should_publish_hybrid_weights(
        3,
        5,
        20,
        evaluation_configured=True,
        eval_interval=10,
        num_rollout_per_epoch=4,
    )


def test_hybrid_weight_publication_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _should_publish_hybrid_weights(0, 0, 20)
