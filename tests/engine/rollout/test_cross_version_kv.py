# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.cross_version_kv import (
    clear_cross_version_kv_task_markers,
    cross_version_kv_abort_retry_interval_seconds,
    cross_version_kv_abort_timeout_seconds,
    cross_version_kv_group_ready_for_finalize,
    cross_version_kv_group_requires_strict_retry,
    cross_version_kv_pause_mode,
    cross_version_kv_protected_drain_timeout_seconds,
    cross_version_kv_strict_refresh,
    mark_cross_version_kv_carry,
    plan_baseline_window_fetch,
    plan_carry_aware_oversampling_seed,
    plan_dp_aligned_extra_groups,
    validate_cross_version_kv_args,
)


def test_cross_version_kv_abort_timeouts_preserve_experiment_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RELAX_CROSS_VERSION_KV_ABORT_RETRY_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("RELAX_CROSS_VERSION_KV_ABORT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("RELAX_CROSS_VERSION_KV_PROTECTED_DRAIN_TIMEOUT_SECONDS", raising=False)

    assert cross_version_kv_abort_retry_interval_seconds() == 0.5
    assert cross_version_kv_abort_timeout_seconds() == 15
    assert cross_version_kv_protected_drain_timeout_seconds() == 600


def test_cross_version_kv_publication_cycle_bounds_gap_two() -> None:
    args = SimpleNamespace(
        enable_cross_version_kv_continuation=True,
        cross_version_kv_max_gap=2,
    )

    modes = [cross_version_kv_pause_mode(args, version) for version in range(1, 8)]

    assert modes == ["abort", "in_place", "in_place", "abort", "in_place", "in_place", "abort"]


def test_cross_version_kv_disabled_preserves_strict_behavior() -> None:
    args = SimpleNamespace(
        enable_cross_version_kv_continuation=False,
        cross_version_kv_max_gap=2,
    )

    assert all(cross_version_kv_pause_mode(args, version) == "abort" for version in range(1, 5))


@pytest.mark.parametrize(("weight_version", "max_gap"), [(0, 2), (1, 0)])
def test_cross_version_kv_rejects_invalid_bounds(weight_version: int, max_gap: int) -> None:
    with pytest.raises(ValueError):
        cross_version_kv_strict_refresh(weight_version, max_gap)


def test_cross_version_kv_carry_assigns_debt_and_preserves_identity() -> None:
    debt = SimpleNamespace(index=1, metadata={})
    fresh = SimpleNamespace(index=2, metadata={})
    groups = [[debt], [fresh]]

    adopted = mark_cross_version_kv_carry(groups, num_old_samples=1)

    assert adopted == 1
    assert groups[0][0] is debt
    assert groups[1][0] is fresh
    assert debt.metadata == {
        "work_origin": "old_debt",
        "cross_version_kv_carried": True,
        "cross_version_kv_carryovers": 1,
    }
    assert fresh.metadata["work_origin"] == "cross_version_fresh"
    assert fresh.metadata["cross_version_kv_carryovers"] == 1


def test_strictly_aborted_carry_starts_new_kv_epoch() -> None:
    sample = SimpleNamespace(
        index=4,
        status="aborted",
        metadata={"cross_version_kv_carryovers": 4},
    )

    adopted = mark_cross_version_kv_carry([[sample]], num_old_samples=1)

    assert adopted == 1
    assert sample.metadata["cross_version_kv_carryovers"] == 0


def test_mixed_group_defers_group_finalize_until_retry_completes() -> None:
    completed = SimpleNamespace(status="completed")
    aborted = SimpleNamespace(status="aborted")

    assert not cross_version_kv_group_ready_for_finalize([completed, aborted])

    aborted.status = "completed"

    assert cross_version_kv_group_ready_for_finalize([completed, aborted])


def test_targeted_retirement_requires_strict_retry_before_carry_adoption() -> None:
    targeted = SimpleNamespace(metadata={"targeted_retirement_aborted": True})
    ordinary = SimpleNamespace(metadata={})

    assert cross_version_kv_group_requires_strict_retry([ordinary, targeted])
    assert not cross_version_kv_group_requires_strict_retry([ordinary])


def test_abort_to_buffer_clears_all_task_lifetime_markers() -> None:
    sample = SimpleNamespace(
        metadata={
            "cross_version_kv_carried": True,
            "cross_version_kv_carryovers": 2,
            "targeted_retirement_aborted": True,
        }
    )

    clear_cross_version_kv_task_markers([sample])
    assert "cross_version_kv_carried" not in sample.metadata
    assert "targeted_retirement_aborted" not in sample.metadata
    assert sample.metadata["cross_version_kv_carryovers"] == 2


@pytest.mark.parametrize(
    ("adopted_current_groups", "missing_debt_groups", "expected"),
    [(0, 0, 16), (8, 0, 8), (16, 0, 0), (4, 2, 14), (16, 3, 3)],
)
def test_carry_aware_seed_preserves_baseline_envelope(
    adopted_current_groups: int,
    missing_debt_groups: int,
    expected: int,
) -> None:
    assert (
        plan_carry_aware_oversampling_seed(
            oversampling_envelope_groups=16,
            adopted_current_groups=adopted_current_groups,
            missing_debt_groups=missing_debt_groups,
        )
        == expected
    )


def test_baseline_window_keeps_fixed_batch_semantics() -> None:
    assert plan_baseline_window_fetch(resident_groups=0, submit_target_groups=8, fetch_batch_groups=16) == 16
    assert plan_baseline_window_fetch(resident_groups=8, submit_target_groups=8, fetch_batch_groups=16) == 0


def test_dp_aligned_extra_groups_never_returns_partial_dp_wave() -> None:
    assert plan_dp_aligned_extra_groups(current_groups=1, available_extra_groups=7, dp_size=8) == 7
    assert plan_dp_aligned_extra_groups(current_groups=1, available_extra_groups=5, dp_size=8) == 0


def test_cross_version_kv_validation_accepts_task22_contract() -> None:
    args = SimpleNamespace(
        enable_cross_version_kv_continuation=True,
        hybrid=True,
        fully_async=False,
        partial_rollout=True,
        colocate=False,
        offload_rollout=False,
        update_weights_interval=1,
        cross_version_kv_max_gap=2,
        max_staleness=2,
    )

    # slime_validate_args normalizes --hybrid before invoking the helper.
    args.fully_async = True
    args.colocate = True
    validate_cross_version_kv_args(args)


def test_cross_version_kv_validation_does_not_require_admission_control() -> None:
    args = SimpleNamespace(
        enable_cross_version_kv_continuation=True,
        hybrid=True,
        fully_async=True,
        partial_rollout=True,
        colocate=True,
        offload_rollout=False,
        update_weights_interval=1,
        cross_version_kv_max_gap=2,
        max_staleness=2,
    )

    validate_cross_version_kv_args(args)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"update_weights_interval": 2}, "update-weights-interval 1"),
        ({"cross_version_kv_max_gap": 3}, "must not exceed"),
        ({"offload_rollout": True}, "incompatible"),
    ],
)
def test_cross_version_kv_validation_rejects_unsafe_contracts(override: dict, message: str) -> None:
    values = {
        "enable_cross_version_kv_continuation": True,
        "hybrid": True,
        "fully_async": True,
        "partial_rollout": True,
        "colocate": True,
        "offload_rollout": False,
        "update_weights_interval": 1,
        "cross_version_kv_max_gap": 2,
        "max_staleness": 2,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        validate_cross_version_kv_args(SimpleNamespace(**values))
