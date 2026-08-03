# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.cross_version_kv import (
    cross_version_kv_group_ready_for_finalize,
    cross_version_kv_pause_mode,
    cross_version_kv_resident_cap,
    cross_version_kv_strict_refresh,
    estimate_cross_version_kv_group_remaining_tokens,
    mark_cross_version_kv_carry,
    plan_cross_version_kv_progress_hedge,
    validate_cross_version_kv_args,
)


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


def test_progress_hedge_prevents_carried_fresh_from_freezing_current_work() -> None:
    assert cross_version_kv_resident_cap(8) == 10
    assert (
        plan_cross_version_kv_progress_hedge(
            adopted_groups=8,
            adopted_debt_groups=0,
            resident_groups=8,
            remaining_fresh_groups=8,
            rollout_batch_size=8,
        )
        == 2
    )


def test_remaining_token_estimate_uses_conditional_tail() -> None:
    group = [SimpleNamespace(response_length=4000), SimpleNamespace(response_length=6000)]

    estimated = estimate_cross_version_kv_group_remaining_tokens(
        group,
        recent_completed_response_lengths=[4500, 5000, 6200, 6500, 7000, 7500, 8000, 8192],
        max_response_length=8192,
    )

    assert estimated == 3500


def test_remaining_token_estimate_falls_back_to_safe_upper_bound() -> None:
    group = [SimpleNamespace(response_length=7000)]

    estimated = estimate_cross_version_kv_group_remaining_tokens(
        group,
        recent_completed_response_lengths=[1000, 2000, 3000],
        max_response_length=8192,
    )

    assert estimated == 1192


def test_remaining_token_estimate_ignores_completed_siblings() -> None:
    group = [
        SimpleNamespace(status="completed", response_length=1000),
        SimpleNamespace(status="pending", response_length=7000),
    ]

    estimated = estimate_cross_version_kv_group_remaining_tokens(
        group,
        recent_completed_response_lengths=[],
        max_response_length=8192,
    )

    assert estimated == 1192


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {
                "adopted_groups": 4,
                "adopted_debt_groups": 4,
                "resident_groups": 4,
                "remaining_fresh_groups": 8,
                "rollout_batch_size": 8,
            },
            0,
        ),
        (
            {
                "adopted_groups": 10,
                "adopted_debt_groups": 0,
                "resident_groups": 10,
                "remaining_fresh_groups": 8,
                "rollout_batch_size": 8,
            },
            0,
        ),
        (
            {
                "adopted_groups": 8,
                "adopted_debt_groups": 0,
                "resident_groups": 9,
                "remaining_fresh_groups": 8,
                "rollout_batch_size": 8,
            },
            1,
        ),
        (
            {
                "adopted_groups": 8,
                "adopted_debt_groups": 0,
                "resident_groups": 8,
                "remaining_fresh_groups": 1,
                "rollout_batch_size": 8,
            },
            1,
        ),
        (
            {
                "adopted_groups": 8,
                "adopted_debt_groups": 0,
                "resident_groups": 8,
                "remaining_fresh_groups": 8,
                "rollout_batch_size": 8,
                "strict_retry_pending": True,
            },
            0,
        ),
        (
            {
                "adopted_groups": 8,
                "adopted_debt_groups": 0,
                "resident_groups": 8,
                "remaining_fresh_groups": 8,
                "rollout_batch_size": 8,
                "estimated_remaining_tokens": 8192,
                "max_response_length": 8192,
            },
            1,
        ),
    ],
)
def test_progress_hedge_is_bounded_and_skips_debt_or_strict_rebuild(values: dict, expected: int) -> None:
    assert plan_cross_version_kv_progress_hedge(**values) == expected


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
    validate_cross_version_kv_args(args, sync_intent_enabled=True)


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
        validate_cross_version_kv_args(SimpleNamespace(**values), sync_intent_enabled=True)
