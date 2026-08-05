# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.cross_version_kv import (
    clear_cross_version_kv_progress_hedge_marker,
    clear_cross_version_kv_task_markers,
    count_cross_version_kv_progress_hedge_groups,
    cross_version_kv_group_ready_for_finalize,
    cross_version_kv_group_requires_strict_retry,
    cross_version_kv_pause_mode,
    cross_version_kv_resident_cap,
    cross_version_kv_strict_refresh,
    estimate_cross_version_kv_group_remaining_tokens,
    mark_cross_version_kv_carry,
    plan_cross_version_kv_progress_hedge,
    plan_joint_carry_admission,
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


def test_targeted_retirement_requires_strict_retry_before_carry_adoption() -> None:
    targeted = SimpleNamespace(metadata={"targeted_retirement_aborted": True})
    ordinary = SimpleNamespace(metadata={})

    assert cross_version_kv_group_requires_strict_retry([ordinary, targeted])
    assert not cross_version_kv_group_requires_strict_retry([ordinary])


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


def test_carried_progress_hedge_rebuilds_inflight_accounting() -> None:
    groups = [
        [SimpleNamespace(metadata={"cross_version_kv_progress_hedge": True})],
        [SimpleNamespace(metadata={})],
        [
            SimpleNamespace(metadata={}),
            SimpleNamespace(metadata={"cross_version_kv_progress_hedge": True}),
        ],
    ]

    assert count_cross_version_kv_progress_hedge_groups(groups) == 2


def test_hedge_completion_preserves_carried_marker_until_fallback_classification() -> None:
    sample = SimpleNamespace(
        metadata={
            "cross_version_kv_progress_hedge": True,
            "cross_version_kv_carried": True,
        }
    )

    assert clear_cross_version_kv_progress_hedge_marker([sample])
    assert "cross_version_kv_progress_hedge" not in sample.metadata
    assert sample.metadata["cross_version_kv_carried"] is True


def test_abort_to_buffer_clears_all_task_lifetime_markers() -> None:
    sample = SimpleNamespace(
        metadata={
            "cross_version_kv_progress_hedge": True,
            "cross_version_kv_carried": True,
            "cross_version_kv_carryovers": 2,
            "targeted_retirement_aborted": True,
        }
    )

    assert clear_cross_version_kv_task_markers([sample])
    assert "cross_version_kv_progress_hedge" not in sample.metadata
    assert "cross_version_kv_carried" not in sample.metadata
    assert "targeted_retirement_aborted" not in sample.metadata
    assert sample.metadata["cross_version_kv_carryovers"] == 2


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


def _plan_joint_for_test(
    *,
    phase: str | None,
    debt_target: int,
    debt_committed: int,
    debt_eligible_inflight: int,
    carry_current_inflight: int,
    current_target: int,
    current_committed: int,
    fresh_current_inflight: int,
    resident_groups: int,
    carry_remaining_tokens: list[int],
    rollout_batch_size: int,
    max_response_length: int,
    strict_retry_pending: bool = False,
    final_backfill: bool = False,
):
    carry_ids = [f"carry-{index}" for index in range(len(carry_remaining_tokens))]
    debt_ids = carry_ids[:debt_eligible_inflight]
    carry_current_ids = carry_ids[debt_eligible_inflight : debt_eligible_inflight + carry_current_inflight]
    fresh_ids = [f"fresh-{index}" for index in range(fresh_current_inflight)]
    known_ids = [*carry_ids, *fresh_ids]
    if len(known_ids) > resident_groups:
        raise ValueError("test resident count is smaller than the disjoint work sets")
    resident_ids = [*known_ids, *(f"other-{index}" for index in range(resident_groups - len(known_ids)))]
    return plan_joint_carry_admission(
        phase=phase,
        debt_target=debt_target,
        debt_committed=debt_committed,
        current_target=current_target,
        current_committed=current_committed,
        resident_group_ids=resident_ids,
        carry_groups=list(zip(carry_ids, carry_remaining_tokens, strict=True)),
        debt_eligible_group_ids=debt_ids,
        carry_current_group_ids=carry_current_ids,
        fresh_current_group_ids=fresh_ids,
        rollout_batch_size=rollout_batch_size,
        max_response_length=max_response_length,
        strict_retry_pending=strict_retry_pending,
        final_backfill=final_backfill,
    )


def test_joint_planner_admits_full_current_batch_without_carry() -> None:
    plan = _plan_joint_for_test(
        phase=None,
        debt_target=0,
        debt_committed=0,
        debt_eligible_inflight=0,
        carry_current_inflight=0,
        current_target=8,
        current_committed=0,
        fresh_current_inflight=0,
        resident_groups=0,
        carry_remaining_tokens=[],
        rollout_batch_size=8,
        max_response_length=8192,
    )

    assert plan.debt_admit_groups == 0
    assert plan.fresh_admit_groups == 8
    assert plan.resident_cap == 10


def test_joint_planner_uses_only_bounded_reserve_behind_full_length_carry() -> None:
    plan = _plan_joint_for_test(
        phase=None,
        debt_target=0,
        debt_committed=0,
        debt_eligible_inflight=0,
        carry_current_inflight=0,
        current_target=8,
        current_committed=0,
        fresh_current_inflight=0,
        resident_groups=8,
        carry_remaining_tokens=[8192] * 8,
        rollout_batch_size=8,
        max_response_length=8192,
    )

    assert plan.carry_work_equivalents == 8
    assert plan.current_reserve == 2
    assert plan.fresh_admit_groups == 2
    assert plan.work_overcommit_equivalents == 2


def test_joint_planner_does_not_double_count_carried_current_work() -> None:
    plan = _plan_joint_for_test(
        phase=None,
        debt_target=0,
        debt_committed=0,
        debt_eligible_inflight=0,
        carry_current_inflight=2,
        current_target=8,
        current_committed=0,
        fresh_current_inflight=0,
        resident_groups=8,
        carry_remaining_tokens=[8192] * 8,
        rollout_batch_size=8,
        max_response_length=8192,
    )

    assert plan.current_deficit == 6
    assert plan.fresh_admit_groups == 0
    assert plan.work_overcommit_equivalents == 0


def test_joint_planner_reports_existing_carry_overcommit_without_reserve() -> None:
    plan = _plan_joint_for_test(
        phase=None,
        debt_target=0,
        debt_committed=0,
        debt_eligible_inflight=0,
        carry_current_inflight=1,
        current_target=8,
        current_committed=0,
        fresh_current_inflight=0,
        resident_groups=9,
        carry_remaining_tokens=[8192] * 9,
        rollout_batch_size=8,
        max_response_length=8192,
    )

    assert plan.current_reserve == 0
    assert plan.fresh_admit_groups == 0
    assert plan.work_overcommit_equivalents == 1


def test_joint_planner_rejects_overlapping_carry_and_fresh_resident_counts() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        plan_joint_carry_admission(
            phase=None,
            debt_target=0,
            debt_committed=0,
            current_target=8,
            current_committed=0,
            resident_group_ids=["shared"],
            carry_groups=[("shared", 8192)],
            debt_eligible_group_ids=[],
            carry_current_group_ids=[],
            fresh_current_group_ids=["shared"],
            rollout_batch_size=8,
            max_response_length=8192,
        )


def test_joint_planner_does_not_recreate_static_filler_under_heavy_debt() -> None:
    plan = _plan_joint_for_test(
        phase=None,
        debt_target=8,
        debt_committed=0,
        debt_eligible_inflight=7,
        carry_current_inflight=0,
        current_target=8,
        current_committed=0,
        fresh_current_inflight=0,
        resident_groups=7,
        carry_remaining_tokens=[8192] * 7,
        rollout_batch_size=8,
        max_response_length=8192,
    )

    assert plan.debt_admit_groups == 1
    assert plan.fresh_admit_groups == 2
    assert plan.debt_admit_groups + plan.fresh_admit_groups <= 3


def test_joint_planner_prioritizes_uncovered_debt_over_current() -> None:
    plan = _plan_joint_for_test(
        phase=None,
        debt_target=8,
        debt_committed=0,
        debt_eligible_inflight=6,
        carry_current_inflight=0,
        current_target=8,
        current_committed=0,
        fresh_current_inflight=0,
        resident_groups=9,
        carry_remaining_tokens=[8192] * 6,
        rollout_batch_size=8,
        max_response_length=8192,
    )

    assert plan.debt_admit_groups == 1
    assert plan.fresh_admit_groups == 0
    assert plan.reason == "debt_first"


@pytest.mark.parametrize(
    ("phase", "strict_retry_pending", "final_backfill", "expected_reason"),
    [
        ("quiesce", False, False, "phase:quiesce"),
        ("weight_sync", False, False, "phase:weight_sync"),
        (None, True, False, "strict_retry"),
        (None, False, True, "final_backfill"),
    ],
)
def test_joint_planner_blocks_current_admission_at_unsafe_boundaries(
    phase: str | None,
    strict_retry_pending: bool,
    final_backfill: bool,
    expected_reason: str,
) -> None:
    plan = _plan_joint_for_test(
        phase=phase,
        debt_target=0,
        debt_committed=0,
        debt_eligible_inflight=0,
        carry_current_inflight=0,
        current_target=8,
        current_committed=0,
        fresh_current_inflight=0,
        resident_groups=0,
        carry_remaining_tokens=[],
        rollout_batch_size=8,
        max_response_length=8192,
        strict_retry_pending=strict_retry_pending,
        final_backfill=final_backfill,
    )

    assert plan.fresh_admit_groups == 0
    assert plan.reason == expected_reason


@pytest.mark.parametrize("remaining", [[-1], [8193], [1.5], [True]])
def test_joint_planner_rejects_invalid_work_estimates(remaining: list[object]) -> None:
    with pytest.raises(ValueError):
        _plan_joint_for_test(
            phase=None,
            debt_target=0,
            debt_committed=0,
            debt_eligible_inflight=0,
            carry_current_inflight=0,
            current_target=8,
            current_committed=0,
            fresh_current_inflight=0,
            resident_groups=1,
            carry_remaining_tokens=remaining,
            rollout_batch_size=8,
            max_response_length=8192,
        )


def test_joint_planner_rejects_non_integer_counts() -> None:
    with pytest.raises(ValueError, match="counts must be integers"):
        plan_joint_carry_admission(
            phase=None,
            debt_target=0.5,
            debt_committed=0,
            current_target=8,
            current_committed=0,
            resident_group_ids=[],
            carry_groups=[],
            debt_eligible_group_ids=[],
            carry_current_group_ids=[],
            fresh_current_group_ids=[],
            rollout_batch_size=8,
            max_response_length=8192,
        )


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
