# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.cross_version_kv import (
    cross_version_kv_pause_mode,
    cross_version_kv_strict_refresh,
    mark_cross_version_kv_carry,
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
