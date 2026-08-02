# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.rollout.sync_intent import (
    SYNC_INTENT_POLICY_ENV,
    SYNC_INTENT_QUIESCE_FLOOR_ENV,
    SYNC_INTENT_QUIESCE_MULTIPLIER_ENV,
    SYNC_INTENT_TTL_ENV,
    SYNC_INTENT_WINDOW_GROUPS_ENV,
    SyncIntentController,
    SyncIntentSnapshot,
    mark_work_origin,
    plan_adaptive_window_fetch,
    plan_intent_guard_fetch,
    resolve_partition_request_priority,
    should_admit_fresh,
    sync_intent_policy_enabled,
)


def test_policy_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SYNC_INTENT_POLICY_ENV, raising=False)
    args = SimpleNamespace(sglang_enable_priority_scheduling=True)
    sample = SimpleNamespace(metadata={})

    assert resolve_partition_request_priority(args, sample) is None
    assert not sync_intent_policy_enabled()


def test_policy_enable_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv(SYNC_INTENT_POLICY_ENV, value)
        assert sync_intent_policy_enabled()


def test_sync_intent_identity_is_idempotent_and_monotonic() -> None:
    controller = SyncIntentController()

    first = controller.begin(sync_id=7, actor_rollout_id=6)
    assert controller.begin(sync_id=7, actor_rollout_id=6) == first
    assert controller.begin(sync_id=6, actor_rollout_id=5) == first
    with pytest.raises(ValueError, match="identity mismatch"):
        controller.begin(sync_id=7, actor_rollout_id=5)

    newer = controller.begin(sync_id=8, actor_rollout_id=7)
    assert newer.sync_id == 8
    assert controller.end(7) == newer
    assert not controller.end(8).active


def test_sync_intent_expires_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SYNC_INTENT_TTL_ENV, "0.001")
    timestamps = iter((100.0, 102.0))
    monkeypatch.setattr("relax.engine.rollout.sync_intent.time.monotonic", lambda: next(timestamps))
    controller = SyncIntentController()
    controller.begin(sync_id=3, actor_rollout_id=2)

    snapshot = controller.snapshot()

    assert not snapshot.active
    assert snapshot.expired


def test_phase_is_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter((10.0, 11.0, 12.0))
    monkeypatch.setattr("relax.engine.rollout.sync_intent.time.monotonic", lambda: next(timestamps))
    controller = SyncIntentController()
    controller.begin(sync_id=6, actor_rollout_id=5)

    training = controller.update_phase(6, "actor_train", estimated_phase_seconds=20.0)
    assert training.phase == "actor_train"
    assert training.phase_started_at == 11.0
    assert training.estimated_phase_seconds == 20.0
    assert controller.update_phase(6, "data_wait").phase == "actor_train"

    quiesced = controller.update_phase(6, "quiesce")
    assert quiesced.phase == "quiesce"
    assert quiesced.phase_started_at == 12.0


def test_guard_is_work_conserving_until_latency_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SYNC_INTENT_QUIESCE_FLOOR_ENV, "2")
    monkeypatch.setenv(SYNC_INTENT_QUIESCE_MULTIPLIER_ENV, "1.25")
    snapshot = SyncIntentSnapshot(
        active=True,
        sync_id=6,
        actor_rollout_id=5,
        phase="actor_train",
        phase_started_at=100.0,
        estimated_phase_seconds=20.0,
    )

    assert (
        plan_intent_guard_fetch(
            snapshot=snapshot,
            physical_rollout_id=5,
            old_debt_groups=4,
            completed_debt_groups=0,
            inflight_debt_groups=0,
            observed_group_latency_seconds=4.0,
            default_fetch_groups=12,
            now=105.0,
        )
        == 12
    )
    assert (
        plan_intent_guard_fetch(
            snapshot=snapshot,
            physical_rollout_id=6,
            old_debt_groups=4,
            completed_debt_groups=0,
            inflight_debt_groups=0,
            observed_group_latency_seconds=4.0,
            default_fetch_groups=12,
            now=116.0,
        )
        == 4
    )


@pytest.mark.parametrize("phase", ("data_wait", None))
def test_fresh_is_work_conserving_while_actor_waits(phase: str | None) -> None:
    snapshot = SyncIntentSnapshot(active=True, sync_id=6, actor_rollout_id=5, phase=phase)

    assert should_admit_fresh(snapshot, observed_group_latency_seconds=100.0)


@pytest.mark.parametrize("phase", ("quiesce", "weight_sync"))
def test_fresh_stops_at_publish_boundary(phase: str) -> None:
    snapshot = SyncIntentSnapshot(active=True, sync_id=6, actor_rollout_id=5, phase=phase)

    assert not should_admit_fresh(snapshot, observed_group_latency_seconds=0.1)


def test_adaptive_window_seeds_once_then_refills_commit_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SYNC_INTENT_WINDOW_GROUPS_ENV, "16")

    assert (
        plan_adaptive_window_fetch(
            resident_groups=0,
            baseline_fetch_groups=8,
            remaining_commit_groups=8,
            hedge_groups=8,
            window_initialized=False,
        )
        == 16
    )
    assert (
        plan_adaptive_window_fetch(
            resident_groups=3,
            baseline_fetch_groups=16,
            remaining_commit_groups=4,
            hedge_groups=8,
            window_initialized=True,
        )
        == 1
    )


def test_old_debt_priority_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    args = SimpleNamespace(sglang_enable_priority_scheduling=True)
    groups = [[SimpleNamespace(metadata={})], [SimpleNamespace(metadata={})]]
    mark_work_origin(groups, old_debt_groups=1, fresh_origin="speculative_fresh")

    monkeypatch.setenv(SYNC_INTENT_POLICY_ENV, "1")
    assert resolve_partition_request_priority(args, groups[0][0]) == 1
    assert resolve_partition_request_priority(args, groups[1][0]) == 0
