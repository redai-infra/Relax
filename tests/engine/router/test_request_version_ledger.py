# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio

import pytest

from relax.engine.router.request_version_ledger import (
    PublicationError,
    RequestVersionLedger,
)


async def test_prepare_targets_only_requests_older_than_max_gap() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("old-a", "engine-a")
    await ledger.register("old-b", "engine-b")
    ledger.current_version = 1
    ledger.current_actor_step = 1
    await ledger.register("safe", "engine-a")

    aborted: list[tuple[str, str]] = []

    async def abort_request(worker: str, rid: str) -> None:
        aborted.append((worker, rid))
        await ledger.unregister(rid)

    plan = await ledger.prepare(
        target_version=2,
        target_actor_step=2,
        max_gap=1,
        max_actor_step_gap=1,
        abort_request=abort_request,
        timeout_seconds=1,
    )

    assert aborted == [("engine-a", "old-a"), ("engine-b", "old-b")]
    assert plan.expired_by_worker == {
        "engine-a": ("old-a",),
        "engine-b": ("old-b",),
    }
    assert "safe" in ledger.active

    await ledger.commit(plan.publication_id, plan.target_version, plan.target_actor_step)
    assert ledger.current_version == 2
    assert ledger.current_actor_step == 2


async def test_prepare_retries_abort_until_late_dispatch_becomes_visible() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("late", "engine-a")
    ledger.current_version = 1
    ledger.current_actor_step = 1
    attempts = 0

    async def abort_request(_worker: str, rid: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            await ledger.unregister(rid)

    plan = await ledger.prepare(
        target_version=2,
        target_actor_step=2,
        max_gap=1,
        max_actor_step_gap=1,
        abort_request=abort_request,
        timeout_seconds=1,
    )

    assert attempts == 2
    await ledger.commit(plan.publication_id, plan.target_version, plan.target_actor_step)


async def test_publication_fence_blocks_new_registration_until_commit() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("expired", "engine-a")
    ledger.current_version = 1
    ledger.current_actor_step = 1
    abort_started = asyncio.Event()

    async def abort_request(_worker: str, _rid: str) -> None:
        abort_started.set()

    prepare_task = asyncio.create_task(
        ledger.prepare(
            target_version=2,
            target_actor_step=2,
            max_gap=1,
            max_actor_step_gap=1,
            abort_request=abort_request,
            timeout_seconds=1,
        )
    )
    await abort_started.wait()
    register_task = asyncio.create_task(ledger.register("new", "engine-a"))
    await asyncio.sleep(0)
    assert not register_task.done()

    await ledger.unregister("expired")
    plan = await prepare_task
    assert not register_task.done()
    await ledger.commit(plan.publication_id, plan.target_version, plan.target_actor_step)

    request = await register_task
    assert request.kv_epoch_version == 2
    assert request.kv_epoch_actor_step == 2


async def test_prepare_surfaces_abort_failure_after_attempting_all_workers() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("expired-a", "engine-a")
    await ledger.register("expired-b", "engine-b")
    ledger.current_version = 1
    ledger.current_actor_step = 1
    attempted: list[str] = []

    async def abort_request(_worker: str, rid: str) -> None:
        attempted.append(rid)
        if rid == "expired-a":
            raise PermissionError("abort denied")
        await ledger.unregister(rid)

    with pytest.raises(
        PublicationError,
        match="targeted retirement abort failed for rid=expired-a: PermissionError: abort denied",
    ):
        await ledger.prepare(
            target_version=2,
            target_actor_step=2,
            max_gap=1,
            max_actor_step_gap=1,
            abort_request=abort_request,
            timeout_seconds=1,
        )

    assert sorted(attempted) == ["expired-a", "expired-b"]
    request = await ledger.register("new", "engine-a")
    assert request.kv_epoch_version == 1
    assert request.kv_epoch_actor_step == 1


async def test_prepare_timeout_rolls_back_without_advancing_version() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("expired", "engine-a")
    ledger.current_version = 1
    ledger.current_actor_step = 1

    async def abort_request(_worker: str, _rid: str) -> None:
        return None

    with pytest.raises(TimeoutError):
        await ledger.prepare(
            target_version=2,
            target_actor_step=2,
            max_gap=1,
            max_actor_step_gap=1,
            abort_request=abort_request,
            timeout_seconds=0.01,
        )

    request = await ledger.register("new", "engine-a")
    assert request.kv_epoch_version == 1
    state = await ledger.snapshot()
    assert state["current_version"] == 1
    assert state["current_actor_step"] == 1
    assert state["publication"] is None
    assert state["failed_reason"].startswith("targeted retirement failed: TimeoutError:")


async def test_prepare_timeout_bounds_stuck_abort_and_releases_fence() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("expired", "engine-a")
    ledger.current_version = 1
    ledger.current_actor_step = 1
    abort_cancelled = asyncio.Event()

    async def abort_request(_worker: str, _rid: str) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            abort_cancelled.set()

    with pytest.raises(TimeoutError):
        await ledger.prepare(
            target_version=2,
            target_actor_step=2,
            max_gap=1,
            max_actor_step_gap=1,
            abort_request=abort_request,
            timeout_seconds=0.01,
        )

    assert abort_cancelled.is_set()
    request = await ledger.register("new", "engine-a")
    assert request.kv_epoch_version == 1


async def test_failed_publication_ends_current_transaction_without_poisoning_router() -> None:
    ledger = RequestVersionLedger(initial_version=2)

    async def abort_request(_worker: str, _rid: str) -> None:
        return None

    plan = await ledger.prepare(
        target_version=3,
        target_actor_step=1,
        max_gap=2,
        max_actor_step_gap=2,
        abort_request=abort_request,
        timeout_seconds=1,
    )
    await ledger.fail(plan.publication_id, plan.target_version, plan.target_actor_step, "DCS broadcast failed")

    request = await ledger.register("rid", "engine-a")
    assert request.kv_epoch_version == 2
    failed_state = await ledger.snapshot()
    assert failed_state["publication"] is None
    assert failed_state["current_version"] == 2
    assert failed_state["current_actor_step"] == 0
    assert failed_state["failed_reason"] == "DCS broadcast failed"


async def test_publication_terminal_operations_are_idempotent_and_mutually_exclusive() -> None:
    ledger = RequestVersionLedger(initial_version=2)

    async def abort_request(_worker: str, _rid: str) -> None:
        return None

    committed = await ledger.prepare(
        target_version=3,
        target_actor_step=1,
        max_gap=2,
        max_actor_step_gap=2,
        abort_request=abort_request,
        timeout_seconds=1,
    )
    first_commit = await ledger.commit(committed.publication_id, committed.target_version, committed.target_actor_step)
    second_commit = await ledger.commit(
        committed.publication_id, committed.target_version, committed.target_actor_step
    )
    assert first_commit == second_commit == committed
    with pytest.raises(PublicationError, match="already committed"):
        await ledger.fail(
            committed.publication_id,
            committed.target_version,
            committed.target_actor_step,
            "late failure",
        )
    with pytest.raises(PublicationError, match="already finalized"):
        await ledger.prepare(
            publication_id=committed.publication_id,
            target_version=4,
            target_actor_step=2,
            max_gap=2,
            max_actor_step_gap=2,
            abort_request=abort_request,
            timeout_seconds=1,
        )

    failed = await ledger.prepare(
        target_version=4,
        target_actor_step=2,
        max_gap=2,
        max_actor_step_gap=2,
        abort_request=abort_request,
        timeout_seconds=1,
    )
    first_fail = await ledger.fail(failed.publication_id, failed.target_version, failed.target_actor_step, "failed")
    second_fail = await ledger.fail(
        failed.publication_id, failed.target_version, failed.target_actor_step, "duplicate"
    )
    assert first_fail == second_fail == failed
    with pytest.raises(PublicationError, match="already failed"):
        await ledger.commit(failed.publication_id, failed.target_version, failed.target_actor_step)

    state = await ledger.snapshot()
    assert state["current_version"] == 3
    assert state["last_publication"]["status"] == "failed"
    assert state["last_publication"]["publication_id"] == failed.publication_id


async def test_publication_version_must_advance_exactly_once() -> None:
    ledger = RequestVersionLedger(initial_version=2)

    async def abort_request(_worker: str, _rid: str) -> None:
        return None

    with pytest.raises(PublicationError, match="advance exactly once"):
        await ledger.prepare(
            target_version=4,
            target_actor_step=1,
            max_gap=2,
            max_actor_step_gap=2,
            abort_request=abort_request,
            timeout_seconds=1,
        )


async def test_actor_step_gap_retires_request_after_skipped_publication() -> None:
    ledger = RequestVersionLedger(initial_version=1, initial_actor_step=0)
    await ledger.register("stale-after-skip", "engine-a")
    aborted: list[str] = []

    async def abort_request(_worker: str, rid: str) -> None:
        aborted.append(rid)
        await ledger.unregister(rid)

    plan = await ledger.prepare(
        target_version=2,
        target_actor_step=4,
        max_gap=1,
        max_actor_step_gap=2,
        abort_request=abort_request,
        timeout_seconds=1,
    )

    assert aborted == ["stale-after-skip"]
    assert plan.publication_gap_expired_rids == ()
    assert plan.actor_step_gap_expired_rids == ("stale-after-skip",)


async def test_short_forced_publication_keeps_request_within_actor_step_gap() -> None:
    ledger = RequestVersionLedger(initial_version=1, initial_actor_step=2)
    request = await ledger.register("safe-forced-publish", "engine-a")
    assert request.kv_epoch_actor_step == 2

    async def abort_request(_worker: str, _rid: str) -> None:
        pytest.fail("safe request was retired")

    plan = await ledger.prepare(
        target_version=2,
        target_actor_step=3,
        max_gap=1,
        max_actor_step_gap=2,
        abort_request=abort_request,
        timeout_seconds=1,
    )

    assert plan.expired_rids == ()
    await ledger.commit(plan.publication_id, plan.target_version, plan.target_actor_step)
    assert ledger.current_actor_step == 3


async def test_commit_rejects_actor_step_identity_mismatch() -> None:
    ledger = RequestVersionLedger(initial_version=1, initial_actor_step=2)

    async def abort_request(_worker: str, _rid: str) -> None:
        return None

    plan = await ledger.prepare(
        target_version=2,
        target_actor_step=4,
        max_gap=1,
        max_actor_step_gap=2,
        abort_request=abort_request,
        timeout_seconds=1,
    )

    with pytest.raises(PublicationError, match="actor-step target mismatch"):
        await ledger.commit(plan.publication_id, plan.target_version, target_actor_step=3)
