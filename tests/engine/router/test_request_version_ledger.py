# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio

import pytest

from relax.engine.router.request_version_ledger import (
    PublicationError,
    PublicationFailed,
    RequestVersionLedger,
)


async def test_prepare_targets_only_requests_older_than_max_gap() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("old-a", "engine-a")
    await ledger.register("old-b", "engine-b")
    ledger.current_version = 1
    await ledger.register("safe", "engine-a")

    aborted: list[tuple[str, str]] = []

    async def abort_request(worker: str, rid: str) -> None:
        aborted.append((worker, rid))
        await ledger.unregister(rid)

    plan = await ledger.prepare(
        target_version=2,
        max_gap=1,
        abort_request=abort_request,
        timeout_seconds=1,
    )

    assert aborted == [("engine-a", "old-a"), ("engine-b", "old-b")]
    assert plan.expired_by_worker == {
        "engine-a": ("old-a",),
        "engine-b": ("old-b",),
    }
    assert "safe" in ledger.active

    await ledger.commit(plan.publication_id, plan.target_version)
    assert ledger.current_version == 2


async def test_prepare_retries_abort_until_late_dispatch_becomes_visible() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("late", "engine-a")
    ledger.current_version = 1
    attempts = 0

    async def abort_request(_worker: str, rid: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            await ledger.unregister(rid)

    plan = await ledger.prepare(
        target_version=2,
        max_gap=1,
        abort_request=abort_request,
        timeout_seconds=1,
    )

    assert attempts == 2
    await ledger.commit(plan.publication_id, plan.target_version)


async def test_publication_fence_blocks_new_registration_until_commit() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("expired", "engine-a")
    ledger.current_version = 1
    abort_started = asyncio.Event()

    async def abort_request(_worker: str, _rid: str) -> None:
        abort_started.set()

    prepare_task = asyncio.create_task(
        ledger.prepare(
            target_version=2,
            max_gap=1,
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
    await ledger.commit(plan.publication_id, plan.target_version)

    request = await register_task
    assert request.kv_epoch_version == 2


async def test_prepare_timeout_poison_fails_closed_without_advancing_version() -> None:
    ledger = RequestVersionLedger(initial_version=0)
    await ledger.register("expired", "engine-a")
    ledger.current_version = 1

    async def abort_request(_worker: str, _rid: str) -> None:
        return None

    with pytest.raises(TimeoutError):
        await ledger.prepare(
            target_version=2,
            max_gap=1,
            abort_request=abort_request,
            timeout_seconds=0.01,
        )

    with pytest.raises(PublicationFailed, match="targeted retirement failed"):
        await ledger.register("new", "engine-a")
    state = await ledger.snapshot()
    assert state["current_version"] == 1
    assert state["publication"] is None


async def test_failed_publication_poison_fails_closed() -> None:
    ledger = RequestVersionLedger(initial_version=2)

    async def abort_request(_worker: str, _rid: str) -> None:
        return None

    plan = await ledger.prepare(
        target_version=3,
        max_gap=2,
        abort_request=abort_request,
        timeout_seconds=1,
    )
    await ledger.fail(plan.publication_id, plan.target_version, "DCS broadcast failed")

    with pytest.raises(PublicationFailed, match="DCS broadcast failed"):
        await ledger.register("rid", "engine-a")


async def test_publication_version_must_advance_exactly_once() -> None:
    ledger = RequestVersionLedger(initial_version=2)

    async def abort_request(_worker: str, _rid: str) -> None:
        return None

    with pytest.raises(PublicationError, match="advance exactly once"):
        await ledger.prepare(
            target_version=4,
            max_gap=2,
            abort_request=abort_request,
            timeout_seconds=1,
        )
