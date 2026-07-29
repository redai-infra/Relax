# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from contextlib import asynccontextmanager

import pytest

from relax.engine.rollout.request_permit import RequestPermitPool, run_request_with_permit


async def _acquire_once(permits: RequestPermitPool) -> None:
    async with permits.acquire():
        pass


@pytest.mark.asyncio
async def test_request_permit_enforces_concurrency_limit():
    permits = RequestPermitPool(2)
    active = 0
    peak = 0
    release = asyncio.Event()
    limit_reached = asyncio.Event()

    async def request():
        nonlocal active, peak
        async with permits.acquire():
            active += 1
            peak = max(peak, active)
            if peak == 2:
                limit_reached.set()
            try:
                await release.wait()
            finally:
                active -= 1

    tasks = [asyncio.create_task(request()) for _ in range(4)]
    try:
        await asyncio.wait_for(limit_reached.wait(), timeout=1)
        assert active == 2
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert peak == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("request failed"), asyncio.CancelledError()])
async def test_request_permit_releases_after_failure(failure):
    permits = RequestPermitPool(1)

    async def fail():
        async with permits.acquire():
            raise failure

    with pytest.raises(type(failure)):
        await fail()

    await asyncio.wait_for(_acquire_once(permits), timeout=1)


@pytest.mark.asyncio
async def test_request_permit_cancellation_while_waiting_does_not_consume_permit():
    permits = RequestPermitPool(1)
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold_permit():
        async with permits.acquire():
            holder_entered.set()
            await release_holder.wait()

    async def wait_for_permit():
        async with permits.acquire():
            pass

    holder = asyncio.create_task(hold_permit())
    await holder_entered.wait()
    waiter = asyncio.create_task(wait_for_permit())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release_holder.set()
    await holder
    await asyncio.wait_for(_acquire_once(permits), timeout=1)


@pytest.mark.asyncio
async def test_request_permit_does_not_cover_environment_work():
    permits = RequestPermitPool(1)
    first_request_done = asyncio.Event()
    environment_done = asyncio.Event()
    short_request_done = asyncio.Event()

    async def long_session():
        async with permits.acquire():
            first_request_done.set()
        await environment_done.wait()
        async with permits.acquire():
            pass

    async def short_session():
        await first_request_done.wait()
        async with permits.acquire():
            short_request_done.set()

    long_task = asyncio.create_task(long_session())
    short_task = asyncio.create_task(short_session())
    await asyncio.wait_for(short_request_done.wait(), timeout=1)
    assert not environment_done.is_set()
    environment_done.set()
    await asyncio.gather(long_task, short_task)


@pytest.mark.parametrize("limit", [0, -1])
def test_request_permit_rejects_non_positive_limit(limit):
    with pytest.raises(ValueError, match="positive"):
        RequestPermitPool(limit)


@pytest.mark.asyncio
async def test_request_is_skipped_when_abort_happens_while_waiting_for_permit():
    permit_entered = asyncio.Event()
    release_permit = asyncio.Event()
    aborted = False
    request_called = False

    @asynccontextmanager
    async def blocking_permit():
        permit_entered.set()
        await release_permit.wait()
        yield

    async def request():
        nonlocal request_called
        request_called = True
        return "response"

    task = asyncio.create_task(run_request_with_permit(blocking_permit, request, should_abort=lambda: aborted))
    await asyncio.wait_for(permit_entered.wait(), timeout=1)
    aborted = True
    release_permit.set()

    result = await asyncio.wait_for(task, timeout=1)

    assert result.aborted
    assert result.value is None
    assert not request_called


@pytest.mark.asyncio
async def test_request_result_reports_permit_wait_and_request_time(monkeypatch):
    permits = RequestPermitPool(1)
    timestamps = iter([10.0, 13.0, 20.0, 25.0])

    async def request():
        return "response"

    monkeypatch.setattr("relax.engine.rollout.request_permit.monotonic", lambda: next(timestamps))

    result = await run_request_with_permit(permits.acquire, request)

    assert not result.aborted
    assert result.value == "response"
    assert result.permit_wait_seconds == 3.0
    assert result.request_seconds == 5.0
