# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio

import pytest

from relax.engine.rollout.request_permit import RequestPermitPool


@pytest.mark.asyncio
async def test_request_permit_enforces_concurrency_limit():
    permits = RequestPermitPool(2)
    active = 0
    peak = 0
    release = asyncio.Event()

    async def request():
        nonlocal active, peak
        async with permits.acquire():
            active += 1
            peak = max(peak, active)
            await release.wait()
            active -= 1

    tasks = [asyncio.create_task(request()) for _ in range(4)]
    while peak < 2:
        await asyncio.sleep(0)
    assert active == 2
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

    async with asyncio.timeout(1):
        async with permits.acquire():
            pass


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
