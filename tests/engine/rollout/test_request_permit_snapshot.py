# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio

from relax.engine.rollout.request_permit import InferencePermitManager


async def test_snapshot_is_observational_and_preserves_fifo() -> None:
    manager = InferencePermitManager(1)
    order = []
    release = asyncio.Event()

    async def holder():
        async with manager.permit():
            order.append("holder")
            await release.wait()

    async def waiter(name):
        async with manager.permit():
            order.append(name)

    holder_task = asyncio.create_task(holder())
    await asyncio.sleep(0)
    first = asyncio.create_task(waiter("first"))
    await asyncio.sleep(0)
    second = asyncio.create_task(waiter("second"))
    await asyncio.sleep(0)

    assert manager.snapshot() == (1, 1, 2)
    release.set()
    await asyncio.gather(holder_task, first, second)
    assert order == ["holder", "first", "second"]
    assert manager.snapshot() == (1, 0, 0)
