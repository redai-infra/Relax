# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import AsyncContextManager, AsyncIterator, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RequestExecution(Generic[T]):
    value: T | None
    aborted: bool
    permit_wait_seconds: float
    request_seconds: float


class RequestPermitPool:
    """Limit only in-flight model requests, not whole rollout sessions."""

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError(f"request permit limit must be positive, got {limit}")
        self._semaphore = asyncio.Semaphore(limit)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        async with self._semaphore:
            yield


async def run_request_with_permit(
    permit: Callable[[], AsyncContextManager[None]],
    request: Callable[[], Awaitable[T]],
    *,
    should_abort: Callable[[], bool] | None = None,
) -> RequestExecution[T]:
    """Run one model request after admission and a post-wait abort check."""
    wait_started = monotonic()
    async with permit():
        wait_finished = monotonic()
        if should_abort is not None and should_abort():
            return RequestExecution(
                value=None,
                aborted=True,
                permit_wait_seconds=wait_finished - wait_started,
                request_seconds=0.0,
            )

        request_started = monotonic()
        value = await request()
        request_finished = monotonic()

    return RequestExecution(
        value=value,
        aborted=False,
        permit_wait_seconds=wait_finished - wait_started,
        request_seconds=request_finished - request_started,
    )
