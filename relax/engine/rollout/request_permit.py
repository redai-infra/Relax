# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


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
