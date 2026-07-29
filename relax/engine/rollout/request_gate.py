# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, TypeVar


__all__: list[str] = []


class GenerationAborted(Exception):
    """Raised when an aborted rollout prevents an inference request from being
    dispatched."""


_GenerateCallable = TypeVar("_GenerateCallable", bound=Callable[..., Any])
_Result = TypeVar("_Result")


def request_scoped_generate(func: _GenerateCallable) -> _GenerateCallable:
    """Declare that a custom generator submits each inference request through
    GenerateState.run_request()."""
    # Keep the legacy marker name; dispatch interprets it as request-scoped admission.
    func.manages_inference_permit = True
    return func


@dataclass
class _Lease:
    owner: asyncio.Task[Any]
    active: bool = True


class InferenceRequestGate:
    """Task-aware concurrency gate for individual inference requests."""

    def __init__(
        self,
        capacity: int,
        is_aborted: Callable[[], bool],
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"InferenceRequestGate capacity must be positive, got {capacity}")
        self.semaphore = asyncio.Semaphore(capacity)
        self._is_aborted = is_aborted
        self._lease_var: ContextVar[_Lease | None] = ContextVar(
            f"inference_request_lease_{id(self)}",
            default=None,
        )

    @asynccontextmanager
    async def permit(self) -> AsyncIterator[None]:
        """Acquire or borrow one request permit for the current task."""
        async with self._lease_scope():
            self._raise_if_aborted()
            yield

    async def run(
        self,
        request_factory: Callable[[], Awaitable[_Result]],
    ) -> _Result:
        """Run a lazily created request while holding a request permit."""
        async with self._lease_scope():
            self._raise_if_aborted()
            return await request_factory()

    @asynccontextmanager
    async def _lease_scope(self) -> AsyncIterator[None]:
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("Inference request permits require an asyncio task")

        inherited_lease = self._lease_var.get()
        if inherited_lease is not None and inherited_lease.active:
            if inherited_lease.owner is not current_task:
                raise RuntimeError("Active inference request lease belongs to another asyncio task")
            yield
            return
        if inherited_lease is not None:
            self._lease_var.set(None)

        await self.semaphore.acquire()

        lease: _Lease | None = None
        token: Token[_Lease | None] | None = None
        try:
            lease = _Lease(owner=current_task)
            token = self._lease_var.set(lease)
            yield
        finally:
            if lease is not None:
                lease.active = False
            try:
                if token is not None:
                    self._lease_var.reset(token)
            finally:
                self.semaphore.release()

    def _raise_if_aborted(self) -> None:
        if self._is_aborted():
            raise GenerationAborted
