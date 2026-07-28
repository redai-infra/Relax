# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Protocol, TypeVar


__all__: list[str] = []


class GenerationAborted(Exception):
    """Raised when an aborted rollout prevents an inference request from being
    dispatched."""


@dataclass(frozen=True)
class RequestEvent:
    request_id: str
    turn_index: int | None
    relative_start_s: float
    permit_wait_s: float
    request_duration_s: float
    total_duration_s: float
    queue_depth_at_start: int
    queue_depth_at_acquire: int
    in_flight_at_start: int
    in_flight_at_end: int
    capacity: int
    outcome: str
    exception_type: str | None
    reentrant: bool


class RequestEventRecorder(Protocol):
    """Synchronous observer for finished inference request attempts."""

    def record(self, event: RequestEvent) -> None: ...


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
    pending_events: list[RequestEvent] = field(default_factory=list)


@dataclass(frozen=True)
class _LeaseUse:
    lease: _Lease
    reentrant: bool
    permit_wait_s: float
    queue_depth_at_acquire: int


class InferenceRequestGate:
    """Task-aware concurrency gate for individual inference requests."""

    def __init__(
        self,
        capacity: int,
        is_aborted: Callable[[], bool],
        recorder: RequestEventRecorder | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"InferenceRequestGate capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.semaphore = asyncio.Semaphore(capacity)
        self._is_aborted = is_aborted
        self._recorder = recorder
        self._lease_var: ContextVar[_Lease | None] = ContextVar(
            f"inference_request_lease_{id(self)}",
            default=None,
        )
        self._queue_depth = 0
        self._in_flight = 0
        self._started_at = monotonic()

    @asynccontextmanager
    async def permit(self) -> AsyncIterator[None]:
        """Acquire or borrow one request permit for the current task."""
        async with self._lease_scope():
            self._raise_if_aborted()
            yield

    async def run(
        self,
        request_factory: Callable[[], Awaitable[_Result]],
        *,
        turn_index: int | None = None,
    ) -> _Result:
        """Run a lazily created request while holding a request permit."""
        if turn_index is not None and type(turn_index) is not int:
            raise TypeError("turn_index must be an int or None")

        request_id = uuid.uuid4().hex
        run_started_at = monotonic()
        queue_depth_at_start = self._queue_depth
        in_flight_at_start = self._in_flight
        request_started_at: float | None = None
        lease_use: _LeaseUse | None = None

        try:
            async with self._lease_scope() as current_lease_use:
                lease_use = current_lease_use
                try:
                    self._raise_if_aborted()
                    request_started_at = monotonic()
                    result = await request_factory()
                except BaseException as error:
                    finished_at = monotonic()
                    lease_use.lease.pending_events.append(
                        self._build_event(
                            request_id=request_id,
                            turn_index=turn_index,
                            run_started_at=run_started_at,
                            request_started_at=request_started_at,
                            finished_at=finished_at,
                            queue_depth_at_start=queue_depth_at_start,
                            queue_depth_at_acquire=lease_use.queue_depth_at_acquire,
                            in_flight_at_start=in_flight_at_start,
                            error=error,
                            reentrant=lease_use.reentrant,
                            permit_wait_s=lease_use.permit_wait_s,
                        )
                    )
                    raise
                else:
                    finished_at = monotonic()
                    lease_use.lease.pending_events.append(
                        self._build_event(
                            request_id=request_id,
                            turn_index=turn_index,
                            run_started_at=run_started_at,
                            request_started_at=request_started_at,
                            finished_at=finished_at,
                            queue_depth_at_start=queue_depth_at_start,
                            queue_depth_at_acquire=lease_use.queue_depth_at_acquire,
                            in_flight_at_start=in_flight_at_start,
                            error=None,
                            reentrant=lease_use.reentrant,
                            permit_wait_s=lease_use.permit_wait_s,
                        )
                    )
                    return result
        except BaseException as error:
            if lease_use is None:
                finished_at = monotonic()
                event = self._build_event(
                    request_id=request_id,
                    turn_index=turn_index,
                    run_started_at=run_started_at,
                    request_started_at=None,
                    finished_at=finished_at,
                    queue_depth_at_start=queue_depth_at_start,
                    queue_depth_at_acquire=self._queue_depth,
                    in_flight_at_start=in_flight_at_start,
                    error=error,
                    reentrant=False,
                    permit_wait_s=max(0.0, finished_at - run_started_at),
                )
                self._defer_or_record(event)
            raise

    @asynccontextmanager
    async def _lease_scope(self) -> AsyncIterator[_LeaseUse]:
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("Inference request permits require an asyncio task")

        inherited_lease = self._lease_var.get()
        if inherited_lease is not None and inherited_lease.active:
            if inherited_lease.owner is not current_task:
                raise RuntimeError("Active inference request lease belongs to another asyncio task")
            yield _LeaseUse(
                lease=inherited_lease,
                reentrant=True,
                permit_wait_s=0.0,
                queue_depth_at_acquire=self._queue_depth,
            )
            return
        if inherited_lease is not None:
            self._lease_var.set(None)

        wait_started_at = monotonic()
        self._queue_depth += 1
        try:
            await self.semaphore.acquire()
        finally:
            self._queue_depth -= 1

        lease: _Lease | None = None
        token: Token[_Lease | None] | None = None
        counted_in_flight = False
        try:
            acquired_at = monotonic()
            lease = _Lease(owner=current_task)
            token = self._lease_var.set(lease)
            self._in_flight += 1
            counted_in_flight = True
            yield _LeaseUse(
                lease=lease,
                reentrant=False,
                permit_wait_s=max(0.0, acquired_at - wait_started_at),
                queue_depth_at_acquire=self._queue_depth,
            )
        finally:
            if lease is not None:
                lease.active = False
            try:
                if token is not None:
                    self._lease_var.reset(token)
            finally:
                if counted_in_flight:
                    self._in_flight -= 1
                self.semaphore.release()
                if lease is not None:
                    self._flush_events(lease)

    def _raise_if_aborted(self) -> None:
        if self._is_aborted():
            raise GenerationAborted

    def _build_event(
        self,
        *,
        request_id: str,
        turn_index: int | None,
        run_started_at: float,
        request_started_at: float | None,
        finished_at: float,
        queue_depth_at_start: int,
        queue_depth_at_acquire: int,
        in_flight_at_start: int,
        error: BaseException | None,
        reentrant: bool,
        permit_wait_s: float,
    ) -> RequestEvent:
        request_duration_s = 0.0 if request_started_at is None else max(0.0, finished_at - request_started_at)
        return RequestEvent(
            request_id=request_id,
            turn_index=turn_index,
            relative_start_s=max(0.0, run_started_at - self._started_at),
            permit_wait_s=permit_wait_s,
            request_duration_s=request_duration_s,
            total_duration_s=max(0.0, finished_at - run_started_at),
            queue_depth_at_start=queue_depth_at_start,
            queue_depth_at_acquire=queue_depth_at_acquire,
            in_flight_at_start=in_flight_at_start,
            in_flight_at_end=self._in_flight,
            capacity=self.capacity,
            outcome=self._event_outcome(error),
            exception_type=None if error is None else type(error).__name__,
            reentrant=reentrant,
        )

    @staticmethod
    def _event_outcome(error: BaseException | None) -> str:
        if error is None:
            return "success"
        if isinstance(error, GenerationAborted):
            return "aborted"
        if isinstance(error, asyncio.CancelledError):
            return "cancelled"
        return "error"

    def _defer_or_record(self, event: RequestEvent) -> None:
        inherited_lease = self._lease_var.get()
        if inherited_lease is not None and inherited_lease.active:
            inherited_lease.pending_events.append(event)
            return
        self._record(event)

    def _flush_events(self, lease: _Lease) -> None:
        events = lease.pending_events
        lease.pending_events = []
        for event in events:
            self._record(event)

    def _record(self, event: RequestEvent) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder.record(event)
        except BaseException:
            pass
