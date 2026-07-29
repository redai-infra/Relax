# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from relax.engine.rollout import request_gate


_TIMEOUT = 1.0
_REQUEST_EVENT_FIELDS = {
    "request_id",
    "turn_index",
    "relative_start_s",
    "permit_wait_s",
    "request_duration_s",
    "total_duration_s",
    "queue_depth_at_start",
    "queue_depth_at_acquire",
    "in_flight_at_start",
    "in_flight_at_end",
    "capacity",
    "outcome",
    "exception_type",
    "reentrant",
}


class _Recorder:
    def __init__(self) -> None:
        self.events: list[request_gate.RequestEvent] = []

    def record(self, event: request_gate.RequestEvent) -> None:
        self.events.append(event)


async def _wait_for_waiter(gate: request_gate.InferenceRequestGate) -> None:
    async def wait() -> None:
        while True:
            waiters = getattr(gate.semaphore, "_waiters", None)
            if waiters is not None and any(not waiter.done() for waiter in waiters):
                return
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=_TIMEOUT)


async def _noop_request() -> None:
    return None


def _assert_event_payload(payload: dict[str, Any]) -> None:
    assert set(payload) == _REQUEST_EVENT_FIELDS
    assert len(payload["request_id"]) == 32
    int(payload["request_id"], 16)
    for field in ("relative_start_s", "permit_wait_s", "request_duration_s", "total_duration_s"):
        assert payload[field] >= 0
    assert payload["total_duration_s"] >= payload["permit_wait_s"]
    assert payload["total_duration_s"] >= payload["request_duration_s"]
    for field in ("queue_depth_at_start", "queue_depth_at_acquire"):
        assert type(payload[field]) is int
        assert payload[field] >= 0
    for field in ("in_flight_at_start", "in_flight_at_end"):
        assert type(payload[field]) is int
        assert 0 <= payload[field] <= payload["capacity"]


async def test_request_gate_enforces_capacity_and_recovers() -> None:
    gate = request_gate.InferenceRequestGate(capacity=2, is_aborted=lambda: False)
    release = asyncio.Event()
    condition = asyncio.Condition()
    in_flight = 0
    peak_in_flight = 0

    async def request() -> None:
        nonlocal in_flight, peak_in_flight
        async with condition:
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            condition.notify_all()
        try:
            await release.wait()
        finally:
            async with condition:
                in_flight -= 1

    async def wait_for_capacity() -> None:
        async with condition:
            await condition.wait_for(lambda: in_flight == 2)

    tasks = [asyncio.create_task(gate.run(request)) for _ in range(3)]
    try:
        await asyncio.wait_for(wait_for_capacity(), timeout=_TIMEOUT)
        await _wait_for_waiter(gate)
        assert in_flight == 2
        assert peak_in_flight == 2
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert in_flight == 0
    assert gate.semaphore._value == 2


async def test_request_gate_releases_after_exception() -> None:
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False)
    error = RuntimeError("request failed")

    async def fail() -> None:
        raise error

    with pytest.raises(RuntimeError) as exc_info:
        await gate.run(fail)

    assert exc_info.value is error
    await asyncio.wait_for(gate.run(_noop_request), timeout=_TIMEOUT)
    assert gate.semaphore._value == 1


async def test_request_gate_releases_after_holding_and_waiting_cancellation() -> None:
    recorder = _Recorder()
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False, recorder=recorder)
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold() -> None:
        holder_entered.set()
        await release_holder.wait()

    holder = asyncio.create_task(gate.run(hold))
    waiter: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(holder_entered.wait(), timeout=_TIMEOUT)
        waiter = asyncio.create_task(gate.run(_noop_request))
        await _wait_for_waiter(gate)

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder
    finally:
        release_holder.set()
        for task in (holder, waiter):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (holder, waiter) if task is not None), return_exceptions=True)

    assert [event.outcome for event in recorder.events].count("cancelled") == 2
    await asyncio.wait_for(gate.run(_noop_request), timeout=_TIMEOUT)
    assert gate.semaphore._value == 1


async def test_request_gate_permit_rechecks_abort_after_wait() -> None:
    aborted = False
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: aborted)
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()
    factory_calls = 0

    async def hold() -> None:
        holder_entered.set()
        await release_holder.wait()

    async def queued_permit() -> None:
        nonlocal factory_calls
        async with gate.permit():
            factory_calls += 1

    holder = asyncio.create_task(gate.run(hold))
    waiter: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(holder_entered.wait(), timeout=_TIMEOUT)
        waiter = asyncio.create_task(queued_permit())
        await _wait_for_waiter(gate)
        aborted = True
        release_holder.set()
        await holder
        with pytest.raises(request_gate.GenerationAborted):
            await waiter
    finally:
        aborted = False
        release_holder.set()
        for task in (holder, waiter):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (holder, waiter) if task is not None), return_exceptions=True)

    assert factory_calls == 0
    async with gate.permit():
        aborted = True
        with pytest.raises(request_gate.GenerationAborted):
            await gate.run(_noop_request)
        aborted = False

    assert factory_calls == 0
    await asyncio.wait_for(gate.run(_noop_request), timeout=_TIMEOUT)
    assert gate.semaphore._value == 1


async def test_request_gate_borrows_only_within_the_same_task() -> None:
    recorder = _Recorder()
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False, recorder=recorder)

    async def inner() -> str:
        return "nested"

    async def outer() -> str:
        result = await gate.run(inner)
        child = asyncio.create_task(gate.run(_noop_request))
        with pytest.raises(RuntimeError, match="belongs to another asyncio task"):
            await child
        return result

    assert await asyncio.wait_for(gate.run(outer), timeout=_TIMEOUT) == "nested"
    assert len(recorder.events) == 3
    assert sum(event.reentrant for event in recorder.events) == 1
    assert [event.outcome for event in recorder.events].count("error") == 1
    assert gate.semaphore._value == 1


async def test_request_gate_clears_stale_inherited_lease() -> None:
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False)
    start_child = asyncio.Event()
    child_created = asyncio.Event()
    child_entered = asyncio.Event()
    blocker_entered = asyncio.Event()
    release_blocker = asyncio.Event()

    async def child() -> None:
        child_created.set()
        await start_child.wait()
        async with gate.permit():
            child_entered.set()
        assert gate._lease_var.get() is None

    async def block() -> None:
        blocker_entered.set()
        await release_blocker.wait()

    child_task: asyncio.Task[None] | None = None

    async def parent() -> None:
        nonlocal child_task
        child_task = asyncio.create_task(child())
        await child_created.wait()

    await gate.run(parent)
    assert child_task is not None
    blocker = asyncio.create_task(gate.run(block))
    try:
        await asyncio.wait_for(blocker_entered.wait(), timeout=_TIMEOUT)
        start_child.set()
        await _wait_for_waiter(gate)
        assert not child_entered.is_set()

        release_blocker.set()
        await asyncio.wait_for(asyncio.gather(blocker, child_task), timeout=_TIMEOUT)
    finally:
        release_blocker.set()
        for task in (blocker, child_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(blocker, child_task, return_exceptions=True)

    assert gate.semaphore._value == 1


async def test_request_event_recorder_reports_requests_after_release() -> None:
    released_values: list[int] = []
    gate: request_gate.InferenceRequestGate

    class ReleaseObservingRecorder(_Recorder):
        def record(self, event: request_gate.RequestEvent) -> None:
            super().record(event)
            released_values.append(gate.semaphore._value)

    recorder = ReleaseObservingRecorder()
    aborted = False
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: aborted, recorder=recorder)

    async def sensitive_request() -> str:
        return "sensitive-response-body"

    assert await gate.run(sensitive_request, turn_index=7) == "sensitive-response-body"

    error = RuntimeError("sensitive-error-message")

    async def fail() -> None:
        raise error

    with pytest.raises(RuntimeError) as exc_info:
        await gate.run(fail)
    assert exc_info.value is error

    factory_calls = 0

    async def skipped() -> None:
        nonlocal factory_calls
        factory_calls += 1

    aborted = True
    with pytest.raises(request_gate.GenerationAborted):
        await gate.run(skipped)

    payloads = [asdict(event) for event in recorder.events]
    assert [payload["outcome"] for payload in payloads] == ["success", "error", "aborted"]
    assert payloads[0]["turn_index"] == 7
    assert payloads[1]["exception_type"] == "RuntimeError"
    assert payloads[2]["request_duration_s"] == 0.0
    assert factory_calls == 0
    for payload in payloads:
        _assert_event_payload(payload)
    assert released_values == [1, 1, 1]
    assert "sensitive-response-body" not in repr(payloads)
    assert "sensitive-error-message" not in repr(payloads)


async def test_request_event_recorder_failure_is_isolated() -> None:
    class FailingRecorder:
        def record(self, event: request_gate.RequestEvent) -> None:
            del event
            raise RuntimeError("recorder failed")

    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False, recorder=FailingRecorder())

    async def request() -> str:
        return "request result"

    assert await gate.run(request) == "request result"
    assert gate.semaphore._value == 1


@pytest.mark.parametrize("turn_index", ["sensitive-user-metadata", True])
async def test_request_event_rejects_non_integer_turn_index(turn_index: Any) -> None:
    recorder = _Recorder()
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False, recorder=recorder)
    factory_calls = 0

    async def request() -> None:
        nonlocal factory_calls
        factory_calls += 1

    with pytest.raises(TypeError, match="turn_index must be an int or None"):
        await gate.run(request, turn_index=turn_index)

    assert factory_calls == 0
    assert recorder.events == []
    assert gate.semaphore._value == 1


async def test_request_gate_releases_when_post_acquire_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False)
    original_monotonic = request_gate.monotonic
    calls = 0

    class SetupFailure(BaseException):
        pass

    def fail_after_acquire() -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SetupFailure
        return original_monotonic()

    monkeypatch.setattr(request_gate, "monotonic", fail_after_acquire)
    with pytest.raises(SetupFailure):
        async with gate.permit():
            pytest.fail("permit body must not run")

    assert gate._queue_depth == 0
    assert gate._in_flight == 0
    assert gate.semaphore._value == 1


def test_request_gate_module_contract() -> None:
    module_path = Path(request_gate.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (module := node.module) is not None
    }
    imported_roots.update(
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert imported_roots <= {
        "__future__",
        "asyncio",
        "collections",
        "contextlib",
        "contextvars",
        "dataclasses",
        "time",
        "typing",
        "uuid",
    }
    assert request_gate.__all__ == []


@pytest.mark.parametrize("capacity", [0, -1])
def test_request_gate_rejects_non_positive_capacity(capacity: int) -> None:
    with pytest.raises(
        ValueError,
        match=rf"InferenceRequestGate capacity must be positive, got {capacity}",
    ):
        request_gate.InferenceRequestGate(capacity=capacity, is_aborted=lambda: False)


def test_request_scoped_generate_marks_same_callable() -> None:
    async def generate() -> None:
        return None

    decorated = request_gate.request_scoped_generate(generate)

    assert decorated is generate
    assert getattr(decorated, "manages_inference_permit", None) is True
