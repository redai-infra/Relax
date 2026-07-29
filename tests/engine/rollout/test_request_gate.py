# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from relax.engine.rollout import request_gate


_TIMEOUT = 1.0


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
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False)
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
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False)

    async def inner() -> str:
        return "nested"

    async def outer() -> str:
        result = await gate.run(inner)
        child = asyncio.create_task(gate.run(_noop_request))
        with pytest.raises(RuntimeError, match="belongs to another asyncio task"):
            await child
        return result

    assert await asyncio.wait_for(gate.run(outer), timeout=_TIMEOUT) == "nested"
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


async def test_request_gate_releases_when_post_acquire_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False)

    class SetupFailure(BaseException):
        pass

    def fail_after_acquire(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SetupFailure

    monkeypatch.setattr(request_gate, "_Lease", fail_after_acquire)
    with pytest.raises(SetupFailure):
        async with gate.permit():
            pytest.fail("permit body must not run")

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
        "typing",
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
