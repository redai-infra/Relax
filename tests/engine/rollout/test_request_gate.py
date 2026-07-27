# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
import asyncio
import importlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


_REQUEST_GATE_MODULE = "relax.engine.rollout.request_gate"
SCENARIO_TIMEOUT_S = 3.0
_ALLOWED_REQUEST_EVENT_FIELDS = {
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


def _load_request_gate_module() -> ModuleType:
    module: ModuleType | None = None
    try:
        module = importlib.import_module(_REQUEST_GATE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _REQUEST_GATE_MODULE:
            raise

    assert module is not None, f"{_REQUEST_GATE_MODULE} must expose the Request Gate v2 API"
    return module


async def _cancel_tasks(tasks: list[asyncio.Task[Any] | None]) -> None:
    existing = [task for task in tasks if task is not None]
    for task in existing:
        if not task.done():
            task.cancel()
    if existing:
        await asyncio.wait_for(
            asyncio.gather(*existing, return_exceptions=True),
            timeout=SCENARIO_TIMEOUT_S,
        )


@contextmanager
def _observe_next_acquire(semaphore: asyncio.Semaphore) -> Iterator[asyncio.Event]:
    acquire_started = asyncio.Event()
    original_acquire = semaphore.acquire

    async def observed_acquire() -> bool:
        acquire_started.set()
        return await original_acquire()

    semaphore.acquire = observed_acquire
    try:
        yield acquire_started
    finally:
        semaphore.acquire = original_acquire


def _event_payload(event: Any) -> dict[str, Any]:
    if is_dataclass(event):
        return asdict(event)
    if hasattr(event, "__dict__"):
        return dict(vars(event))
    raise AssertionError(f"Request event must expose structured fields, got {type(event).__name__}")


def _assert_event_contract(
    payload: dict[str, Any],
    *,
    outcome: str,
    exception_type: str | None,
    reentrant: bool,
) -> None:
    assert set(payload) == _ALLOWED_REQUEST_EVENT_FIELDS
    assert payload["outcome"] == outcome
    assert payload["exception_type"] == exception_type
    assert payload["reentrant"] is reentrant
    assert payload["capacity"] == 1
    assert len(payload["request_id"]) == 32
    int(payload["request_id"], 16)
    for field in ("queue_depth_at_start", "queue_depth_at_acquire"):
        assert type(payload[field]) is int
        assert payload[field] >= 0
    for field in ("in_flight_at_start", "in_flight_at_end"):
        assert type(payload[field]) is int
        assert 0 <= payload[field] <= payload["capacity"]
    for field in ("relative_start_s", "permit_wait_s", "request_duration_s", "total_duration_s"):
        assert payload[field] >= 0
    assert payload["total_duration_s"] >= payload["permit_wait_s"]
    assert payload["total_duration_s"] >= payload["request_duration_s"]


def _event_with_exception(events: list[Any], exception_type: str | None) -> dict[str, Any]:
    matches = [payload for event in events if (payload := _event_payload(event))["exception_type"] == exception_type]
    assert len(matches) == 1
    return matches[0]


class _GateState:
    def __init__(self, request_gate: ModuleType, capacity: int) -> None:
        self.capacity = capacity
        self.aborted = False
        self.waiting = 0
        self.waiting_changed = asyncio.Event()
        self.gate = request_gate.InferenceRequestGate(
            capacity=capacity,
            is_aborted=lambda: self.aborted,
        )
        self.semaphore = self.gate.semaphore


class _GateScenarioAdapter:
    def __init__(self, request_gate: ModuleType) -> None:
        self._request_gate = request_gate
        self._states: list[_GateState] = []

    def make_request_state(self, capacity: int) -> _GateState:
        state = _GateState(self._request_gate, capacity)
        self._states.append(state)
        return state

    def assert_capacity_recovered(self) -> None:
        for state in self._states:
            assert getattr(state.semaphore, "_value", None) == state.capacity

    @asynccontextmanager
    async def request_scope(self, state: _GateState) -> AsyncIterator[None]:
        waiting = True
        state.waiting += 1
        state.waiting_changed.set()
        try:
            async with state.gate.permit():
                state.waiting -= 1
                waiting = False
                yield
        finally:
            if waiting:
                state.waiting -= 1

    @staticmethod
    def set_aborted(state: _GateState, aborted: bool) -> None:
        state.aborted = aborted

    def generation_aborted_type(self) -> type[BaseException] | None:
        aborted_type = getattr(self._request_gate, "GenerationAborted", None)
        if isinstance(aborted_type, type) and issubclass(aborted_type, BaseException):
            return aborted_type
        return None

    @staticmethod
    async def wait_for_waiter(state: _GateState) -> None:
        async def _wait() -> None:
            while state.waiting == 0:
                state.waiting_changed.clear()
                if state.waiting == 0:
                    await state.waiting_changed.wait()

        await asyncio.wait_for(_wait(), timeout=SCENARIO_TIMEOUT_S)

    async def observe_gate_run_contract(self) -> dict[str, Any]:
        gate_type = self._request_gate.InferenceRequestGate
        aborted_type = self.generation_aborted_type()
        assert aborted_type is not None

        aborted = False
        success_factory_calls = 0
        abort_factory_calls = 0
        queued_holder_release = asyncio.Event()
        queued_holder_task = None
        queued_waiter_task = None
        gate = gate_type(capacity=1, is_aborted=lambda: aborted)
        permit_available = callable(getattr(gate, "permit", None))
        permit_entered = False
        borrowed_abort_factory_calls = 0
        borrowed_abort_exception = None
        queued_abort_factory_calls = 0
        queued_abort_exception = None

        try:
            if permit_available:
                async with gate.permit():
                    permit_entered = True
                    aborted = True

                    async def borrowed_abort_factory() -> None:
                        nonlocal borrowed_abort_factory_calls
                        borrowed_abort_factory_calls += 1

                    with pytest.raises(aborted_type) as exc_info:
                        await gate.run(borrowed_abort_factory)
                    borrowed_abort_exception = type(exc_info.value).__name__
                    aborted = False

            queued_holder_entered = asyncio.Event()

            async def queued_holder() -> None:
                async with gate.permit():
                    queued_holder_entered.set()
                    await queued_holder_release.wait()

            async def queued_abort_factory() -> None:
                nonlocal queued_abort_factory_calls
                queued_abort_factory_calls += 1

            if permit_available:
                queued_holder_task = asyncio.create_task(queued_holder())
                await asyncio.wait_for(queued_holder_entered.wait(), timeout=SCENARIO_TIMEOUT_S)
                with _observe_next_acquire(gate.semaphore) as acquire_started:
                    queued_waiter_task = asyncio.create_task(gate.run(queued_abort_factory))
                    await asyncio.wait_for(acquire_started.wait(), timeout=SCENARIO_TIMEOUT_S)
                    aborted = True
                    queued_holder_release.set()
                    await asyncio.wait_for(queued_holder_task, timeout=SCENARIO_TIMEOUT_S)
                    with pytest.raises(aborted_type) as exc_info:
                        await asyncio.wait_for(queued_waiter_task, timeout=SCENARIO_TIMEOUT_S)
                    queued_abort_exception = type(exc_info.value).__name__
                aborted = False

            async def success_factory() -> str:
                nonlocal success_factory_calls
                success_factory_calls += 1
                return "factory-result"

            success_result = await gate.run(success_factory)
            aborted = True

            async def abort_factory() -> str:
                nonlocal abort_factory_calls
                abort_factory_calls += 1
                return "must-not-run"

            with pytest.raises(aborted_type) as exc_info:
                await gate.run(abort_factory)
            abort_exception = type(exc_info.value).__name__

            return {
                "available": True,
                "timed_out": False,
                "error_type": None,
                "permit_available": permit_available,
                "permit_entered": permit_entered,
                "borrowed_abort_factory_calls": borrowed_abort_factory_calls,
                "borrowed_abort_exception": borrowed_abort_exception,
                "queued_abort_factory_calls": queued_abort_factory_calls,
                "queued_abort_exception": queued_abort_exception,
                "semaphore_count": len(
                    {id(value) for value in vars(gate).values() if isinstance(value, asyncio.Semaphore)}
                ),
                "success_result": success_result,
                "success_factory_calls": success_factory_calls,
                "abort_factory_calls": abort_factory_calls,
                "abort_exception": abort_exception,
                "recovered": getattr(gate.semaphore, "_value", None) == 1,
            }
        finally:
            aborted = False
            queued_holder_release.set()
            await _cancel_tasks([queued_holder_task, queued_waiter_task])

    async def observe_recorder_contract(self) -> dict[str, Any]:
        gate_type = self._request_gate.InferenceRequestGate
        aborted_type = self.generation_aborted_type()
        assert aborted_type is not None

        events: list[Any] = []
        released_values: list[int | None] = []
        gate = None

        class Recorder:
            def record(self, event: Any) -> None:
                events.append(event)
                released_values.append(getattr(gate.semaphore, "_value", None))

        class RecorderBaseException(BaseException):
            pass

        class ThrowingRecorder:
            def __init__(
                self,
                recorded_events: list[Any],
                permit_released: list[bool],
                failure_type: type[BaseException] = RuntimeError,
            ) -> None:
                self.recorded_events = recorded_events
                self.permit_released = permit_released
                self.failure_type = failure_type
                self.gate: Any | None = None

            def record(self, event: Any) -> None:
                self.recorded_events.append(event)
                assert self.gate is not None
                self.permit_released.append(not self.gate.semaphore.locked())
                raise self.failure_type("synthetic recorder failure")

        gate = gate_type(capacity=1, is_aborted=lambda: False, recorder=Recorder())
        gates = [gate]

        async def sensitive_factory() -> str:
            return "sensitive-response-body"

        await gate.run(sensitive_factory, turn_index=7)
        event_payload = _event_payload(events[0])
        _assert_event_contract(
            event_payload,
            outcome="success",
            exception_type=None,
            reentrant=False,
        )
        assert event_payload["turn_index"] == 7

        failure_events: list[Any] = []
        failure_permit_released: list[bool] = []
        failure_recorder = ThrowingRecorder(failure_events, failure_permit_released)
        failing_gate = gate_type(
            capacity=1,
            is_aborted=lambda: False,
            recorder=failure_recorder,
        )
        failure_recorder.gate = failing_gate
        gates.append(failing_gate)

        async def isolated_factory() -> str:
            return "recorder-isolated"

        recorder_failure_result = await failing_gate.run(isolated_factory)
        request_error = RuntimeError("sensitive-exception-message")

        async def failing_factory() -> str:
            raise request_error

        try:
            await failing_gate.run(failing_factory)
        except RuntimeError as exc:
            request_exception_identity_preserved = exc is request_error
        else:
            request_exception_identity_preserved = False

        holding_release = asyncio.Event()
        holding_entered = asyncio.Event()

        async def holding_factory() -> None:
            holding_entered.set()
            await holding_release.wait()

        holding_task = asyncio.create_task(failing_gate.run(holding_factory))
        try:
            await asyncio.wait_for(holding_entered.wait(), timeout=SCENARIO_TIMEOUT_S)
            holding_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(holding_task, timeout=SCENARIO_TIMEOUT_S)
            holding_cancellation_preserved = True
        finally:
            holding_release.set()
            await _cancel_tasks([holding_task])

        base_exception_events: list[Any] = []
        base_exception_permit_released: list[bool] = []
        base_exception_recorder = ThrowingRecorder(
            base_exception_events,
            base_exception_permit_released,
            RecorderBaseException,
        )
        base_exception_gate = gate_type(
            capacity=1,
            is_aborted=lambda: False,
            recorder=base_exception_recorder,
        )
        base_exception_recorder.gate = base_exception_gate
        gates.append(base_exception_gate)
        base_exception_result = await base_exception_gate.run(isolated_factory)

        waiting_release = asyncio.Event()
        waiting_holder_entered = asyncio.Event()
        waiting_factory_calls = 0
        waiting_events: list[Any] = []
        waiting_permit_released: list[bool] = []
        waiting_recorder = ThrowingRecorder(waiting_events, waiting_permit_released)
        waiting_gate = gate_type(
            capacity=1,
            is_aborted=lambda: False,
            recorder=waiting_recorder,
        )
        waiting_recorder.gate = waiting_gate
        gates.append(waiting_gate)

        async def waiting_holder_factory() -> None:
            waiting_holder_entered.set()
            await waiting_release.wait()

        async def waiting_factory() -> None:
            nonlocal waiting_factory_calls
            waiting_factory_calls += 1

        waiting_holder_task = asyncio.create_task(waiting_gate.run(waiting_holder_factory))
        waiting_task = None
        try:
            await asyncio.wait_for(waiting_holder_entered.wait(), timeout=SCENARIO_TIMEOUT_S)
            with _observe_next_acquire(waiting_gate.semaphore) as acquire_started:
                waiting_task = asyncio.create_task(waiting_gate.run(waiting_factory))
                await asyncio.wait_for(acquire_started.wait(), timeout=SCENARIO_TIMEOUT_S)
                waiting_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(waiting_task, timeout=SCENARIO_TIMEOUT_S)
            waiting_cancellation_preserved = True
            waiting_release.set()
            await asyncio.wait_for(waiting_holder_task, timeout=SCENARIO_TIMEOUT_S)
        finally:
            waiting_release.set()
            await _cancel_tasks([waiting_holder_task, waiting_task])

        abort_factory_calls = 0
        abort_events: list[Any] = []
        abort_permit_released: list[bool] = []
        abort_recorder = ThrowingRecorder(abort_events, abort_permit_released)
        abort_gate = gate_type(
            capacity=1,
            is_aborted=lambda: True,
            recorder=abort_recorder,
        )
        abort_recorder.gate = abort_gate
        gates.append(abort_gate)

        async def abort_factory() -> None:
            nonlocal abort_factory_calls
            abort_factory_calls += 1

        with pytest.raises(aborted_type) as exc_info:
            await abort_gate.run(abort_factory)
        abort_exception = type(exc_info.value).__name__

        nested_events: list[Any] = []
        nested_release_counts: list[int] = []
        nested_release_count = 0

        class NestedRecorder:
            def record(self, event: Any) -> None:
                nested_events.append(event)
                nested_release_counts.append(nested_release_count)

        nested_gate = gate_type(capacity=1, is_aborted=lambda: False, recorder=NestedRecorder())
        gates.append(nested_gate)
        original_nested_release = nested_gate.semaphore.release

        def observed_nested_release() -> None:
            nonlocal nested_release_count
            nested_release_count += 1
            original_nested_release()

        nested_gate.semaphore.release = observed_nested_release
        nested_inner_finished = asyncio.Event()
        nested_outer_release = asyncio.Event()
        nested_contender_entered = asyncio.Event()
        nested_contender_release = asyncio.Event()

        async def nested_factory() -> str:
            return "nested-result"

        async def outer_factory() -> str:
            result = await nested_gate.run(nested_factory, turn_index=1)
            nested_inner_finished.set()
            await nested_outer_release.wait()
            return result

        async def nested_contender() -> None:
            async with nested_gate.permit():
                nested_contender_entered.set()
                await nested_contender_release.wait()

        nested_outer_task = asyncio.create_task(nested_gate.run(outer_factory, turn_index=0))
        nested_contender_task = None
        try:
            await asyncio.wait_for(nested_inner_finished.wait(), timeout=SCENARIO_TIMEOUT_S)
            assert nested_events == []
            with _observe_next_acquire(nested_gate.semaphore) as nested_acquire_started:
                nested_contender_task = asyncio.create_task(nested_contender())
                await asyncio.wait_for(nested_acquire_started.wait(), timeout=SCENARIO_TIMEOUT_S)
                assert not nested_contender_entered.is_set()
                assert nested_events == []
            nested_outer_release.set()
            nested_result = await asyncio.wait_for(nested_outer_task, timeout=SCENARIO_TIMEOUT_S)
            await asyncio.wait_for(nested_contender_entered.wait(), timeout=SCENARIO_TIMEOUT_S)
            assert len(nested_events) == 2
            nested_contender_release.set()
            await asyncio.wait_for(nested_contender_task, timeout=SCENARIO_TIMEOUT_S)
        finally:
            nested_outer_release.set()
            nested_contender_release.set()
            await _cancel_tasks([nested_outer_task, nested_contender_task])
        nested_payloads = [_event_payload(event) for event in nested_events]

        isolated_payload = _event_with_exception(failure_events, None)
        error_payload = _event_with_exception(failure_events, "RuntimeError")
        holding_cancelled_payload = _event_with_exception(failure_events, "CancelledError")
        waiting_success_payload = _event_with_exception(waiting_events, None)
        waiting_cancelled_payload = _event_with_exception(waiting_events, "CancelledError")
        abort_payload = _event_with_exception(abort_events, aborted_type.__name__)
        base_exception_payload = _event_with_exception(base_exception_events, None)

        _assert_event_contract(
            isolated_payload,
            outcome="success",
            exception_type=None,
            reentrant=False,
        )
        _assert_event_contract(
            error_payload,
            outcome="error",
            exception_type="RuntimeError",
            reentrant=False,
        )
        for cancelled_payload in (holding_cancelled_payload, waiting_cancelled_payload):
            _assert_event_contract(
                cancelled_payload,
                outcome="cancelled",
                exception_type="CancelledError",
                reentrant=False,
            )
        _assert_event_contract(
            waiting_success_payload,
            outcome="success",
            exception_type=None,
            reentrant=False,
        )
        _assert_event_contract(
            abort_payload,
            outcome="aborted",
            exception_type=aborted_type.__name__,
            reentrant=False,
        )
        _assert_event_contract(
            base_exception_payload,
            outcome="success",
            exception_type=None,
            reentrant=False,
        )
        assert len(nested_payloads) == 2
        assert sum(payload["reentrant"] is True for payload in nested_payloads) == 1
        assert sum(payload["reentrant"] is False for payload in nested_payloads) == 1
        for nested_payload in nested_payloads:
            _assert_event_contract(
                nested_payload,
                outcome="success",
                exception_type=None,
                reentrant=bool(nested_payload["reentrant"]),
            )

        all_payloads = [
            event_payload,
            *[_event_payload(event) for event in failure_events],
            *[_event_payload(event) for event in waiting_events],
            *[_event_payload(event) for event in abort_events],
            *[_event_payload(event) for event in base_exception_events],
            *nested_payloads,
        ]
        request_ids = [payload["request_id"] for payload in all_payloads]
        assert len(request_ids) == len(set(request_ids))
        assert failure_permit_released == [True, True, True]
        assert base_exception_permit_released == [True]
        assert abort_permit_released == [True]
        assert waiting_permit_released == [False, True]
        serialized_events = json.dumps(all_payloads, sort_keys=True)

        return {
            "available": True,
            "timed_out": False,
            "error_type": None,
            "event_fields": sorted(event_payload),
            "event_count": len(events),
            "event_outcome": event_payload.get("outcome"),
            "release_before_record": released_values == [1],
            "recorder_failure_result": recorder_failure_result,
            "base_exception_recorder_result": base_exception_result,
            "acquired_paths_release_before_record": (
                failure_permit_released == [True, True, True]
                and base_exception_permit_released == [True]
                and abort_permit_released == [True]
                and waiting_permit_released == [False, True]
            ),
            "request_exception_identity_preserved": request_exception_identity_preserved,
            "holding_cancellation_preserved": holding_cancellation_preserved,
            "waiting_cancellation_preserved": waiting_cancellation_preserved,
            "abort_exception": abort_exception,
            "abort_factory_calls": abort_factory_calls,
            "nested_result": nested_result,
            "nested_event_count": len(nested_payloads),
            "nested_reentrant_count": sum(payload.get("reentrant") is True for payload in nested_payloads),
            "nested_release_before_record": nested_release_counts == [1, 1],
            "contains_sensitive_body": any(
                sensitive_value in serialized_events
                for sensitive_value in ("sensitive-response-body", "sensitive-exception-message")
            ),
            "recovered": (
                waiting_factory_calls == 0
                and all(getattr(current_gate.semaphore, "_value", None) == 1 for current_gate in gates)
            ),
        }


class _RequestProbe:
    def __init__(self) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0
        self.started = 0
        self._condition = asyncio.Condition()

    async def execute(self, release: asyncio.Event) -> None:
        async with self._condition:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.started += 1
            self._condition.notify_all()
        try:
            await release.wait()
        finally:
            async with self._condition:
                self.in_flight -= 1
                self._condition.notify_all()

    async def wait_for_in_flight(self, count: int) -> None:
        async def _wait() -> None:
            async with self._condition:
                await self._condition.wait_for(lambda: self.in_flight >= count)

        await asyncio.wait_for(_wait(), timeout=SCENARIO_TIMEOUT_S)


async def _assert_waiter_queued(
    adapter: _GateScenarioAdapter,
    state: _GateState,
    entered: asyncio.Event,
) -> None:
    waiter_observed = asyncio.create_task(adapter.wait_for_waiter(state))
    body_entered = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait(
            (waiter_observed, body_entered),
            timeout=SCENARIO_TIMEOUT_S,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if body_entered in done:
            raise AssertionError("request entered while configured capacity was full")
        if waiter_observed not in done:
            raise TimeoutError("request did not become an observable semaphore waiter")
        await waiter_observed
    finally:
        await _cancel_tasks([waiter_observed, body_entered])


async def _assert_exact_capacity(adapter: _GateScenarioAdapter, state: _GateState, capacity: int) -> None:
    probe = _RequestProbe()
    release = asyncio.Event()
    extra_attempted = asyncio.Event()
    extra_entered = asyncio.Event()

    async def _holder() -> None:
        async with adapter.request_scope(state):
            await probe.execute(release)

    async def _extra() -> None:
        extra_attempted.set()
        async with adapter.request_scope(state):
            extra_entered.set()
            await probe.execute(release)

    holders = [asyncio.create_task(_holder()) for _ in range(capacity)]
    extra = None
    try:
        await probe.wait_for_in_flight(capacity)
        extra = asyncio.create_task(_extra())
        await asyncio.wait_for(extra_attempted.wait(), timeout=SCENARIO_TIMEOUT_S)
        await _assert_waiter_queued(adapter, state, extra_entered)
        assert probe.peak_in_flight == capacity
        release.set()
        await asyncio.wait_for(asyncio.gather(*holders, extra), timeout=SCENARIO_TIMEOUT_S)
    finally:
        release.set()
        await _cancel_tasks([*holders, extra])

    assert probe.started == capacity + 1
    assert probe.peak_in_flight == capacity
    assert probe.in_flight == 0


async def _scenario_capacity_and_success_release(adapter: _GateScenarioAdapter) -> None:
    state = adapter.make_request_state(2)
    await _assert_exact_capacity(adapter, state, capacity=2)


async def _scenario_exception_release(adapter: _GateScenarioAdapter) -> None:
    state = adapter.make_request_state(1)

    try:
        async with adapter.request_scope(state):
            raise RuntimeError("scenario request failure")
    except RuntimeError as exc:
        assert str(exc) == "scenario request failure"
    else:
        raise AssertionError("request exception was swallowed")

    await _assert_exact_capacity(adapter, state, capacity=1)


async def _scenario_holding_cancellation_release(adapter: _GateScenarioAdapter) -> None:
    state = adapter.make_request_state(1)
    entered = asyncio.Event()
    hold = asyncio.Event()

    async def _holder() -> None:
        async with adapter.request_scope(state):
            entered.set()
            await hold.wait()

    task = asyncio.create_task(_holder())
    try:
        await asyncio.wait_for(entered.wait(), timeout=SCENARIO_TIMEOUT_S)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        hold.set()
        await _cancel_tasks([task])

    await _assert_exact_capacity(adapter, state, capacity=1)


async def _scenario_waiting_cancellation_no_overrelease(adapter: _GateScenarioAdapter) -> None:
    state = adapter.make_request_state(1)
    holder_entered = asyncio.Event()
    waiter_attempted = asyncio.Event()
    waiter_entered = asyncio.Event()
    release = asyncio.Event()

    async def _holder() -> None:
        async with adapter.request_scope(state):
            holder_entered.set()
            await release.wait()

    async def _waiter() -> None:
        waiter_attempted.set()
        async with adapter.request_scope(state):
            waiter_entered.set()

    holder = asyncio.create_task(_holder())
    waiter = None
    try:
        await asyncio.wait_for(holder_entered.wait(), timeout=SCENARIO_TIMEOUT_S)
        waiter = asyncio.create_task(_waiter())
        await asyncio.wait_for(waiter_attempted.wait(), timeout=SCENARIO_TIMEOUT_S)
        await _assert_waiter_queued(adapter, state, waiter_entered)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await holder
    finally:
        release.set()
        await _cancel_tasks([holder, waiter])

    await _assert_exact_capacity(adapter, state, capacity=1)


async def _scenario_same_task_reentrant_borrow(adapter: _GateScenarioAdapter) -> None:
    state = adapter.make_request_state(1)
    entered = 0

    async def _nested() -> None:
        nonlocal entered
        async with adapter.request_scope(state):
            async with adapter.request_scope(state):
                entered += 1

    await asyncio.wait_for(_nested(), timeout=SCENARIO_TIMEOUT_S)
    assert entered == 1
    await _assert_exact_capacity(adapter, state, capacity=1)


async def _scenario_cross_task_active_lease_fails_fast(adapter: _GateScenarioAdapter) -> None:
    state = adapter.make_request_state(1)
    child = None

    async def _child() -> None:
        async with adapter.request_scope(state):
            raise AssertionError("cross-task active lease entered its request body")

    try:
        async with adapter.request_scope(state):
            child = asyncio.create_task(_child())
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(child, timeout=SCENARIO_TIMEOUT_S / 2)
    finally:
        await _cancel_tasks([child])

    await _assert_exact_capacity(adapter, state, capacity=1)


async def _scenario_stale_inherited_lease_acquires_independently(adapter: _GateScenarioAdapter) -> None:
    state = adapter.make_request_state(1)
    child_created = asyncio.Event()
    child_attempted = asyncio.Event()
    start_child = asyncio.Event()
    child_entered = asyncio.Event()
    blocker_entered = asyncio.Event()
    release_blocker = asyncio.Event()

    async def _child() -> None:
        child_created.set()
        await start_child.wait()
        child_attempted.set()
        async with adapter.request_scope(state):
            child_entered.set()

    async def _blocker() -> None:
        async with adapter.request_scope(state):
            blocker_entered.set()
            await release_blocker.wait()

    async with adapter.request_scope(state):
        child = asyncio.create_task(_child())
        await asyncio.wait_for(child_created.wait(), timeout=SCENARIO_TIMEOUT_S)

    blocker = asyncio.create_task(_blocker())
    try:
        await asyncio.wait_for(blocker_entered.wait(), timeout=SCENARIO_TIMEOUT_S)
        start_child.set()
        await asyncio.wait_for(child_attempted.wait(), timeout=SCENARIO_TIMEOUT_S)
        await _assert_waiter_queued(adapter, state, child_entered)
        assert not child_entered.is_set(), "stale inherited lease borrowed without acquiring capacity"
        release_blocker.set()
        await asyncio.wait_for(child_entered.wait(), timeout=SCENARIO_TIMEOUT_S)
        await asyncio.wait_for(child, timeout=SCENARIO_TIMEOUT_S)
    finally:
        release_blocker.set()
        await _cancel_tasks([child, blocker])


async def _scenario_abort_after_wait_skips_request(adapter: _GateScenarioAdapter) -> None:
    aborted_type = adapter.generation_aborted_type()
    assert aborted_type is not None

    state = adapter.make_request_state(1)
    holder_entered = asyncio.Event()
    waiter_attempted = asyncio.Event()
    waiter_body_entered = asyncio.Event()
    release = asyncio.Event()

    async def _holder() -> None:
        async with adapter.request_scope(state):
            holder_entered.set()
            await release.wait()

    async def _waiter() -> None:
        waiter_attempted.set()
        async with adapter.request_scope(state):
            waiter_body_entered.set()

    holder = asyncio.create_task(_holder())
    waiter = None
    try:
        await asyncio.wait_for(holder_entered.wait(), timeout=SCENARIO_TIMEOUT_S)
        waiter = asyncio.create_task(_waiter())
        await asyncio.wait_for(waiter_attempted.wait(), timeout=SCENARIO_TIMEOUT_S)
        await _assert_waiter_queued(adapter, state, waiter_body_entered)
        adapter.set_aborted(state, True)
        release.set()
        with pytest.raises(aborted_type):
            await asyncio.wait_for(waiter, timeout=SCENARIO_TIMEOUT_S)
    finally:
        adapter.set_aborted(state, False)
        release.set()
        await _cancel_tasks([holder, waiter])

    assert not waiter_body_entered.is_set()
    await _assert_exact_capacity(adapter, state, capacity=1)


_GATE_SCENARIOS = (
    _scenario_capacity_and_success_release,
    _scenario_exception_release,
    _scenario_holding_cancellation_release,
    _scenario_waiting_cancellation_no_overrelease,
    _scenario_same_task_reentrant_borrow,
    _scenario_cross_task_active_lease_fails_fast,
    _scenario_stale_inherited_lease_acquires_independently,
    _scenario_abort_after_wait_skips_request,
)
_GATE_SCENARIO_NAMES = tuple(scenario.__name__.removeprefix("_scenario_") for scenario in _GATE_SCENARIOS)


def _assert_clean_observation(observation: dict[str, Any]) -> None:
    assert observation.get("timed_out") is False, observation
    assert observation.get("error_type") is None, observation
    assert observation.get("recovered") is True, observation


@pytest.mark.parametrize("scenario", _GATE_SCENARIOS, ids=_GATE_SCENARIO_NAMES)
async def test_inference_request_gate_contract(
    scenario: Callable[[_GateScenarioAdapter], Awaitable[None]],
) -> None:
    request_gate = _load_request_gate_module()
    adapter = _GateScenarioAdapter(request_gate)
    await scenario(adapter)
    adapter.assert_capacity_recovered()


async def test_inference_request_gate_run_contract() -> None:
    request_gate = _load_request_gate_module()
    observation = await _GateScenarioAdapter(request_gate).observe_gate_run_contract()

    assert observation["available"] is True, observation
    _assert_clean_observation(observation)
    assert observation["permit_available"] is True, observation
    assert observation["permit_entered"] is True, observation
    assert observation["borrowed_abort_factory_calls"] == 0, observation
    assert observation["borrowed_abort_exception"] == "GenerationAborted", observation
    assert observation["queued_abort_factory_calls"] == 0, observation
    assert observation["queued_abort_exception"] == "GenerationAborted", observation
    assert observation["semaphore_count"] == 1, observation
    assert observation["success_result"] == "factory-result", observation
    assert observation["success_factory_calls"] == 1, observation
    assert observation["abort_factory_calls"] == 0, observation
    assert observation["abort_exception"] == "GenerationAborted", observation


async def test_request_event_recorder_contract() -> None:
    request_gate = _load_request_gate_module()
    observation = await _GateScenarioAdapter(request_gate).observe_recorder_contract()

    assert observation["available"] is True, observation
    _assert_clean_observation(observation)
    assert set(observation["event_fields"]) == _ALLOWED_REQUEST_EVENT_FIELDS, observation
    assert observation["event_count"] == 1, observation
    assert observation["event_outcome"] == "success", observation
    assert observation["release_before_record"] is True, observation
    assert observation["recorder_failure_result"] == "recorder-isolated", observation
    assert observation["request_exception_identity_preserved"] is True, observation
    assert observation["holding_cancellation_preserved"] is True, observation
    assert observation["waiting_cancellation_preserved"] is True, observation
    assert observation["abort_exception"] == "GenerationAborted", observation
    assert observation["abort_factory_calls"] == 0, observation
    assert observation["nested_result"] == "nested-result", observation
    assert observation["nested_event_count"] == 2, observation
    assert observation["nested_reentrant_count"] == 1, observation
    assert observation["nested_release_before_record"] is True, observation
    assert observation["contains_sensitive_body"] is False, observation


@pytest.mark.parametrize("turn_index", ["sensitive-user-metadata", True])
async def test_request_event_rejects_non_integer_turn_index_without_recording(turn_index: Any) -> None:
    request_gate = _load_request_gate_module()
    events: list[Any] = []
    factory_calls = 0

    class Recorder:
        def record(self, event: Any) -> None:
            events.append(event)

    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False, recorder=Recorder())

    async def factory() -> None:
        nonlocal factory_calls
        factory_calls += 1

    with pytest.raises(TypeError, match="turn_index must be an int or None"):
        await gate.run(factory, turn_index=turn_index)

    assert factory_calls == 0
    assert events == []
    assert gate.semaphore._value == 1


async def test_request_gate_releases_when_post_acquire_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    request_gate = _load_request_gate_module()
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

    assert calls == 2
    assert gate._queue_depth == 0
    assert gate._in_flight == 0
    assert gate.semaphore._value == 1


async def test_request_gate_clears_inactive_inherited_lease() -> None:
    request_gate = _load_request_gate_module()
    gate = request_gate.InferenceRequestGate(capacity=1, is_aborted=lambda: False)
    start_child = asyncio.Event()

    async def child() -> None:
        await start_child.wait()
        async with gate.permit():
            pass
        assert gate._lease_var.get() is None

    async with gate.permit():
        child_task = asyncio.create_task(child())

    start_child.set()
    await asyncio.wait_for(child_task, timeout=SCENARIO_TIMEOUT_S)
    assert gate.semaphore._value == 1


def test_request_gate_uses_only_standard_library_imports() -> None:
    request_gate = _load_request_gate_module()
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


def test_request_scoped_generate_marks_same_callable() -> None:
    request_gate = _load_request_gate_module()
    decorator = getattr(request_gate, "request_scoped_generate", None)
    assert callable(decorator)

    async def generate() -> None:
        return None

    decorated = decorator(generate)
    assert decorated is generate
    assert getattr(decorated, "manages_inference_permit", None) is True
