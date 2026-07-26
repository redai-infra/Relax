# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# CPU async unit tests for ModelRequestScheduler and request_model_aware contract.

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from relax.engine.rollout.sglang_rollout import (
    ModelRequestScheduler,
    RolloutRequestAborted,
    _is_request_model_aware,
    _validate_request_model_signature,
    request_model_aware,
)


class _FakeState:
    def __init__(self) -> None:
        self.aborted = False


def _scheduler(capacity: int = 1) -> tuple[ModelRequestScheduler, _FakeState]:
    state = _FakeState()
    return ModelRequestScheduler(state, capacity), state


class _ActiveCounter:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def enter(self) -> None:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)

    def leave(self) -> None:
        self.active -= 1


@pytest.mark.asyncio
async def test_model_request_scheduler_respects_capacity() -> None:
    scheduler, _ = _scheduler(capacity=2)
    counter = _ActiveCounter()
    release = asyncio.Event()
    entered = asyncio.Event()

    async def fake_post(url, payload, headers=None):
        counter.enter()
        if counter.active == 2:
            entered.set()
        await release.wait()
        counter.leave()
        return {"ok": True}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=fake_post):
        tasks = [
            asyncio.create_task(scheduler.request("http://x", {"i": i})) for i in range(3)
        ]
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        assert counter.max_active <= 2
        assert counter.active == 2
        release.set()
        await asyncio.gather(*tasks)
        assert counter.max_active <= 2
        assert counter.calls == 3


@pytest.mark.asyncio
async def test_model_request_scheduler_releases_on_success() -> None:
    scheduler, _ = _scheduler(capacity=1)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def fake_post(url, payload, headers=None):
        if payload["id"] == 1:
            first_entered.set()
            await release_first.wait()
            return {"id": 1}
        second_entered.set()
        return {"id": 2}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=fake_post):
        t1 = asyncio.create_task(scheduler.request("http://x", {"id": 1}))
        await asyncio.wait_for(first_entered.wait(), timeout=1.0)
        t2 = asyncio.create_task(scheduler.request("http://x", {"id": 2}))
        await asyncio.sleep(0)
        assert not second_entered.is_set()
        release_first.set()
        await asyncio.wait_for(second_entered.wait(), timeout=1.0)
        assert await t1 == {"id": 1}
        assert await t2 == {"id": 2}


@pytest.mark.asyncio
async def test_model_request_scheduler_releases_on_exception() -> None:
    scheduler, _ = _scheduler(capacity=1)

    async def boom(url, payload, headers=None):
        raise RuntimeError("http failed")

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=boom):
        with pytest.raises(RuntimeError, match="http failed"):
            await scheduler.request("http://x", {})

    async def ok(url, payload, headers=None):
        return {"ok": True}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=ok):
        assert await scheduler.request("http://x", {}) == {"ok": True}


@pytest.mark.asyncio
async def test_model_request_scheduler_releases_on_cancel_while_holding() -> None:
    scheduler, _ = _scheduler(capacity=1)
    holding = asyncio.Event()

    async def fake_post(url, payload, headers=None):
        holding.set()
        await asyncio.Event().wait()  # block forever
        return {}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=fake_post):
        task = asyncio.create_task(scheduler.request("http://x", {}))
        await asyncio.wait_for(holding.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    done = asyncio.Event()

    async def ok(url, payload, headers=None):
        done.set()
        return {"ok": True}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=ok):
        await asyncio.wait_for(scheduler.request("http://x", {}), timeout=1.0)
        assert done.is_set()


@pytest.mark.asyncio
async def test_model_request_scheduler_cancel_while_waiting() -> None:
    scheduler, _ = _scheduler(capacity=1)
    first_holding = asyncio.Event()
    release_first = asyncio.Event()

    async def fake_post(url, payload, headers=None):
        if payload.get("id") == 1:
            first_holding.set()
            await release_first.wait()
            return {"id": 1}
        return {"id": 2}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=fake_post):
        t1 = asyncio.create_task(scheduler.request("http://x", {"id": 1}))
        await asyncio.wait_for(first_holding.wait(), timeout=1.0)
        t2 = asyncio.create_task(scheduler.request("http://x", {"id": 2}))
        await asyncio.sleep(0)
        t2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t2
        release_first.set()
        assert await t1 == {"id": 1}


@pytest.mark.asyncio
async def test_model_request_scheduler_abort_after_acquire() -> None:
    scheduler, state = _scheduler(capacity=1)
    first_holding = asyncio.Event()
    release_first = asyncio.Event()
    http_calls = 0

    async def fake_post(url, payload, headers=None):
        nonlocal http_calls
        http_calls += 1
        if payload.get("id") == 1:
            first_holding.set()
            await release_first.wait()
            return {"id": 1}
        return {"id": 2}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=fake_post):
        t1 = asyncio.create_task(scheduler.request("http://x", {"id": 1}))
        await asyncio.wait_for(first_holding.wait(), timeout=1.0)
        t2 = asyncio.create_task(scheduler.request("http://x", {"id": 2}))
        await asyncio.sleep(0)
        state.aborted = True
        release_first.set()
        assert await t1 == {"id": 1}
        with pytest.raises(RolloutRequestAborted):
            await t2
        assert http_calls == 1


@pytest.mark.asyncio
async def test_model_request_scheduler_evaluation_ignores_abort() -> None:
    scheduler, state = _scheduler(capacity=1)
    state.aborted = True

    async def ok(url, payload, headers=None):
        return {"ok": True}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=ok):
        with pytest.raises(RolloutRequestAborted):
            await scheduler.request("http://x", {}, evaluation=False)
        assert await scheduler.request("http://x", {}, evaluation=True) == {"ok": True}


@pytest.mark.asyncio
async def test_model_request_scheduler_retry_holds_slot_for_logical_post() -> None:
    # One logical post() (with internal retry/backoff) occupies one slot.
    scheduler, _ = _scheduler(capacity=1)
    counter = _ActiveCounter()
    attempt = 0
    second_started = asyncio.Event()

    async def flaky_post(url, payload, headers=None):
        nonlocal attempt
        counter.enter()
        try:
            attempt += 1
            if attempt == 1:
                await asyncio.sleep(0.05)
                raise RuntimeError("transient")
            return {"ok": True}
        finally:
            counter.leave()

    # Simulate http_utils.post wrapping retries inside one awaitable call.
    async def logical_post(url, payload, headers=None):
        try:
            return await flaky_post(url, payload, headers=headers)
        except RuntimeError:
            await asyncio.sleep(0.05)
            return await flaky_post(url, payload, headers=headers)

    async def other_post(url, payload, headers=None):
        second_started.set()
        counter.enter()
        counter.leave()
        return {"other": True}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=logical_post):
        t1 = asyncio.create_task(scheduler.request("http://x", {"id": 1}))
        await asyncio.sleep(0)
        with patch("relax.engine.rollout.sglang_rollout.post", side_effect=other_post):
            t2 = asyncio.create_task(scheduler.request("http://x", {"id": 2}))
            await asyncio.sleep(0.02)
            assert not second_started.is_set()
            await t1
            await asyncio.wait_for(second_started.wait(), timeout=1.0)
            await t2
        assert counter.max_active <= 1


@pytest.mark.asyncio
async def test_model_request_scheduler_run_legacy_holds_session_slot() -> None:
    scheduler, _ = _scheduler(capacity=1)
    legacy_entered = asyncio.Event()
    release_legacy = asyncio.Event()
    other_entered = asyncio.Event()

    async def legacy_op():
        legacy_entered.set()
        await release_legacy.wait()
        return "legacy"

    async def other_post(url, payload, headers=None):
        other_entered.set()
        return {"ok": True}

    t_legacy = asyncio.create_task(scheduler.run_legacy(legacy_op, evaluation=False))
    await asyncio.wait_for(legacy_entered.wait(), timeout=1.0)
    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=other_post):
        t_other = asyncio.create_task(scheduler.request("http://x", {}))
        await asyncio.sleep(0)
        assert not other_entered.is_set()
        release_legacy.set()
        assert await t_legacy == "legacy"
        await asyncio.wait_for(other_entered.wait(), timeout=1.0)
        assert await t_other == {"ok": True}


@pytest.mark.asyncio
async def test_interleave_short_request_between_turns() -> None:
    # capacity=1: short request completes during long session env wait.
    scheduler, _ = _scheduler(capacity=1)
    turn1_done = asyncio.Event()
    env_gate = asyncio.Event()
    short_done = asyncio.Event()
    order: list[str] = []

    async def fake_post(url, payload, headers=None):
        label = payload["label"]
        order.append(f"start:{label}")
        if label == "long-1":
            turn1_done.set()
        if label == "short":
            short_done.set()
        order.append(f"end:{label}")
        return {"label": label}

    @request_model_aware
    async def long_session(args, sample, sampling_params, *, request_model):
        await request_model("http://x", {"label": "long-1"})
        await env_gate.wait()
        await request_model("http://x", {"label": "long-2"})
        return sample

    async def short_session():
        await scheduler.request("http://x", {"label": "short"})

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=fake_post):
        from functools import partial

        request_model = partial(scheduler.request, evaluation=False)
        long_task = asyncio.create_task(
            long_session(None, SimpleNamespace(), {}, request_model=request_model)
        )
        await asyncio.wait_for(turn1_done.wait(), timeout=1.0)
        short_task = asyncio.create_task(short_session())
        await asyncio.wait_for(short_done.wait(), timeout=1.0)
        assert short_done.is_set()
        assert not env_gate.is_set()
        env_gate.set()
        await long_task
        await short_task

    assert order == [
        "start:long-1",
        "end:long-1",
        "start:short",
        "end:short",
        "start:long-2",
        "end:long-2",
    ]


def test_request_model_aware_marker_and_signature_validation() -> None:
    async def unmarked(args, sample, sampling_params, *, request_model):
        return sample

    assert not _is_request_model_aware(unmarked)

    @request_model_aware
    async def good(args, sample, sampling_params, *, request_model):
        return sample

    assert _is_request_model_aware(good)
    _validate_request_model_signature(good)

    @request_model_aware
    async def missing(args, sample, sampling_params):
        return sample

    with pytest.raises(TypeError, match="missing"):
        _validate_request_model_signature(missing)

    @request_model_aware
    async def positional(args, sample, sampling_params, request_model):
        return sample

    with pytest.raises(TypeError, match="keyword-only"):
        _validate_request_model_signature(positional)

    @request_model_aware
    async def optional(args, sample, sampling_params, *, request_model=None):
        return sample

    with pytest.raises(TypeError, match="no default"):
        _validate_request_model_signature(optional)


@pytest.mark.asyncio
async def test_bypass_request_model_not_in_framework_guarantee() -> None:
    # Third-party HTTP that bypasses request_model is outside capacity accounting.
    scheduler, _ = _scheduler(capacity=1)
    counter = _ActiveCounter()
    holding = asyncio.Event()
    release = asyncio.Event()

    async def slotted(url, payload, headers=None):
        counter.enter()
        holding.set()
        await release.wait()
        counter.leave()
        return {"slotted": True}

    async def bypass_http():
        # Simulates custom code calling post() directly — not admitted.
        counter.enter()
        await asyncio.sleep(0)
        counter.leave()
        return {"bypass": True}

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=slotted):
        t1 = asyncio.create_task(scheduler.request("http://x", {}))
        await asyncio.wait_for(holding.wait(), timeout=1.0)
        bypass_result = await bypass_http()
        assert bypass_result == {"bypass": True}
        # Framework-managed active is 1, but raw counter may be 2 during bypass overlap.
        assert counter.max_active >= 2
        release.set()
        await t1
