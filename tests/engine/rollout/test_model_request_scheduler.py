# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# CPU async unit tests for ModelRequestScheduler and request_model_aware contract.

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from relax.engine.rollout import sglang_rollout as rollout_mod
from relax.engine.rollout.sglang_rollout import (
    ModelRequestScheduler,
    RolloutRequestAborted,
    _is_request_model_aware,
    _model_request_capacity,
    _validate_request_model_signature,
    request_model_aware,
)


class _FakeState:
    def __init__(self) -> None:
        self.aborted = False


def _scheduler(capacity: int = 1) -> tuple[ModelRequestScheduler, _FakeState]:
    state = _FakeState()
    return ModelRequestScheduler(state, capacity), state


@pytest.mark.parametrize("capacity", [0, -1])
def test_model_request_scheduler_rejects_non_positive_capacity(capacity: int) -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        _scheduler(capacity=capacity)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sglang_server_concurrency": 0}, "sglang_server_concurrency"),
        ({"rollout_num_gpus": 0}, "rollout_num_gpus"),
        ({"rollout_num_gpus_per_engine": 0}, "rollout_num_gpus_per_engine"),
        ({"rollout_num_gpus": 1, "rollout_num_gpus_per_engine": 2}, "capacity must be positive"),
    ],
)
def test_model_request_capacity_rejects_invalid_configuration(overrides: dict[str, int], message: str) -> None:
    values = {
        "sglang_server_concurrency": 2,
        "rollout_num_gpus": 8,
        "rollout_num_gpus_per_engine": 2,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        _model_request_capacity(SimpleNamespace(**values))


def test_model_request_capacity_uses_configured_engine_count() -> None:
    args = SimpleNamespace(
        sglang_server_concurrency=4,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=2,
    )
    assert _model_request_capacity(args) == 16


def test_generate_keeps_request_model_optional_for_direct_callers() -> None:
    parameter = inspect.signature(rollout_mod.generate).parameters["request_model"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


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
        await asyncio.Event().wait()
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
async def test_model_request_scheduler_retry_holds_slot_for_real_post(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real http_utils.post/_post retry+backoff still occupies one scheduler slot.
    import httpx

    import relax.utils.http_utils as http_utils

    scheduler, _ = _scheduler(capacity=1)
    backoff_entered = asyncio.Event()
    release_backoff = asyncio.Event()
    second_http_started = asyncio.Event()
    real_sleep = asyncio.sleep

    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        async def post(self, url, json=None, headers=None):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("transient")
            if self.calls >= 3:
                second_http_started.set()
            request = httpx.Request("POST", url)
            return httpx.Response(200, request=request, json={"ok": True, "call": self.calls})

    async def gated_sleep(seconds: float) -> None:
        if seconds < 0.5:
            await real_sleep(seconds)
            return
        backoff_entered.set()
        await release_backoff.wait()

    monkeypatch.setattr(http_utils, "_http_client", _FlakyClient())
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)
    monkeypatch.setattr(http_utils.asyncio, "sleep", gated_sleep)

    t1 = asyncio.create_task(scheduler.request("http://x", {"id": 1}))
    await asyncio.wait_for(backoff_entered.wait(), timeout=1.0)

    t2 = asyncio.create_task(scheduler.request("http://x", {"id": 2}))
    await real_sleep(0)
    assert not second_http_started.is_set()

    release_backoff.set()
    assert await t1 == {"ok": True, "call": 2}
    await asyncio.wait_for(second_http_started.wait(), timeout=1.0)
    assert await t2 == {"ok": True, "call": 3}


@pytest.mark.asyncio
async def test_mixed_ordinary_aware_legacy_respect_shared_capacity() -> None:
    # ordinary + request-aware + legacy share one capacity=2 pool.
    scheduler, _ = _scheduler(capacity=2)
    counter = _ActiveCounter()
    release = asyncio.Event()
    two_holding = asyncio.Event()

    async def fake_post(url, payload, headers=None):
        counter.enter()
        if counter.active == 2:
            two_holding.set()
        await release.wait()
        counter.leave()
        return {"kind": payload.get("kind")}

    async def legacy_op():
        counter.enter()
        if counter.active == 2:
            two_holding.set()
        await release.wait()
        counter.leave()
        return "legacy"

    request_model = scheduler.request

    @request_model_aware
    async def aware_session(args, sample, sampling_params, *, request_model):
        await request_model("http://x", {"kind": "aware"})
        return sample

    with patch("relax.engine.rollout.sglang_rollout.post", side_effect=fake_post):
        t_legacy = asyncio.create_task(scheduler.run_legacy(legacy_op))
        t_ordinary = asyncio.create_task(scheduler.request("http://x", {"kind": "ordinary"}))
        await asyncio.wait_for(two_holding.wait(), timeout=1.0)
        assert counter.active == 2
        assert counter.max_active <= 2

        t_aware = asyncio.create_task(aware_session(None, SimpleNamespace(), {}, request_model=request_model))
        await asyncio.sleep(0)
        assert counter.active == 2
        assert counter.max_active <= 2

        release.set()
        await asyncio.gather(t_legacy, t_ordinary, t_aware)
        assert counter.max_active <= 2
        assert counter.calls == 3


def test_request_model_aware_accepts_valid_signature() -> None:
    async def unmarked(args, sample, sampling_params, *, request_model):
        return sample

    assert not _is_request_model_aware(unmarked)

    @request_model_aware
    async def good(args, sample, sampling_params, *, request_model):
        return sample

    assert _is_request_model_aware(good)
    _validate_request_model_signature(good)


def test_request_model_aware_rejects_invalid_signature() -> None:
    @request_model_aware
    async def missing(args, sample, sampling_params):
        return sample

    @request_model_aware
    async def positional(args, sample, sampling_params, request_model):
        return sample

    @request_model_aware
    async def optional(args, sample, sampling_params, *, request_model=None):
        return sample

    with pytest.raises(TypeError, match="missing"):
        _validate_request_model_signature(missing)
    with pytest.raises(TypeError, match="keyword-only"):
        _validate_request_model_signature(positional)
    with pytest.raises(TypeError, match="no default"):
        _validate_request_model_signature(optional)
