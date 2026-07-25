# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from examples.deepeyes import rollout


class _PermitProbe:
    def __init__(self) -> None:
        self.held = False

    @asynccontextmanager
    async def acquire(self):
        self.held = True
        try:
            yield
        finally:
            self.held = False


@pytest.mark.asyncio
async def test_deepeyes_inference_holds_request_permit(monkeypatch):
    probe = _PermitProbe()
    state = SimpleNamespace(request_permit=probe.acquire)

    async def fake_inference(*args, **kwargs):
        del args, kwargs
        assert probe.held
        return "response"

    monkeypatch.setattr(rollout, "_run_inference_step", fake_inference)

    result = await rollout._run_inference_step_with_permit(state, "url")

    assert result == "response"
    assert not probe.held


@pytest.mark.asyncio
async def test_deepeyes_inference_releases_request_permit_on_error(monkeypatch):
    probe = _PermitProbe()
    state = SimpleNamespace(request_permit=probe.acquire)

    async def failing_inference(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("inference failed")

    monkeypatch.setattr(rollout, "_run_inference_step", failing_inference)

    with pytest.raises(RuntimeError, match="inference failed"):
        await rollout._run_inference_step_with_permit(state, "url")

    assert not probe.held


@pytest.mark.asyncio
async def test_deepeyes_does_not_infer_when_aborted_while_waiting_for_permit(monkeypatch):
    waiting = asyncio.Event()
    release = asyncio.Event()
    inference_called = False

    @asynccontextmanager
    async def blocking_permit():
        waiting.set()
        await release.wait()
        yield

    state = SimpleNamespace(request_permit=blocking_permit, aborted=False)

    async def fake_inference(*args, **kwargs):
        nonlocal inference_called
        del args, kwargs
        inference_called = True

    monkeypatch.setattr(rollout, "_run_inference_step", fake_inference)

    task = asyncio.create_task(rollout._run_inference_step_with_permit(state, "url"))
    await asyncio.wait_for(waiting.wait(), timeout=1)
    state.aborted = True
    release.set()

    result = await asyncio.wait_for(task, timeout=1)

    assert result is None
    assert not inference_called
