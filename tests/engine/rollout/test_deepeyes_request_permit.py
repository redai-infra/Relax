# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from examples.deepeyes import rollout
from relax.engine.rollout.request_permit import RequestPermitPool


@pytest.mark.asyncio
async def test_deepeyes_inference_holds_request_permit(monkeypatch):
    permits = RequestPermitPool(1)
    state = SimpleNamespace(request_permit=permits.acquire)

    async def fake_inference(*args, **kwargs):
        del args, kwargs
        assert permits._semaphore.locked()
        return "response"

    monkeypatch.setattr(rollout, "_run_inference_step", fake_inference)

    result = await rollout._run_inference_step_with_permit(state, "url")

    assert result == "response"
    async with permits.acquire():
        pass


@pytest.mark.asyncio
async def test_deepeyes_inference_releases_request_permit_on_error(monkeypatch):
    permits = RequestPermitPool(1)
    state = SimpleNamespace(request_permit=permits.acquire)

    async def failing_inference(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("inference failed")

    monkeypatch.setattr(rollout, "_run_inference_step", failing_inference)

    with pytest.raises(RuntimeError, match="inference failed"):
        await rollout._run_inference_step_with_permit(state, "url")

    async with permits.acquire():
        pass
