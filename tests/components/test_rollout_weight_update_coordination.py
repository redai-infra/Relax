# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


pytest.importorskip("ray", reason="Ray is an optional test dependency")

from relax.components.base import Base  # noqa: E402, I001
from relax.components.rollout import Rollout

RolloutClass = Rollout.func_or_class


@pytest.mark.asyncio
async def test_weight_update_prepare_failure_restores_rollout_state() -> None:
    rollout = RolloutClass.__new__(RolloutClass)
    Base.__init__(rollout)
    rollout.step = 1
    rollout.status = "running"
    rollout._weight_update_ready = asyncio.Event()
    rollout._async_check_production_for_update_weight = AsyncMock(return_value=True)
    rollout.rollout_manager = SimpleNamespace(
        health_monitoring_pause=SimpleNamespace(remote=AsyncMock()),
        set_weight_updating=SimpleNamespace(remote=AsyncMock(side_effect=[RuntimeError("prepare failed"), None])),
    )
    with pytest.raises(RuntimeError, match="prepare failed"):
        await rollout.can_do_update_weight_for_async()

    assert rollout.status == "running"
    assert rollout._weight_update_ready.is_set()
    assert rollout.rollout_manager.set_weight_updating.remote.await_args_list[1].args == (False,)
