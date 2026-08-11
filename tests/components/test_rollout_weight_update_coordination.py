# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


pytest.importorskip("ray", reason="Ray is an optional test dependency")

from relax.components.base import Base  # noqa: E402, I001
from relax.components.rollout import Rollout

RolloutClass = Rollout.func_or_class


def test_rollout_evaluation_uses_periodic_and_epoch_boundaries() -> None:
    rollout = RolloutClass.__new__(RolloutClass)
    Base.__init__(rollout)
    rollout.config = SimpleNamespace(
        eval_interval=10,
        eval_prompt_data=["aime", "/data/aime.jsonl"],
        num_rollout=20,
    )
    rollout.num_rollout_per_epoch = 4

    assert not rollout._should_eval(2)
    assert rollout._should_eval(3)
    assert rollout._should_eval(9)
    assert rollout._should_eval(19)


def test_rollout_evaluation_does_not_force_final_step() -> None:
    rollout = RolloutClass.__new__(RolloutClass)
    Base.__init__(rollout)
    rollout.config = SimpleNamespace(
        eval_interval=10,
        eval_prompt_data=["aime", "/data/aime.jsonl"],
        num_rollout=25,
    )
    rollout.num_rollout_per_epoch = None

    assert rollout._should_eval(9)
    assert rollout._should_eval(19)
    assert not rollout._should_eval(24)


def test_rollout_evaluation_requires_complete_configuration() -> None:
    rollout = RolloutClass.__new__(RolloutClass)
    Base.__init__(rollout)
    rollout.config = SimpleNamespace(eval_interval=10, eval_prompt_data=None, num_rollout=20)
    rollout.num_rollout_per_epoch = 4

    assert not rollout._should_eval(3)


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
        health_monitoring_resume=SimpleNamespace(remote=AsyncMock()),
        set_weight_updating=SimpleNamespace(remote=AsyncMock(side_effect=[RuntimeError("prepare failed"), None])),
    )
    with pytest.raises(RuntimeError, match="prepare failed"):
        await rollout.can_do_update_weight_for_async()

    assert rollout.status == "running"
    assert rollout._weight_update_ready.is_set()
    assert rollout.rollout_manager.set_weight_updating.remote.await_args_list[1].args == (False,)
    rollout.rollout_manager.health_monitoring_resume.remote.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_operation_cleanup_waits_for_inflight_prepare() -> None:
    rollout = RolloutClass.__new__(RolloutClass)
    Base.__init__(rollout)
    rollout.step = 1
    rollout.status = "running"
    rollout._weight_update_ready = asyncio.Event()
    rollout._weight_update_ready.set()
    rollout._weight_update_prepare_lock = asyncio.Lock()
    rollout._weight_update_prepare_tasks = {}
    rollout._ended_weight_update_operations = {}
    rollout._async_check_production_for_update_weight = AsyncMock(return_value=True)

    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()

    async def pause_health_monitoring() -> None:
        prepare_started.set()
        await release_prepare.wait()

    rollout.rollout_manager = SimpleNamespace(
        health_monitoring_pause=SimpleNamespace(remote=AsyncMock(side_effect=pause_health_monitoring)),
        health_monitoring_resume=SimpleNamespace(remote=AsyncMock()),
        set_weight_updating=SimpleNamespace(remote=AsyncMock()),
    )

    prepare = asyncio.create_task(rollout.can_do_update_weight_for_async(operation_id="actor-step-7"))
    await prepare_started.wait()
    cleanup = asyncio.create_task(rollout.end_update_weight(operation_id="actor-step-7"))
    await asyncio.sleep(0)
    assert not cleanup.done()

    release_prepare.set()
    assert await prepare == 1
    await cleanup

    assert rollout.status == "running"
    assert [call.args for call in rollout.rollout_manager.set_weight_updating.remote.await_args_list] == [
        (True,),
        (False,),
    ]


@pytest.mark.asyncio
async def test_late_prepare_cannot_overtake_operation_cleanup() -> None:
    rollout = RolloutClass.__new__(RolloutClass)
    Base.__init__(rollout)
    rollout.step = 1
    rollout.status = "running"
    rollout._weight_update_ready = asyncio.Event()
    rollout._weight_update_ready.set()
    rollout._weight_update_prepare_lock = asyncio.Lock()
    rollout._weight_update_prepare_tasks = {}
    rollout._ended_weight_update_operations = {}
    rollout._async_check_production_for_update_weight = AsyncMock(return_value=True)
    rollout.rollout_manager = SimpleNamespace(
        health_monitoring_pause=SimpleNamespace(remote=AsyncMock()),
        health_monitoring_resume=SimpleNamespace(remote=AsyncMock()),
        set_weight_updating=SimpleNamespace(remote=AsyncMock()),
    )

    await rollout.end_update_weight(operation_id="actor-step-9")
    result = await rollout.can_do_update_weight_for_async(operation_id="actor-step-9")

    assert result == 0
    rollout._async_check_production_for_update_weight.assert_not_awaited()
    rollout.rollout_manager.set_weight_updating.remote.assert_not_awaited()
