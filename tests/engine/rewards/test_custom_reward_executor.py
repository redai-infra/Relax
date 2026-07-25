# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import os
from types import SimpleNamespace

import pytest
import ray

import relax.engine.rewards as rewards_module
from relax.agentic.pipeline import reward as agentic_reward
from relax.engine.rewards import RewardExecutor, async_rm, batched_async_rm
from relax.utils.types import Sample


_FIXTURE_MODULE = "tests.engine.rewards.task19_custom_rewards"


def _make_args(custom_name: str, **overrides) -> SimpleNamespace:
    defaults = {
        "custom_rm_path": f"{_FIXTURE_MODULE}.{custom_name}",
        "group_rm": False,
        "reward_max_concurrency": 4,
        "reward_num_workers": 2,
        "rm_type": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_sample(index: int, response: str = "ok") -> Sample:
    return Sample(index=index, group_index=index // 2, response=response, label=str(index))


def _kill_executor_workers() -> None:
    executor = RewardExecutor._instance
    if executor is None:
        return
    for worker in executor._workers:
        try:
            ray.kill(worker)
        except Exception:
            pass
    executor._workers = []


@pytest.fixture(scope="module", autouse=True)
def _ray_runtime():
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=4, ignore_reinit_error=True)
    yield
    if started_here:
        ray.shutdown()


@pytest.fixture(autouse=True)
def _reset_executor():
    _kill_executor_workers()
    RewardExecutor._instance = None
    yield
    _kill_executor_workers()
    RewardExecutor._instance = None


@pytest.mark.asyncio
async def test_sync_custom_reward_runs_in_worker_process_and_forwards_kwargs():
    args = _make_args("sync_process_reward")
    sample = _make_sample(7)

    result = await async_rm(args, sample, marker="forwarded")

    assert result["pid"] != os.getpid()
    assert result["index"] == 7
    assert result["response"] == "ok"
    assert result["marker"] == "forwarded"


@pytest.mark.asyncio
async def test_async_custom_reward_is_awaited_in_caller_process():
    args = _make_args("async_process_reward")
    sample = _make_sample(3)

    result = await async_rm(args, sample, marker="direct")

    assert result == {"pid": os.getpid(), "index": 3, "marker": "direct"}
    assert RewardExecutor._instance._workers == []


@pytest.mark.asyncio
async def test_custom_function_is_loaded_once_in_caller(monkeypatch):
    load_count = 0
    original_load_function = rewards_module.load_function

    def counted_load_function(path):
        nonlocal load_count
        load_count += 1
        return original_load_function(path)

    monkeypatch.setattr(rewards_module, "load_function", counted_load_function)
    args = _make_args("async_process_reward")

    await async_rm(args, _make_sample(0))
    await async_rm(args, _make_sample(1))

    assert load_count == 1


@pytest.mark.asyncio
async def test_sync_custom_function_is_loaded_once_per_worker():
    args = _make_args(
        "counted_sync_reward",
        reward_max_concurrency=4,
        reward_num_workers=2,
    )

    results = await batched_async_rm(args, [_make_sample(index) for index in range(6)])

    assert len({result["pid"] for result in results}) == 2
    assert all(result["load_count"] == 1 for result in results)


@pytest.mark.asyncio
async def test_reward_num_workers_limits_worker_process_count():
    args = _make_args(
        "sync_timed_reward",
        reward_max_concurrency=8,
        reward_num_workers=3,
        test_reward_delay=0.05,
    )

    results = await asyncio.wait_for(
        batched_async_rm(args, [_make_sample(index) for index in range(9)]),
        timeout=15,
    )

    assert len({result["pid"] for result in results}) == 3


def _peak_parallel_intervals(results: list[dict]) -> int:
    events = []
    for result in results:
        events.append((result["started_at"], 1))
        events.append((result["finished_at"], -1))
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return peak


@pytest.mark.asyncio
async def test_reward_max_concurrency_limits_inflight_custom_rewards():
    args = _make_args(
        "sync_timed_reward",
        reward_max_concurrency=2,
        reward_num_workers=4,
        test_reward_delay=0.1,
    )

    results = await asyncio.wait_for(
        batched_async_rm(args, [_make_sample(index) for index in range(8)]),
        timeout=15,
    )

    assert _peak_parallel_intervals(results) == 2


@pytest.mark.asyncio
async def test_sync_custom_reward_error_identifies_sample_and_releases_limit():
    args = _make_args(
        "sync_maybe_failing_reward",
        reward_max_concurrency=1,
        reward_num_workers=1,
    )
    failing_sample = _make_sample(17, response="fail")

    with pytest.raises(RuntimeError, match=r"index=17"):
        await asyncio.wait_for(async_rm(args, failing_sample), timeout=10)

    result = await asyncio.wait_for(async_rm(args, _make_sample(18)), timeout=10)
    assert result == 18.0


@pytest.mark.asyncio
async def test_non_group_batch_scores_each_sync_sample_in_input_order():
    args = _make_args("sync_maybe_failing_reward")
    samples = [_make_sample(index) for index in range(6)]

    results = await batched_async_rm(args, samples)

    assert results == [float(index) for index in range(6)]


@pytest.mark.asyncio
async def test_group_custom_reward_keeps_batch_call_compatibility():
    args = _make_args("sync_group_reward", group_rm=True)
    samples = [_make_sample(index) for index in range(4)]

    results = await batched_async_rm(args, samples)

    assert results == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_async_group_custom_reward_keeps_batch_call_compatibility():
    args = _make_args("async_group_reward", group_rm=True)
    samples = [_make_sample(index) for index in range(4)]

    results = await batched_async_rm(args, samples)

    assert results == [0.0, 1.0, 2.0, 3.0]
    assert RewardExecutor._instance._workers == []


@pytest.mark.asyncio
async def test_batched_custom_reward_error_identifies_batch_position():
    args = _make_args("sync_maybe_failing_reward")
    samples = [_make_sample(0), _make_sample(1, response="fail"), _make_sample(2)]

    with pytest.raises(RuntimeError, match=r"batch position 1"):
        await batched_async_rm(args, samples)


@pytest.mark.asyncio
async def test_agentic_reward_helpers_delegate_to_shared_executor(monkeypatch):
    args = _make_args("async_process_reward")
    sample = _make_sample(5)
    calls = []

    async def fake_async_rm(received_args, received_sample):
        calls.append(("single", received_args, received_sample))
        return 5.0

    async def fake_batched_async_rm(received_args, received_samples):
        calls.append(("batch", received_args, received_samples))
        return [5.0]

    monkeypatch.setattr(rewards_module, "async_rm", fake_async_rm)
    monkeypatch.setattr(rewards_module, "batched_async_rm", fake_batched_async_rm)

    assert await agentic_reward._async_rm(args, sample) == 5.0
    assert await agentic_reward._batched_async_rm(args, [sample]) == [5.0]
    assert calls == [
        ("single", args, sample),
        ("batch", args, [sample]),
    ]
