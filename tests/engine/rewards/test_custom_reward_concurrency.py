# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU/Ray tests for custom reward concurrency (task 19)."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ray = pytest.importorskip("ray")

if not ray.is_initialized():
    ray.init(address="local", num_cpus=8, include_dashboard=False, ignore_reinit_error=True)

from relax.engine.rewards import (  # noqa: E402
    CustomRewardError,
    RewardExecutor,
    async_rm,
    batched_async_rm,
)
from relax.engine.rewards.custom_reward import CustomRewardResolver  # noqa: E402
from relax.utils.reload_utils import ReloadGenerationRegistry  # noqa: E402
from relax.utils.types import Sample  # noqa: E402


FIXTURE = "tests.engine.rewards.custom_reward_fixtures"


@pytest.fixture(autouse=True)
async def _reset_executor():
    await RewardExecutor.reset()
    yield
    await RewardExecutor.reset()


def _args(path: str | None, **overrides) -> SimpleNamespace:
    defaults = {
        "custom_rm_path": path,
        "reward_max_concurrency": 8,
        "reward_num_workers": 4,
        "rm_type": "math",
        "rm_url": None,
        "reward_key": None,
        "group_rm": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _sample(response: str = "ok", label: str = "ok", index: int = 0) -> Sample:
    return Sample(response=response, label=label, index=index, group_index=0, session_id=f"s{index}")


@ray.remote
class _OverlapCounter:
    def __init__(self):
        self.current = 0
        self.peak = 0

    def enter(self):
        self.current += 1
        self.peak = max(self.peak, self.current)
        return self.current

    def leave(self):
        self.current -= 1

    def get_peak(self):
        return self.peak


def _fresh_overlap_counter(name: str):
    try:
        ray.kill(ray.get_actor(name))
    except ValueError:
        pass
    return _OverlapCounter.options(name=name, get_if_exists=False).remote()


async def _measured_overlap_peak(
    *,
    counter_name: str,
    max_concurrency: int,
    num_workers: int,
    n_samples: int = 6,
) -> tuple[list[float], int]:
    counter = _fresh_overlap_counter(counter_name)
    args = _args(
        f"{FIXTURE}.sync_slow_reward_tracked",
        reward_max_concurrency=max_concurrency,
        reward_num_workers=num_workers,
    )
    executor = RewardExecutor.get_or_create(max_concurrency=max_concurrency, num_workers=num_workers)
    opts = {"custom_options": {"overlap_counter_name": counter_name}}
    await executor.execute_custom_sample(args, _sample(), **opts)
    counter = _fresh_overlap_counter(counter_name)

    rewards = await asyncio.gather(
        *[executor.execute_custom_sample(args, _sample(index=i), **opts) for i in range(n_samples)]
    )
    peak = ray.get(counter.get_peak.remote())
    ray.kill(counter)
    return rewards, peak


@pytest.mark.asyncio
async def test_sync_custom_runs_in_worker_process():
    args = _args(f"{FIXTURE}.sync_reward_with_pid", reward_key="score", reward_threshold=0.5)
    result = await async_rm(args, _sample())
    assert isinstance(result, dict)
    assert result["pid"] != os.getpid()
    assert result["has_rm_type"] is True
    assert result["threshold"] == 0.5


@pytest.mark.asyncio
async def test_async_custom_does_not_call_worker():
    args = _args(f"{FIXTURE}.async_reward_with_marker", reward_key="score", reward_num_workers=2)
    executor = RewardExecutor.get_or_create(max_concurrency=8, num_workers=2)
    result = await executor.execute_custom_sample(args, _sample())
    assert result["async"] is True
    assert result["full_args"] is True
    assert executor._workers == []


@pytest.mark.asyncio
async def test_legacy_batch_custom_list_call_even_when_group_rm_false():
    args = _args(f"{FIXTURE}.sync_group_reward", group_rm=False)
    samples = [_sample("a", "a", 0), _sample("b", "x", 1)]
    rewards = await batched_async_rm(args, samples)
    assert rewards == [1.0, 0.0]


@pytest.mark.asyncio
async def test_ignore_custom_uses_builtin():
    args = _args(f"{FIXTURE}.sync_raises", rm_type="dummy")
    reward = await async_rm(args, _sample(), ignore_custom=True)
    assert reward == 0.0


@pytest.mark.asyncio
async def test_sync_exception_includes_sample_index():
    args = _args(f"{FIXTURE}.sync_raises")
    with pytest.raises(CustomRewardError) as exc_info:
        await async_rm(args, _sample(index=7))
    assert "index=7" in str(exc_info.value)
    assert FIXTURE in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,call",
    [
        (f"{FIXTURE}.sync_group_bad_length", "group"),
        (f"{FIXTURE}.sync_returns_awaitable", "single"),
    ],
)
async def test_custom_reward_contract_errors(path: str, call: str):
    args = _args(path)
    with pytest.raises(CustomRewardError):
        if call == "group":
            await batched_async_rm(args, [_sample(), _sample(index=1)])
        else:
            await async_rm(args, _sample())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_concurrency,num_workers",
    [
        (2, 4),
        (8, 2),
    ],
)
async def test_sync_overlap_capped_by_min_concurrency_and_workers(max_concurrency: int, num_workers: int):
    cap = min(max_concurrency, num_workers)
    rewards, peak = await _measured_overlap_peak(
        counter_name=f"task19_overlap_{max_concurrency}_{num_workers}",
        max_concurrency=max_concurrency,
        num_workers=num_workers,
    )
    assert rewards == [1.0] * 6
    assert 1 <= peak <= cap


@pytest.mark.asyncio
async def test_bind_generation_provider_idempotent_for_bound_method():
    registry = ReloadGenerationRegistry()
    executor = RewardExecutor.get_or_create(max_concurrency=4, num_workers=2)
    executor.bind_generation_provider(registry.current)
    executor.bind_generation_provider(registry.current)
    assert executor.current_custom_generation(f"{FIXTURE}.sync_reward") == 0

    other = ReloadGenerationRegistry()
    with pytest.raises(RuntimeError, match="already bound"):
        executor.bind_generation_provider(other.current)


@pytest.mark.asyncio
async def test_resolver_keeps_older_generation_cache_entries():
    resolver = CustomRewardResolver()
    path = f"{FIXTURE}.sync_reward"
    load_calls: list[str] = []

    def _fake_load(p: str):
        load_calls.append(p)
        return lambda args, sample, **kwargs: 1.0

    with patch("relax.engine.rewards.custom_reward.load_function", side_effect=_fake_load):
        first = resolver.ensure_loaded(path, 0)
        second = resolver.ensure_loaded(path, 1)
        again_old = resolver.ensure_loaded(path, 0)

    assert first is again_old
    assert first is not second
    assert load_calls == [path, path]
    assert (path, 0) in resolver._cache
    assert (path, 1) in resolver._cache


@pytest.mark.asyncio
async def test_agentic_dispatch_does_not_double_semaphore():
    from relax.agentic.pipeline import reward as agentic_reward

    args = _args(f"{FIXTURE}.async_reward", reward_max_concurrency=1)
    domain = agentic_reward.RewardDomain(args=args, group_filter=None, max_submissions_per_step=1)

    async def run_one(idx: int):
        return await domain._run_sample_reward(_sample(index=idx))

    results = await asyncio.wait_for(asyncio.gather(run_one(0), run_one(1)), timeout=5)
    assert all(r.reward == 1.0 for r in results)


@pytest.mark.asyncio
async def test_sync_recovers_after_custom_exception():
    fail_args = _args(f"{FIXTURE}.sync_raises", reward_num_workers=2)
    ok_args = _args(f"{FIXTURE}.sync_reward", reward_num_workers=2)
    with pytest.raises(CustomRewardError):
        await async_rm(fail_args, _sample(index=1))
    reward = await asyncio.wait_for(async_rm(ok_args, _sample()), timeout=10)
    assert reward == 1.0


@pytest.mark.asyncio
async def test_worker_loads_custom_function_once_per_generation():
    path = f"{FIXTURE}.sync_reward"
    args = _args(path, reward_num_workers=1)
    executor = RewardExecutor.get_or_create(max_concurrency=4, num_workers=1)
    await executor.execute_custom_sample(args, _sample(index=0))
    await executor.execute_custom_sample(args, _sample(index=1))
    worker = executor._workers[0]
    assert ray.get(worker.custom_load_count.remote(path)) == 1


@pytest.mark.asyncio
async def test_worker_init_does_not_block_event_loop():
    ticks: list[float] = []

    async def heartbeat():
        for _ in range(8):
            ticks.append(1.0)
            await asyncio.sleep(0.01)

    args = _args(f"{FIXTURE}.sync_reward", reward_num_workers=4)
    executor = RewardExecutor.get_or_create(max_concurrency=4, num_workers=4)
    reward, _ = await asyncio.wait_for(
        asyncio.gather(executor.execute_custom_sample(args, _sample()), heartbeat()),
        timeout=30,
    )
    assert reward == 1.0
    assert len(ticks) >= 2


@pytest.mark.asyncio
async def test_auto_pick_primitives_visible_on_worker_args():
    from relax.engine.rewards.custom_reward import build_reward_worker_config, pick_primitive_arg_attrs

    args = _args(
        f"{FIXTURE}.sync_reward",
        reward_threshold=0.75,
        task_name="math",
        tokenizer=object(),
    )
    picked = pick_primitive_arg_attrs(args)
    assert picked["reward_threshold"] == 0.75
    assert picked["task_name"] == "math"
    assert "tokenizer" not in picked

    config = build_reward_worker_config(args, custom_options={"task_name": "override"})
    ns = config.as_namespace()
    assert ns.reward_threshold == 0.75
    assert ns.task_name == "override"
