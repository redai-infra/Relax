# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU/Ray tests for custom reward concurrency (task 19)."""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import pytest


ray = pytest.importorskip("ray")

if not ray.is_initialized():
    # Force a local cluster so a stale /tmp/ray/ray_current_cluster does not hang CI/dev boxes.
    ray.init(address="local", num_cpus=8, include_dashboard=False, ignore_reinit_error=True)

from relax.engine.rewards import (  # noqa: E402
    CustomRewardError,
    RewardExecutor,
    async_rm,
    batched_async_rm,
)
from relax.engine.rewards.custom_reward import CustomRewardError as _CustomRewardError  # noqa: E402
from relax.utils.reload_utils import ReloadableMixin, ReloadGenerationRegistry  # noqa: E402
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


@pytest.mark.asyncio
async def test_sync_custom_runs_in_worker_process():
    args = _args(f"{FIXTURE}.sync_reward_with_pid", reward_key="score")
    result = await async_rm(args, _sample())
    assert isinstance(result, dict)
    assert result["pid"] != os.getpid()
    assert result["has_rm_type"] is True


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
    with pytest.raises((CustomRewardError, _CustomRewardError)) as exc_info:
        await async_rm(args, _sample(index=7))
    assert "index=7" in str(exc_info.value)
    assert FIXTURE in str(exc_info.value)


@pytest.mark.asyncio
async def test_group_length_mismatch_raises():
    args = _args(f"{FIXTURE}.sync_group_bad_length")
    with pytest.raises((CustomRewardError, _CustomRewardError)):
        await batched_async_rm(args, [_sample(), _sample(index=1)])


@pytest.mark.asyncio
async def test_sync_returns_awaitable_raises_contract_error():
    args = _args(f"{FIXTURE}.sync_returns_awaitable")
    with pytest.raises((CustomRewardError, _CustomRewardError)):
        await async_rm(args, _sample())


@pytest.mark.asyncio
async def test_reward_max_concurrency_limits_overlap():
    @ray.remote
    class Counter:
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

    counter = Counter.remote()

    # Patch via a dedicated fixture-like function defined inline is hard;
    # instead run slow rewards under concurrency=2 and check wall clock + peak via sleep.
    args = _args(f"{FIXTURE}.sync_slow_reward", reward_max_concurrency=2, reward_num_workers=4)
    executor = RewardExecutor.get_or_create(max_concurrency=2, num_workers=4)
    # Warm workers/cache outside the timed window.
    await executor.execute_custom_sample(args, _sample())

    samples = [_sample(index=i) for i in range(6)]
    start = time.perf_counter()
    rewards = await asyncio.gather(*[executor.execute_custom_sample(args, s) for s in samples])
    elapsed = time.perf_counter() - start
    assert rewards == [1.0] * 6
    # Serial would be ~0.6s; with concurrency 2 expect roughly >= 0.3s and < 0.55s after warmup.
    assert elapsed < 0.55
    assert elapsed >= 0.25
    del counter


@pytest.mark.asyncio
async def test_reload_generation_bumps_and_propagates():
    class _Host(ReloadableMixin):
        def __init__(self, path: str):
            self.args = SimpleNamespace(custom_rm_path=path)
            self._reload_generations = ReloadGenerationRegistry()

    path = f"{FIXTURE}.sync_reward"
    host = _Host(path)
    executor = RewardExecutor.get_or_create(max_concurrency=4, num_workers=2)
    executor.bind_generation_provider(host._reload_generations.current)

    assert host._reload_generations.current("custom_rm", path) == 0
    result = host.reload_function_by_name("custom_rm")
    assert result["success"] is True
    assert result["generation"] == 1
    assert host._reload_generations.current("custom_rm", path) == 1

    reward = await executor.execute_custom_sample(_args(path), _sample())
    assert reward == 1.0


@pytest.mark.asyncio
async def test_agentic_dispatch_does_not_double_semaphore():
    from relax.agentic.pipeline import reward as agentic_reward

    args = _args(f"{FIXTURE}.async_reward", reward_max_concurrency=1)
    # With concurrency 1, two overlapping agentic sample rewards must still complete:
    # dispatch path must not acquire RewardExecutor semaphore again.
    domain = agentic_reward.RewardDomain(args=args, group_filter=None, max_submissions_per_step=1)

    async def run_one(idx: int):
        sample = _sample(index=idx)
        return await domain._run_sample_reward(sample)

    results = await asyncio.wait_for(asyncio.gather(run_one(0), run_one(1)), timeout=5)
    assert all(r.reward == 1.0 for r in results)
