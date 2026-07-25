# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
from types import SimpleNamespace

import pytest


_HAS_FULL_PIPELINE = False

try:
    import ray

    if not ray.is_initialized():
        try:
            ray.init(num_cpus=8, ignore_reinit_error=True)
        except ValueError as exc:
            if "When connecting to an existing cluster" in str(exc):
                ray.init(ignore_reinit_error=True)
            else:
                raise

    from relax.engine.rewards import RewardExecutor, async_rm, batched_async_rm
    from relax.utils.types import Sample

    _HAS_FULL_PIPELINE = True
except ImportError:
    ray = None
    RewardExecutor = None
    Sample = None
    async_rm = None
    batched_async_rm = None


pytestmark = pytest.mark.skipif(not _HAS_FULL_PIPELINE, reason="Full reward pipeline dependencies are not installed")

HELPERS = "tests.engine.rewards.custom_reward_helpers"


def _make_args(**overrides):
    defaults = {
        "rm_type": "openr1mm",
        "custom_rm_path": None,
        "group_rm": False,
        "reward_max_concurrency": 64,
        "reward_num_workers": 4,
        "rm_url": None,
        "reward_key": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_sample(index: int, response: str = "ok", label: str = "ok", metadata: dict | None = None):
    return Sample(
        index=index,
        group_index=index // 10,
        response=response,
        label=label,
        metadata=metadata or {},
    )


def _kill_named_reward_workers(max_workers: int = 16):
    for idx in range(max_workers):
        try:
            worker = ray.get_actor(f"reward_worker_{idx}")
        except ValueError:
            continue
        ray.kill(worker)


@pytest.fixture(autouse=True)
def _reset_executor():
    _kill_named_reward_workers()
    RewardExecutor._instance = None
    yield
    inst = RewardExecutor._instance
    if inst is not None:
        for worker in inst._workers:
            try:
                ray.kill(worker)
            except Exception:
                pass
        inst._workers = []
    RewardExecutor._instance = None
    _kill_named_reward_workers()


@pytest.mark.asyncio
async def test_sync_custom_reward_runs_in_worker_process():
    args = _make_args(custom_rm_path=f"{HELPERS}.sync_reward", reward_num_workers=1)
    sample = _make_sample(index=3)

    result = await async_rm(args, sample, tag="single")

    assert result["score"] == 1.0
    assert result["index"] == 3
    assert result["tag"] == "single"
    assert result["pid"] != os.getpid()


@pytest.mark.asyncio
async def test_async_custom_reward_is_awaited_without_workers():
    args = _make_args(custom_rm_path=f"{HELPERS}.async_reward", reward_num_workers=2)
    samples = [_make_sample(index=idx) for idx in range(3)]

    rewards = await batched_async_rm(args, samples, tag="async")

    assert [reward["index"] for reward in rewards] == [0, 1, 2]
    assert all(reward["score"] == 2.0 for reward in rewards)
    assert {reward["pid"] for reward in rewards} == {os.getpid()}
    assert RewardExecutor._instance._workers == []


@pytest.mark.asyncio
async def test_batched_sync_custom_reward_uses_worker_pool_and_preserves_order():
    args = _make_args(
        custom_rm_path=f"{HELPERS}.sync_reward",
        reward_max_concurrency=4,
        reward_num_workers=2,
    )
    samples = [_make_sample(index=idx) for idx in range(6)]

    rewards = await batched_async_rm(args, samples)

    assert [reward["index"] for reward in rewards] == list(range(6))
    worker_pids = {reward["pid"] for reward in rewards}
    assert len(worker_pids) == 2
    assert os.getpid() not in worker_pids


@pytest.mark.asyncio
async def test_group_sync_custom_reward_keeps_batch_semantics_in_worker():
    args = _make_args(
        custom_rm_path=f"{HELPERS}.sync_batch_reward",
        group_rm=True,
        reward_num_workers=2,
    )
    samples = [_make_sample(index=idx) for idx in range(4)]

    rewards = await batched_async_rm(args, samples, tag="group")

    assert [reward["index"] for reward in rewards] == list(range(4))
    assert all(reward["tag"] == "group" for reward in rewards)
    worker_pids = {reward["pid"] for reward in rewards}
    assert len(worker_pids) == 1
    assert os.getpid() not in worker_pids


@pytest.mark.asyncio
async def test_reward_max_concurrency_limits_sync_custom_reward(tmp_path):
    counter_path = tmp_path / "counter.json"
    args = _make_args(
        custom_rm_path=f"{HELPERS}.metered_sync_reward",
        reward_max_concurrency=1,
        reward_num_workers=3,
        reward_counter_path=str(counter_path),
        reward_sleep_s=0.05,
    )
    samples = [_make_sample(index=idx) for idx in range(5)]

    rewards = await batched_async_rm(args, samples)

    assert [reward["index"] for reward in rewards] == list(range(5))
    state = json.loads(counter_path.read_text(encoding="utf-8"))
    assert state["max_active"] == 1


@pytest.mark.asyncio
async def test_sync_custom_reward_exception_includes_sample_context():
    args = _make_args(custom_rm_path=f"{HELPERS}.failing_sync_reward", reward_num_workers=1)
    sample = _make_sample(index=7, metadata={"source": "unit"})

    with pytest.raises(Exception, match="index=7"):
        await async_rm(args, sample)


@pytest.mark.asyncio
async def test_custom_reward_function_is_loaded_once_per_executor(monkeypatch):
    import relax.engine.rewards as reward_module

    calls = []
    original_load_function = reward_module.load_function

    def counting_load_function(path):
        calls.append(path)
        return original_load_function(path)

    monkeypatch.setattr(reward_module, "load_function", counting_load_function)
    custom_path = f"{HELPERS}.async_reward"
    args = _make_args(custom_rm_path=custom_path)
    samples = [_make_sample(index=idx) for idx in range(4)]

    await batched_async_rm(args, samples)

    assert calls == [custom_path]
