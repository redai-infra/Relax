# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Custom reward fixtures used by RewardExecutor integration tests."""

import asyncio
import functools
import os
import time


_DYNAMIC_LOAD_COUNT = 0
_RELOAD_GENERATION = globals().get("_RELOAD_GENERATION", 0) + 1


def sync_process_reward(args, sample, **kwargs):
    return {
        "pid": os.getpid(),
        "index": sample.index,
        "response": sample.response,
        "marker": kwargs.get("marker"),
    }


def sync_timed_reward(args, sample, **kwargs):
    started_at = time.monotonic()
    time.sleep(args.test_reward_delay)
    return {
        "pid": os.getpid(),
        "index": sample.index,
        "started_at": started_at,
        "finished_at": time.monotonic(),
    }


def sync_cancellable_reward(args, sample, **kwargs):
    if sample.response == "block":
        import ray

        ray.get(args.test_probe.mark_started.remote(os.getpid()))
        time.sleep(args.test_reward_delay)
    return {"pid": os.getpid(), "index": sample.index}


def sync_maybe_failing_reward(args, sample, **kwargs):
    if sample.response == "fail":
        raise ValueError("intentional custom reward failure")
    return float(sample.index)


def sync_group_reward(args, samples, **kwargs):
    return [float(sample.index) for sample in samples]


async def async_process_reward(args, sample, **kwargs):
    await asyncio.sleep(0)
    return {
        "pid": os.getpid(),
        "index": sample.index,
        "marker": kwargs.get("marker"),
    }


def _wrap_async_reward(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


@_wrap_async_reward
async def decorated_async_process_reward(args, sample, **kwargs):
    await asyncio.sleep(0)
    return {
        "pid": os.getpid(),
        "index": sample.index,
    }


async def async_group_reward(args, samples, **kwargs):
    await asyncio.sleep(0)
    return [float(sample.index) for sample in samples]


def sync_reloadable_reward(args, sample, _generation=_RELOAD_GENERATION, **kwargs):
    return {"generation": _generation, "pid": os.getpid()}


async def async_reloadable_reward(args, sample, _generation=_RELOAD_GENERATION, **kwargs):
    await asyncio.sleep(0)
    return {"generation": _generation, "pid": os.getpid()}


def __getattr__(name):
    """Expose a function whose attribute lookup count is observable per
    process."""
    if name != "counted_sync_reward":
        raise AttributeError(name)

    global _DYNAMIC_LOAD_COUNT
    _DYNAMIC_LOAD_COUNT += 1

    def counted_sync_reward(args, sample, **kwargs):
        return {
            "pid": os.getpid(),
            "load_count": _DYNAMIC_LOAD_COUNT,
        }

    return counted_sync_reward
