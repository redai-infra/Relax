# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Importable custom reward fixtures for concurrency tests."""

from __future__ import annotations

import os
import time
from typing import Any

from relax.utils.types import Sample


def sync_reward(args, sample: Sample, **kwargs) -> float:
    return 1.0 if str(sample.response) == str(sample.label) else 0.0


def sync_reward_with_pid(args, sample: Sample, **kwargs) -> dict[str, Any]:
    return {
        "score": 1.0,
        "pid": os.getpid(),
        "has_rm_type": hasattr(args, "rm_type"),
        "threshold": getattr(args, "reward_threshold", None),
    }


def sync_slow_reward_tracked(args, sample: Sample, **kwargs) -> float:
    import ray

    name = getattr(args, "overlap_counter_name", None)
    counter = ray.get_actor(name) if name else None
    if counter is not None:
        ray.get(counter.enter.remote())
    try:
        time.sleep(0.1)
        return 1.0
    finally:
        if counter is not None:
            ray.get(counter.leave.remote())


def sync_group_reward(args, samples: list[Sample], **kwargs) -> list[float]:
    return [1.0 if str(s.response) == str(s.label) else 0.0 for s in samples]


def sync_group_bad_length(args, samples: list[Sample], **kwargs) -> list[float]:
    return [1.0]


def sync_raises(args, sample: Sample, **kwargs) -> float:
    raise ValueError(f"boom index={sample.index}")


def sync_returns_awaitable(args, sample: Sample, **kwargs):
    async def _inner():
        return 1.0

    return _inner()


async def async_reward(args, sample: Sample, **kwargs) -> float:
    return 1.0 if str(sample.response) == str(sample.label) else 0.0


async def async_reward_with_marker(args, sample: Sample, **kwargs) -> dict[str, Any]:
    return {"score": 1.0, "async": True, "full_args": hasattr(args, "custom_rm_path")}
