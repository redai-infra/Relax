# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Importable custom reward fixtures for concurrency tests."""

from __future__ import annotations

import os
import time
from typing import Any

from relax.utils.types import Sample


_LOAD_COUNTER = 0


def reset_load_counter() -> None:
    global _LOAD_COUNTER
    _LOAD_COUNTER = 0


def get_load_counter() -> int:
    return _LOAD_COUNTER


# Count module (re)loads via side effect at import/reload time is hard;
# instead track calls to ensure_loaded via a marker on first function body.


def sync_reward(args, sample: Sample, **kwargs) -> float:
    """Simple sync reward: 1.0 if response == label else 0.0."""
    return 1.0 if str(sample.response) == str(sample.label) else 0.0


def sync_reward_with_pid(args, sample: Sample, **kwargs) -> dict[str, Any]:
    return {"score": 1.0, "pid": os.getpid(), "has_rm_type": hasattr(args, "rm_type")}


def sync_slow_reward(args, sample: Sample, **kwargs) -> float:
    time.sleep(0.1)
    return 1.0


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


async def async_group_reward(args, samples: list[Sample], **kwargs) -> list[float]:
    return [float(i) for i in range(len(samples))]
