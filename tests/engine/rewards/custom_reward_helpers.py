# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
import os
import time
from pathlib import Path


def sync_reward(args, sample, **kwargs):
    return {
        "score": 1.0 if sample.response == sample.label else 0.0,
        "pid": os.getpid(),
        "index": sample.index,
        "tag": kwargs.get("tag"),
    }


async def async_reward(args, sample, **kwargs):
    await asyncio.sleep(0)
    return {
        "score": 2.0,
        "pid": os.getpid(),
        "index": sample.index,
        "tag": kwargs.get("tag"),
    }


def sync_batch_reward(args, samples, **kwargs):
    pid = os.getpid()
    return [
        {
            "score": 1.0 if sample.response == sample.label else 0.0,
            "pid": pid,
            "index": sample.index,
            "tag": kwargs.get("tag"),
        }
        for sample in samples
    ]


def failing_sync_reward(args, sample, **kwargs):
    del args, kwargs
    raise ValueError(f"boom for sample {sample.index}")


def _update_counter(path: str, delta: int) -> None:
    import fcntl

    counter_path = Path(path)
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    with counter_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        state = json.loads(raw) if raw else {"active": 0, "max_active": 0}
        state["active"] += delta
        state["max_active"] = max(state["max_active"], state["active"])
        handle.seek(0)
        handle.truncate()
        json.dump(state, handle)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def metered_sync_reward(args, sample, **kwargs):
    del kwargs
    _update_counter(args.reward_counter_path, 1)
    try:
        time.sleep(args.reward_sleep_s)
        return {
            "score": 1.0,
            "pid": os.getpid(),
            "index": sample.index,
        }
    finally:
        _update_counter(args.reward_counter_path, -1)
