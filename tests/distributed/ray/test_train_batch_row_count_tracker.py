# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio

import pytest

from relax.distributed.ray.utils import TrainBatchRowCountTracker


@pytest.mark.asyncio
async def test_train_batch_row_tracker_waits_for_matching_rollout():
    tracker = TrainBatchRowCountTracker()
    tracker.start(0)
    tracker.complete(0, 24)

    # A completed previous step must not satisfy a waiter for the next step.
    waiter = asyncio.create_task(tracker.wait(1))
    await asyncio.sleep(0)
    assert waiter.done() is False

    # Starting after wait() must reuse the same event object.
    tracker.start(1)
    tracker.complete(1, 32)
    assert await waiter == 32
    assert await tracker.wait(0) == 24


@pytest.mark.asyncio
async def test_train_batch_row_tracker_propagates_failure_to_waiter():
    tracker = TrainBatchRowCountTracker()
    tracker.start(7)
    waiter = asyncio.create_task(tracker.wait(7))
    await asyncio.sleep(0)
    tracker.fail(7, ValueError("converter failed"))

    with pytest.raises(RuntimeError, match="rollout_id=7.*converter failed"):
        await waiter


def test_train_batch_row_tracker_rejects_non_positive_count():
    tracker = TrainBatchRowCountTracker()
    tracker.start(3)
    with pytest.raises(ValueError, match="invalid train row count"):
        tracker.complete(3, 0)
