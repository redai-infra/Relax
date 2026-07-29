# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import time
from collections.abc import Callable
from typing import Any


def execute_hybrid_forward_mini(
    *,
    chunks_per_mini: int,
    batch_index_for_chunk: Callable[[int], int],
    restore_actor: Callable[[int], None],
    fetch_chunk: Callable[[int], tuple[Any, list[int]]],
    forward_chunk: Callable[[Any, int, list[int]], None],
) -> list[tuple[Any, list[int]]]:
    """Restore once, then fetch and forward each fixed actor chunk in order."""
    if chunks_per_mini <= 0:
        raise ValueError(f"chunks_per_mini must be positive, got {chunks_per_mini}")

    first_batch_index = batch_index_for_chunk(0)
    restore_actor(first_batch_index)

    chunks = []
    for chunk_index in range(chunks_per_mini):
        batch_index = batch_index_for_chunk(chunk_index)
        batch, global_indexes = fetch_chunk(batch_index)
        forward_chunk(batch, batch_index, global_indexes)
        chunks.append((batch, global_indexes))
    return chunks


def fetch_exact_chunk_with_timeout(
    *,
    fetch_once: Callable[[], tuple[Any | None, Any]],
    expected_samples: int,
    timeout_s: float,
    error_context: str,
    poll_interval_s: float = 0.1,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, Any, float]:
    """Poll an exact sample-count fetch and fail instead of spinning
    forever."""
    if expected_samples <= 0:
        raise ValueError(f"expected_samples must be positive, got {expected_samples}")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive and finite, got {timeout_s}")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError(f"poll_interval_s must be positive and finite, got {poll_interval_s}")

    start = clock()
    while True:
        batch, metadata = fetch_once()
        if batch is not None:
            actual_samples = len(batch.get("total_lengths", []))
            if actual_samples != expected_samples:
                raise RuntimeError(f"{error_context}, expected={expected_samples}, last_returned={actual_samples}")
            return batch, metadata, clock() - start

        elapsed = clock() - start
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"{error_context}, expected={expected_samples}, last_returned=0, elapsed={elapsed:.3f}s"
            )
        sleep(poll_interval_s)
