# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import time
from collections.abc import Callable, Sequence
from typing import Any


def execute_hybrid_forward_mini(
    *,
    chunks_per_mini: int,
    batch_index_for_chunk: Callable[[int], int],
    restore_actor: Callable[[int], None],
    fetch_chunk: Callable[[int], tuple[Any, list[int]]],
    forward_chunk: Callable[[Any, int, list[int]], None],
    overlap_producer: bool = True,
) -> list[tuple[Any, list[int]]]:
    """Restore once, then execute a matched chunk schedule with optional
    overlap."""
    if chunks_per_mini <= 0:
        raise ValueError(f"chunks_per_mini must be positive, got {chunks_per_mini}")
    if type(overlap_producer) is not bool:
        raise TypeError(f"overlap_producer must be bool, got {overlap_producer!r}")

    batch_indexes = [batch_index_for_chunk(chunk_index) for chunk_index in range(chunks_per_mini)]
    restore_actor(batch_indexes[0])

    chunks: list[tuple[Any, list[int]]] = []
    for batch_index in batch_indexes:
        batch, global_indexes = fetch_chunk(batch_index)
        chunks.append((batch, global_indexes))
        if overlap_producer:
            forward_chunk(batch, batch_index, global_indexes)

    if not overlap_producer:
        for (batch, global_indexes), batch_index in zip(chunks, batch_indexes, strict=True):
            forward_chunk(batch, batch_index, global_indexes)
    return chunks


def canonicalize_hybrid_microbatch_schedule(
    chunk_schedules: Sequence[tuple[Sequence[int], Sequence[Sequence[int]]]],
    canonical_global_indexes: Sequence[int],
) -> list[list[int]]:
    """Translate chunk-local forward schedules into merged-batch indexes.

    The actor old-logprob forward chooses its dynamic microbatches
    independently for each producer chunk. Training must replay those exact
    sample groups and their order; otherwise batch-shape-dependent numerics
    create an artificial PPO ratio even though the weights did not change.
    """
    canonical_indexes = list(canonical_global_indexes)
    if not canonical_indexes:
        raise ValueError("canonical_global_indexes must not be empty")
    if not all(type(index) is int for index in canonical_indexes):
        raise TypeError("canonical_global_indexes must contain only int values")
    if len(set(canonical_indexes)) != len(canonical_indexes):
        raise ValueError("canonical_global_indexes must not contain duplicates")

    canonical_positions = {index: position for position, index in enumerate(canonical_indexes)}
    observed_global_indexes: list[int] = []
    merged_schedule: list[list[int]] = []

    for chunk_index, (global_indexes, local_schedule) in enumerate(chunk_schedules):
        chunk_global_indexes = list(global_indexes)
        if not chunk_global_indexes:
            raise ValueError(f"chunk {chunk_index} global_indexes must not be empty")
        if not all(type(index) is int for index in chunk_global_indexes):
            raise TypeError(f"chunk {chunk_index} global_indexes must contain only int values")

        flattened_local_indexes: list[int] = []
        for microbatch_index, local_indexes in enumerate(local_schedule):
            normalized_local_indexes = list(local_indexes)
            if not normalized_local_indexes:
                raise ValueError(f"chunk {chunk_index} microbatch {microbatch_index} must not be empty")
            if not all(type(index) is int for index in normalized_local_indexes):
                raise TypeError(f"chunk {chunk_index} microbatch {microbatch_index} must contain only int indexes")
            try:
                merged_schedule.append(
                    [
                        canonical_positions[chunk_global_indexes[local_index]]
                        for local_index in normalized_local_indexes
                    ]
                )
            except IndexError as exc:
                raise ValueError(
                    f"chunk {chunk_index} microbatch {microbatch_index} contains an out-of-range local index"
                ) from exc
            except KeyError as exc:
                raise ValueError(
                    f"chunk {chunk_index} references global index {exc.args[0]} outside canonical_global_indexes"
                ) from exc
            flattened_local_indexes.extend(normalized_local_indexes)

        expected_local_indexes = list(range(len(chunk_global_indexes)))
        if sorted(flattened_local_indexes) != expected_local_indexes:
            raise ValueError(f"chunk {chunk_index} microbatch schedule must cover each local sample exactly once")
        observed_global_indexes.extend(chunk_global_indexes)

    if sorted(observed_global_indexes) != sorted(canonical_indexes):
        raise ValueError("chunk schedules must cover each canonical global index exactly once")
    if sorted(index for microbatch in merged_schedule for index in microbatch) != list(range(len(canonical_indexes))):
        raise ValueError("merged microbatch schedule must cover the merged batch exactly once")
    return merged_schedule


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
