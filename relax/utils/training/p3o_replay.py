# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay guards for P3O's two-pass optimizer step.

P3O computes ESS over a whole optimizer step, so the data window must be read
twice: once to accumulate the importance-ratio moments, once to train. These two
context managers make the second read identical to what a single-pass run would
have seen -- same tokens, same RNG stream. They are deliberately free of any
Megatron import so the invariants can be tested on CPU.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def preserved_rng_state() -> Iterator[None]:
    """Snapshot and restore CPU / CUDA / Megatron RNG around the stats pass.

    The train pass must see exactly the RNG stream it would have seen without a
    pre-pass, otherwise any stochastic op (dropout, MoE jitter) would
    desynchronize the two forwards -- and under tensor parallelism, the ranks
    within one forward.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    tracker = None
    tracker_states = None
    try:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        tracker = get_cuda_rng_tracker()
        tracker_states = tracker.get_states()
    except (ImportError, AssertionError, RuntimeError):
        # Tracker unavailable or uninitialized (CPU tests, no model-parallel init).
        tracker = None

    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        if tracker is not None and tracker_states is not None:
            tracker.set_states(tracker_states)


@contextmanager
def preserved_iterator_positions(data_iterator: Sequence[Any] | Any) -> Iterator[None]:
    """Snapshot and restore data-iterator offsets, deduplicated by identity.

    Under virtual pipeline parallelism the same iterator instance is passed once
    per model chunk. Restoring it twice would be harmless, but snapshotting it
    twice and restoring in the wrong order would not, so dedupe on ``id``.

    The restore runs in ``finally``: a pre-pass that raises must still leave the
    window replayable, so the error surfaces as itself rather than as a confusing
    downstream shape mismatch.

    Raises:
        RuntimeError: If an iterator cannot report its position, which would
            silently make the train pass consume different tokens.
    """
    iterators = data_iterator if isinstance(data_iterator, (list, tuple)) else [data_iterator]

    unique: dict[int, Any] = {}
    for iterator in iterators:
        if iterator is not None:
            unique.setdefault(id(iterator), iterator)

    for iterator in unique.values():
        if not (hasattr(iterator, "snapshot_position") and hasattr(iterator, "restore_position")):
            raise RuntimeError(
                f"P3O: data iterator {type(iterator).__name__} is not replayable (missing "
                "snapshot_position/restore_position). The optimizer-step ESS pre-pass must read "
                "the window twice; materialize the window or disable --advantage-estimator p3o."
            )

    positions = {key: iterator.snapshot_position() for key, iterator in unique.items()}
    try:
        yield
        # WARNING: callers must not advance or otherwise mutate any of the
        # tracked iterators *outside* this context manager while the with-block
        # is open.  External advancement between snapshot and restore will
        # silently corrupt the replay: restore_position rewinds to the saved
        # offset, causing the train pass to re-consume tokens that were already
        # consumed by the external caller rather than the tokens this pre-pass
        # saw.  Only the pre-pass (the model forward) should drive the iterators
        # while this context is live.
    finally:
        for key, iterator in unique.items():
            iterator.restore_position(positions[key])
