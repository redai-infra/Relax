# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.utils.training.hybrid_forward_pipeline import (
    canonicalize_hybrid_microbatch_schedule,
    execute_hybrid_forward_mini,
    fetch_exact_chunk_with_timeout,
)


def test_first_chunk_forward_runs_before_later_chunk_is_ready():
    events = []
    chunk_one_ready = False

    def restore_actor(batch_index):
        events.append(("restore", batch_index))

    def fetch_chunk(batch_index):
        nonlocal chunk_one_ready
        if batch_index == 1:
            assert chunk_one_ready, "chunk 1 was fetched before chunk 0 forward completed"
        events.append(("fetch", batch_index))
        return {"total_lengths": [batch_index + 1]}, [100 + batch_index]

    def forward_chunk(batch, batch_index, global_indexes):
        nonlocal chunk_one_ready
        events.append(("forward", batch_index, global_indexes))
        if batch_index == 0:
            chunk_one_ready = True

    chunks = execute_hybrid_forward_mini(
        chunks_per_mini=2,
        batch_index_for_chunk=lambda chunk_index: chunk_index,
        restore_actor=restore_actor,
        fetch_chunk=fetch_chunk,
        forward_chunk=forward_chunk,
    )

    assert events == [
        ("restore", 0),
        ("fetch", 0),
        ("forward", 0, [100]),
        ("fetch", 1),
        ("forward", 1, [101]),
    ]
    assert chunks == [
        ({"total_lengths": [1]}, [100]),
        ({"total_lengths": [2]}, [101]),
    ]


def test_restore_occurs_once_per_optimizer_mini():
    restore_calls = []

    for mini_index in range(3):
        execute_hybrid_forward_mini(
            chunks_per_mini=2,
            batch_index_for_chunk=lambda chunk_index, mini=mini_index: mini * 2 + chunk_index,
            restore_actor=restore_calls.append,
            fetch_chunk=lambda batch_index: ({"total_lengths": [1]}, [batch_index]),
            forward_chunk=lambda batch, batch_index, global_indexes: None,
        )

    assert restore_calls == [0, 2, 4]


def test_no_overlap_control_fetches_all_chunks_before_matched_forwards():
    events = []

    chunks = execute_hybrid_forward_mini(
        chunks_per_mini=3,
        batch_index_for_chunk=lambda chunk_index: 10 + chunk_index,
        restore_actor=lambda batch_index: events.append(("restore", batch_index)),
        fetch_chunk=lambda batch_index: (
            events.append(("fetch", batch_index)) or {"total_lengths": [batch_index]},
            [100 + batch_index],
        ),
        forward_chunk=lambda _batch, batch_index, global_indexes: events.append(
            ("forward", batch_index, global_indexes)
        ),
        overlap_producer=False,
    )

    assert events == [
        ("restore", 10),
        ("fetch", 10),
        ("fetch", 11),
        ("fetch", 12),
        ("forward", 10, [110]),
        ("forward", 11, [111]),
        ("forward", 12, [112]),
    ]
    assert [indexes for _, indexes in chunks] == [[110], [111], [112]]


def test_chunk_microbatch_schedule_is_replayed_on_canonical_batch():
    schedule = canonicalize_hybrid_microbatch_schedule(
        [
            ([12, 10], [[1], [0]]),
            ([13, 11], [[0, 1]]),
        ],
        [10, 11, 12, 13],
    )

    assert schedule == [[0], [2], [3, 1]]


@pytest.mark.parametrize(
    ("chunk_schedules", "canonical_indexes", "error"),
    [
        ([([10, 11], [[0]])], [10, 11], "cover each local sample exactly once"),
        ([([10, 11], [[0, 0], [1]])], [10, 11], "cover each local sample exactly once"),
        ([([10, 11], [[0, 2]])], [10, 11], "out-of-range local index"),
        ([([10], [[0]])], [10, 11], "cover each canonical global index exactly once"),
        ([([10, 12], [[0, 1]])], [10, 11], "outside canonical_global_indexes"),
    ],
)
def test_chunk_microbatch_schedule_rejects_incomplete_or_invalid_coverage(
    chunk_schedules,
    canonical_indexes,
    error,
):
    with pytest.raises(ValueError, match=error):
        canonicalize_hybrid_microbatch_schedule(chunk_schedules, canonical_indexes)


def test_invalid_chunk_count_fails_before_restore():
    restore_calls = []

    with pytest.raises(ValueError, match="chunks_per_mini must be positive"):
        execute_hybrid_forward_mini(
            chunks_per_mini=0,
            batch_index_for_chunk=lambda chunk_index: chunk_index,
            restore_actor=restore_calls.append,
            fetch_chunk=lambda batch_index: ({}, []),
            forward_chunk=lambda batch, batch_index, global_indexes: None,
        )

    assert restore_calls == []


def test_exact_fetch_retries_then_returns_complete_chunk():
    attempts = iter(
        [
            (None, None),
            ({"total_lengths": [1, 2]}, "metadata"),
        ]
    )
    now = [0.0]

    batch, metadata, elapsed = fetch_exact_chunk_with_timeout(
        fetch_once=lambda: next(attempts),
        expected_samples=2,
        timeout_s=5,
        error_context="rollout_id=7, mini_index=0, chunk_index=1",
        clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert batch == {"total_lengths": [1, 2]}
    assert metadata == "metadata"
    assert elapsed == 0.1


def test_exact_fetch_rejects_underfilled_chunk():
    with pytest.raises(RuntimeError, match="expected=2, last_returned=1"):
        fetch_exact_chunk_with_timeout(
            fetch_once=lambda: ({"total_lengths": [1]}, "metadata"),
            expected_samples=2,
            timeout_s=5,
            error_context="rollout_id=7, mini_index=0, chunk_index=1",
        )


def test_exact_fetch_timeout_contains_actionable_context():
    now = [0.0]

    with pytest.raises(
        TimeoutError,
        match="rollout_id=7, mini_index=0, chunk_index=1, expected=2, last_returned=0",
    ):
        fetch_exact_chunk_with_timeout(
            fetch_once=lambda: (None, None),
            expected_samples=2,
            timeout_s=0.2,
            error_context="rollout_id=7, mini_index=0, chunk_index=1",
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )


@pytest.mark.parametrize("timeout_s", [float("nan"), float("inf"), 0, -1])
def test_exact_fetch_rejects_invalid_timeout(timeout_s):
    with pytest.raises(ValueError, match="positive and finite"):
        fetch_exact_chunk_with_timeout(
            fetch_once=lambda: (None, None),
            expected_samples=2,
            timeout_s=timeout_s,
            error_context="rollout_id=7",
        )
