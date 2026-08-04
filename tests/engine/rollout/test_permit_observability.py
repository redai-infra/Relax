# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

from relax.engine.rollout.permit_observability import (
    begin_wait,
    capture_sglang_timing,
    export_rows,
    mark_engine_request_started,
    mark_granted,
    mark_terminal,
)


def _sample():
    return SimpleNamespace(
        response_length=17,
        tokens=list(range(41)),
        group_index=101,
        index=202,
        abort_count=1,
        status="completed",
        metadata={},
        prompt="secret prompt must never be serialized",
    )


def test_permit_row_records_only_shape_and_lifecycle(tmp_path) -> None:
    sample = _sample()
    row = begin_wait(
        sample,
        physical_rollout_id=3,
        permit_snapshot={"capacity": 64, "in_use": 64, "waiting": 2},
    )
    mark_granted(row, {"capacity": 64, "in_use": 64, "waiting": 1})
    mark_engine_request_started(sample)
    capture_sglang_timing(
        sample,
        {
            "queue_time": 0.25,
            "forward_entry_time": 1_722_000_000.0,
            "prefill_finished_time": 1_722_000_000.75,
        },
    )
    sample.response_length = 29
    sample.tokens.extend(range(41, 53))
    mark_terminal(row, sample)

    assert row["attempt_kind"] == "resume"
    assert row["context_tokens_before"] == 41
    assert row["generated_tokens_this_attempt"] == 12
    assert row["engine_request_started"] is True
    assert row["sglang_queue_plus_prefill_seconds"] == 1.0
    assert sample.metadata == {}
    assert "prompt" not in row

    path = export_rows([row], directory=str(tmp_path), physical_rollout_id=3)
    exported = json.loads(path.read_text(encoding="utf-8"))
    assert exported["permit_wait_id"] == row["permit_wait_id"]
    assert "secret prompt" not in path.read_text(encoding="utf-8")


def test_cancelled_before_grant_remains_observable() -> None:
    sample = _sample()
    row = begin_wait(sample, physical_rollout_id=4, permit_snapshot=None)
    mark_terminal(row, sample, RuntimeError("cancelled"))

    assert row["permit_wait_status"] == "cancelled_before_grant"
    assert row["request_wall_seconds"] is None
    assert row["exception_type"] == "RuntimeError"


def test_fresh_context_is_recovered_after_tokenization() -> None:
    sample = _sample()
    sample.response_length = 0
    sample.tokens = []
    row = begin_wait(sample, physical_rollout_id=5, permit_snapshot=None)

    sample.tokens = list(range(23))
    sample.response_length = 7
    mark_terminal(row, sample)

    assert row["generated_tokens_this_attempt"] == 7
    assert row["context_tokens_before"] == 16


def test_terminal_before_engine_request_is_explicit() -> None:
    sample = _sample()
    row = begin_wait(sample, physical_rollout_id=6, permit_snapshot=None)
    mark_granted(row, None)
    mark_terminal(row, sample)

    assert row["permit_wait_status"] == "terminal"
    assert row["engine_request_started"] is False
    assert row["sglang_queue_plus_prefill_seconds"] is None
