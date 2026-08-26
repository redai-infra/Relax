# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import pytest
import torch

from relax.utils.training import hybrid_pipeline_trace


def test_global_index_fingerprint_is_order_independent_and_composable():
    combined = hybrid_pipeline_trace.fingerprint_global_indexes([11, 12, 11])
    reordered = hybrid_pipeline_trace.fingerprint_global_indexes([12, 11, 11])
    first = hybrid_pipeline_trace.fingerprint_global_indexes([11])
    second = hybrid_pipeline_trace.fingerprint_global_indexes([12, 11])

    assert combined == reordered
    assert combined != hybrid_pipeline_trace.fingerprint_global_indexes([11, 12])
    assert int(combined, 16) == (int(first, 16) + int(second, 16)) % (1 << 128)


def test_trace_disabled_creates_no_files(tmp_path):
    args = SimpleNamespace(hybrid_pipeline_trace_dir=None)

    record = hybrid_pipeline_trace.emit_hybrid_pipeline_event(
        args,
        "actor_forward_start",
        rollout_id=1,
        role="actor",
    )

    assert record is None
    assert list(tmp_path.iterdir()) == []


def test_trace_records_metrics_without_sample_content(tmp_path, monkeypatch):
    hybrid_pipeline_trace._close_writers()
    monkeypatch.setattr(hybrid_pipeline_trace.socket, "gethostname", lambda: "test-host")
    monkeypatch.setattr(hybrid_pipeline_trace.os, "getpid", lambda: 123)
    monkeypatch.setattr(hybrid_pipeline_trace, "_global_rank", lambda: 4)
    monkeypatch.setattr(hybrid_pipeline_trace, "_cuda_memory_peaks", lambda: (1024, 2048))
    args = SimpleNamespace(hybrid_pipeline_trace_dir=str(tmp_path))
    batch = {
        "total_lengths": [10, 20],
        "response_lengths": [4, 5],
        "multimodal_train_inputs": [
            {"pixel_values": torch.zeros((2, 3), dtype=torch.float32)},
            None,
        ],
        "prompt": "SECRET_SAMPLE_CONTENT",
    }

    record = hybrid_pipeline_trace.emit_hybrid_pipeline_event(
        args,
        "chunk_fetch_end",
        rollout_id=7,
        role="actor",
        chunk_index=1,
        batch=batch,
        global_indexes=[11, 12],
        monotonic_ns=99,
        details={"mini_index": 0, "is_last": False},
    )

    assert record["sample_count"] == 2
    assert record["total_tokens"] == 30
    assert record["response_tokens"] == 9
    assert record["multimodal_tensor_bytes"] == 24
    assert record["global_indexes_fingerprint"] == hybrid_pipeline_trace.fingerprint_global_indexes([11, 12])
    assert record["cuda_max_allocated_bytes"] == 1024
    assert record["cuda_max_reserved_bytes"] == 2048

    trace_files = list(tmp_path.glob("*.jsonl"))
    assert [path.name for path in trace_files] == ["hybrid-pipeline-actor-test-host-pid123-rank4.jsonl"]
    payload = trace_files[0].read_text(encoding="utf-8")
    assert "SECRET_SAMPLE_CONTENT" not in payload
    assert json.loads(payload)["event"] == "chunk_fetch_end"
    hybrid_pipeline_trace._close_writers()


def test_trace_rejects_non_scalar_details(tmp_path):
    args = SimpleNamespace(hybrid_pipeline_trace_dir=str(tmp_path))

    with pytest.raises(TypeError, match="trace details must be scalar"):
        hybrid_pipeline_trace.emit_hybrid_pipeline_event(
            args,
            "event",
            rollout_id=1,
            role="actor",
            details={"sample": {"must": "not be serialized"}},
        )


def test_trace_rejects_reserved_detail_fields(tmp_path):
    args = SimpleNamespace(hybrid_pipeline_trace_dir=str(tmp_path))

    with pytest.raises(ValueError, match="cannot replace record fields"):
        hybrid_pipeline_trace.emit_hybrid_pipeline_event(
            args,
            "event",
            rollout_id=1,
            role="actor",
            details={"event": "overwritten"},
        )
