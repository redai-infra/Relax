# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from examples.mem_agent.metrics import aggregate, exact_match, f1_score, sub_exact_match
from examples.mem_agent.prepare_data import (
    convert_file,
    convert_row,
    download_hf_file,
    read_rows,
    update_manifest,
)


def test_convert_row_supports_training_and_ruler_formats():
    training = convert_row(
        {
            "prompt": [{"role": "user", "content": "Who?"}],
            "context": "Document text",
            "reward_model": {"ground_truth": ["Alice", "A. Alice"]},
            "extra_info": {"num_docs": 200},
        }
    )
    ruler = convert_row({"input": "Where?", "answers": ["Paris"], "context": "Context", "num_docs": 50})

    assert training["prompt"] == "Who?"
    assert training["metadata"]["ground_truth"] == ["Alice", "A. Alice"]
    assert training["metadata"]["num_docs"] == 200
    assert ruler["label"] == "Paris"
    assert ruler["metadata"]["question"] == "Where?"


def test_convert_file_and_manifest_are_deterministic(tmp_path):
    source = tmp_path / "eval.json"
    source.write_text(
        json.dumps([{"input": "Where?", "answers": ["Paris"], "context": "Context", "num_docs": 50}]),
        encoding="utf-8",
    )
    output = tmp_path / "eval.jsonl"
    manifest = tmp_path / "artifact_manifest.json"

    assert convert_file(source, output) == (1, 0)
    update_manifest(manifest, source, source.name, "dataset/id", "revision", output, written=1, skipped=0)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["artifacts"][0]["filename"] == "eval.json"
    assert len(payload["artifacts"][0]["sha256"]) == 64
    assert payload["artifacts"][0]["converted"]["written_rows"] == 1
    assert len(payload["artifacts"][0]["converted"]["sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8"))["metadata"]["ground_truth"] == ["Paris"]


def test_parquet_reader_and_huggingface_download_are_pinned_and_normalizable(monkeypatch, tmp_path):
    parquet_row = {
        "prompt": [{"role": "user", "content": "Who?"}],
        "context": "Document",
        "reward_model": {"ground_truth": ["Alice"]},
    }

    class FakeFrame:
        def to_dict(self, orient):
            assert orient == "records"
            return [parquet_row]

    monkeypatch.setitem(
        sys.modules,
        "pandas",
        SimpleNamespace(read_parquet=lambda path: FakeFrame()),
    )
    assert convert_row(list(read_rows(tmp_path / "train.parquet"))[0])["label"] == "Alice"

    captured = {}
    fake_hub = ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / kwargs["filename"])

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    path = download_hf_file("dataset/id", "fixed-revision", "train.parquet", tmp_path / "cache")
    assert path == tmp_path / "train.parquet"
    assert captured == {
        "repo_id": "dataset/id",
        "repo_type": "dataset",
        "revision": "fixed-revision",
        "filename": "train.parquet",
        "cache_dir": str(tmp_path / "cache"),
    }


def test_ruler_metrics_match_expected_semantics():
    assert exact_match("The Eiffel Tower", "Eiffel Tower") == 1.0
    assert sub_exact_match("located in Paris France", "Paris") == 1.0
    assert f1_score("Paris France", "Paris") == 2 / 3
    summary = aggregate(
        [
            {"judge_f1": 1.0, "judge_em": 1.0, "judge_sub_em": 1.0, "judge_boxed_em": 1.0},
            {"judge_f1": 0.0, "judge_em": 0.0, "judge_sub_em": 0.0, "judge_boxed_em": 0.0},
            {"error": "boom"},
        ]
    )
    assert summary["total"] == 3
    assert summary["successful"] == 2
    assert summary["errors"] == 1
    assert summary["sub_em_pct"] == pytest.approx(100 / 3)
    assert summary["boxed_em_pct"] == pytest.approx(100 / 3)
