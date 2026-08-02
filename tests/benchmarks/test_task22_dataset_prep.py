# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the Task 22 GSM8K dataset prep helper."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


pd = pytest.importorskip("pandas")

MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "task22_hybrid_async_text" / "prepare_task22_dataset.py"
SPEC = importlib.util.spec_from_file_location("task22_dataset_prep", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
dataset_prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dataset_prep)


def test_extract_final_answer_normalizes_whitespace() -> None:
    assert dataset_prep.extract_final_answer("The work is shown below.\n####   42  \n") == "42"


def test_subset_records_uses_first_limit_rows() -> None:
    frame = pd.DataFrame(
        {
            "question": [f"Question {idx}" for idx in range(20)],
            "answer": [f"Reasoning...\n#### {idx}\n" for idx in range(20)],
        }
    )

    records = dataset_prep.subset_records(frame, 16)

    assert len(records) == 16
    assert records[0] == {"prompt": "Question 0", "label": "0"}
    assert records[-1] == {"prompt": "Question 15", "label": "15"}


def test_main_writes_jsonl_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    download_dir = tmp_path / "download"
    source_dir = download_dir / "main"
    source_dir.mkdir(parents=True)
    (source_dir / "train-00000-of-00001.parquet").write_text("placeholder", encoding="utf-8")

    frame = pd.DataFrame(
        {
            "question": [f"Question {idx}" for idx in range(18)],
            "answer": [f"Working...\n#### {idx}\n" for idx in range(18)],
        }
    )
    output = tmp_path / "task22_gsm8k_main16.jsonl"

    monkeypatch.setattr(
        dataset_prep,
        "parse_args",
        lambda: argparse.Namespace(
            repo_id="AI-ModelScope/gsm8k",
            split="main",
            limit=16,
            output=output,
            download_dir=download_dir,
            force=False,
        ),
    )
    monkeypatch.setattr(dataset_prep, "download_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(dataset_prep.pd, "read_parquet", lambda path: frame)

    assert dataset_prep.main() == 0

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 16
    assert json.loads(lines[0]) == {"prompt": "Question 0", "label": "0"}
    assert json.loads(lines[-1]) == {"prompt": "Question 15", "label": "15"}
