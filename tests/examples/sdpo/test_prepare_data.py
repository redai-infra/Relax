# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import sys

from examples.on_policy_distillation.sdpo.prepare_data import _read_rows, main


def test_read_rows_accepts_jsonl_with_json_suffix(tmp_path) -> None:
    path = tmp_path / "train.json"
    path.write_text(
        "\n".join(json.dumps(row) for row in ({"idx": 1}, {"idx": 2})) + "\n",
        encoding="utf-8",
    )

    assert _read_rows(path) == [{"idx": 1}, {"idx": 2}]


def test_read_rows_accepts_json_array(tmp_path) -> None:
    path = tmp_path / "train.json"
    path.write_text(json.dumps([{"idx": 1}, {"idx": 2}]), encoding="utf-8")

    assert _read_rows(path) == [{"idx": 1}, {"idx": 2}]


def test_prepare_data_main_writes_normalized_jsonl(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "chemistry" / "train.json"
    output_path = tmp_path / "prepared" / "chemistry.jsonl"
    input_path.parent.mkdir()
    input_path.write_text(
        json.dumps(
            {
                "idx": 4,
                "dataset": "sciknoweval",
                "kind": "mcq",
                "answer": "B",
                "prompt": "Choose B.",
                "system": "Return only the answer.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--dataset",
            "sciknoweval",
            "--input",
            str(input_path),
            "--domain",
            "chemistry",
            "--source-split",
            "train",
            "--output",
            str(output_path),
        ],
    )

    main()

    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert row["label"] == "B"
    assert row["metadata"]["source_split"] == "train"
    assert row["metadata"]["domain"] == "Chemistry"


def test_prepare_data_main_limits_rows_for_smoke_runs(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "prepared.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "idx": index,
                    "dataset": "sciknoweval",
                    "kind": "mcq",
                    "answer": "A",
                    "prompt": f"Question {index}",
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--dataset",
            "sciknoweval",
            "--input",
            str(input_path),
            "--domain",
            "chemistry",
            "--source-split",
            "train",
            "--max-rows",
            "1",
            "--output",
            str(output_path),
        ],
    )

    main()

    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1


def test_prepare_data_main_limits_rows_after_filtering(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "prepared.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "idx": index,
                    "question": f"Question {index}",
                    "answerKey": "A",
                    "choices": {"label": ["A"], "text": ["answer"]},
                    "domain": "Physics",
                    "details": {"level": level},
                }
            )
            for index, level in enumerate(("L2", "L3", "L3"))
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--dataset",
            "sciknoweval",
            "--input",
            str(input_path),
            "--source-split",
            "test",
            "--max-rows",
            "1",
            "--output",
            str(output_path),
        ],
    )

    main()

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt"].startswith("Question 1")


def test_normalize_sciknoweval_uses_answer_when_answer_key_is_empty() -> None:
    from examples.on_policy_distillation.sdpo.prepare_data import normalize_rows

    normalized = normalize_rows(
        "sciknoweval",
        [
            {
                "question": "Fill it.",
                "answerKey": "",
                "answer": "H2O",
                "domain": "Chemistry",
                "details": {"level": "L3"},
            }
        ],
        source_split="test",
    )

    assert normalized[0]["label"] == "H2O"
    assert normalized[0]["metadata"]["answer_key"] == "H2O"
