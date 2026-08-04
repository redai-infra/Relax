# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from examples.mem_agent.compare_results import compare_baseline, compare_pair, validate_compatible_summaries


def test_compare_pair_uses_absolute_percentage_points():
    passed = compare_pair("50", {"sub_em_pct": 41.0}, {"sub_em_pct": 43.9})
    failed = compare_pair("50", {"sub_em_pct": 41.0}, {"sub_em_pct": 44.1})

    assert passed["absolute_gap_pp"] == pytest.approx(2.9)
    assert passed["passed"] is True
    assert failed["passed"] is False


def test_compare_baseline_requires_strict_improvement():
    improved = compare_baseline("50", {"sub_em_pct": 41.0}, {"sub_em_pct": 41.1}, "sub_em_pct")
    tied = compare_baseline("50", {"sub_em_pct": 41.0}, {"sub_em_pct": 41.0}, "sub_em_pct")

    assert improved["improvement_pp"] == pytest.approx(0.1)
    assert improved["passed"] is True
    assert tied["passed"] is False


def test_comparator_rejects_control_variable_mismatch():
    common = {
        "data_file": "eval_50.jsonl",
        "data_sha256": "a" * 64,
        "evaluator_schema_version": "mem-agent-vime-eval-v1",
        "mode": "recurrent",
        "tokenizer": "frozen-tokenizer",
        "temperature": 0.7,
        "top_p": 0.95,
        "sampling_count": 1,
        "chunk_tokens": 2048,
        "max_memory_tokens": 1024,
        "max_final_tokens": 256,
        "max_chunks": 64,
        "max_input_tokens": 7936,
        "server_max_model_len": 8192,
        "total": 128,
        "successful": 128,
        "errors": 0,
    }
    mismatched = {**common, "chunk_tokens": 4096}

    validate_compatible_summaries(common, dict(common))
    with pytest.raises(ValueError, match="chunk_tokens"):
        validate_compatible_summaries(common, mismatched)

    changed_data = {**common, "data_sha256": "b" * 64}
    with pytest.raises(ValueError, match="data_sha256"):
        validate_compatible_summaries(common, changed_data)

    incomplete = {**common, "successful": 127, "errors": 1}
    with pytest.raises(ValueError, match="incomplete"):
        validate_compatible_summaries(common, incomplete)
