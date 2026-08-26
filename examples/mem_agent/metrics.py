# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""RULER-HQA metrics used by MemAgent evaluation."""

from __future__ import annotations

from collections import Counter

from examples.mem_agent.reward import normalize_answer


def exact_match(prediction: str, ground_truth: str) -> float:
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    if prediction in ("yes", "no", "noanswer") and prediction != ground_truth:
        return 0.0
    if ground_truth in ("yes", "no", "noanswer") and prediction != ground_truth:
        return 0.0
    return float(prediction == ground_truth)


def sub_exact_match(prediction: str, ground_truth: str) -> float:
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    if not prediction or not ground_truth:
        return 0.0
    return float(ground_truth in prediction or prediction in ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    overlap = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def aggregate(records: list[dict]) -> dict[str, float | int]:
    successful = [record for record in records if not record.get("error")]
    # Evaluation failures are zero-score examples, not dropped observations.
    # Keeping the original denominator prevents transient serving errors from
    # making a run look artificially better.
    total = len(records)
    result: dict[str, float | int] = {
        "total": total,
        "successful": len(successful),
        "errors": total - len(successful),
    }
    for output_key, record_key in (
        ("f1", "judge_f1"),
        ("em", "judge_em"),
        ("sub_em", "judge_sub_em"),
        ("boxed_em", "judge_boxed_em"),
    ):
        value = sum(float(record.get(record_key, 0.0)) for record in successful) / total if total else 0.0
        result[output_key] = value
        result[f"{output_key}_pct"] = value * 100
    return result
