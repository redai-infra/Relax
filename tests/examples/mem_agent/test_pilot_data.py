# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import pytest

from examples.mem_agent.prepare_pilot_data import build_candidates, select_pilot_sets


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


def _row(index: int, context: str, answer: str = "x"):
    return {
        "prompt": f"Question {index}?",
        "label": answer,
        "metadata": {"question": f"Question {index}?", "context": context, "ground_truth": [answer]},
    }


def test_candidate_filter_requires_short_multichunk_answer_visible_rows():
    rows = [
        _row(0, "abcx"),  # one chunk: too short
        _row(1, "abcdxefg"),  # two chunks: eligible
        _row(2, "abcdefghxijk"),  # three chunks: eligible
        _row(3, "abcdefghijklmnopx"),  # five chunks: too long
        _row(4, "abcdefgh", answer="z"),  # answer absent
    ]

    candidates, manifest = build_candidates(
        rows,
        CharacterTokenizer(),
        chunk_tokens=4,
        min_chunks=2,
        max_chunks=3,
        candidate_count=2,
        seed=42,
    )

    assert {row["metadata"]["pilot"]["source_index"] for row in candidates} == {1, 2}
    assert all(2 <= row["metadata"]["pilot"]["num_chunks"] <= 3 for row in candidates)
    assert manifest["rejected"] == {"answer_not_in_context": 1, "too_long": 1, "too_short": 1}


def test_pass_at_n_selection_guarantees_success_failure_and_disjoint_split():
    candidates = [_row(index, f"abcdx{index}ef") | {"_id": f"q{index}"} for index in range(5)]
    records = []
    success_counts = {"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4}
    for candidate in candidates:
        for sample_index in range(4):
            records.append(
                {
                    "_id": candidate["_id"],
                    "sample_index": sample_index,
                    "judge_boxed_em": float(sample_index < success_counts[candidate["_id"]]),
                }
            )

    train_rows, eval_rows, manifest = select_pilot_sets(
        candidates,
        records,
        samples_per_item=4,
        train_count=2,
        eval_count=1,
        seed=7,
        preferred_min_successes=2,
        preferred_max_successes=2,
    )

    selected = train_rows + eval_rows
    assert len(train_rows) == 2
    assert len(eval_rows) == 1
    assert not ({row["_id"] for row in train_rows} & {row["_id"] for row in eval_rows})
    assert all(0 < row["metadata"]["pilot"]["baseline_successes"] < 4 for row in selected)
    assert all(row["metadata"]["pilot"]["baseline_pass_at_n"] is True for row in selected)
    assert {entry["status"] for entry in manifest["screening"]} >= {"preferred", "no_reward_variance"}


def test_pass_at_n_selection_fails_before_gpu_training_when_pool_is_too_hard():
    candidates = [_row(index, f"abcdx{index}ef") | {"_id": f"q{index}"} for index in range(2)]
    records = [
        {"_id": candidate["_id"], "sample_index": sample_index, "judge_boxed_em": 0.0}
        for candidate in candidates
        for sample_index in range(4)
    ]

    with pytest.raises(ValueError, match="non-degenerate Pass@4"):
        select_pilot_sets(
            candidates,
            records,
            samples_per_item=4,
            train_count=1,
            eval_count=1,
            seed=42,
        )
