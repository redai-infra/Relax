# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.mem_agent.reward import extract_last_boxed, normalize_answer, reward_func
from relax.utils.types import Sample


def test_extract_last_boxed_supports_nested_braces_and_uses_last_answer():
    assert extract_last_boxed(r"first \boxed{wrong}; final \boxed{New {York}}") == "New {York}"
    assert extract_last_boxed(r"incomplete \boxed{answer") == ""


def test_normalize_answer_uses_hotpotqa_rules():
    assert normalize_answer("The, Eiffel Tower!") == "eiffel tower"


@pytest.mark.asyncio
async def test_reward_scores_only_final_output_against_all_ground_truths():
    sample = Sample(
        response=r"ignored \boxed{wrong}",
        label="wrong",
        metadata={"final_output": r"answer: \boxed{The Eiffel Tower}", "ground_truth": ["Paris", "Eiffel Tower"]},
        train_metadata={"mem_agent_turns": [{"response": "unscored memory"}]},
    )
    result = await reward_func(SimpleNamespace(), sample)
    assert result["score"] == 1.0
    assert result["pred"] == "The Eiffel Tower"
    assert result["diagnostic"] == "matched"


@pytest.mark.asyncio
async def test_reward_reports_missing_boxed_as_zero():
    sample = Sample(response="plain answer", label="answer")
    result = await reward_func(SimpleNamespace(), sample)
    assert result["score"] == 0.0
    assert result["diagnostic"] == "missing_boxed"
