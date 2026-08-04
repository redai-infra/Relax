# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Rule-based final-answer reward for MemAgent."""

from __future__ import annotations

import re
import string
from typing import Any


def extract_last_boxed(text: str) -> str:
    """Extract the payload of the last balanced ``\\boxed{...}``."""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return ""
    left = text.find("{", start)
    depth = 0
    for position in range(left, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[left + 1 : position].strip()
    return ""


def normalize_answer(answer: str) -> str:
    """Apply the standard HotpotQA exact-match normalization."""
    answer = answer.lower()
    answer = "".join(character for character in answer if character not in set(string.punctuation))
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def exact_match_any(prediction: str, ground_truths: list[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(str(answer)) for answer in ground_truths)


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> dict[str, Any]:
    """Score only the final boxed answer; memory turns receive no direct
    score."""
    del args, kwargs
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    final_output = str(metadata.get("final_output") or sample.response or "")
    ground_truths = metadata.get("ground_truth") or ([] if sample.label is None else [sample.label])
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    prediction = extract_last_boxed(final_output[-300:])
    score = float(bool(prediction) and exact_match_any(prediction, list(ground_truths)))
    return {
        "score": score,
        "pred": prediction,
        "gt": str(ground_truths[0]) if ground_truths else "",
        "diagnostic": "matched" if score else ("missing_boxed" if not prediction else "answer_mismatch"),
    }
