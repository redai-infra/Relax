import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[3] / "relax" / "engine" / "rewards" / "math_dapo_utils.py"
_SPEC = importlib.util.spec_from_file_location("math_dapo_utils", _MODULE_PATH)
assert _SPEC is not None
math_dapo_utils = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(math_dapo_utils)

compute_score = math_dapo_utils.compute_score
is_correct_strict_box = math_dapo_utils.is_correct_strict_box
last_boxed_only_string = math_dapo_utils.last_boxed_only_string
normalize_final_answer = math_dapo_utils.normalize_final_answer
remove_boxed = math_dapo_utils.remove_boxed


def test_last_boxed_only_string_returns_last_complete_boxed_expression():
    text = "first \\boxed{3}, reasoning continues, final \\boxed{\\frac{9}{3}}"

    boxed = last_boxed_only_string(text)

    assert boxed == "\\boxed{\\frac{9}{3}}"


def test_last_boxed_only_string_returns_none_without_boxed_answer():
    boxed = last_boxed_only_string("推理过程里没有最终 boxed 答案")

    assert boxed is None


def test_last_boxed_only_string_returns_none_for_incomplete_boxed_expression():
    boxed = last_boxed_only_string("推理过程...... 最终答案是 \\boxed{9")

    assert boxed is None


def test_remove_boxed_returns_inner_answer():
    answer = remove_boxed("\\boxed{42}")

    assert answer == "42"


def test_remove_boxed_rejects_non_boxed_input():
    with pytest.raises(AssertionError, match="box error"):
        remove_boxed("42")


def test_normalize_final_answer_removes_units_spaces_and_commas():
    normalized = normalize_final_answer("answer = 1,234 dollars")

    assert normalized == "1234"


def test_normalize_final_answer_takes_rhs_of_equation():
    normalized = normalize_final_answer("x = 42")

    assert normalized == "42"


def test_normalize_final_answer_unwraps_text():
    normalized = normalize_final_answer(r"\text{apples}")

    assert normalized == "apples"


def test_is_correct_strict_box_accepts_matching_boxed_answer():
    score, extracted = is_correct_strict_box("推理过程...... 最终答案是 \\boxed{9}", "9")

    assert score == 1
    assert extracted == "9"


def test_is_correct_strict_box_rejects_wrong_boxed_answer():
    score, extracted = is_correct_strict_box("推理过程...... 最终答案是 \\boxed{8}", "9")

    assert score == -1
    assert extracted == "8"


def test_is_correct_strict_box_uses_pause_token_window():
    pred = "outdated answer \\boxed{8}" + ("x" * 120) + " final answer \\boxed{9}"
    pause_tokens_index = [0, 10, 20, len(pred)]

    score, extracted = is_correct_strict_box(pred, "9", pause_tokens_index=pause_tokens_index)

    assert score == 1
    assert extracted == "9"


def test_compute_score_returns_positive_dict_for_strict_boxed_answer():
    result = compute_score("推理过程...... 最终答案是 \\boxed{9}", "9", strict_box_verify=True)

    assert result == {"score": 1.0, "acc": True, "pred": "9"}


def test_compute_score_returns_negative_dict_when_boxed_answer_is_missing():
    result = compute_score("推理过程...... 最终答案是 9", "9", strict_box_verify=True)

    assert result == {"score": -1.0, "acc": False, "pred": None}


def test_compute_score_supports_default_answer_pattern():
    result = compute_score("We solve the equation step by step.\nAnswer: 9", "9")

    assert result == {"score": 1.0, "acc": True, "pred": "9"}


def test_compute_score_returns_negative_dict_for_default_wrong_answer():
    result = compute_score("We solve the equation step by step.\nAnswer: 8", "9")

    assert result == {"score": -1.0, "acc": False, "pred": "8"}
