# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from examples.on_policy_distillation.sdpo.prepare_data import normalize_rows
from examples.on_policy_distillation.sdpo.reward import score


def _sample(metadata: dict, response: str) -> SimpleNamespace:
    return SimpleNamespace(metadata=metadata, response=response, label="")


def test_normalize_sciknoweval_filters_l3_domains_and_preserves_split() -> None:
    rows = [
        {
            "question": "Which option?",
            "choices": {"text": ["one", "two"], "label": ["A", "B"]},
            "answerKey": "B",
            "type": "mcq-2-choices",
            "domain": "Physics",
            "details": {"level": "L3"},
        },
        {
            "question": "filtered",
            "choices": {"text": [], "label": []},
            "answerKey": "A",
            "type": "mcq-2-choices",
            "domain": "Physics",
            "details": {"level": "L2"},
        },
    ]

    normalized = normalize_rows("sciknoweval", rows, source_split="train")

    assert len(normalized) == 1
    assert normalized[0]["metadata"]["source_split"] == "train"
    assert normalized[0]["metadata"]["domain"] == "Physics"


def test_normalize_rows_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="expected 'train' or 'test'"):
        normalize_rows("sciknoweval", [], source_split="validation")


def test_normalize_sciknoweval_accepts_reference_flat_domain_format() -> None:
    rows = [
        {
            "idx": 7,
            "dataset": "sciknoweval",
            "kind": "mcq",
            "answer": "C",
            "prompt": "Question\n\nA: one\nB: two\nC: three\nD: four",
            "system": "Return only the answer tag.",
        }
    ]

    normalized = normalize_rows("sciknoweval", rows, source_split="train", domain="material")

    assert len(normalized) == 1
    assert normalized[0]["label"] == "C"
    assert normalized[0]["metadata"]["domain"] == "Materials"
    assert "Return only the answer tag." in normalized[0]["prompt"]


def test_toolalpaca_reward_accepts_canonical_action_input() -> None:
    sample = _sample(
        {
            "data_source": "toolalpaca",
            "golden_answer": [{"Action": "search", "Action_Input": '{"query": "relax"}'}],
        },
        'Action: search\nAction Input: {"query": "relax"}',
    )

    result = score(None, sample)

    assert result["score"] == 1.0
    assert result["feedback"] == ""


def test_toolalpaca_reward_parses_nested_action_input_json() -> None:
    sample = _sample(
        {
            "data_source": "toolalpaca",
            "golden_answer": [
                {
                    "Action": "sendHttpRequest",
                    "Action_Input": (
                        '{"method": "POST", "data": {"name": "John Doe", "email": "john.doe@example.com"}}'
                    ),
                }
            ],
        },
        (
            "Action: sendHttpRequest\nAction Input: "
            '{"method": "POST", "data": '
            '{"name": "John Doe", "email": "john.doe@example.com"}}'
        ),
    )

    result = score(None, sample)

    assert result["score"] == 1.0
    assert result["feedback"] == ""


def test_toolalpaca_reward_accepts_action_names_with_spaces_and_brackets() -> None:
    for action in ("Get Task", "Search Twitch", "[Optional]updateUserProfile"):
        sample = _sample(
            {
                "data_source": "toolalpaca",
                "golden_answer": [{"Action": action, "Action_Input": "{}"}],
            },
            f"Action: {action}\nAction Input: {{}}",
        )

        assert score(None, sample)["score"] == 1.0


def test_reference_tooluse_row_is_normalized_for_the_same_reward() -> None:
    rows = [
        {
            "idx": 3,
            "dataset": "tooluse",
            "kind": "tooluse",
            "prompt": "Action: <tool>\nAction Input: <JSON>",
            "answer": '[{"Action": "search", "Action_Input": "{\\"query\\": \\"relax\\"}"}]',
        }
    ]

    normalized = normalize_rows("tooluse", rows, source_split="train")

    assert len(normalized) == 1
    assert normalized[0]["metadata"]["data_source"] == "tooluse"
    assert normalized[0]["metadata"]["golden_answer"][0]["Action"] == "search"


def test_toolalpaca_reward_wrong_action_has_no_feedback() -> None:
    sample = _sample(
        {
            "data_source": "toolalpaca",
            "golden_answer": [{"Action": "search", "Action_Input": '{"query": "relax"}'}],
        },
        'Action: lookup\nAction Input: {"query": "relax"}',
    )

    result = score(None, sample)

    assert result["score"] == 0.0
    assert result["format_error"] == 0
    assert result["feedback"] == ""
    assert "search" not in result["feedback"]


_MULTI_STEP_GOLDEN = [
    {"Action": "getClientRequestData", "Action_Input": '{"url": "https://httpbin.org/get"}'},
    {"Action": "sendHttpRequest", "Action_Input": '{"method": "POST", "url": "https://httpbin.org/post"}'},
]


def _multi_step_response() -> str:
    return (
        "Action: getClientRequestData\n"
        'Action Input: {"url": "https://httpbin.org/get"}\n'
        "Action: sendHttpRequest\n"
        'Action Input: {"method": "POST", "url": "https://httpbin.org/post"}'
    )


def test_tooluse_reward_accepts_correct_multi_step_trajectory() -> None:
    sample = _sample({"data_source": "tooluse", "golden_answer": _MULTI_STEP_GOLDEN}, _multi_step_response())

    result = score(None, sample)

    assert result["score"] == 1.0
    assert result["format_error"] == 0


def test_tooluse_reward_rejects_swapped_step_order() -> None:
    swapped = (
        "Action: sendHttpRequest\n"
        'Action Input: {"method": "POST", "url": "https://httpbin.org/post"}\n'
        "Action: getClientRequestData\n"
        'Action Input: {"url": "https://httpbin.org/get"}'
    )
    sample = _sample({"data_source": "tooluse", "golden_answer": _MULTI_STEP_GOLDEN}, swapped)

    result = score(None, sample)

    assert result["score"] == 0.0


def test_tooluse_reward_rejects_cross_step_value_swap() -> None:
    swapped_values = (
        "Action: getClientRequestData\n"
        'Action Input: {"method": "POST", "url": "https://httpbin.org/post"}\n'
        "Action: sendHttpRequest\n"
        'Action Input: {"url": "https://httpbin.org/get"}'
    )
    sample = _sample({"data_source": "tooluse", "golden_answer": _MULTI_STEP_GOLDEN}, swapped_values)

    result = score(None, sample)

    assert result["score"] == 0.0


def _sciknoweval_sample(response: str, *, expected: str = "B", task_type: str = "mcq") -> SimpleNamespace:
    return _sample(
        {
            "data_source": "sciknoweval",
            "answer_key": expected,
            "task_type": task_type,
        },
        response,
    )


def test_sciknoweval_reward_accepts_answer_tag() -> None:
    result = score(None, _sciknoweval_sample("The answer is <answer>B</answer>."))

    assert result["score"] == 1.0
    assert result["format_error"] == 0
    assert result["feedback"] == ""


def test_sciknoweval_reward_missing_answer_tag_is_format_error() -> None:
    result = score(None, _sciknoweval_sample("The answer is B."))

    assert result["score"] == 0.0
    assert result["format_error"] == 1
    assert "wrong format" in result["feedback"]


def test_sciknoweval_reward_wrong_answer_has_no_feedback() -> None:
    result = score(None, _sciknoweval_sample("The answer is <answer>C</answer>."))

    assert result["score"] == 0.0
    assert result["format_error"] == 0
    assert result["feedback"] == ""


def test_sciknoweval_reward_truncation_feedback_takes_priority() -> None:
    from relax.utils.types import Sample

    sample = _sciknoweval_sample("The answer is B.")
    sample.status = Sample.Status.TRUNCATED

    result = score(None, sample)

    assert result["score"] == 0.0
    assert "truncated" in result["feedback"]
    assert "wrong format" not in result["feedback"]


def test_toolalpaca_reward_missing_format_has_feedback() -> None:
    sample = _sample(
        {
            "data_source": "toolalpaca",
            "golden_answer": [{"Action": "search", "Action_Input": '{"query": "relax"}'}],
        },
        "just a text answer",
    )

    result = score(None, sample)

    assert result["score"] == 0.0
    assert result["format_error"] == 1
    assert "format" in result["feedback"]


def test_toolalpaca_reward_truncation_has_feedback() -> None:
    from relax.utils.types import Sample

    sample = _sample(
        {
            "data_source": "toolalpaca",
            "golden_answer": [{"Action": "search", "Action_Input": '{"query": "relax"}'}],
        },
        'Action: lookup\nAction Input: {"query": "relax"}',
    )
    sample.status = Sample.Status.TRUNCATED

    result = score(None, sample)

    assert "truncated" in result["feedback"]
