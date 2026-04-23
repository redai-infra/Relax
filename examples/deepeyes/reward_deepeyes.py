from __future__ import annotations

import asyncio
import os
import random
import re

from relax.utils.types import Sample


TIMEOUT = int(os.environ.get("DEEPEYES_JUDGE_TIMEOUT", "120"))


def extract_answer(text: str):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgment: 1
"""  # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgment: 1
"""  # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgment: 1
"""  # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgment: 1
"""  # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgment: 1
"""  # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgment: 0
"""  # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgment: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgment is 1; if they are different, Judgment is 0. Just output Judgment and don't output anything else.\n\n
"""
    return chat_template


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgment:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt


def _get_judge_client():
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("openai package is required for DeepEyes judge scoring.") from exc

    api_key = os.environ.get("DEEPEYES_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPEYES_JUDGE_API_KEY or OPENAI_API_KEY for DeepEyes judge scoring.")

    base_url = os.environ.get("DEEPEYES_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    models = os.environ.get("DEEPEYES_JUDGE_MODELS") or os.environ.get("DEEPEYES_JUDGE_MODEL") or "gpt-4o"
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    return client, model_list


def compute_score(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    is_format_error = False
    format_error_reasons: list[str] = []

    # ensure think token
    if not predict_str.startswith("<think>"):
        predict_str = "<think>" + predict_str

    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True
        format_error_reasons.append("think_tag_mismatch")
    if count_think_1 == 0 or count_think_2 == 0:
        is_format_error = True
        format_error_reasons.append("think_tag_missing")

    count_vision_1 = predict_str.count("<tool_response>")
    count_vision_2 = predict_str.count("</tool_response>")
    if count_vision_1 != count_vision_2:
        is_format_error = True
        format_error_reasons.append("tool_response_tag_mismatch")

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True
        format_error_reasons.append("answer_tag_mismatch")
    if count_answer_1 == 0 or count_answer_2 == 0:
        is_format_error = True
        format_error_reasons.append("answer_tag_missing")

    answer_text = extract_answer(predict_no_think)
    if not answer_text:
        is_format_error = True
        format_error_reasons.append("answer_extract_failed")

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if answer_text and len(answer_text) >= 300:
        is_format_error = True
        format_error_reasons.append("answer_too_long")

    if is_format_error:
        acc_reward = 0.0
        response = ""
    else:
        if not isinstance(extra_info, dict) or "question" not in extra_info:
            raise ValueError("extra_info with 'question' is required for DeepEyes judge scoring.")
        question_text = extra_info["question"]
        full_prompt = get_prompt(answer_text, ground_truth, question_text)

        client, model_list = _get_judge_client()
        model_name = random.choice(model_list)
        response = "error"
        for attempt in range(3):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": full_prompt},
                    ],
                    seed=random.randint(0, 1000000),
                    temperature=0.3,
                    timeout=TIMEOUT,
                )
                response = chat_response.choices[0].message.content.strip()
                break
            except BaseException as exc:
                print(f"[ERROR] request model={model_name} attempt={attempt + 1}/3 error: {exc}")
                if attempt == 2:
                    response = "error"

        # print(response)
        if "Judgment:" in response:
            response = response.split("Judgment:")[-1].strip()
            if "1" in response:
                acc_reward = 1.0
            elif "0" in response:
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error response={response}")
                acc_reward = 0.0
        else:
            if response == "1":
                acc_reward = 1.0
            elif response == "0":
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error response={response}")
                acc_reward = 0.0

    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    final_score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
    format_error_reason = ",".join(sorted(set(format_error_reasons)))
    return {
        "score": final_score,
        "acc": acc_reward,
        "format": format_reward,
        "tool": tool_reward,
        "judge_response": response,
        "format_error_reason": format_error_reason,
        "count_vision_1": count_vision_1,
        "predict_str": predict_str,
        "ground_truth": ground_truth,
    }


async def reward_func(args, sample: Sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    question = sample.metadata.get("question")
    ground_truth = sample.metadata.get("answer")
    if question is None or ground_truth is None:
        raise ValueError(f"question or answer is missing, {question=}, {ground_truth=}")
    return await asyncio.to_thread(compute_score, sample.response, ground_truth, {"question": question})
