# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DeepEyesV2 reward function.

Routes scoring on ``sample.metadata["data_source"]``:

* ``perception`` -> :func:`compute_score`        (LLM-as-judge,
                                                  ``COMMON_VERIFY_PROMPT``)
* ``reason``     -> :func:`compute_score_math`   (math-verify + judge
                                                  fallback, ``MATH_VERIFY_PROMPT``)
* ``search``     -> :func:`compute_score_search` (judge + search penalty)
* ``vstar-test`` -> :func:`compute_score_acc`    (string match + judge fallback)

Final score:

* perception / reason : ``0.6 * acc + 0.2 * format + 0.2 * tool``
* search              : ``0.8 * acc * (1 - 0.1 * search_penalty) + 0.2 * format``
* ``format`` is ``1.0`` (good) / ``0.0`` (bad); ``search_penalty`` is ``0.1``
  whenever the response contains at least one ``<tool_call>``.
* ``tool`` (perception/reason only) is ``1.0`` if the response contains at
  least one well-formed ``<tool_call>`` with a known tool ``name``
  (``python_exec`` / ``search`` / ``image_search``) **AND** ``acc >= 0.5``,
  else ``0.0``. Binary by design — repeated tool calls do not stack. Gated
  on acc to match upstream DeepEyesV2's (dead-code) intent and prevent
  reward hacking where the model wraps any output in ``<tool_call>`` to
  farm the 0.2 bonus regardless of whether the tool call actually helped.
  Diverges from upstream DeepEyesV2 (which drops the bonus entirely)
  because we skip the cold-start SFT stage that taught upstream's model
  to call tools.

Judge backend env vars:

* ``DEEPEYES_JUDGE_API_KEY``  / ``OPENAI_API_KEY``
* ``DEEPEYES_JUDGE_BASE_URL`` / ``OPENAI_BASE_URL`` / ``LLM_AS_A_JUDGE_BASE``
* ``DEEPEYES_JUDGE_MODELS``   / ``DEEPEYES_JUDGE_MODEL`` (comma-separated)
* ``DEEPEYES_JUDGE_TIMEOUT``  (default 120s)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from typing import Any

from relax.utils.types import Sample


logger = logging.getLogger(__name__)

TIMEOUT = int(os.environ.get("DEEPEYES_JUDGE_TIMEOUT", "120"))
MAX_ANSWER_LEN = 300

# ---------------------------------------------------------------------------
# Embedded judge prompts
# ---------------------------------------------------------------------------

COMMON_VERIFY_PROMPT = """# 角色
你是一个判断专家，专注于判断输入的2个答案是否一致。

# 任务
你的任务是判断[模型回答]与[参考答案]是否一致，不需要思考[问题]真正的正确答案，以下是详细的判断步骤：
1. 问题理解：仔细阅读[问题]，并按照[判断标准]对问题进行分类，找出问题中包含多少个提问。
    - 问题中可能包含占位符“<image_*>”（其中“ * ”为数字），代表问题中有图片输入。注意：此类问题不用进行问题理解。
2. 答案对照：按照问题中的提问顺序，将[模型回答]与[参考答案]一一进行判断，对比是否一致。若存在一处不一致，则视为不一致。

# 判断标准

## 简答类
答案不唯一或不具体，需要根据材料、条件，自行组织语言回答问题或进行解答题目、证明结论。

### 简答（描述）
简答（描述）类问题，如材料题、写作题、图片描述等，[参考答案]与[模型回答]不需要完全一致，[模型答案]中包含[参考答案]中的要点，且表意一致（例如：参考答案为神态描写，则模型答案也必须为神态描写，否则判断为不一致），即判断为一致。

### 解答（证明）
解答（证明）类问题，如数学、物理解答（证明）题，[模型回答]必须与[参考答案]结论一致，且给出严格的解答（证明）过程。
- 如果[参考答案]包含解答（证明）过程，[模型答案]未给出解答（证明）过程，则判为不一致；反之，如果[参考答案]不包含解答（证明）过程，[模型答案]也未给出解答（证明）过程，则判为一致。

## 客观类
存在明确、客观的答案，在多个答案中选择正确答案或通过常识、计算推理直接给出答案，如科学知识、数学、物理等。
- 可以忽略答案组织形式（排版、分隔方式、是否使用Latex、大小写等）。例如：计算题只需要最终结果数值一致即可（例：“6棵”、“6”、“six”等视为一致）。

### 选择题
给出答案选项，答案选项可能用字母（A、B、C、D、...）、罗马数字（I、II、III、IV、...）或阿拉伯数字（1、2、3、4、...）标记，选择其中一个或多个选项。[模型回答]中的答案只需要与[参考答案]中对应的标记一致，即判断为一致。

### 填空题
根据[问题]补充完整陈述内容，将合适的内容填入空缺(缺省)中，题目结构通常包含明确的已知条件和需要填入的数值，或需自行组织语言。[模型回答]必须与[参考答案]中答案顺序需对应且正确，否则判断为不一致。
- 可以忽略答案组织形式（排版、分隔方式、是否使用Latex、大小写等）。例如：计算题只需要最终结果数值一致即可（例：“6棵”、“6”、“six”等视为一致）。

### 分类（判断）题
判断[问题]中指定内容是否正确，或对[问题]中给出的元素根据指定类型进行分类。[模型回答]必须给出明确的判断（或分类），且必须与[参考答案]对应，否则判为不一致。

## 图片输入选择类
仅判断[模型回答]与[参考答案]是否一致。禁止分析问题中的图片序号。
- 可以忽略答案组织形式（如排版、分隔方式、是否使用Latex等）。

# 输出
1. 在<think></think>标签中输出你的思考过程。
2. 结论输出：用一个词（是或否）在最后得出结论，格式为 \\boxed{Yes} 或 \\boxed{No}。
3. 注意：你输出的结论必须与思考过程中得到的结论一致。思考过程结论为：一致/yes，则最终结论输出：\\boxed{Yes}。

## 输出示例
<最终结果>
\\boxed{Yes/No}
<\\最终结果>


以下是输入内容：
"""

JUDGE_USER_TEMPLATE = """[问题]:{question}
[参考答案]:{answer}
[模型回答]:{prediction}"""

MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer.
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""

# vstar-test uses the same English chat template as v1
ACC_CHAT_TEMPLATE = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""

ACC_ICE_EXAMPLES = [
    """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""",
    """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""",
    """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""",
    """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""",
    """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""",
    """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""",
    """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""",
]


def _build_acc_chat_prompt(predict_str: str, ground_truth: str, question: str) -> str:
    demo = ACC_CHAT_TEMPLATE
    for ex in ACC_ICE_EXAMPLES:
        demo += ex + "\n\n"
    test = f"\n[Question]: {question}\n[Standard Answer]: {ground_truth}\n[Model_answer] : {predict_str}\nJudgement:"
    return demo + test


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------


_judge_client_cache: tuple | None = None  # (client, model_list)


def _get_judge_client():
    """Lazy build (and cache) an OpenAI-compatible client + model id list."""
    global _judge_client_cache
    if _judge_client_cache is not None:
        return _judge_client_cache

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("openai package is required for DeepEyesV2 judge scoring.") from exc

    api_key = (
        os.environ.get("DEEPEYES_JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"  # local sglang judge typically accepts EMPTY
    )
    base_url = (
        os.environ.get("DEEPEYES_JUDGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_AS_A_JUDGE_BASE")
    )
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    models_env = os.environ.get("DEEPEYES_JUDGE_MODELS") or os.environ.get("DEEPEYES_JUDGE_MODEL") or ""
    model_list = [m.strip() for m in models_env.split(",") if m.strip()]
    if not model_list:
        # Auto-discover the served model id
        try:
            models = client.models.list()
            model_list = [m.id for m in models.data]
        except Exception as exc:
            logger.warning(f"[reward_deepeyes_v2] failed to auto-discover judge models: {exc}")
    if not model_list:
        model_list = ["gpt-4o"]

    _judge_client_cache = (client, model_list)
    return _judge_client_cache


def _judge_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 8192, retries: int = 3) -> str:
    """Run a single OpenAI chat completion with retries; returns the response
    string or ``"error"`` on failure."""
    client, model_list = _get_judge_client()
    model_name = random.choice(model_list)
    response = "error"
    for attempt in range(retries):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                seed=random.randint(0, 1000000),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=TIMEOUT,
            )
            response = chat_response.choices[0].message.content.strip()
            return response
        except BaseException as exc:
            logger.warning(
                f"[reward_deepeyes_v2] judge model={model_name} attempt={attempt + 1}/{retries} error: {exc}"
            )
            if attempt == retries - 1:
                response = "error"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_SEARCH_TOOL_NAMES = frozenset({"search", "image_search"})
_VALID_TOOL_NAMES = frozenset({"python_exec", "search", "image_search"})


def extract_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def _has_valid_tool_call(predict_str: str) -> bool:
    """Return True iff at least one ``<tool_call>`` block parses as JSON with a
    known tool ``name``.

    Binary by design: per ``compute_score`` / ``compute_score_math``, repeated
    tool calls in a trajectory do not stack — the perception/reason tool bonus
    saturates at the first valid call (avoids rewarding spammy chains).
    Malformed JSON or unknown names don't trigger the bonus because they never
    actually run a tool — the model just emitted the tag.
    """
    for m in _TOOL_CALL_RE.finditer(predict_str):
        try:
            payload = json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("name") in _VALID_TOOL_NAMES:
            return True
    return False


def _count_search_tool_calls(predict_str: str) -> int:
    """Count <tool_call> blocks whose JSON ``name`` is search-like.

    Under the unified schema, python execution also uses ``<tool_call>`` with
    ``name="python_exec"`` — those must NOT count toward search_penalty,
    otherwise every code-using trajectory gets penalised as if it had searched.
    Malformed JSON / missing name are ignored (they're caught by other checks).
    """
    n = 0
    for m in _TOOL_CALL_RE.finditer(predict_str):
        try:
            payload = json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("name") in _SEARCH_TOOL_NAMES:
            n += 1
    return n


def _common_format_check(predict_str: str) -> tuple[bool, list[str], int]:
    """Run the format checks shared by perception/reason/search.

    Returns ``(is_format_error, reasons, search_call_count)`` where
    ``search_call_count`` is the number of ``<tool_call>`` blocks whose
    ``name`` is ``search`` or ``image_search`` (NOT ``python_exec``).
    :func:`compute_score_search` uses this for the search penalty.

    The legacy ``<code>...</code>`` tag-mismatch check is gone: under the
    unified <tool_call>-only schema (see ``app/prompt.py``), the model never
    emits ``<code>``, so the check was always a no-op for new data.
    """
    reasons: list[str] = []
    is_err = False

    if not predict_str.startswith("<think>"):
        predict_str = "<think>" + predict_str
    c1 = predict_str.count("<think>")
    c2 = predict_str.count("</think>")
    if c1 != c2:
        is_err = True
        reasons.append("think_tag_mismatch")
    if c1 == 0 or c2 == 0:
        is_err = True
        reasons.append("think_tag_missing")

    no_think = predict_str.split("</think>")[-1].strip()
    a1 = no_think.count("<answer>")
    a2 = no_think.count("</answer>")
    if a1 != a2:
        is_err = True
        reasons.append("answer_tag_mismatch")
    if a1 == 0 or a2 == 0:
        is_err = True
        reasons.append("answer_tag_missing")

    return is_err, reasons, _count_search_tool_calls(predict_str)


def _parse_chinese_judge(response: str) -> int:
    """Parse the bilingual judge response.

    Returns 1 (Yes) or 0 (No).
    """
    f = response
    if "<最终结果>" in f:
        f = f.split("<最终结果>")[-1].strip().split("<\\最终结果>")[0].strip()
    if "boxed" in f:
        f = f.split("boxed{")[-1].strip().split("}")[0].strip()
    return 1 if "Yes" in f else 0


def _parse_english_acc_judge(response: str) -> tuple[bool, float]:
    """Parse the English yes/no judge response used by ``vstar-test``.

    Returns ``(matched, acc_reward)``:

    - ``matched`` is ``False`` when the response cannot be parsed -- the
      caller should retry the LLM judge call.
    - ``acc_reward`` is ``1.0`` / ``0.0`` when ``matched`` is ``True``;
      ``0.0`` (placeholder) otherwise.
    """
    if "Judgement:" in response:
        tail = response.split("Judgement:")[-1].strip()
        if "1" in tail:
            return True, 1.0
        if "0" in tail:
            return True, 0.0
        return False, 0.0
    if response == "1":
        return True, 1.0
    if response == "0":
        return True, 0.0
    return False, 0.0


# ---------------------------------------------------------------------------
# compute_score (perception)
# ---------------------------------------------------------------------------


def compute_score(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    is_format_error, reasons, _search_calls = _common_format_check(predict_str)
    no_think = predict_str.split("</think>")[-1].strip()
    answer_text = extract_answer(no_think) or extract_answer(predict_str)

    if not answer_text:
        is_format_error = True
        reasons.append("answer_extract_failed")

    if answer_text and len(answer_text) >= MAX_ANSWER_LEN:
        is_format_error = True
        reasons.append("answer_too_long")
        acc_reward = 0.0
        judge_response = ""
    else:
        if not isinstance(extra_info, dict) or "question" not in extra_info:
            raise ValueError("extra_info with 'question' is required for DeepEyesV2 perception scoring.")
        question = extra_info["question"]
        user_prompt = JUDGE_USER_TEMPLATE.format(question=question, answer=ground_truth, prediction=answer_text)
        judge_response = _judge_chat(
            messages=[
                {"role": "system", "content": COMMON_VERIFY_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        acc_reward = float(_parse_chinese_judge(judge_response))

    format_reward = 0.0 if is_format_error else 1.0
    tool_reward = 1.0 if _has_valid_tool_call(predict_str) and acc_reward >= 0.5 else 0.0
    final_score = 0.6 * acc_reward + 0.2 * format_reward + 0.2 * tool_reward
    return {
        "score": final_score,
        "acc": acc_reward,
        "format": format_reward,
        "tool": tool_reward,
        "judge_response": judge_response,
        "format_error_reason": ",".join(sorted(set(reasons))),
        "predict_str": predict_str,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# compute_score_math (reason)
# ---------------------------------------------------------------------------


# Constants for math answer normalization (verbatim from v2 source)
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """LaTeX/math answer normaliser (verbatim port of v2)."""
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def _math_generative_verify(question: str, ground_truth: str, model_answer: str) -> bool:
    """Use ``MATH_VERIFY_PROMPT`` to ask the judge whether ``model_answer`` is
    equivalent to ``ground_truth``."""
    full_prompt = MATH_VERIFY_PROMPT.format(query=question, gold_ans=ground_truth, pred_ans=model_answer)
    for _ in range(8):
        response = _judge_chat(
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.5,
            retries=1,
        )
        if response == "error":
            continue
        judgement = response.split("## Equivalence Judgement")[-1].lower()
        if "true" in judgement and "false" not in judgement:
            return True
        if "false" in judgement and "true" not in judgement:
            return False
    return False


def compute_score_math(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    is_format_error, reasons, _search_calls = _common_format_check(predict_str)
    no_think = predict_str.split("</think>")[-1].strip()
    answer_text = extract_answer(no_think) or extract_answer(predict_str)

    if not answer_text:
        is_format_error = True
        reasons.append("answer_extract_failed")
        acc_reward = 0.0
    else:
        if not isinstance(extra_info, dict) or "question" not in extra_info:
            raise ValueError("extra_info with 'question' is required for DeepEyesV2 reason scoring.")
        final_answer = normalize_final_answer(answer_text)
        if not final_answer or not ground_truth:
            acc_reward = 0.0
        else:
            acc_reward = 1.0 if _math_generative_verify(extra_info["question"], ground_truth, final_answer) else 0.0

    format_reward = 0.0 if is_format_error else 1.0
    tool_reward = 1.0 if _has_valid_tool_call(predict_str) and acc_reward >= 0.5 else 0.0
    final_score = 0.6 * acc_reward + 0.2 * format_reward + 0.2 * tool_reward
    return {
        "score": final_score,
        "acc": acc_reward,
        "format": format_reward,
        "tool": tool_reward,
        "format_error_reason": ",".join(sorted(set(reasons))),
        "predict_str": predict_str,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# compute_score_search (search)
# ---------------------------------------------------------------------------


def compute_score_search(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    is_format_error, reasons, search_call_count = _common_format_check(predict_str)
    # Search-split-only: <tool_call> opening/closing tag pair must balance.
    # Compares raw counts (any name, including python_exec) — this is a syntax
    # check, not a semantic one. Perception / reason scorers skip it because
    # stray tool_call tags shouldn't penalise non-search splits.
    raw_open = predict_str.count("<tool_call>")
    raw_close = predict_str.count("</tool_call>")
    if raw_open != raw_close:
        is_format_error = True
        reasons.append("tool_call_tag_mismatch")
    search_penalty = 0.1 if search_call_count > 0 else 0.0

    no_think = predict_str.split("</think>")[-1].strip()
    answer_text = extract_answer(no_think) or extract_answer(predict_str)

    if not answer_text:
        is_format_error = True
        reasons.append("answer_extract_failed")
    if answer_text and len(answer_text) >= MAX_ANSWER_LEN:
        is_format_error = True
        reasons.append("answer_too_long")
        acc_reward = 0.0
        judge_response = ""
    else:
        if not isinstance(extra_info, dict) or "question" not in extra_info:
            raise ValueError("extra_info with 'question' is required for DeepEyesV2 search scoring.")
        question = extra_info["question"]
        user_prompt = JUDGE_USER_TEMPLATE.format(question=question, answer=ground_truth, prediction=answer_text or "")
        judge_response = _judge_chat(
            messages=[
                {"role": "system", "content": COMMON_VERIFY_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        acc_reward = float(_parse_chinese_judge(judge_response))

    format_reward = 0.0 if is_format_error else 1.0
    final_score = 0.8 * acc_reward * (1 - search_penalty) + 0.2 * format_reward
    return {
        "score": final_score,
        "acc": acc_reward,
        "format": format_reward,
        "search_penalty": search_penalty,
        "judge_response": judge_response,
        "format_error_reason": ",".join(sorted(set(reasons))),
        "predict_str": predict_str,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# compute_score_acc (vstar-test)
# ---------------------------------------------------------------------------


def compute_score_acc(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    no_think = predict_str.split("</think>")[-1].strip()
    answer_text = extract_answer(no_think) or extract_answer(predict_str)
    if not answer_text:
        return {"score": 0.0, "acc": 0.0}

    if answer_text == ground_truth:
        return {"score": 1.0, "acc": 1.0}
    if answer_text.strip().lower().startswith(str(ground_truth).strip().lower()):
        return {"score": 1.0, "acc": 1.0}

    if not isinstance(extra_info, dict) or "question" not in extra_info:
        raise ValueError("extra_info with 'question' is required for DeepEyesV2 vstar-test scoring.")
    full_prompt = _build_acc_chat_prompt(answer_text, ground_truth, extra_info["question"])

    for _ in range(32):
        response = _judge_chat(
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.5,
            retries=1,
        )
        if response == "error":
            continue
        matched, acc_reward = _parse_english_acc_judge(response)
        if matched:
            return {"score": acc_reward, "acc": acc_reward}
    return {"score": 0.0, "acc": 0.0}


# ---------------------------------------------------------------------------
# Routing entry-point
# ---------------------------------------------------------------------------


# Routing table: ``extra_info["data_source"]`` -> sub-task scorer.
# Keys must match exactly the values stored in the v2 RL dataset:
#   perception_all_*.parquet -> "perception"
#   reason.parquet           -> "reason"
#   search.parquet           -> "search"
#   vstar_test.parquet       -> "vstar-test"  (hyphen, not underscore)
_SCORER_REGISTRY = {
    "perception": compute_score,
    "reason": compute_score_math,
    "search": compute_score_search,
    "vstar-test": compute_score_acc,
}


# ---------------------------------------------------------------------------
# Field resolvers
# ---------------------------------------------------------------------------
#
# The DeepEyesV2 RL dataset (DeepEyesV2_RL_with_datasource) is **100%
# guaranteed** to expose the following keys -- verified by inspecting the
# parquet schemas of all four train/eval files (perception/reason/search/
# vstar_test):
#
#   reward_model = {"ground_truth": str, "style": "rule"}    -> sample.label
#   extra_info   = {"answer", "data_source", "index",        -> sample.metadata
#                   "question", "split", ...}
#
# So these resolvers do NOT fall back through alias keys -- if any required
# field is missing it almost certainly means the dataset was preprocessed
# differently and the run should fail loud rather than silently scoring 0.


def _resolve_ground_truth(sample: Sample) -> str:
    """Extract the ground-truth answer string from ``sample.label``.

    Relax wires ``--label-key reward_model`` so ``sample.label`` is the
    ``reward_model`` struct ``{"ground_truth": str, "style": "rule"}``.
    """
    label = sample.label
    if isinstance(label, dict) and "ground_truth" in label and label["ground_truth"] is not None:
        return str(label["ground_truth"])
    raise ValueError(
        f"DeepEyesV2 reward expects sample.label to be a dict with a "
        f"non-null 'ground_truth' key (from the dataset's reward_model "
        f"column), got: {label!r}"
    )


def _resolve_question(metadata: dict, sample: Sample) -> str:
    """Extract the natural-language question from ``sample.metadata``.

    Relax wires ``--metadata-key extra_info`` so ``sample.metadata`` is the
    ``extra_info`` struct which always contains ``question``.
    """
    if isinstance(metadata, dict) and "question" in metadata:
        return str(metadata["question"] or "")
    raise ValueError(
        f"DeepEyesV2 reward expects sample.metadata to be the extra_info "
        f"dict containing a 'question' field, got: {metadata!r}"
    )


def _resolve_data_source(metadata: dict) -> str:
    """Extract the dataset source tag for routing to the correct scorer.

    The v2 dataset always sets ``extra_info["data_source"]`` to one of
    ``{"perception", "reason", "search", "vstar-test"}`` -- note the hyphen in
    ``vstar-test``. We strip + lowercase to be defensive against any accidental
    capitalisation but keep the hyphen so the registry key matches.
    """
    if isinstance(metadata, dict) and metadata.get("data_source"):
        return str(metadata["data_source"]).strip().lower()
    raise ValueError(
        f"DeepEyesV2 reward expects sample.metadata['data_source'] to be "
        f"one of {{perception, reason, search, vstar-test}}, but extra_info "
        f"is {metadata!r}"
    )


def _compute_one(sample: Sample) -> dict:
    metadata = sample.metadata or {}
    # ``extra_info`` is forwarded verbatim to the scorers (which only
    # actually look up "question") so that future per-task fields like
    # ``category`` (vstar-test) remain accessible.
    extra_info = {
        **metadata,
        "question": _resolve_question(metadata, sample),
    }
    ground_truth = _resolve_ground_truth(sample)
    data_source = _resolve_data_source(metadata)

    scorer = _SCORER_REGISTRY.get(data_source)
    if scorer is None:
        # Unknown data_source means a new task was added without updating
        # the routing table. Fall back to perception (the most general
        # LLM-as-judge) so training keeps running, with a loud warning.
        logger.warning(
            f"[reward_deepeyes_v2] unknown data_source={data_source!r} (expected one of "
            f"{sorted(_SCORER_REGISTRY.keys())}); falling back to compute_score (perception)."
        )
        scorer = compute_score

    try:
        result = scorer(sample.response, ground_truth, extra_info)
    except Exception as exc:
        logger.error(f"[reward_deepeyes_v2] scorer={scorer.__name__} failed on data_source={data_source}: {exc}")
        result = {"score": 0.0, "acc": 0.0, "format": 0.0, "error": str(exc)}

    if not isinstance(result, dict):
        result = {"score": float(result), "acc": float(result)}
    result.setdefault("data_source", data_source)
    return result


async def reward_func(args: Any, sample: Sample, **kwargs) -> dict:
    """Async entry-point invoked by Relax via ``--custom-rm-path``.

    Routes the scoring to the matching v2 sub-task scorer based on
    ``sample.metadata['data_source']``.
    """
    if not isinstance(sample, Sample):
        raise TypeError("`sample` must be an instance of relax.utils.types.Sample.")
    return await asyncio.to_thread(_compute_one, sample)
