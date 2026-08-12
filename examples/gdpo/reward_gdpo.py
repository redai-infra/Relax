# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-component reward for the GDPO example: correctness and format.

GDPO standardizes each component within its prompt group before combining them,
so a group whose rollouts differ in their reward *components* but share the same
*summed* reward still carries a learning signal. Example: (correct, badly
formatted) and (wrong, well formatted) both sum to 1 -- GRPO sees one constant
summed reward and the whole group contributes nothing, while GDPO keeps each
component's signal. (If every component is constant within the group, GDPO
returns zero too.)

Wire it up with::

    --advantage-estimator gdpo
    --gdpo-reward-keys correctness format
    --custom-rm-path examples.gdpo.reward_gdpo.reward_func
    --reward-key score

``score`` is what ``--reward-key`` selects for metrics and for the ``raw_reward``
column; the training signal comes from the two components.
"""

import re
from typing import Any


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_answer(response: str) -> str | None:
    match = _ANSWER_RE.search(response)
    return match.group(1).strip() if match else None


def _final_answer(label: Any) -> str:
    """Normalise a GSM8K label to just the final answer.

    GSM8K ships ``answer`` as the full worked solution ending in ``#### 36``, so
    comparing a model's ``<answer>36</answer>`` against the whole string makes
    ``correctness`` zero for every rollout. That collapses the component in every
    group, and GDPO silently degrades to the single ``format`` reward -- which is
    the one thing this example exists to demonstrate it does not do.

    Handling it here rather than only in a data-prep step keeps the example
    working against the dataset it names. Labels without the marker pass through
    unchanged, so a pre-cleaned dataset behaves identically.
    """
    text = str(label).strip()
    return text.rsplit("####", 1)[-1].strip() if "####" in text else text


def compute_gdpo_reward(response: str, label: Any) -> dict[str, float]:
    """Score one response on answer correctness and on output format.

    The two components are deliberately decorrelated: a response can be correct
    without the expected tags, and well-formatted while wrong. That is the
    situation GDPO handles better than a summed reward.
    """
    answer = _extract_answer(response)
    correctness = 1.0 if answer is not None and answer == _final_answer(label) else 0.0

    has_think = _THINK_RE.search(response) is not None
    format_score = 0.5 * float(has_think) + 0.5 * float(answer is not None)

    return {
        "score": correctness,
        "correctness": correctness,
        "format": format_score,
    }


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> dict[str, float]:
    """Entry point for ``--custom-rm-path``."""
    return compute_gdpo_reward(sample.response, sample.label)
