# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-component reward for the GDPO example: correctness and format.

GDPO standardizes each component within its prompt group before combining them.
Two things survive that GRPO's single standardization of the summed reward
destroys:

* **Relative strength between groups.** Standardizing per group forces every
  group to unit variance, so a group where only `correctness` varies and one
  where both components vary come out identical. Standardizing each component
  first makes the latter twice the amplitude, and step 3 whitens across the
  *batch*, so the difference reaches the final advantage.
* **Scale disparity between components.** A `correctness` in {0, 1} added to a
  reward in the hundreds gives a sum whose variance is essentially the large
  component's, so GRPO's direction is decided by it alone.

What GDPO does *not* do: rescue a group whose components sum to a constant.
There `format = C - correctness` forces the standardized values to be exact
opposites, and equal weights cancel them to zero -- the same answer GRPO
gives. Only unequal weights break that tie. (If every component is constant
within the group, GDPO returns zero too.)

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
    without the expected tags, and well-formatted while wrong.

    ``score`` is ``correctness`` rather than the sum, on purpose. It feeds
    ``--reward-key``, which selects the scalar for metrics and the
    ``raw_reward`` column only -- it does not participate in the GDPO
    computation, which reads the two components directly. Reporting accuracy
    there is more legible on a dashboard than a blended number. Note this means
    ``rollout/raw_reward`` tracks correctness alone, not overall reward.
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
