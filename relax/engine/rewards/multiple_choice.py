# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re

from relax.engine.rewards.registry import register_reward

ANS_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def extract_answer(text: str) -> str:
    m = ANS_TAG.search(text)
    return m.group(1).strip() if m else ""


def get_multiple_choice_reward(response, label):
    response = extract_answer(response)
    label = extract_answer(label)
    reward = 1.0 if response == label else 0.0
    return reward


@register_reward("multiple_choice")
def multiple_choice_reward(response, label, metadata=None):
    """Reward function for the ``multiple_choice`` type.

    Registered via ``@register_reward("multiple_choice")`` so it
    participates in the format-aware reward routing system.
    """
    return get_multiple_choice_reward(response, label)
