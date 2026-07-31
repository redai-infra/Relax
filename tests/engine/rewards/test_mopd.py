# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for GSM8K scoring through the MOPD reward."""

from relax.engine.rewards.mopd import get_mopd_reward


GSM8K_SOLUTION = (
    "Brady makes 1500 + 450 = <<1500+450=1950>>1950 dollars.\n"
    "Together they make 1950 + 1500 = <<1950+1500=3450>>3450 dollars.\n"
    "#### 3,450"
)


def test_mopd_gsm8k_accepts_boxed_answer_against_full_solution_label():
    response = "### Final Answer\n\n$$\n\\boxed{3450}\n$$<|im_end|>"

    assert get_mopd_reward(response, GSM8K_SOLUTION, {"data_source": "gsm8k"}) == 1.0


def test_mopd_gsm8k_rejects_wrong_boxed_answer_against_full_solution_label():
    response = "### Final Answer\n\n$$\n\\boxed{3451}\n$$<|im_end|>"

    assert get_mopd_reward(response, GSM8K_SOLUTION, {"data_source": "gsm8k"}) == -1.0


def test_mopd_gsm8k_accepts_dict_ground_truth_with_full_solution_label():
    response = "Reasoning... Answer: 3450"
    label = {"ground_truth": GSM8K_SOLUTION}

    assert get_mopd_reward(response, label, {"data_source": "gsm8k"}) == 1.0
