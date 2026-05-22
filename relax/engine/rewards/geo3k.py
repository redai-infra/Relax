# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re

from mathruler.grader import extract_boxed_content, grade_answer


def _extract_ground_truth(label):
    if isinstance(label, dict):
        gt = label.get("ground_truth") or label.get("answer", "")
        return str(gt)
    return str(label)


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )


def get_geo3k_reward(response, label):
    ground_truth = _extract_ground_truth(label)
    if not ground_truth:
        return 0.0
    return compute_score(response, ground_truth)
