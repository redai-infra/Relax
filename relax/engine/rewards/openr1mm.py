# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re

from math_verify import parse, verify


def get_openr1mm_rule_based_reward(response, label):
    # Reject reward-hacking: model repeats </think>/<answer> blocks to inflate
    # match probability. Only a single think/answer pair is allowed.
    if response.count("</think>") > 1 or response.count("<answer>") > 1 or response.count("</answer>") > 1:
        return 0.0

    # -------------------------
    # 1. symbolic verification
    # -------------------------
    try:
        answer = parse(response)
        solution = parse(label)
        score = float(verify(answer, solution))
        if score > 0:
            return 1.0
    except Exception:
        print("parse error", flush=True)
        pass  # same as original: silently fallback

    # -------------------------
    # 2. string-based matching
    # -------------------------
    try:
        # extract ground truth
        sol_match = re.search(r"<answer>(.*?)</answer>", label)
        ground_truth = sol_match.group(1).strip() if sol_match else label.strip()

        # extract model answer
        content_match = re.search(r"<answer>(.*?)</answer>", response)
        student_answer = content_match.group(1).strip() if content_match else response.strip()

        if student_answer == ground_truth:
            return 1.0

    except Exception:
        pass

    return 0.0


def MiniR1Format(response, label):
    try:
        completion = ensure_think_prefix(response)
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer><|im_end|>$"

        m = re.search(regex, completion, re.DOTALL)
        # if the format is not correct, reward is 0
        if m is None or len(m.groups()) != 2:
            return 0.0
        else:
            return 1.0
    except Exception:
        return 0.0


def ensure_think_prefix(s):
    s = s.strip()
    think = "<think>"
    if s[: len(think)] != think:
        return think + s
    return s
