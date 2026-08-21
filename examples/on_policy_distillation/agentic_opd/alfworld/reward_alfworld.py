# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward function for the ALFWorld agentic OPD recipe.

The ALFWorld outcome is produced by the environment during rollout and carried
back through ``sample.metadata`` (written by ``app/agent.py``). The reward is the
binary task success (``won``), matching the SDAR / verl-agent convention where the
episode-level signal is ``10 * won`` up to a positive scale. We emit the raw
success as ``score`` and add a per-task-type breakdown so the six ALFWorld
categories can be tracked as separate metrics during eval.
"""

from __future__ import annotations

from relax.utils.types import Sample


# Canonical ALFWorld task-type names (from prepare_data.py / AlfredTWEnv).
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


def compute_score(metadata: dict) -> dict:
    success = float(metadata.get("success", 1.0 if metadata.get("won") else 0.0))
    task_type = metadata.get("task_type")

    result = {
        "score": success,
        "acc": success,
        "won": success,
    }
    # Per-category success so eval can report the paper's six ALFWorld tasks.
    if task_type in TASK_TYPES:
        result[f"success/{task_type}"] = success
    return result


async def reward_func(args, sample: Sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return compute_score(metadata)
