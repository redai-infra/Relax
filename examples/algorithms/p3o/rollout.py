# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Controlled behavior-policy sampling for the Task40 mismatch experiment."""

from argparse import Namespace
from typing import Any

from relax.engine.rollout.sglang_rollout import generate as _sglang_generate
from relax.utils.types import Sample


BEHAVIOR_TEMPERATURE = 1.2
BEHAVIOR_TOP_P = 1.0


def behavior_sampling_params(sampling_params: dict[str, Any], *, evaluation: bool) -> dict[str, Any]:
    """Return isolated sampling parameters for Task40 rollout generation."""
    updated = sampling_params.copy()
    if not evaluation:
        updated["temperature"] = BEHAVIOR_TEMPERATURE
        updated["top_p"] = BEHAVIOR_TOP_P
    return updated


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """Generate with behavior-only mismatch while preserving evaluation
    settings."""
    return await _sglang_generate(
        args,
        sample,
        behavior_sampling_params(sampling_params, evaluation=evaluation),
        evaluation=evaluation,
    )
