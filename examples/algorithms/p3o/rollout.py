# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Controlled behavior-policy sampling for the P3O mismatch experiment."""

import math
import os
from argparse import Namespace
from typing import Any

from relax.utils.types import Sample


async def _sglang_generate(*args: Any, **kwargs: Any) -> Sample:
    """Import the heavyweight rollout backend only when generation starts."""
    from relax.engine.rollout.sglang_rollout import generate

    return await generate(*args, **kwargs)


def _behavior_temperature() -> float:
    raw_value = os.environ.get("P3O_BEHAVIOR_TEMPERATURE")
    if raw_value is None:
        raise ValueError("P3O_BEHAVIOR_TEMPERATURE must be set when temperature override is enabled")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("P3O_BEHAVIOR_TEMPERATURE must be numeric") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("P3O_BEHAVIOR_TEMPERATURE must be finite and greater than zero")
    return value


def behavior_sampling_params(sampling_params: dict[str, Any], *, evaluation: bool) -> dict[str, Any]:
    """Return isolated sampling parameters for P3O rollout generation."""
    updated = sampling_params.copy()
    if not evaluation:
        updated["temperature"] = _behavior_temperature()
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
