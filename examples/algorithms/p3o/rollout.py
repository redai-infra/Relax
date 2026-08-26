# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Controlled behavior-policy sampling for the P3O mismatch experiment."""

import math
import os
from argparse import Namespace
from typing import Any

from relax.utils.types import Sample


_P3O_TRUNCATION_SAMPLING_KEYS = (
    "min_p",
    "top_a",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
)


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
        if updated.get("top_p", 1.0) != 1.0 or updated.get("top_k", -1) != -1:
            raise ValueError(
                "P3O behavior sampling requires top_p=1.0 and top_k=-1 so rollout log-probs describe "
                "the untruncated distribution"
            )
        unsupported = sorted(key for key in _P3O_TRUNCATION_SAMPLING_KEYS if key in updated)
        if unsupported:
            raise ValueError(
                "P3O behavior sampling does not support additional distribution-truncation parameters: "
                f"{', '.join(unsupported)}"
            )
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


# _dispatch_generate checks this explicit opt-in before running custom P3O
# generation. The wrapper above rejects known truncation knobs and delegates to
# the built-in SGLang path, which returns sampled-token behavior log-probs.
generate.p3o_behavior_logprob_contract = True
