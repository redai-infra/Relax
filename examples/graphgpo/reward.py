# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward passthrough for ALFWorld explicit turn rows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any


async def reward_func(
    _args: Any,
    samples: Sequence[Any],
    **_kwargs: Any,
) -> list[float]:
    """Return the environment reward already attached to every turn row.

    Agentic group-RM evaluation always invokes the configured batched reward
    function, even when the managed agent exported a reward.  GraphGPO's
    environment is the reward source, so this adapter validates and forwards
    those values without scoring the model response a second time.
    """

    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    if not samples:
        raise ValueError("samples must not be empty")

    rewards: list[float] = []
    for index, sample in enumerate(samples):
        reward = getattr(sample, "reward", None)
        if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(float(reward)):
            raise ValueError(f"samples[{index}].reward must be a finite real number")
        rewards.append(float(reward))
    return rewards
