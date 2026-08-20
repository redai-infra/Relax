# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from examples.graphgpo.reward import reward_func


def test_reward_func_forwards_environment_rewards_in_order() -> None:
    samples = [
        SimpleNamespace(reward=10),
        SimpleNamespace(reward=-0.2),
        SimpleNamespace(reward=0.0),
    ]

    assert asyncio.run(reward_func(None, samples)) == [10.0, -0.2, 0.0]


@pytest.mark.parametrize(
    "samples",
    [
        "not-a-group",
        [],
        [SimpleNamespace()],
        [SimpleNamespace(reward=True)],
        [SimpleNamespace(reward=math.nan)],
        [SimpleNamespace(reward=math.inf)],
        [SimpleNamespace(reward={"score": 1.0})],
    ],
)
def test_reward_func_rejects_invalid_group_rewards(samples: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(reward_func(None, samples))  # type: ignore[arg-type]
