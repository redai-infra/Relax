# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Registration and rollout reward-path tests for P3O."""

import argparse
from types import SimpleNamespace

from relax.core.registry import ALGOS
from relax.utils.arguments import get_slime_extra_args_provider
from relax.utils.types import Sample
from relax.utils.utils import post_process_rewards


def test_p3o_registry_parser_accepts_estimator():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    action = next(action for action in parser._actions if action.dest == "advantage_estimator")

    assert "p3o" in action.choices
    parsed, unknown = parser.parse_known_args(["--advantage-estimator", "p3o"])
    assert parsed.advantage_estimator == "p3o"
    assert unknown == []


def test_p3o_registry_uses_grpo_service_roles():
    assert "p3o" in ALGOS
    assert ALGOS["p3o"].keys() == ALGOS["grpo"].keys()
    for role in ALGOS["grpo"]:
        assert ALGOS["p3o"][role] is ALGOS["grpo"][role]


def _normalized_rewards(estimator: str):
    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        advantage_estimator=estimator,
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
        reward_key=None,
    )
    samples = [
        Sample(group_index=0, reward=1.0),
        Sample(group_index=0, reward=3.0),
        Sample(group_index=1, reward=2.0),
        Sample(group_index=1, reward=6.0),
    ]
    return post_process_rewards(args, samples)


def test_p3o_registry_uses_grpo_group_reward_normalization():
    p3o_raw, p3o_normalized = _normalized_rewards("p3o")
    grpo_raw, grpo_normalized = _normalized_rewards("grpo")

    assert p3o_raw == grpo_raw == [1.0, 3.0, 2.0, 6.0]
    assert p3o_normalized == grpo_normalized
