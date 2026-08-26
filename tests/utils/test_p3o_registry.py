# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Registration and rollout reward-path tests for P3O."""

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# `relax.core.registry` eagerly imports `relax.components.advantages`, which imports
# `megatron.core` at module level. CI installs no megatron, so the import runs under
# the shared stub; the registry mapping and reward path under test are pure Python.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backends" / "megatron"))

from _megatron_stub import stubbed_megatron_modules  # noqa: E402


with stubbed_megatron_modules(
    ("megatron", "ray", "tensordict", "transfer_queue", "sglang", "sglang_router", "pybase64")
):
    from relax.core.registry import ALGOS  # noqa: E402
    from relax.utils.arguments import get_slime_extra_args_provider  # noqa: E402
    from relax.utils.types import Sample  # noqa: E402
    from relax.utils.utils import post_process_rewards  # noqa: E402


def test_p3o_registry_parser_accepts_estimator():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    action = next(action for action in parser._actions if action.dest == "advantage_estimator")

    assert "p3o" in action.choices
    parsed, unknown = parser.parse_known_args(["--advantage-estimator", "p3o"])
    assert parsed.advantage_estimator == "p3o"
    assert unknown == []


def test_p3o_registry_parser_exposes_active_plan_defaults_and_modes():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())

    defaults, unknown = parser.parse_known_args([])
    configured, configured_unknown = parser.parse_known_args(
        [
            "--p3o-ess-scope",
            "step",
            "--p3o-kl-mode",
            "proxy_safe",
            "--clip-low",
            "0.1",
            "--clip-high",
            "0.3",
        ]
    )

    assert unknown == configured_unknown == []
    assert defaults.p3o_ess_scope == "micro-batch"
    assert defaults.p3o_kl_mode == "proxy"
    assert defaults.clip_low == defaults.clip_high == 0.2
    assert configured.p3o_ess_scope == "step"
    assert configured.p3o_kl_mode == "proxy_safe"
    assert configured.clip_low == 0.1
    assert configured.clip_high == 0.3


def test_p3o_registry_uses_grpo_service_roles():
    assert "p3o" in ALGOS
    assert ALGOS["p3o"].keys() == ALGOS["grpo"].keys()
    for role in ALGOS["grpo"]:
        assert ALGOS["p3o"][role] is ALGOS["grpo"][role]


def _normalized_rewards(
    estimator: str,
    *,
    rewards: tuple[float, ...] = (1.0, 3.0, 2.0, 6.0),
    n_samples_per_prompt: int = 2,
    grpo_std_normalization: bool = True,
):
    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        advantage_estimator=estimator,
        rewards_normalization=True,
        grpo_std_normalization=grpo_std_normalization,
        n_samples_per_prompt=n_samples_per_prompt,
        reward_key=None,
    )
    samples = [
        Sample(group_index=position // n_samples_per_prompt, reward=reward) for position, reward in enumerate(rewards)
    ]
    return post_process_rewards(args, samples)


def test_p3o_registry_uses_feynrl_sample_std_independent_of_grpo_flag():
    p3o_raw, p3o_normalized = _normalized_rewards("p3o")
    _, p3o_without_grpo_flag = _normalized_rewards("p3o", grpo_std_normalization=False)

    assert p3o_raw == [1.0, 3.0, 2.0, 6.0]
    expected = [-1 / math.sqrt(2), 1 / math.sqrt(2)] * 2
    assert p3o_normalized == pytest.approx(expected, abs=1e-6)
    assert p3o_without_grpo_flag == pytest.approx(expected, abs=1e-6)


def test_p3o_registry_preserves_raw_reward_for_single_sample_groups():
    raw, normalized = _normalized_rewards(
        "p3o",
        rewards=(1.5, -2.0),
        n_samples_per_prompt=1,
    )

    assert raw == normalized == [1.5, -2.0]


def test_grpo_registry_normalization_is_unchanged():
    _, normalized = _normalized_rewards("grpo", grpo_std_normalization=False)

    assert normalized == [-1.0, 1.0, -2.0, 2.0]
