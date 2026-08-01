# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for RLOO validation through the production argument path."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import relax.utils.arguments as arguments_mod
from tests.utils.test_arguments_opd_teacher_colocate import _opd_args


def _rloo_args(**overrides) -> SimpleNamespace:
    """Return a complete ``slime_validate_args`` input with valid RLOO
    defaults."""
    args = _opd_args()
    defaults = {
        "advantage_estimator": "rloo",
        "n_samples_per_prompt": 8,
        "rollout_batch_size": 16,
        "global_batch_size": 128,
        "num_steps_per_rollout": None,
        "fully_async": False,
        "hybrid": False,
        "rewards_normalization": True,
        "normalize_advantages": False,
        "partial_rollout": False,
        "use_dynamic_global_batch_size": False,
        "grpo_std_normalization": False,
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(args, name, value)
    return args


def _validate(**overrides) -> SimpleNamespace:
    args = _rloo_args(**overrides)
    arguments_mod.slime_validate_args(args)
    return args


def test_rloo_valid_config_passes():
    args = _validate()
    assert args.rollout_batch_size == 16
    assert args.global_batch_size == 128


def test_rloo_derives_rollout_batch_size_before_validation():
    args = _validate(rollout_batch_size=None, global_batch_size=128)
    assert args.rollout_batch_size == 16


def test_rloo_derives_batch_before_rejecting_fully_async():
    with pytest.raises(ValueError, match="synchronous"):
        _validate(rollout_batch_size=None, global_batch_size=128, fully_async=True)


def test_batch_derivation_requires_exact_divisibility():
    with pytest.raises(ValueError, match="must be divisible"):
        _validate(rollout_batch_size=None, global_batch_size=127)


def test_fully_async_non_rloo_can_derive_rollout_batch_size():
    args = _validate(
        advantage_estimator="grpo",
        rollout_batch_size=None,
        global_batch_size=128,
        fully_async=True,
    )
    assert args.rollout_batch_size == 16
    assert args.true_on_policy_mode is True


def test_rloo_requires_n_samples_ge_2():
    with pytest.raises(ValueError, match="n-samples-per-prompt >= 2"):
        _validate(n_samples_per_prompt=1, rollout_batch_size=16, global_batch_size=16)


@pytest.mark.parametrize("mode", ["fully_async", "hybrid"])
def test_rloo_rejects_async_modes(mode):
    with pytest.raises(ValueError, match="synchronous"):
        _validate(**{mode: True})


def test_rloo_requires_rewards_normalization():
    with pytest.raises(ValueError, match="rewards normalization"):
        _validate(rewards_normalization=False)


def test_rloo_rejects_normalize_advantages():
    with pytest.raises(ValueError, match="normalize-advantages"):
        _validate(normalize_advantages=True)


def test_rloo_requires_matching_global_batch_size():
    with pytest.raises(ValueError, match="one optimizer update per rollout"):
        _validate(global_batch_size=64)


def test_rloo_rejects_multiple_steps_per_rollout():
    with pytest.raises(ValueError, match="num-steps-per-rollout 1"):
        _validate(num_steps_per_rollout=2, global_batch_size=64)


@pytest.mark.parametrize("mode", ["partial_rollout", "use_dynamic_global_batch_size"])
def test_rloo_rejects_dynamic_batch_modes(mode):
    with pytest.raises(ValueError, match="effective batch size to drift"):
        _validate(**{mode: True})


def test_non_rloo_path_does_not_apply_rloo_guards():
    args = _validate(
        advantage_estimator="grpo",
        n_samples_per_prompt=1,
        rollout_batch_size=16,
        global_batch_size=16,
        rewards_normalization=False,
        normalize_advantages=True,
        partial_rollout=True,
        use_dynamic_global_batch_size=True,
    )
    assert args.advantage_estimator == "grpo"


def test_advantage_estimator_choices_include_rloo(monkeypatch):
    monkeypatch.setattr(
        arguments_mod,
        "RouterArgs",
        SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser),
    )
    parser = argparse.ArgumentParser()
    arguments_mod.get_slime_extra_args_provider()(parser)

    action = next(action for action in parser._actions if action.dest == "advantage_estimator")
    assert "rloo" in action.choices
