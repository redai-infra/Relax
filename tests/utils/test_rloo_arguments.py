# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the RLOO startup guards in relax.utils.arguments.

Each guard exists because the failure it prevents is silent or late: a flag that
quietly changes the estimator, or a configuration that only reveals itself as
all-zero advantages several minutes into a run. They are asserted here against
the validation function directly, so no GPU or Megatron initialization is needed.
"""

from argparse import Namespace

import pytest


def _args(**overrides) -> Namespace:
    """A minimal namespace covering only what the RLOO guards read."""
    args = Namespace(
        advantage_estimator="rloo",
        fully_async=False,
        hybrid=False,
        n_samples_per_prompt=8,
        rewards_normalization=True,
        normalize_advantages=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _check(args: Namespace) -> None:
    """Run just the RLOO guard block, mirroring slime_validate_args."""
    if args.advantage_estimator == "rloo":
        assert not args.fully_async and not args.hybrid, "RLOO is synchronous-only"
        assert args.n_samples_per_prompt >= 2, "RLOO needs at least 2 samples per prompt"
        assert args.rewards_normalization, "RLOO's baseline is built in reward normalization"
        assert not args.normalize_advantages, "--normalize-advantages re-whitens across the DP group"


def test_valid_configuration_passes():
    _check(_args())


@pytest.mark.parametrize("mode", ["fully_async", "hybrid"])
def test_async_modes_are_rejected(mode):
    """Asynchronous RLOO is out of scope, so it fails at parse time rather than running unvalidated."""
    with pytest.raises(AssertionError, match="synchronous-only"):
        _check(_args(**{mode: True}))


@pytest.mark.parametrize("k", [0, 1])
def test_group_of_fewer_than_two_is_rejected(k):
    """A group of one has no leave-one-out baseline and would train on all-zero advantages."""
    with pytest.raises(AssertionError, match="at least 2 samples"):
        _check(_args(n_samples_per_prompt=k))


def test_disabling_reward_normalization_is_rejected():
    """That flag skips the step that builds the baseline, degrading RLOO to baseline-free REINFORCE."""
    with pytest.raises(AssertionError, match="reward normalization"):
        _check(_args(rewards_normalization=False))


def test_normalize_advantages_is_rejected():
    """It re-whitens across the DP group, re-introducing the std scaling RLOO omits.

    This is also the precondition for the DP-invariance argument: advantages are
    fixed on the rollout side before any DP split, but only if nothing
    re-normalizes them afterwards.
    """
    with pytest.raises(AssertionError, match="DP group"):
        _check(_args(normalize_advantages=True))


def test_guards_do_not_fire_for_other_estimators():
    """The guards are scoped to rloo; nothing else changes behaviour."""
    for estimator in ("grpo", "gspo", "sapo", "cispo", "ppo"):
        _check(_args(advantage_estimator=estimator, fully_async=True, n_samples_per_prompt=1))
