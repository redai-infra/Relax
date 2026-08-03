# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the RLOO startup guards.

Each guard exists because the failure it prevents is silent or late: a flag that
quietly changes the estimator, or a configuration that only reveals itself as
all-zero advantages several minutes into a run.

These call the production validator (`validate_rloo_args`, which
`slime_validate_args` invokes) rather than re-stating the guards, so deleting or
breaking a production guard fails these tests. It is a pure function on `args`,
so no GPU or Megatron initialization is needed.
"""

from argparse import Namespace

import pytest

from relax.utils.training.ppo_utils import validate_rloo_args


def _args(**overrides) -> Namespace:
    """A valid synchronous RLOO namespace, covering what the guards read."""
    args = Namespace(
        advantage_estimator="rloo",
        fully_async=False,
        hybrid=False,
        n_samples_per_prompt=8,
        rewards_normalization=True,
        normalize_advantages=False,
        calculate_per_token_loss=True,
        num_steps_per_rollout=None,
        rollout_batch_size=4,
        global_batch_size=32,
        kl_coef=0.0,
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        custom_convert_samples_to_train_data_path=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_valid_configuration_passes():
    validate_rloo_args(_args())


@pytest.mark.parametrize("mode", ["fully_async", "hybrid"])
def test_async_modes_are_rejected(mode):
    """Asynchronous RLOO is out of scope, so it fails at parse time rather than
    running unvalidated."""
    with pytest.raises(AssertionError, match="synchronous-only"):
        validate_rloo_args(_args(**{mode: True}))


@pytest.mark.parametrize("k", [0, 1])
def test_group_of_fewer_than_two_is_rejected(k):
    """A group of one has no leave-one-out baseline and would train on all-zero
    advantages."""
    with pytest.raises(AssertionError, match="at least 2 samples"):
        validate_rloo_args(_args(n_samples_per_prompt=k))


def test_disabling_reward_normalization_is_rejected():
    """That flag skips the step that builds the baseline, degrading RLOO to
    baseline-free REINFORCE."""
    with pytest.raises(AssertionError, match="reward-normalization step"):
        validate_rloo_args(_args(rewards_normalization=False))


def test_normalize_advantages_is_rejected():
    """It re-whitens across the DP group, re-introducing the std scaling RLOO
    omits.

    This is also the precondition for the DP-invariance argument: advantages
    are fixed on the rollout side before any DP split, but only if nothing re-
    normalizes them afterwards.
    """
    with pytest.raises(AssertionError, match="data-parallel group"):
        validate_rloo_args(_args(normalize_advantages=True))


def test_per_token_loss_is_required():
    """Without the flag the reduction is a per-sample mean, which re-weights
    samples by response length -- a different estimator, and switched silently
    because `uses_completion_level_reduction` keys off the same flag.

    Megatron only forces it when CP > 1, so a CP=1 run would otherwise train the
    wrong objective without any error.
    """
    with pytest.raises(AssertionError, match="requires --calculate-per-token-loss"):
        validate_rloo_args(_args(calculate_per_token_loss=False))


@pytest.mark.parametrize("num_steps_per_rollout", [2, 4])
def test_multiple_steps_per_rollout_are_rejected_explicitly(num_steps_per_rollout):
    """RLOO's loss has no importance ratio, so the second step within a rollout
    trains at updated weights against log-probabilities sampled from the old
    ones."""
    with pytest.raises(AssertionError, match="exactly one optimizer step per rollout"):
        validate_rloo_args(_args(num_steps_per_rollout=num_steps_per_rollout))


def test_one_step_per_rollout_is_accepted():
    validate_rloo_args(_args(num_steps_per_rollout=1))


def test_batch_sizes_implying_more_than_one_step_are_rejected():
    """A batch shape that implies two steps is rejected even without the flag.

    With `--num-steps-per-rollout` unset the step count is implicit:
    `rollout_batch_size * n_samples_per_prompt // global_batch_size`. A GRPO
    configuration that legitimately yields two steps must not silently become a
    two-step RLOO run.
    """
    with pytest.raises(AssertionError, match="exactly one optimizer step per rollout"):
        validate_rloo_args(_args(rollout_batch_size=4, n_samples_per_prompt=8, global_batch_size=16))


def test_batch_sizes_are_not_checked_before_they_are_derived():
    """`rollout_batch_size` / `global_batch_size` may still be None on
    namespaces the validator sees outside `slime_validate_args`; that must not
    raise."""
    validate_rloo_args(_args(rollout_batch_size=None, global_batch_size=None))


@pytest.mark.parametrize("kl_coef", [1e-4, 0.1])
def test_non_zero_kl_coef_is_rejected(kl_coef):
    """`get_grpo_returns` uses only the shape of the reference KL, so a non-
    zero coefficient would pay for the reference forward and change nothing."""
    with pytest.raises(AssertionError, match="has no effect under RLOO"):
        validate_rloo_args(_args(kl_coef=kl_coef))


def test_kl_via_the_loss_term_is_allowed():
    """`--use-kl-loss` is the supported route: `policy_loss_function` adds it
    as a separate term, so it is not silently dropped the way `--kl-coef`
    is."""
    validate_rloo_args(_args(use_kl_loss=True, kl_loss_coef=0.001))


@pytest.mark.parametrize(
    "attribute",
    [
        "custom_reward_post_process_path",
        "agentic_custom_advantage_path",
        "custom_convert_samples_to_train_data_path",
    ],
)
def test_hooks_that_bypass_the_baseline_are_rejected(attribute):
    """Each hook returns from (or replaces) `post_process_rewards` before the
    RLOO branch runs, so the leave-one-out baseline is never built and training
    silently becomes baseline-free REINFORCE."""
    with pytest.raises(AssertionError, match="bypasses RLOO's leave-one-out baseline"):
        validate_rloo_args(_args(**{attribute: "my_module.my_func"}))


def test_guards_do_not_fire_for_other_estimators():
    """The guards are scoped to rloo; nothing else changes behaviour."""
    for estimator in ("grpo", "gspo", "sapo", "cispo", "ppo"):
        validate_rloo_args(
            _args(
                advantage_estimator=estimator,
                fully_async=True,
                n_samples_per_prompt=1,
                custom_reward_post_process_path="my_module.my_func",
                calculate_per_token_loss=False,
                kl_coef=0.1,
                num_steps_per_rollout=2,
            )
        )


def test_missing_estimator_attribute_is_tolerated():
    """The validator runs on namespaces built by other code paths (SFT, tests),
    so an absent `advantage_estimator` must be a no-op rather than an
    AttributeError."""
    validate_rloo_args(Namespace())
