# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the RLOO helpers in relax.utils.training.ppo_utils.

RLOO replaces GRPO's whole-group baseline with a leave-one-out baseline and drops
the standard-deviation scaling. Everything under test is pure CPU torch, so it is
imported at module scope and needs no GPU or Megatron.

``get_rloo_advantages`` computes the baseline directly from raw rewards, while
``scale_centered_rewards_for_rloo`` rescales already-centered rewards. They are
algebraically identical, which the first test exploits as a cross-check. The last
three tests carry the chain further, through ``compute_policy_loss``, to pin the
resulting ``pg_loss`` against hand-computed references.
"""

import math

import pytest
import torch

from relax.utils.training.ppo_utils import (
    compute_policy_loss,
    get_rloo_advantages,
    scale_centered_rewards_for_rloo,
)


def _centered(rewards: torch.Tensor) -> torch.Tensor:
    return rewards - rewards.mean()


@pytest.mark.parametrize("k", [2, 3, 5, 8, 16])
def test_two_formulations_agree(k):
    """A_i = R_i - mean(R_j, j!=i) equals k/(k-1) * (R_i - mean(R)) for every group size."""
    torch.manual_seed(0)
    rewards = torch.randn(k, dtype=torch.float64)
    direct = get_rloo_advantages(rewards)
    rescaled = scale_centered_rewards_for_rloo(_centered(rewards))
    assert torch.allclose(direct, rescaled, atol=1e-12), (
        f"k={k}: leave-one-out form {direct.tolist()} != rescaled form {rescaled.tolist()}"
    )


def test_matches_hand_computed_reference():
    """k=8 with three correct answers matches the reference values."""
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    advantages = get_rloo_advantages(rewards)
    # A correct sample is compared against the other seven: mean([1,1,0,0,0,0,0]) = 2/7.
    expected_correct = 1.0 - 2.0 / 7.0
    # A wrong sample is compared against mean([1,1,1,0,0,0,0]) = 3/7.
    expected_wrong = 0.0 - 3.0 / 7.0
    expected = torch.tensor([expected_correct] * 3 + [expected_wrong] * 5, dtype=torch.float64)
    assert torch.allclose(advantages, expected, atol=1e-12), f"expected {expected.tolist()}, got {advantages.tolist()}"
    assert pytest.approx(expected_correct, abs=1e-4) == 0.7143, "correct-sample advantage should be ~+0.7143"
    assert pytest.approx(expected_wrong, abs=1e-4) == -0.4286, "wrong-sample advantage should be ~-0.4286"


@pytest.mark.parametrize("k", [2, 4, 8])
def test_group_sums_to_zero(k):
    """Leave-one-out advantages are centered, so each group sums to zero."""
    torch.manual_seed(1)
    rewards = torch.randn(k, dtype=torch.float64)
    assert abs(float(get_rloo_advantages(rewards).sum())) < 1e-12, "RLOO advantages must sum to zero per group"


def test_single_sample_group_is_zero_not_nan():
    """k=1 has no baseline; both helpers return zero, not a division."""
    rewards = torch.tensor([3.5], dtype=torch.float64)
    for name, out in (
        ("get_rloo_advantages", get_rloo_advantages(rewards)),
        ("scale_centered_rewards_for_rloo", scale_centered_rewards_for_rloo(_centered(rewards))),
    ):
        assert out.shape == rewards.shape, f"{name} must preserve shape for k=1"
        assert torch.all(torch.isfinite(out)), f"{name} must not produce NaN/Inf for k=1"
        assert float(out.abs().sum()) == 0.0, f"{name} must return exactly zero for k=1"


def test_zero_variance_group_is_zero():
    """All-equal rewards carry no signal, so every advantage is zero."""
    for value in (0.0, 1.0, -2.5):
        rewards = torch.full((8,), value, dtype=torch.float64)
        advantages = get_rloo_advantages(rewards)
        assert torch.all(torch.isfinite(advantages)), f"reward={value} produced non-finite advantages"
        assert float(advantages.abs().max()) < 1e-12, f"reward={value} must give all-zero advantages"


def test_no_std_normalization_unlike_grpo():
    """RLOO must not divide by the group std, only rescale by k/(k-1)."""
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    centered = _centered(rewards)
    rloo = get_rloo_advantages(rewards)
    grpo = centered / (rewards.std() + 1e-6)  # mirrors relax/utils/utils.py group norm

    ratio = rloo / centered
    assert torch.allclose(ratio, torch.full_like(ratio, 8 / 7), atol=1e-12), (
        "RLOO/centered ratio must be exactly k/(k-1) = 8/7"
    )
    assert not torch.allclose(rloo, grpo, atol=1e-3), "RLOO must differ from std-normalized GRPO advantages"


def test_dtype_and_device_preserved():
    """Helpers keep the caller's dtype, so float32 callers are unaffected."""
    for dtype in (torch.float32, torch.float64):
        rewards = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=dtype)
        assert get_rloo_advantages(rewards).dtype == dtype, f"dtype {dtype} not preserved"
        assert scale_centered_rewards_for_rloo(_centered(rewards)).dtype == dtype, f"dtype {dtype} not preserved"


def test_integer_rewards_promote_instead_of_truncating():
    """Integer rewards must promote to float, not floor the division."""
    rewards = torch.tensor([1, 0, 0, 0], dtype=torch.int64)
    advantages = get_rloo_advantages(rewards)
    assert advantages.is_floating_point(), "integer rewards must promote to a float dtype, not truncate"
    expected = torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3], dtype=torch.float64)
    assert torch.allclose(advantages.double(), expected, atol=1e-6), (
        f"expected {expected.tolist()}, got {advantages.tolist()}"
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_reward_contaminates_only_its_own_group(bad):
    """A non-finite reward propagates across its whole group.

    Documented as a known limitation rather than silently masked: the baseline
    is a group-wide sum, so one bad sample poisons its group. Callers must
    filter upstream; zeroing it here would hide a broken reward function.
    """
    rewards = torch.tensor([1.0, 0.0, bad, 0.0], dtype=torch.float64)
    advantages = get_rloo_advantages(rewards)
    assert not torch.all(torch.isfinite(advantages)), (
        "a non-finite reward must remain visible in the advantages, not be silently zeroed"
    )
    clean = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.all(torch.isfinite(get_rloo_advantages(clean))), "a clean group must stay finite"


@pytest.mark.parametrize("shape", [(2, 4), (8, 1), (1, 1, 8)])
def test_non_1d_input_raises_instead_of_silently_using_the_wrong_group_size(shape):
    """A 2-D batch must fail loudly, not be reduced as one oversized group.

    Before the guard, ``[[1, 0], [0, 1]]`` returned +-0.667 -- the answer for a
    single group of four -- where the correct per-row answer is +-1.0. Nothing
    raised, so the caller had no way to notice.
    """
    values = torch.zeros(shape, dtype=torch.float64)
    for name, fn in (
        ("get_rloo_advantages", get_rloo_advantages),
        ("scale_centered_rewards_for_rloo", scale_centered_rewards_for_rloo),
    ):
        with pytest.raises(ValueError, match="1-D"):
            fn(values)
        assert True, name  # keep the name in the failure output


def test_1d_group_of_two_still_works_after_the_shape_guard():
    """Sanity check that the guard does not reject the smallest valid group."""
    advantages = get_rloo_advantages(torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(advantages, torch.tensor([1.0, -1.0], dtype=torch.float64), atol=1e-12)


def test_policy_loss_from_rloo_advantages_at_ratio_one():
    """reward -> advantage -> pg_loss matches hand-computed values at ratio 1.

    With one correct answer out of four, RLOO gives A = [1, -1/3, -1/3, -1/3].
    At ``ppo_kl = 0`` the ratio is exactly 1, so ``pg_loss = -A`` element-wise and
    nothing is clipped.
    """
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    advantages = get_rloo_advantages(rewards)
    assert torch.allclose(advantages, torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3], dtype=torch.float64), atol=1e-12)

    ppo_kl = torch.zeros_like(advantages)
    pg_losses, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip=0.2, eps_clip_high=0.2)

    expected = torch.tensor([-1.0, 1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
    assert torch.allclose(pg_losses, expected, atol=1e-12), f"expected {expected.tolist()}, got {pg_losses.tolist()}"
    assert float(clipfrac.sum()) == 0.0, "an unchanged policy must not clip any token"


def test_policy_loss_from_rloo_advantages_clips_positive_side_only():
    """At ratio 1.5 the positive-advantage token clips; negative ones do not.

    Reference, with ``eps_clip = eps_clip_high = 0.2`` so the ratio clamps to 1.2:

    * ``A = +1``:   ``max(-1.5 * 1, -1.2 * 1) = -1.2``          -> clipped
    * ``A = -1/3``: ``max(-1.5 * -1/3, -1.2 * -1/3) = 0.5``     -> unclipped
    """
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    advantages = get_rloo_advantages(rewards)
    ppo_kl = torch.full_like(advantages, -math.log(1.5))  # ratio = exp(-ppo_kl) = 1.5

    pg_losses, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip=0.2, eps_clip_high=0.2)

    expected = torch.tensor([-1.2, 0.5, 0.5, 0.5], dtype=torch.float64)
    assert torch.allclose(pg_losses, expected, atol=1e-12), f"expected {expected.tolist()}, got {pg_losses.tolist()}"
    assert clipfrac.tolist() == [1.0, 0.0, 0.0, 0.0], (
        f"only the positive-advantage token should clip, got {clipfrac.tolist()}"
    )


def test_policy_loss_rloo_and_grpo_differ_only_by_advantage_scale():
    """At ratio 1 the RLOO/GRPO pg_loss ratio equals their advantage ratio.

    Confirms RLOO changes the loss only through the advantage, leaving the
    clipped objective itself untouched.
    """
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    centered = _centered(rewards)
    rloo_adv = get_rloo_advantages(rewards)
    grpo_adv = centered / (rewards.std() + 1e-6)

    ppo_kl = torch.zeros_like(rloo_adv)
    rloo_loss, _ = compute_policy_loss(ppo_kl, rloo_adv, eps_clip=0.2, eps_clip_high=0.2)
    grpo_loss, _ = compute_policy_loss(ppo_kl, grpo_adv, eps_clip=0.2, eps_clip_high=0.2)

    assert torch.allclose(rloo_loss / grpo_loss, rloo_adv / grpo_adv, atol=1e-12), (
        "the loss ratio must track the advantage ratio exactly"
    )
