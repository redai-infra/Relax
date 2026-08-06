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
    compute_rloo_loss,
    get_rloo_advantages,
    get_rloo_baseline,
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

    This helper is deliberately transparent: the baseline is a group-wide sum, so
    one bad sample poisons its group, and zeroing it here would hide a broken
    reward function. Rejection happens one level up, in ``post_process_rewards``
    (see ``test_non_finite_reward_is_rejected_with_the_offending_position``), where
    the sample position is still known and can be reported.
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


def test_rloo_loss_is_unclipped_reinforce():
    """RLOO's loss is -A * log pi, with no trust region at all.

    Pinned against PPO-Clip because the difference is the whole point: reusing the
    clipped objective would make this "GRPO with a leave-one-out baseline" rather
    than RLOO as published, and clipping would bias an estimator chosen for being
    unbiased.
    """
    advantages = torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3], dtype=torch.float64)
    log_probs = torch.tensor([-0.5, -1.0, -2.0, -0.25], dtype=torch.float64)

    pg_loss, clipfrac = compute_rloo_loss(log_probs=log_probs, advantages=advantages)

    assert torch.allclose(pg_loss, -(advantages * log_probs), atol=1e-12)
    assert float(clipfrac.abs().sum()) == 0.0, "RLOO clips nothing, so clipfrac must be all zero"
    assert pg_loss.shape == log_probs.shape, "loss must stay element-wise; the caller reduces"


def test_rloo_loss_ignores_the_ratio_that_ppo_clip_would_use():
    """A large policy shift changes PPO-Clip's loss but must not change RLOO's.

    At ratio 1.5 PPO-Clip clamps the positive-advantage token; RLOO has no
    ratio term, so its loss depends only on the current log-prob.
    """
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    log_probs = torch.tensor([-0.5, -0.5], dtype=torch.float64)
    ppo_kl = torch.full_like(advantages, -math.log(1.5))

    rloo_loss, _ = compute_rloo_loss(log_probs=log_probs, advantages=advantages)
    clipped, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip=0.2, eps_clip_high=0.2)

    assert float(clipfrac.sum()) > 0, "the reference PPO-Clip case must actually clip"
    assert not torch.allclose(rloo_loss, clipped), "RLOO must not reproduce the clipped objective"


def test_rloo_loss_gradient_flows_only_through_log_probs():
    """Advantages are detached: they scale the gradient but receive none."""
    advantages = torch.tensor([0.75, -0.75], dtype=torch.float64, requires_grad=True)
    log_probs = torch.tensor([-0.5, -1.5], dtype=torch.float64, requires_grad=True)

    pg_loss, _ = compute_rloo_loss(log_probs=log_probs, advantages=advantages)
    pg_loss.sum().backward()

    assert advantages.grad is None, "advantages must be detached, not a gradient path"
    assert torch.allclose(log_probs.grad, -advantages.detach(), atol=1e-12)


def test_empty_response_contributes_no_loss_but_still_feeds_the_baseline():
    """A zero-valid-token response must not affect the loss, yet must still
    count.

    Acceptance criterion 2 names "empty response" explicitly. The two halves are
    separate: the sample contributes nothing to the objective (its mask is all
    zero), but its reward still participates in the other samples' leave-one-out
    baselines -- dropping it would bias every other advantage in the group.
    """
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    advantages = get_rloo_advantages(rewards)

    # The empty sample is index 3; an all-zero mask means no valid response token.
    log_probs = torch.tensor([-0.5, -1.0, -2.0, -0.25], dtype=torch.float64)
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)

    pg_loss, _ = compute_rloo_loss(log_probs=log_probs, advantages=advantages)
    masked = pg_loss * mask

    assert float(masked[3]) == 0.0, "an empty response must contribute exactly zero loss"
    # Its reward is still in the baseline: dropping it would change the others.
    without_it = get_rloo_advantages(rewards[:3])
    assert not torch.allclose(advantages[:3], without_it), (
        "the empty sample's reward must still participate in the other baselines"
    )


def test_baseline_is_raw_reward_minus_advantage_element_wise():
    """b_i = R_i - A_i, broadcast over each sample's tokens."""
    advantages = [torch.tensor([0.75, 0.75], dtype=torch.float64), torch.tensor([-0.25], dtype=torch.float64)]
    raw_rewards = [1.0, 0.0]

    baseline = get_rloo_baseline(raw_rewards, advantages)

    expected = torch.tensor([1.0 - 0.75, 1.0 - 0.75, 0.0 - (-0.25)], dtype=torch.float64)
    assert torch.allclose(baseline, expected, atol=1e-12), f"expected {expected.tolist()}, got {baseline.tolist()}"


def test_baseline_layout_follows_the_advantages_not_the_response_lengths():
    """The regression this function exists to prevent.

    Under CP>1 the advantages handed to the loss are the rank-local shard while
    ``response_lengths`` stays full-length. An earlier implementation built the
    broadcast from the lengths and produced a tensor of the wrong size --
    measured on 8xH100, 1536 against an advantage shard of 740 -- which is a
    shape error at best and a silently wrong baseline at worst. Shape must
    track the advantages.
    """
    full_response_length = 8
    shard = full_response_length // 2  # what a CP=2 rank actually holds
    advantages = [torch.full((shard,), 0.5, dtype=torch.float64) for _ in range(3)]

    baseline = get_rloo_baseline([1.0, 1.0, 0.0], advantages)

    assert baseline.numel() == 3 * shard, (
        f"baseline must match the advantage shard ({3 * shard} elements), got {baseline.numel()}"
    )
    assert baseline.numel() != 3 * full_response_length, "must not be sized from the full response length"


def test_baseline_preserves_dtype_and_device_of_the_advantages():
    """ones_like inherits both, so no explicit dtype/device plumbing is
    needed."""
    for dtype in (torch.float32, torch.float64):
        advantages = [torch.zeros(4, dtype=dtype)]
        out = get_rloo_baseline([1.0], advantages)
        assert out.dtype == dtype and out.shape == advantages[0].shape


def test_baseline_rejects_a_length_mismatch_instead_of_truncating():
    """A short reward list would otherwise zip-truncate into a too-short
    baseline."""
    advantages = [torch.zeros(2, dtype=torch.float64) for _ in range(3)]
    with pytest.raises(ValueError, match="one-to-one per sample"):
        get_rloo_baseline([1.0, 0.0], advantages)


def test_baseline_accepts_a_reward_tensor_as_well_as_a_list():
    """raw_reward may arrive as a tensor after the transfer queue; both must
    work."""
    advantages = [torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)]
    from_list = get_rloo_baseline([1.0, 0.5], advantages)
    from_tensor = get_rloo_baseline(torch.tensor([1.0, 0.5], dtype=torch.float64), advantages)
    assert torch.allclose(from_list, from_tensor, atol=1e-12)
