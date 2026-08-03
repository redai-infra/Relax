# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.utils.training.ppo_utils import (
    compute_rloo_baselines_and_advantages,
)
from relax.utils.training.ppo_utils import (
    compute_rloo_loss as _compiled_compute_rloo_loss,
)


compute_rloo_loss = torch.compiler.disable(_compiled_compute_rloo_loss)


def test_rloo_baseline_and_advantage_match_reference_with_interleaved_groups():
    rewards = torch.tensor([1.0, -1.0, 0.0, 2.0, 0.5, 1.0])
    group_indices = [10, 20, 10, 20, 10, 20]

    baselines, advantages = compute_rloo_baselines_and_advantages(rewards, group_indices, group_size=3)

    expected_baselines = torch.tensor([0.25, 1.5, 0.75, 0.0, 0.5, 0.5])
    expected_advantages = rewards - expected_baselines
    assert torch.allclose(baselines, expected_baselines, atol=1e-6)
    assert torch.allclose(advantages, expected_advantages, atol=1e-6)


def test_rloo_tensor_group_indices_preserve_gathered_sample_order():
    rewards = torch.tensor([1.0, -1.0, 0.0, 2.0, 0.5, 1.0])
    group_indices = torch.tensor([10, 20, 10, 20, 10, 20])

    baselines, advantages = compute_rloo_baselines_and_advantages(rewards, group_indices, group_size=3)

    expected_baselines = torch.tensor([0.25, 1.5, 0.75, 0.0, 0.5, 0.5])
    assert torch.allclose(baselines, expected_baselines, atol=1e-6)
    assert torch.allclose(advantages, rewards - expected_baselines, atol=1e-6)


def test_rloo_group_size_two_and_float64_are_exact():
    rewards = torch.tensor([3.0, -2.0], dtype=torch.float64)

    baselines, advantages = compute_rloo_baselines_and_advantages(rewards, [0, 0], group_size=2)

    assert baselines.dtype == torch.float64
    assert torch.equal(baselines, torch.tensor([-2.0, 3.0], dtype=torch.float64))
    assert torch.equal(advantages, torch.tensor([5.0, -5.0], dtype=torch.float64))


@pytest.mark.parametrize("group_size", [0, 1])
def test_rloo_rejects_group_size_below_two(group_size):
    with pytest.raises(ValueError, match="group_size >= 2"):
        compute_rloo_baselines_and_advantages(torch.tensor([1.0]), [0], group_size)


@pytest.mark.parametrize(
    ("rewards", "group_indices", "message"),
    [
        (torch.tensor([1.0, 0.0]), [None, 0], "group_index"),
        (torch.tensor([1.0, 0.0, 0.5]), [0, 0, 1], "group 1 has 1 samples"),
        (torch.tensor([1.0, float("nan")]), [0, 0], "finite"),
        (torch.tensor([1.0, float("inf")]), [0, 0], "finite"),
        (torch.tensor([1.0, -float("inf")]), [0, 0], "finite"),
    ],
)
def test_rloo_rejects_invalid_groups_and_rewards(rewards, group_indices, message):
    with pytest.raises(ValueError, match=message):
        compute_rloo_baselines_and_advantages(rewards, group_indices, group_size=2)


def test_rloo_sequence_loss_matches_masked_reference_and_empty_response():
    log_probs = [
        torch.tensor([-0.2, -0.3, -0.4]),
        torch.tensor([], dtype=torch.float32),
        torch.tensor([-0.5, -0.1]),
    ]
    masks = [
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([], dtype=torch.float32),
        torch.tensor([1.0, 1.0]),
    ]
    advantages = torch.tensor([0.75, -0.75, 0.0])
    sequence_log_probs = torch.stack([(values * mask).sum() for values, mask in zip(log_probs, masks, strict=True)])

    actual = compute_rloo_loss(sequence_log_probs, advantages)

    expected = torch.tensor([0.45, 0.0, 0.0])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_rloo_precomputed_advantages_are_dp_partition_invariant():
    rewards = torch.tensor([1.0, 0.0, 0.5, -1.0, 2.0, 1.0])
    groups = [0, 0, 0, 1, 1, 1]
    baselines, advantages = compute_rloo_baselines_and_advantages(rewards, groups, group_size=3)

    dp0 = (baselines[:3], advantages[:3])
    dp1 = (baselines[3:], advantages[3:])

    assert torch.equal(torch.cat([dp0[0], dp1[0]]), baselines)
    assert torch.equal(torch.cat([dp0[1], dp1[1]]), advantages)


def test_rloo_cp_split_preserves_sequence_loss_and_gradient():
    full_log_probs = torch.tensor([-0.2, -0.3, -0.4, -0.1], requires_grad=True)
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    advantage = torch.tensor([0.75])

    full_loss = compute_rloo_loss((full_log_probs * mask).sum().reshape(1), advantage).sum()
    full_loss.backward()
    expected_grad = full_log_probs.grad.clone()

    cp_log_probs = full_log_probs.detach().clone().requires_grad_(True)
    shard_sum = (cp_log_probs[:2] * mask[:2]).sum() + (cp_log_probs[2:] * mask[2:]).sum()
    cp_loss = compute_rloo_loss(shard_sum.reshape(1), advantage).sum()
    cp_loss.backward()

    assert torch.allclose(cp_loss, full_loss.detach(), atol=1e-6)
    assert torch.allclose(cp_log_probs.grad, expected_grad, atol=1e-6)
