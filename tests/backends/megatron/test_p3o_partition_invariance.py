# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Logical partition-invariance tests for optimizer-step P3O."""

import math

import pytest
import torch

from relax.utils.training.p3o_utils import (
    P3OSufficientStats,
    compute_p3o_sufficient_stats,
    compute_p3o_token_terms,
    finalize_p3o_step_context,
)


TOKEN_COUNT = 11
INDICES = torch.arange(TOKEN_COUNT)
BEHAVIOR_LOG_PROBS = torch.full((TOKEN_COUNT,), -2.0)
LOG_RATIOS = torch.tensor([math.log(value) for value in (1.0, 2.0, 0.5, 4.0, 0.8, 1.4, 0.25, 3.0, 1.1, 0.6, 2.5)])
ADVANTAGES = torch.tensor([1.0, -1.0, 0.5, 2.0, -0.2, 0.7, -1.5, 1.2, 0.4, -0.8, 1.8])
VALID_MASK = torch.tensor([True, True, False, True, True, False, True, True, True, False, True])


def _evaluate(shards: list[torch.Tensor]):
    log_probs = (BEHAVIOR_LOG_PROBS + LOG_RATIOS).clone().requires_grad_(True)
    stats = P3OSufficientStats.zeros()
    for shard in shards:
        if shard.numel() == 0:
            stats = stats + P3OSufficientStats.zeros()
            continue
        stats = stats + compute_p3o_sufficient_stats(
            log_probs[shard],
            BEHAVIOR_LOG_PROBS[shard],
            VALID_MASK[shard],
        )
    context = finalize_p3o_step_context(stats)

    total = 0.0 * log_probs.sum()
    for shard in shards:
        if shard.numel() == 0:
            continue
        terms = compute_p3o_token_terms(
            log_probs[shard],
            BEHAVIOR_LOG_PROBS[shard],
            ADVANTAGES[shard],
            VALID_MASK[shard],
            context,
        )
        total = total + terms.score_loss.sum() + terms.adaptive_kl_loss.sum()
    loss = total / context.valid_token_count
    loss.backward()
    return context, loss.detach(), log_probs.grad.detach()


def _assert_matches_oracle(shards: list[torch.Tensor]):
    expected_context, expected_loss, expected_grad = _evaluate([INDICES])
    actual_context, actual_loss, actual_grad = _evaluate(shards)

    torch.testing.assert_close(actual_context.normalized_ess, expected_context.normalized_ess)
    torch.testing.assert_close(actual_context.adaptive_cap, expected_context.adaptive_cap)
    torch.testing.assert_close(actual_context.ratio_mean, expected_context.ratio_mean)
    torch.testing.assert_close(actual_context.ratio_std, expected_context.ratio_std)
    torch.testing.assert_close(actual_context.valid_token_count, expected_context.valid_token_count)
    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
def test_p3o_partition_invariance_fixed_micro_batches(micro_batch_size):
    shards = list(torch.split(INDICES, micro_batch_size))
    _assert_matches_oracle(shards)


def test_p3o_partition_invariance_ragged_and_dummy_micro_batches():
    shards = [
        INDICES[0:3],
        torch.empty(0, dtype=torch.long),
        INDICES[3:4],
        INDICES[4:9],
        torch.empty(0, dtype=torch.long),
        INDICES[9:],
    ]
    _assert_matches_oracle(shards)


@pytest.mark.parametrize("data_parallel_size", [1, 2, 4])
def test_p3o_partition_invariance_logical_data_parallel_shards(data_parallel_size):
    _assert_matches_oracle(list(torch.tensor_split(INDICES, data_parallel_size)))


def test_p3o_partition_invariance_static_dp2_cp2_zigzag_shards():
    shards = [
        torch.tensor([0, 7, 8]),
        torch.tensor([1, 6, 9]),
        torch.tensor([2, 5, 10]),
        torch.tensor([3, 4]),
    ]
    _assert_matches_oracle(shards)


def test_p3o_partition_invariance_dynamic_cp_and_zero_local_tokens():
    shards = [
        torch.tensor([0, 1, 6]),
        torch.tensor([2, 5, 7, 9]),
        torch.tensor([3, 4, 8, 10]),
        torch.empty(0, dtype=torch.long),
    ]
    _assert_matches_oracle(shards)
