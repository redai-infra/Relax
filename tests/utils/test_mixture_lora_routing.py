# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math

import pytest
import torch

from relax.utils.mixture_lora import RoutingDecision, compute_routing_statistics, route_topk


def _normalized_entropy(values):
    if len(values) <= 1:
        return 0.0
    return -sum(value * math.log(value) for value in values if value > 0) / math.log(len(values))


def test_route_topk_returns_fp32_normalized_weights():
    logits = torch.tensor([[4.0, 1.0, 3.0, 2.0], [0.0, 5.0, 2.0, 1.0]], dtype=torch.bfloat16)

    decision = route_topk(logits, top_k=2, temperature=1.0)

    assert decision.pre_topk_probs.shape == (2, 4)
    assert decision.topk_indices.shape == (2, 2)
    assert decision.post_topk_weights.shape == (2, 2)
    assert decision.pre_topk_probs.dtype == torch.float32
    assert decision.post_topk_weights.dtype == torch.float32
    assert decision.topk_indices.dtype == torch.long
    assert torch.equal(decision.topk_indices, torch.tensor([[0, 2], [1, 2]]))
    assert torch.allclose(decision.post_topk_weights.sum(dim=-1), torch.ones(2))

    dense_weights = decision.dense_weights()
    assert torch.allclose(dense_weights.sum(dim=-1), torch.ones(2))
    assert torch.equal(dense_weights == 0, torch.tensor([[False, True, False, True], [True, False, False, True]]))


def test_route_topk_temperature_changes_distribution():
    logits = torch.tensor([[3.0, 2.0, 1.0]])

    cold = route_topk(logits, top_k=2, temperature=0.5)
    warm = route_topk(logits, top_k=2, temperature=2.0)

    assert cold.pre_topk_probs.max() > warm.pre_topk_probs.max()
    assert cold.post_topk_weights.max() > warm.post_topk_weights.max()


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_balance_loss_uniform_baseline_does_not_depend_on_top_k(top_k):
    decision = route_topk(torch.zeros(8, 4), top_k=top_k, temperature=1.0)
    stats = compute_routing_statistics(decision, torch.ones(8, dtype=torch.bool))

    assert stats.balance_loss == pytest.approx(1.0)
    assert stats.selection_share.sum() == pytest.approx(1.0)


def test_routing_statistics_use_only_response_tokens():
    pre_topk_probs = torch.tensor(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
        ]
    )
    topk_indices = torch.tensor([[0, 1], [1, 2], [2, 1]])
    post_topk_weights = torch.tensor([[2 / 3, 1 / 3], [5 / 8, 3 / 8], [7 / 9, 2 / 9]])
    decision = RoutingDecision(pre_topk_probs, topk_indices, post_topk_weights)

    stats = compute_routing_statistics(decision, torch.tensor([1, 0, 1], dtype=torch.bool))

    assert stats.valid_token_count == 2
    assert torch.allclose(stats.pre_topk_mean_prob, torch.tensor([0.35, 0.25, 0.40]))
    assert torch.allclose(stats.post_topk_mean_weight, torch.tensor([1 / 3, 5 / 18, 7 / 18]))
    assert torch.allclose(stats.selection_share, torch.tensor([0.25, 0.50, 0.25]))
    assert torch.allclose(stats.top1_fraction, torch.tensor([0.50, 0.0, 0.50]))
    assert stats.balance_loss == pytest.approx(0.9375)

    expected_pre_entropy = (_normalized_entropy([0.6, 0.3, 0.1]) + _normalized_entropy([0.1, 0.2, 0.7])) / 2
    expected_post_entropy = (_normalized_entropy([2 / 3, 1 / 3]) + _normalized_entropy([7 / 9, 2 / 9])) / 2
    assert stats.pre_topk_normalized_entropy == pytest.approx(expected_pre_entropy)
    assert stats.post_topk_normalized_entropy == pytest.approx(expected_post_entropy)


def test_balance_loss_backpropagates_through_pre_topk_probabilities():
    logits = torch.tensor([[3.0, 2.0, 1.0], [2.0, 0.0, 1.0]], requires_grad=True)
    decision = route_topk(logits, top_k=2, temperature=1.0)

    compute_routing_statistics(decision, torch.ones(2, dtype=torch.bool)).balance_loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_k_one_has_zero_post_topk_entropy():
    decision = route_topk(torch.tensor([[2.0, 1.0], [0.0, 3.0]]), top_k=1, temperature=1.0)
    stats = compute_routing_statistics(decision, torch.ones(2, dtype=torch.bool))

    assert stats.post_topk_normalized_entropy == 0


def test_no_response_tokens_returns_zero_statistics_and_loss():
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0]], requires_grad=True)
    decision = route_topk(logits, top_k=1, temperature=1.0)
    stats = compute_routing_statistics(decision, torch.zeros(2, dtype=torch.bool))

    assert stats.valid_token_count == 0
    assert torch.count_nonzero(stats.pre_topk_mean_prob) == 0
    assert torch.count_nonzero(stats.post_topk_mean_weight) == 0
    assert stats.balance_loss == 0

    stats.balance_loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize(
    ("top_k", "temperature", "error"),
    [
        (0, 1.0, ValueError),
        (4, 1.0, ValueError),
        (1, 0.0, ValueError),
        (1, float("inf"), ValueError),
        (True, 1.0, ValueError),
        (1, True, TypeError),
    ],
)
def test_route_topk_rejects_invalid_configuration(top_k, temperature, error):
    with pytest.raises(error):
        route_topk(torch.ones(2, 3), top_k=top_k, temperature=temperature)


def test_route_topk_rejects_invalid_logits():
    with pytest.raises(ValueError, match="shape"):
        route_topk(torch.ones(2, 3, 4), top_k=2, temperature=1.0)
    with pytest.raises(TypeError, match="floating point"):
        route_topk(torch.ones(2, 3, dtype=torch.long), top_k=2, temperature=1.0)


def test_statistics_reject_mask_with_wrong_shape():
    decision = route_topk(torch.ones(2, 3), top_k=2, temperature=1.0)

    with pytest.raises(ValueError, match="response_mask"):
        compute_routing_statistics(decision, torch.ones(2, 1))
