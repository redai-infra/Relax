# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tiny-model acceptance gate for P3O's on-policy degeneration."""

import copy

import torch
from torch import nn

from relax.utils.training.p3o_utils import (
    compute_p3o_sufficient_stats,
    compute_p3o_token_terms,
    finalize_p3o_step_context,
)


def _flatten_gradients(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.flatten() for parameter in model.parameters()])


def test_p3o_on_policy_matches_policy_gradient_and_parameter_update():
    torch.manual_seed(42)
    base_model = nn.Linear(3, 1, bias=True)
    pg_model = copy.deepcopy(base_model)
    p3o_model = copy.deepcopy(base_model)
    features = torch.tensor(
        [
            [0.2, -0.5, 1.0],
            [1.5, 0.3, -0.7],
            [-0.4, 0.8, 0.1],
            [0.9, -1.2, 0.6],
            [-0.8, -0.2, 1.3],
            [0.5, 0.7, -0.9],
        ],
        dtype=torch.float32,
    )
    advantages = torch.tensor([1.0, -0.5, 0.75, -1.25, 0.4, 0.9])
    valid_mask = torch.ones(features.size(0), dtype=torch.bool)
    behavior_log_probs = base_model(features).squeeze(-1).detach()

    pg_optimizer = torch.optim.SGD(pg_model.parameters(), lr=0.05)
    pg_log_probs = pg_model(features).squeeze(-1)
    pg_loss = -(pg_log_probs * advantages).mean()
    pg_loss.backward()
    pg_gradients = _flatten_gradients(pg_model).clone()

    p3o_optimizer = torch.optim.SGD(p3o_model.parameters(), lr=0.05)
    p3o_log_probs = p3o_model(features).squeeze(-1)
    context = finalize_p3o_step_context(compute_p3o_sufficient_stats(p3o_log_probs, behavior_log_probs, valid_mask))
    terms = compute_p3o_token_terms(
        p3o_log_probs,
        behavior_log_probs,
        advantages,
        valid_mask,
        context,
    )
    p3o_loss = (terms.score_loss + terms.adaptive_kl_loss).mean()
    p3o_loss.backward()
    p3o_gradients = _flatten_gradients(p3o_model).clone()

    cosine = torch.nn.functional.cosine_similarity(pg_gradients, p3o_gradients, dim=0)
    relative_l2 = torch.linalg.vector_norm(p3o_gradients - pg_gradients) / torch.linalg.vector_norm(pg_gradients)
    assert float(cosine) >= 0.9999
    assert float(relative_l2) <= 1e-4
    assert float(terms.adaptive_kl_loss.detach().abs().max()) <= 1e-7

    pg_optimizer.step()
    p3o_optimizer.step()
    for pg_parameter, p3o_parameter in zip(pg_model.parameters(), p3o_model.parameters(), strict=True):
        torch.testing.assert_close(p3o_parameter, pg_parameter, rtol=1e-4, atol=1e-6)
