# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Smoke coverage for RLOO dispatch through Megatron's policy loss."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")


def test_policy_loss_function_dispatches_rloo_objective(monkeypatch):
    import relax.backends.megatron.loss as loss_module

    log_probs = torch.tensor([-0.7, -0.4, -0.2, -0.1], dtype=torch.float64, requires_grad=True)
    entropy = torch.zeros_like(log_probs)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float64)

    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (None, {"log_probs": [log_probs], "entropy": [entropy]}),
    )
    monkeypatch.setattr(loss_module, "resolve_opd_gather_topk_token_ids", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        loss_module,
        "compute_policy_opd_loss",
        lambda **_kwargs: (None, {}),
    )

    args = SimpleNamespace(
        advantage_estimator="rloo",
        true_on_policy_mode=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    batch = {
        "advantages": advantages,
        "log_probs": [torch.zeros_like(log_probs)],
        "response_lengths": [log_probs.numel()],
        "total_lengths": [log_probs.numel() + 1],
        "unconcat_tokens": [torch.arange(log_probs.numel() + 1)],
        "loss_masks": [torch.ones_like(log_probs)],
    }

    loss, metrics = loss_module.policy_loss_function(
        args,
        batch,
        logits=torch.empty(1, 1, 1),
        sum_of_sample_mean=lambda values: values.sum(),
    )

    expected = -(advantages * log_probs).sum()
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["pg_loss"], expected.detach())
    assert metrics["pg_clipfrac"].item() == 0.0

    loss.backward()
    assert torch.allclose(log_probs.grad, -advantages)
