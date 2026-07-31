# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Metric-contract tests for the Megatron P3O loss branch."""

from argparse import Namespace

import torch

from relax.backends.megatron import loss as loss_module
from relax.utils.training.p3o_utils import P3OStepContext


REQUIRED_P3O_METRICS = {
    "p3o/normalized_ess",
    "p3o/adaptive_cap",
    "p3o/ratio_mean",
    "p3o/ratio_std",
    "p3o/cap_fraction",
    "p3o/score_loss",
    "p3o/behavior_kl_proxy",
    "p3o/adaptive_kl_loss",
    "p3o/reference_kl",
    "p3o/entropy",
    "p3o/valid_tokens",
    "p3o/total_loss",
}


def test_p3o_loss_reports_complete_schema_without_reference_kl(monkeypatch):
    step_context = P3OStepContext(
        normalized_ess=torch.tensor(0.75, dtype=torch.float64),
        adaptive_cap=torch.tensor(0.75, dtype=torch.float64),
        valid_token_count=torch.tensor(2.0, dtype=torch.float64),
        ratio_mean=torch.tensor(1.0, dtype=torch.float64),
        ratio_std=torch.tensor(0.0, dtype=torch.float64),
    )
    args = Namespace(
        _p3o_step_context=step_context,
        entropy_coef=0.0,
        qkv_format="thd",
        use_kl_loss=False,
    )
    log_probs = torch.tensor([-0.4, -0.8], requires_grad=True)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (
            torch.empty(0),
            {
                "log_probs": [log_probs],
                "entropy": [torch.tensor([0.2, 0.3])],
            },
        ),
    )
    monkeypatch.setattr(
        loss_module,
        "get_cp_local_valid_mask",
        lambda *args, **kwargs: torch.tensor([True, True]),
    )
    batch = {
        "advantages": torch.tensor([1.0, -1.0]),
        "rollout_log_probs": [log_probs.detach().clone()],
        "unconcat_tokens": [torch.tensor([1, 2])],
        "total_lengths": [2],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
    }

    _, metrics = loss_module.p3o_loss_function(args, batch, torch.zeros(1), torch.sum)

    assert REQUIRED_P3O_METRICS <= metrics.keys()
    assert torch.equal(metrics["p3o/reference_kl"], torch.zeros(()))
    assert not metrics["p3o/reference_kl"].requires_grad
