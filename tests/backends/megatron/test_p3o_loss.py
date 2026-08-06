# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Metric-contract tests for the Megatron P3O loss branch.

``relax.backends.megatron.loss`` imports ``megatron.core`` at module scope, and
CI installs no megatron. The branch under test only consumes token terms, so
the megatron surface is stubbed for the import and restored afterwards --
keeping these assertions running in CI instead of silently skipping.
"""

from argparse import Namespace

import torch

from tests.backends.megatron._megatron_stub import stubbed_megatron_modules


with stubbed_megatron_modules(("megatron", "ray")):
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


def test_p3o_loss_function_normalizes_by_true_valid_tokens(monkeypatch):
    """All-masked samples must not add phantom tokens to P3O's normalizer."""
    args = Namespace(
        advantage_estimator="p3o",
        allgather_cp=False,
        calculate_per_token_loss=True,
        global_batch_size=2,
        loss_type="policy_loss",
        qkv_format="thd",
        recompute_loss_function=False,
    )
    batch = {
        "loss_masks": [torch.zeros(2), torch.tensor([1.0, 0.0])],
        "response_lengths": [2, 2],
        "total_lengths": [3, 3],
    }
    monkeypatch.setattr(loss_module, "get_cp_local_num_tokens", lambda *args, **kwargs: torch.tensor(2.0))
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(
        loss_module,
        "get_cp_local_valid_mask",
        lambda *args, **kwargs: torch.tensor([False, False, True, False]),
    )
    monkeypatch.setattr(
        loss_module,
        "p3o_loss_function",
        lambda *args, **kwargs: (torch.tensor(3.0, requires_grad=True), {"loss": torch.tensor(3.0)}),
    )

    _, normalizer, logging_dict = loss_module.loss_function(args, batch, 1, torch.zeros(1))

    assert normalizer.item() == 1
    assert logging_dict["values"][0].item() == 1
