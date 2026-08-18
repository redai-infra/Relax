# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Metric-contract tests for the Megatron P3O loss branch.

``relax.backends.megatron.loss`` imports ``megatron.core`` at module scope, and
CI installs no megatron. The branch under test only consumes token terms, so
the megatron surface is stubbed for the import and restored afterwards --
keeping these assertions running in CI instead of silently skipping.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from tests.backends.megatron._megatron_stub import stubbed_megatron_modules


with stubbed_megatron_modules(("megatron", "ray", "tensordict")):
    from relax.backends.megatron import loss as loss_module

from relax.utils.training.p3o_utils import P3OStepContext


REQUIRED_P3O_METRICS = {
    "p3o/normalized_ess",
    "p3o/adaptive_cap",
    "p3o/ratio_mean",
    "p3o/ratio_std",
    "p3o/cap_fraction",
    "p3o/clip_fraction",
    "p3o/score_loss",
    "p3o/behavior_kl_proxy",
    "p3o/adaptive_kl_loss",
    "p3o/reference_kl",
    "p3o/entropy",
    "p3o/valid_tokens",
    "p3o/total_loss",
}


def test_get_p3o_context_computes_micro_batch_scope_without_prepass():
    args = Namespace(p3o_ess_scope="micro-batch")
    log_probs = torch.tensor([-0.4, -0.8])
    behavior_log_probs = torch.tensor([-0.5, -0.7])
    valid_mask = torch.tensor([True, True])

    context = loss_module.get_p3o_context(args, log_probs, behavior_log_probs, valid_mask)

    assert context.valid_token_count.item() == 2
    assert 0.0 < context.normalized_ess.item() <= 1.0
    assert torch.equal(context.normalized_ess, context.adaptive_cap)


def test_get_p3o_context_zeroes_dummy_micro_batch_before_collective(monkeypatch):
    captured = {}

    def capture_stats(stats, invalid_count, **kwargs):
        captured["stats"] = stats.as_vector()
        captured["invalid_count"] = invalid_count
        return stats

    monkeypatch.setattr(loss_module, "synchronize_p3o_stats", capture_stats)
    context = loss_module.get_p3o_context(
        Namespace(p3o_ess_scope="micro-batch"),
        torch.tensor([float("nan")]),
        torch.tensor([0.0]),
        torch.tensor([True]),
        is_dummy=True,
    )

    assert torch.equal(captured["stats"], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(captured["invalid_count"], torch.zeros((), dtype=torch.float64))
    assert context.valid_token_count.item() == 0


def test_get_p3o_context_rejects_unknown_scope():
    args = Namespace(p3o_ess_scope="window")

    with pytest.raises(ValueError, match="micro-batch.*step"):
        loss_module.get_p3o_context(args, torch.zeros(1), torch.zeros(1), torch.ones(1, dtype=torch.bool))


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
        p3o_ess_scope="step",
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
    dummy_masks = []
    dummy_flags = []

    def capture_context(*args, is_dummy=False):
        dummy_masks.append(args[3])
        dummy_flags.append(is_dummy)
        return step_context

    monkeypatch.setattr(loss_module, "get_p3o_context", capture_context)
    batch = {
        "advantages": torch.tensor([1.0, -1.0]),
        "rollout_log_probs": [torch.full_like(log_probs.detach(), float("nan"))],
        "unconcat_tokens": [torch.tensor([1, 2])],
        "total_lengths": [2],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "__is_dummy__": True,
    }

    loss, metrics = loss_module.p3o_loss_function(args, batch, torch.zeros(1), torch.sum)

    assert REQUIRED_P3O_METRICS <= metrics.keys()
    assert loss.dtype is torch.float64
    assert metrics["p3o/score_loss"].dtype is torch.float64
    assert metrics["p3o/total_loss"].dtype is torch.float64
    assert metrics["p3o/normalized_ess"].dtype is torch.float64
    assert not any(metric.startswith("opd/") for metric in metrics)
    assert torch.equal(metrics["p3o/reference_kl"], torch.zeros(()))
    assert not metrics["p3o/reference_kl"].requires_grad
    assert torch.isfinite(loss)
    assert torch.equal(dummy_masks[0], torch.zeros(2, dtype=torch.bool))
    assert dummy_flags == [True]


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
        use_opd=False,
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
    monkeypatch.setattr(
        loss_module,
        "policy_loss_function",
        lambda *args, **kwargs: pytest.fail("P3O must not use the ordinary policy-loss path"),
    )
    monkeypatch.setattr(
        loss_module,
        "compute_policy_opd_loss",
        lambda *args, **kwargs: pytest.fail("P3O must not call compute_policy_opd_loss"),
    )

    _, normalizer, logging_dict = loss_module.loss_function(args, batch, 1, torch.zeros(1))

    assert normalizer.item() == 1
    assert logging_dict["values"][0].item() == 1


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_p3o_loss_uses_one_canonical_full_cp_graph(monkeypatch, cp_rank):
    """Only CP0's full loss survives, while full response masks stay
    ungathered."""
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda: cp_rank)

    full_logits = torch.arange(12.0, requires_grad=True).reshape(1, 2, 6)
    local_logits = torch.arange(6.0, requires_grad=True).reshape(1, 1, 6)
    rollout = torch.tensor([-0.2])
    advantages = torch.tensor([0.5])
    full_mask = torch.tensor([1.0, 0.0])
    gathered_inputs = []

    def gather(value, *args, **kwargs):
        del args, kwargs
        gathered_inputs.append(value)
        return torch.cat([value, value])

    reducer_inputs = {}

    def build_reducer(*args, **kwargs):
        reducer_inputs["args"] = args
        reducer_inputs["kwargs"] = kwargs
        return torch.sum

    impl_inputs = {}

    def canonical_impl(args, batch, logits, reducer):
        del args
        impl_inputs.update(batch=batch, logits=logits, reducer=reducer)
        loss = logits.sum()
        return loss, {"loss": loss.detach()}

    monkeypatch.setattr(loss_module, "all_gather_with_cp", gather)
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", build_reducer)
    monkeypatch.setattr(loss_module, "_p3o_loss_function_impl", canonical_impl)
    batch = {
        "packed_seq_params": SimpleNamespace(_relax_p3o_canonical_full_logits=full_logits),
        "rollout_log_probs": [rollout],
        "advantages": [advantages],
        "loss_masks": [full_mask],
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
    }
    args = Namespace(qkv_format="thd", calculate_per_token_loss=True)

    loss, metrics = loss_module.p3o_loss_function(args, batch, local_logits, torch.sum)

    assert gathered_inputs == [rollout, advantages]
    if cp_rank == 0:
        assert impl_inputs["batch"]["loss_masks"][0] is full_mask
    else:
        assert torch.equal(impl_inputs["batch"]["loss_masks"][0], torch.zeros_like(full_mask))
    assert impl_inputs["batch"]["dynamic_cp_size"] == 1
    assert impl_inputs["batch"]["dynamic_cp_rank"] == 0
    assert impl_inputs["logits"] is full_logits
    assert reducer_inputs["args"][2][0] is impl_inputs["batch"]["loss_masks"][0]
    assert reducer_inputs["kwargs"] == {"dynamic_cp_size": 1, "dynamic_cp_rank": 0}
    if cp_rank == 0:
        assert torch.equal(loss, full_logits.sum())
        assert torch.equal(metrics["loss"], full_logits.sum().detach())
    else:
        assert torch.equal(loss, torch.zeros(()))
        assert torch.equal(metrics["loss"], torch.zeros(()))


def test_policy_loss_dispatch_selects_dedicated_p3o_path():
    args = Namespace(advantage_estimator="p3o", use_opd=False)

    assert loss_module._select_policy_loss_function(args) is loss_module.p3o_loss_function


def test_policy_loss_dispatch_rejects_p3o_with_opd():
    args = Namespace(advantage_estimator="p3o", use_opd=True)

    with pytest.raises(ValueError, match="P3O and OPD are mutually exclusive"):
        loss_module._select_policy_loss_function(args)


def test_policy_loss_dispatch_preserves_opd_for_non_p3o_estimators():
    args = Namespace(advantage_estimator="grpo", use_opd=True)

    assert loss_module._select_policy_loss_function(args) is loss_module.policy_loss_function
