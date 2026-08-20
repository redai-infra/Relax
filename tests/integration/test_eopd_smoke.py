# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Smoke test: end-to-end EOPD FKL loss computation + gradient backprop."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _install_fake_megatron(monkeypatch):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_world_size = lambda: 1
    core.mpu = mpu
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)


@pytest.fixture()
def opd_utils(monkeypatch):
    _install_fake_megatron(monkeypatch)
    sys.modules.pop("relax.utils.opd.opd_utils", None)
    module = importlib.import_module("relax.utils.opd.opd_utils")
    yield module
    sys.modules.pop("relax.utils.opd.opd_utils", None)


def _make_args(**overrides):
    defaults = {
        "use_eopd": True,
        "eopd_entropy_threshold": 0.8,
        "eopd_fkl_coef": 1.0,
        "opd_norm_mode": "trunc",
        "opd_log_prob_min_clamp": None,
        "opd_token_selection": "teacher_topk",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_eopd_smoke_forward_and_backward(opd_utils):
    """Full chain: compute_eopd_fkl_loss → reduce_opd_loss → backward → non-
    zero grads."""
    K = 32
    seq_len = 10

    student_topk_lp = torch.randn(seq_len, K, requires_grad=True)
    teacher_topk_lp = torch.randn(seq_len, K)
    teacher_entropy = torch.rand(seq_len) * 2.0

    batch = {
        "opd_topk_teacher_log_probs": [teacher_topk_lp],
        "teacher_entropy": [teacher_entropy],
        "response_lengths": [seq_len],
        "loss_masks": [torch.ones(seq_len)],
    }
    log_probs_and_entropy = {
        "topk_log_probs": [student_topk_lp],
    }

    args = _make_args()
    loss, metrics = opd_utils.compute_eopd_fkl_loss(
        args=args, batch=batch, log_probs_and_entropy=log_probs_and_entropy
    )

    assert loss is not None, "EOPD loss should not be None"
    assert loss.ndim == 0, "loss should be a scalar"
    assert loss.item() >= 0.0, "FKL loss should be non-negative"

    loss.backward()
    assert student_topk_lp.grad is not None, "student log-probs should have gradients"
    assert student_topk_lp.grad.abs().sum() > 0, "gradients should be non-zero"

    assert "eopd_fkl_loss" in metrics
    assert "eopd_high_entropy_frac" in metrics
    assert "eopd_teacher_entropy_mean" in metrics


def test_eopd_smoke_zero_loss_below_threshold(opd_utils):
    """All entropy below threshold → loss is zero, but function still returns a
    valid tensor."""
    K = 32
    seq_len = 5

    student_topk_lp = torch.randn(seq_len, K, requires_grad=True)
    teacher_topk_lp = torch.randn(seq_len, K)
    teacher_entropy = torch.full((seq_len,), 0.1)

    batch = {
        "opd_topk_teacher_log_probs": [teacher_topk_lp],
        "teacher_entropy": [teacher_entropy],
        "response_lengths": [seq_len],
        "loss_masks": [torch.ones(seq_len)],
    }
    log_probs_and_entropy = {"topk_log_probs": [student_topk_lp]}

    args = _make_args(eopd_entropy_threshold=0.8)
    loss, metrics = opd_utils.compute_eopd_fkl_loss(
        args=args, batch=batch, log_probs_and_entropy=log_probs_and_entropy
    )

    assert loss is not None
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_eopd_smoke_multi_sample(opd_utils):
    """Multiple samples in a batch are handled correctly."""
    K = 8
    s1_len, s2_len = 6, 4

    student_lp_1 = torch.randn(s1_len, K, requires_grad=True)
    student_lp_2 = torch.randn(s2_len, K, requires_grad=True)
    teacher_lp_1 = torch.randn(s1_len, K)
    teacher_lp_2 = torch.randn(s2_len, K)
    ent_1 = torch.full((s1_len,), 1.5)
    ent_2 = torch.full((s2_len,), 1.5)

    batch = {
        "opd_topk_teacher_log_probs": [teacher_lp_1, teacher_lp_2],
        "teacher_entropy": [ent_1, ent_2],
        "response_lengths": [s1_len, s2_len],
        "loss_masks": [torch.ones(s1_len), torch.ones(s2_len)],
    }
    log_probs_and_entropy = {"topk_log_probs": [student_lp_1, student_lp_2]}

    args = _make_args()
    loss, metrics = opd_utils.compute_eopd_fkl_loss(
        args=args, batch=batch, log_probs_and_entropy=log_probs_and_entropy
    )

    assert loss is not None
    loss.backward()
    assert student_lp_1.grad is not None
    assert student_lp_2.grad is not None
