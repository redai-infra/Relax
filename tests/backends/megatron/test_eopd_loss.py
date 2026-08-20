# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for EOPD (Entropy-aware OPD) forward KL loss."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


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
def opd_utils_module(monkeypatch):
    torch = pytest.importorskip("torch", exc_type=ImportError)
    _install_fake_megatron(monkeypatch)
    sys.modules.pop("relax.utils.opd.opd_utils", None)
    module = importlib.import_module("relax.utils.opd.opd_utils")
    yield module, torch
    sys.modules.pop("relax.utils.opd.opd_utils", None)


def _make_args(**overrides):
    defaults = dict(
        use_eopd=True,
        eopd_entropy_threshold=0.8,
        eopd_fkl_coef=1.0,
        eopd_fkl_top_k=0,
        opd_log_prob_top_k=4,
        opd_token_selection="teacher_topk",
        opd_norm_mode="tail",
        opd_log_prob_min_clamp=None,
        opd_loss_coef=1.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_batch(torch, teacher_entropy_vals, response_lengths, teacher_topk_lp, loss_masks=None):
    batch = {
        "response_lengths": response_lengths,
        "total_lengths": [r + 2 for r in response_lengths],
        "loss_masks": loss_masks or [torch.ones(r, dtype=torch.float32) for r in response_lengths],
        "teacher_entropy": [torch.tensor(e, dtype=torch.float32) for e in teacher_entropy_vals],
        "opd_topk_teacher_log_probs": [torch.tensor(lp, dtype=torch.float32) for lp in teacher_topk_lp],
    }
    return batch


def test_eopd_fkl_loss_basic(opd_utils_module):
    opd_utils, torch = opd_utils_module
    args = _make_args()

    teacher_topk_lp = [
        [[-1.0, -2.0, -3.0, -4.0], [-0.5, -1.5, -2.5, -3.5]],
    ]
    student_topk_lp = [
        [[-1.1, -2.1, -3.1, -4.1], [-0.6, -1.6, -2.6, -3.6]],
    ]
    teacher_entropy = [[1.0, 0.5]]

    batch = _make_batch(torch, teacher_entropy, [2], teacher_topk_lp)
    log_probs_and_entropy = {
        "topk_log_probs": [torch.tensor(student_topk_lp[0], dtype=torch.float32)],
    }

    loss, reported = opd_utils.compute_eopd_fkl_loss(
        args=args, batch=batch, log_probs_and_entropy=log_probs_and_entropy
    )
    assert loss is not None
    assert loss.item() > 0.0
    assert "eopd_fkl_loss" in reported
    assert "eopd_high_entropy_frac" in reported
    assert "eopd_teacher_entropy_mean" in reported
    assert abs(reported["eopd_high_entropy_frac"].item() - 0.5) < 1e-5


def test_eopd_fkl_loss_all_below_threshold(opd_utils_module):
    opd_utils, torch = opd_utils_module
    args = _make_args(eopd_entropy_threshold=2.0)

    teacher_topk_lp = [[[-1.0, -2.0, -3.0, -4.0], [-0.5, -1.5, -2.5, -3.5]]]
    student_topk_lp = [[[-1.1, -2.1, -3.1, -4.1], [-0.6, -1.6, -2.6, -3.6]]]
    teacher_entropy = [[0.5, 0.3]]

    batch = _make_batch(torch, teacher_entropy, [2], teacher_topk_lp)
    log_probs_and_entropy = {
        "topk_log_probs": [torch.tensor(student_topk_lp[0], dtype=torch.float32)],
    }

    loss, reported = opd_utils.compute_eopd_fkl_loss(
        args=args, batch=batch, log_probs_and_entropy=log_probs_and_entropy
    )
    assert loss is not None
    assert abs(loss.item()) < 1e-6
    assert abs(reported["eopd_high_entropy_frac"].item()) < 1e-5


def test_eopd_fkl_loss_all_above_threshold(opd_utils_module):
    opd_utils, torch = opd_utils_module
    args = _make_args(eopd_entropy_threshold=0.1)

    teacher_topk_lp = [[[-1.0, -2.0, -3.0, -4.0], [-0.5, -1.5, -2.5, -3.5]]]
    student_topk_lp = [[[-1.1, -2.1, -3.1, -4.1], [-0.6, -1.6, -2.6, -3.6]]]
    teacher_entropy = [[1.0, 0.5]]

    batch = _make_batch(torch, teacher_entropy, [2], teacher_topk_lp)
    log_probs_and_entropy = {
        "topk_log_probs": [torch.tensor(student_topk_lp[0], dtype=torch.float32)],
    }

    loss_gated, _ = opd_utils.compute_eopd_fkl_loss(
        args=args, batch=batch, log_probs_and_entropy=log_probs_and_entropy
    )

    full_fkl_chunks = []
    for s, t in zip(student_topk_lp[0], teacher_topk_lp[0]):
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(0)
        full_fkl_chunks.append(opd_utils.compute_opd_kl_topk(s_t, t_t, kl_type="forward_kl"))
    full_fkl = torch.cat(full_fkl_chunks, dim=0).mean()

    assert loss_gated is not None
    assert torch.isclose(loss_gated, full_fkl, atol=1e-5)


def test_eopd_fkl_loss_normalization(opd_utils_module):
    opd_utils, torch = opd_utils_module
    args = _make_args(eopd_entropy_threshold=0.0)

    teacher_topk_lp = [
        [[-1.0, -2.0, -3.0, -4.0], [-0.5, -1.5, -2.5, -3.5]],
        [[-1.2, -2.2, -3.2, -4.2]],
    ]
    student_topk_lp = [
        [[-1.1, -2.1, -3.1, -4.1], [-0.6, -1.6, -2.6, -3.6]],
        [[-1.3, -2.3, -3.3, -4.3]],
    ]
    teacher_entropy = [[1.0, 1.0], [1.0]]
    loss_masks = [torch.ones(2), torch.tensor([1.0])]

    batch = _make_batch(torch, teacher_entropy, [2, 1], teacher_topk_lp, loss_masks=loss_masks)
    log_probs_and_entropy = {
        "topk_log_probs": [
            torch.tensor(student_topk_lp[0], dtype=torch.float32),
            torch.tensor(student_topk_lp[1], dtype=torch.float32),
        ],
    }

    loss, _ = opd_utils.compute_eopd_fkl_loss(args=args, batch=batch, log_probs_and_entropy=log_probs_and_entropy)
    assert loss is not None
    assert loss.item() > 0.0


def test_eopd_fkl_loss_disabled(opd_utils_module):
    opd_utils, torch = opd_utils_module
    args = _make_args(use_eopd=False)

    loss, reported = opd_utils.compute_eopd_fkl_loss(args=args, batch={}, log_probs_and_entropy={})
    assert loss is None
    assert reported == {}


def test_eopd_reported_metrics(opd_utils_module):
    opd_utils, torch = opd_utils_module
    args = _make_args()

    teacher_topk_lp = [[[-1.0, -2.0, -3.0, -4.0]]]
    student_topk_lp = [[[-1.1, -2.1, -3.1, -4.1]]]
    teacher_entropy = [[1.0]]

    batch = _make_batch(torch, teacher_entropy, [1], teacher_topk_lp)
    log_probs_and_entropy = {
        "topk_log_probs": [torch.tensor(student_topk_lp[0], dtype=torch.float32)],
    }

    _, reported = opd_utils.compute_eopd_fkl_loss(args=args, batch=batch, log_probs_and_entropy=log_probs_and_entropy)
    assert "eopd_fkl_loss" in reported
    assert "eopd_high_entropy_frac" in reported
    assert "eopd_teacher_entropy_mean" in reported
    assert reported["eopd_high_entropy_frac"].item() == 1.0
    assert abs(reported["eopd_teacher_entropy_mean"].item() - 1.0) < 1e-5
