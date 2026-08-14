# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Element-wise parity tests: Relax EOPD vs reference forward-KL formulas from the paper."""

import importlib
import sys
from types import ModuleType

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


def _reference_forward_kl_trunc(teacher_lp: torch.Tensor, student_lp: torch.Tensor) -> torch.Tensor:
    """Reference forward KL per token: sum_k p_teacher(k) * (log p_teacher(k) - log p_student(k)).

    This matches the paper formula for truncated (top-K) forward KL without tail correction.
    """
    t = teacher_lp.float()
    s = student_lp.float()
    per_k = t.exp() * (t - s)
    return per_k.sum(dim=-1)


def _reference_entropy_mask(teacher_entropy: torch.Tensor, threshold: float) -> torch.Tensor:
    """Reference entropy gate: 1 where entropy >= threshold, else 0."""
    return (teacher_entropy >= threshold).float()


def _reference_reduce(values: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Reference masked reduction: sum(values * mask) / max(sum(mask), 1)."""
    masked = values * loss_mask
    return masked.sum() / torch.clamp_min(loss_mask.sum(), 1)


class TestForwardKLParity:
    """Verify compute_opd_kl_topk(forward_kl, trunc) matches reference per-token forward KL."""

    def test_basic_parity(self, opd_utils):
        torch.manual_seed(42)
        K = 32
        R = 20
        teacher_lp = torch.log_softmax(torch.randn(R, K), dim=-1)
        student_lp = torch.log_softmax(torch.randn(R, K), dim=-1)

        relax_result = opd_utils.compute_opd_kl_topk(student_lp, teacher_lp, kl_type="forward_kl", norm_mode="trunc")
        ref_result = _reference_forward_kl_trunc(teacher_lp, student_lp)

        torch.testing.assert_close(relax_result, ref_result, atol=1e-5, rtol=1e-5)

    def test_identical_distributions(self, opd_utils):
        K = 16
        R = 10
        lp = torch.log_softmax(torch.randn(R, K), dim=-1)

        relax_result = opd_utils.compute_opd_kl_topk(lp, lp, kl_type="forward_kl", norm_mode="trunc")
        ref_result = _reference_forward_kl_trunc(lp, lp)

        torch.testing.assert_close(relax_result, ref_result, atol=1e-6, rtol=1e-6)
        assert relax_result.abs().max() < 1e-5, "KL(p||p) should be ~0"

    def test_various_k_sizes(self, opd_utils):
        for K in [4, 8, 16, 32, 64]:
            teacher_lp = torch.log_softmax(torch.randn(5, K), dim=-1)
            student_lp = torch.log_softmax(torch.randn(5, K), dim=-1)

            relax_result = opd_utils.compute_opd_kl_topk(
                student_lp, teacher_lp, kl_type="forward_kl", norm_mode="trunc"
            )
            ref_result = _reference_forward_kl_trunc(teacher_lp, student_lp)
            torch.testing.assert_close(relax_result, ref_result, atol=1e-5, rtol=1e-5)


class TestEntropyMaskParity:
    """Verify entropy gating matches reference: mask = (entropy >= tau)."""

    def test_mask_parity(self, opd_utils):
        entropy = torch.tensor([0.1, 0.5, 0.8, 1.0, 1.5, 0.79, 0.81])
        threshold = 0.8

        ref_mask = _reference_entropy_mask(entropy, threshold)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        torch.testing.assert_close(ref_mask, expected)

    def test_mask_applied_to_fkl(self, opd_utils):
        """FKL * entropy_mask matches element-wise reference."""
        K = 8
        R = 7
        teacher_lp = torch.log_softmax(torch.randn(R, K), dim=-1)
        student_lp = torch.log_softmax(torch.randn(R, K), dim=-1)
        entropy = torch.tensor([0.1, 0.5, 0.8, 1.0, 1.5, 0.79, 0.81])
        threshold = 0.8

        per_token_fkl = opd_utils.compute_opd_kl_topk(student_lp, teacher_lp, kl_type="forward_kl", norm_mode="trunc")
        ref_fkl = _reference_forward_kl_trunc(teacher_lp, student_lp)
        ref_mask = _reference_entropy_mask(entropy, threshold)
        ref_gated = ref_fkl * ref_mask

        ent_mask = (entropy >= threshold).float()
        relax_gated = per_token_fkl * ent_mask

        torch.testing.assert_close(relax_gated, ref_gated, atol=1e-5, rtol=1e-5)


class TestReductionParity:
    """Verify reduce_opd_loss matches reference: masked_sum / mask_count."""

    def test_single_sample(self, opd_utils):
        values = torch.tensor([0.5, 1.0, 0.2, 0.8, 0.3])
        loss_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        batch = {"response_lengths": [5], "loss_masks": [loss_mask]}

        relax_result = opd_utils.reduce_opd_loss(batch, values)
        ref_result = _reference_reduce(values, loss_mask)
        torch.testing.assert_close(relax_result, ref_result, atol=1e-6, rtol=1e-6)

    def test_multi_sample(self, opd_utils):
        v1 = torch.tensor([0.5, 1.0, 0.2])
        v2 = torch.tensor([0.8, 0.3])
        m1 = torch.tensor([1.0, 1.0, 0.0])
        m2 = torch.tensor([1.0, 1.0])

        values = torch.cat([v1, v2])
        batch = {"response_lengths": [3, 2], "loss_masks": [m1, m2]}

        relax_result = opd_utils.reduce_opd_loss(batch, values)

        masked = torch.cat([v1 * m1, v2 * m2])
        ref_result = masked.sum() / (m1.sum() + m2.sum())
        torch.testing.assert_close(relax_result, ref_result, atol=1e-6, rtol=1e-6)

    def test_all_masked(self, opd_utils):
        values = torch.tensor([1.0, 2.0, 3.0])
        loss_mask = torch.zeros(3)
        batch = {"response_lengths": [3], "loss_masks": [loss_mask]}

        relax_result = opd_utils.reduce_opd_loss(batch, values)
        assert relax_result.item() == pytest.approx(0.0, abs=1e-7)


class TestEndToEndParity:
    """Full EOPD chain: FKL + entropy gate + reduction matches hand-computed reference."""

    def test_full_chain(self, opd_utils):
        torch.manual_seed(123)
        K, R = 32, 10
        threshold = 0.8

        teacher_lp = torch.log_softmax(torch.randn(R, K), dim=-1)
        student_lp = torch.log_softmax(torch.randn(R, K), dim=-1)
        entropy = torch.rand(R) * 2.0
        loss_mask = torch.ones(R)
        loss_mask[3] = 0.0
        loss_mask[7] = 0.0

        ref_fkl = _reference_forward_kl_trunc(teacher_lp, student_lp)
        ref_mask = _reference_entropy_mask(entropy, threshold)
        ref_gated = ref_fkl * ref_mask
        ref_loss = _reference_reduce(ref_gated, loss_mask)

        per_token_fkl = opd_utils.compute_opd_kl_topk(student_lp, teacher_lp, kl_type="forward_kl", norm_mode="trunc")
        ent_mask = (entropy >= threshold).float()
        gated = per_token_fkl * ent_mask

        batch = {"response_lengths": [R], "loss_masks": [loss_mask]}
        relax_loss = opd_utils.reduce_opd_loss(batch, gated)

        torch.testing.assert_close(relax_loss, ref_loss, atol=1e-5, rtol=1e-5)
