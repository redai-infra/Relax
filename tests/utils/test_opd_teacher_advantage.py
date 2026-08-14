# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for teacher advantage computation in OPD."""

import importlib
import sys
from types import ModuleType

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


def test_teacher_advantage_replace(opd_utils_module):
    """teacher_advantage = (teacher_lp - student_lp).detach(), replaces original advantage."""
    opd_utils, torch = opd_utils_module

    teacher_lp = [torch.tensor([-1.0, -2.0, -3.0])]
    student_lp = [torch.tensor([-1.5, -2.5, -3.5])]
    original_adv = [torch.tensor([10.0, 20.0, 30.0])]

    rollout_data = {"teacher_log_probs": teacher_lp, "rollout_log_probs": student_lp}
    advantages = [a.clone() for a in original_adv]

    opd_utils._apply_teacher_advantage(rollout_data, advantages, additive=False)

    expected = teacher_lp[0] - student_lp[0]
    assert torch.allclose(advantages[0], expected, atol=1e-6)
    assert not torch.allclose(advantages[0], original_adv[0])


def test_teacher_advantage_additive(opd_utils_module):
    """additive=True adds teacher advantage to original instead of
    replacing."""
    opd_utils, torch = opd_utils_module

    teacher_lp = [torch.tensor([-1.0, -2.0])]
    student_lp = [torch.tensor([-1.5, -2.5])]
    original_adv = [torch.tensor([10.0, 20.0])]

    rollout_data = {"teacher_log_probs": teacher_lp, "rollout_log_probs": student_lp}
    advantages = [a.clone() for a in original_adv]

    opd_utils._apply_teacher_advantage(rollout_data, advantages, additive=True)

    teacher_adv = teacher_lp[0] - student_lp[0]
    expected = original_adv[0] + teacher_adv
    assert torch.allclose(advantages[0], expected, atol=1e-6)


def test_teacher_advantage_detached(opd_utils_module):
    """Result should be detached (no gradient)."""
    opd_utils, torch = opd_utils_module

    teacher_lp = [torch.tensor([-1.0, -2.0], requires_grad=True)]
    student_lp = [torch.tensor([-1.5, -2.5], requires_grad=True)]
    original_adv = [torch.tensor([10.0, 20.0])]

    rollout_data = {"teacher_log_probs": teacher_lp, "rollout_log_probs": student_lp}
    advantages = [a.clone() for a in original_adv]

    opd_utils._apply_teacher_advantage(rollout_data, advantages, additive=False)

    assert not advantages[0].requires_grad


def test_teacher_advantage_missing_data(opd_utils_module):
    """Gracefully handle missing teacher/student log_probs."""
    opd_utils, torch = opd_utils_module

    original_adv = [torch.tensor([10.0, 20.0])]
    advantages = [a.clone() for a in original_adv]

    opd_utils._apply_teacher_advantage({"teacher_log_probs": None, "rollout_log_probs": None}, advantages)
    assert torch.allclose(advantages[0], original_adv[0])

    opd_utils._apply_teacher_advantage({}, advantages)
    assert torch.allclose(advantages[0], original_adv[0])


def test_teacher_advantage_multi_sample(opd_utils_module):
    """Works correctly with multiple samples."""
    opd_utils, torch = opd_utils_module

    teacher_lp = [torch.tensor([-1.0, -2.0]), torch.tensor([-0.5])]
    student_lp = [torch.tensor([-1.5, -2.5]), torch.tensor([-1.0])]
    advantages = [torch.zeros(2), torch.zeros(1)]

    rollout_data = {"teacher_log_probs": teacher_lp, "rollout_log_probs": student_lp}

    opd_utils._apply_teacher_advantage(rollout_data, advantages, additive=False)

    assert torch.allclose(advantages[0], torch.tensor([0.5, 0.5]), atol=1e-6)
    assert torch.allclose(advantages[1], torch.tensor([0.5]), atol=1e-6)


def test_teacher_advantage_empty_tensor(opd_utils_module):
    """Skip samples with empty tensors."""
    opd_utils, torch = opd_utils_module

    teacher_lp = [torch.tensor([])]
    student_lp = [torch.tensor([])]
    original_adv = [torch.tensor([10.0])]
    advantages = [a.clone() for a in original_adv]

    rollout_data = {"teacher_log_probs": teacher_lp, "rollout_log_probs": student_lp}

    opd_utils._apply_teacher_advantage(rollout_data, advantages, additive=False)
    assert torch.allclose(advantages[0], original_adv[0])
