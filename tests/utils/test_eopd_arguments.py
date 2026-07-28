# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validation tests for EOPD CLI arguments."""

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
    _install_fake_megatron(monkeypatch)
    sys.modules.pop("relax.utils.opd.opd_utils", None)
    module = importlib.import_module("relax.utils.opd.opd_utils")
    yield module
    sys.modules.pop("relax.utils.opd.opd_utils", None)


def _base_args(**overrides):
    defaults = dict(
        use_opd=True,
        opd_type="megatron",
        opd_teacher_timeout_s=600,
        opd_log_prob_top_k=4,
        opd_token_selection="teacher_topk",
        opd_kl_type="reverse_kl",
        opd_kl_coef=0.0,
        opd_loss_coef=1.0,
        opd_teacher_prompt_key=None,
        opd_teacher_image_key=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
        opd_teacher_load="/fake/teacher/ckpt",
        use_eopd=True,
        eopd_entropy_threshold=0.8,
        eopd_fkl_coef=1.0,
        eopd_fkl_top_k=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_eopd_requires_megatron_type(opd_utils_module, monkeypatch, tmp_path):
    args = _base_args(opd_type="sglang", opd_teacher_load=None, opd_teacher_url="http://t/generate")
    monkeypatch.setattr("os.path.exists", lambda p: True)
    with pytest.raises(ValueError, match="megatron"):
        opd_utils_module.validate_opd_args(args, is_sft=False)


def test_eopd_requires_loss_mode(opd_utils_module, monkeypatch):
    args = _base_args(opd_kl_coef=1.0, opd_loss_coef=0.0)
    monkeypatch.setattr("os.path.exists", lambda p: True)
    with pytest.raises(ValueError, match="opd-loss-coef"):
        opd_utils_module.validate_opd_args(args, is_sft=False)


def test_eopd_requires_topk(opd_utils_module, monkeypatch):
    args = _base_args(opd_log_prob_top_k=0, eopd_fkl_top_k=0)
    monkeypatch.setattr("os.path.exists", lambda p: True)
    with pytest.raises(ValueError, match="top-k"):
        opd_utils_module.validate_opd_args(args, is_sft=False)


def test_eopd_valid_config(opd_utils_module, monkeypatch):
    args = _base_args()
    monkeypatch.setattr("os.path.exists", lambda p: True)
    opd_utils_module.validate_opd_args(args, is_sft=False)
