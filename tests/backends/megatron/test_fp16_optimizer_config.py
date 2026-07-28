# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.backends.megatron import model  # noqa: E402


def test_build_optimizer_config_kwargs_does_not_rewrite_precision_values():
    args = Namespace(
        fp16=True,
        bf16=False,
        params_dtype=torch.float16,
        initial_loss_scale=65536.0,
        min_loss_scale=2.0,
        use_precision_aware_optimizer=False,
        store_param_remainders=True,
    )

    kwargs = model._build_optimizer_config_kwargs(args)

    assert kwargs["bf16"] is False
    assert kwargs["fp16"] is True
    assert kwargs["params_dtype"] is torch.float16
    assert kwargs["initial_loss_scale"] == 65536.0
    assert kwargs["min_loss_scale"] == 2.0
    assert kwargs["use_precision_aware_optimizer"] is False
    assert kwargs["store_param_remainders"] is True


def test_build_optimizer_config_kwargs_builds_real_fp16_optimizer_config():
    args = Namespace(
        fp16=True,
        bf16=False,
        params_dtype=torch.float16,
        initial_loss_scale=65536.0,
        min_loss_scale=2.0,
        use_precision_aware_optimizer=True,
        store_param_remainders=True,
        use_distributed_optimizer=True,
    )

    kwargs = model._build_optimizer_config_kwargs(args)
    config = model.OptimizerConfig(**kwargs)

    assert config.fp16 is True
    assert config.bf16 is False
    assert config.params_dtype is torch.float16
    assert config.initial_loss_scale == 65536.0
    assert config.min_loss_scale == 2.0
    assert config.use_precision_aware_optimizer is True
    assert config.store_param_remainders is True


def test_build_optimizer_config_kwargs_builds_real_bf16_optimizer_config():
    args = Namespace(
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        initial_loss_scale=131072.0,
        min_loss_scale=4.0,
        use_precision_aware_optimizer=True,
        store_param_remainders=True,
        use_distributed_optimizer=True,
    )

    kwargs = model._build_optimizer_config_kwargs(args)
    config = model.OptimizerConfig(**kwargs)

    assert config.bf16 is True
    assert config.fp16 is False
    assert config.params_dtype is torch.bfloat16
    assert config.initial_loss_scale == 131072.0
    assert config.min_loss_scale == 4.0
    assert config.use_precision_aware_optimizer is True
    assert config.store_param_remainders is True


def test_setup_model_and_optimizer_preserves_built_precision_config(monkeypatch):
    expected_kwargs = {
        "fp16": True,
        "bf16": False,
        "params_dtype": torch.float16,
        "initial_loss_scale": 65536.0,
        "min_loss_scale": 2.0,
        "use_precision_aware_optimizer": False,
        "store_param_remainders": True,
    }
    captured_kwargs = {}

    class CapturingOptimizerConfig:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.timers = object()

    monkeypatch.setattr(model, "get_model_provider_func", lambda _args, _role: "provider")
    monkeypatch.setattr(model, "wrap_model_provider_with_freeze", lambda provider, _args: provider)
    monkeypatch.setattr(model, "get_model", lambda *_args, **_kwargs: ["model"])
    monkeypatch.setattr(model, "ModelType", SimpleNamespace(encoder_or_decoder=object()))
    monkeypatch.setattr(model, "_build_optimizer_config_kwargs", lambda _args: expected_kwargs.copy())
    monkeypatch.setattr(model, "OptimizerConfig", CapturingOptimizerConfig)
    monkeypatch.setattr(model, "get_megatron_optimizer", lambda **_kwargs: "optimizer")
    monkeypatch.setattr(model, "get_optimizer_param_scheduler", lambda _args, _optimizer: "scheduler")

    args = Namespace(
        moe_use_upcycling=False,
        load="/checkpoint",
        pretrained_checkpoint=None,
        dynamic_context_parallel=False,
        context_parallel_size=1,
        only_load_weight=False,
        fp16=True,
        use_gloo_process_groups=False,
    )

    result = model.setup_model_and_optimizer(args)

    assert captured_kwargs == expected_kwargs
    assert result == (["model"], "optimizer", "scheduler")
