# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import torch

from relax.backends.megatron import model


def test_build_optimizer_config_kwargs_preserves_explicit_fp16_optimizer_values():
    args = Namespace(
        fp16=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        initial_loss_scale=65536.0,
        min_loss_scale=2.0,
        use_precision_aware_optimizer=False,
        store_param_remainders=False,
    )

    kwargs = model._build_optimizer_config_kwargs(args)

    assert kwargs["bf16"] is False
    assert kwargs["fp16"] is True
    assert kwargs["params_dtype"] is torch.float16
    assert kwargs["initial_loss_scale"] == 65536.0
    assert kwargs["min_loss_scale"] == 2.0
    assert kwargs["use_precision_aware_optimizer"] is False
    assert kwargs["store_param_remainders"] is False


def test_build_optimizer_config_kwargs_keeps_bf16_values_unchanged():
    args = Namespace(
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        initial_loss_scale=131072.0,
        min_loss_scale=4.0,
        use_precision_aware_optimizer=True,
        store_param_remainders=True,
    )

    kwargs = model._build_optimizer_config_kwargs(args)

    assert kwargs["bf16"] is True
    assert kwargs["fp16"] is False
    assert kwargs["params_dtype"] is torch.bfloat16
    assert kwargs["initial_loss_scale"] == 131072.0
    assert kwargs["min_loss_scale"] == 4.0
    assert kwargs["use_precision_aware_optimizer"] is True
    assert kwargs["store_param_remainders"] is True
