# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest


# The default CPU CI does not install Megatron. The validation tests run in
# the official Relax image with the real Megatron argument definitions.
pytest.importorskip("megatron.training.arguments")

from relax.backends.megatron.arguments import _hf_validate_args  # noqa: E402


def test_mixture_lora_rejects_multimodal_hf_config():
    args = SimpleNamespace(lora_num_experts=4, apply_rope_fusion=False)
    hf_config = SimpleNamespace(text_config=SimpleNamespace())

    with pytest.raises(AssertionError, match="text-only base models"):
        _hf_validate_args(args, hf_config)


def test_mixture_lora_rejects_moe_hf_config():
    args = SimpleNamespace(lora_num_experts=4)
    hf_config = SimpleNamespace(num_experts=8)

    with pytest.raises(AssertionError, match="dense base models"):
        _hf_validate_args(args, hf_config)


def test_mixture_lora_rejects_mtp_layers():
    args = SimpleNamespace(lora_num_experts=4, mtp_num_layers=1)

    with pytest.raises(AssertionError, match="does not currently support MTP"):
        _hf_validate_args(args, SimpleNamespace())


def test_mixture_lora_accepts_disabled_mtp():
    args = SimpleNamespace(lora_num_experts=4, mtp_num_layers=None)

    _hf_validate_args(args, SimpleNamespace())
