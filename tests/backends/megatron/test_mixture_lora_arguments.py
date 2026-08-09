# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.backends.megatron.arguments import _hf_validate_args


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
