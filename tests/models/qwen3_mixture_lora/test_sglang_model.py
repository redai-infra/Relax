# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F
from torch import nn


# This suite needs the SGLang Qwen3 model definition. It is available in the
# official Relax image but intentionally absent from the default CPU CI.
pytest.importorskip("sglang.srt.models.qwen3")

from relax.models.mixture_lora_sglang import MixtureLoraSGLangModelMixin  # noqa: E402
from relax.models.qwen3_mixture_lora.sglang.model import EntryClass  # noqa: E402
from relax.utils.mixture_lora_common import MixtureLoraConfig  # noqa: E402


def _config():
    return MixtureLoraConfig(
        num_experts=4,
        rank=2,
        top_k=2,
        temperature=0.8,
        aux_loss_coef=0.01,
        alpha=4.0,
        target_modules=("linear_qkv", "linear_proj"),
    )


class _TupleLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        return F.linear(x, self.weight), None


def test_external_entry_class_overrides_the_checkpoint_architecture():
    assert EntryClass.__name__ == "Qwen3ForCausalLM"


def test_qwen3_model_reuses_the_shared_mixture_lora_machinery():
    # Everything except the site mapping below is architecture-independent, so
    # a second architecture only has to supply its own linears.
    assert issubclass(EntryClass, MixtureLoraSGLangModelMixin)
    assert EntryClass.__mro__.index(MixtureLoraSGLangModelMixin) < EntryClass.__mro__.index(nn.Module)
    assert "install_mixture_lora" not in vars(EntryClass)
    assert "load_weights" not in vars(EntryClass)


def test_qwen3_sites_map_to_the_attention_projections():
    model = EntryClass.__new__(EntryClass)
    nn.Module.__init__(model)
    model.mixture_lora_config = _config()
    model.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.qkv_proj = _TupleLinear(4, 6)
    layer.self_attn.o_proj = _TupleLinear(6, 4)
    model.model.layers = nn.ModuleList([layer])

    site_modules = model.mixture_lora_site_modules(0)
    model.install_mixture_lora()

    assert site_modules == {
        "linear_qkv": layer.self_attn.qkv_proj,
        "linear_proj": layer.self_attn.o_proj,
    }
    assert layer.self_attn.qkv_proj.mixture_lora.site_id == "decoder.layers.0.self_attention.linear_qkv"
    assert layer.self_attn.o_proj.mixture_lora.site_id == "decoder.layers.0.self_attention.linear_proj"
