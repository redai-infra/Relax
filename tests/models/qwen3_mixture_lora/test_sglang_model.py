# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F
from torch import nn


# This parity suite needs both model backends. They are available in the
# official Relax image but intentionally absent from the default CPU CI.
pytest.importorskip("megatron.bridge")
pytest.importorskip("sglang.srt.models.qwen3")

from relax.backends.megatron.mixture_lora import MixtureLoRAAdapter  # noqa: E402
from relax.models.qwen3_mixture_lora.sglang.model import (  # noqa: E402
    EntryClass,
    SGLangMixtureLoRA,
    attach_sglang_mixture_lora,
    load_sglang_mixture_lora_weights,
)
from relax.utils.mixture_lora import MixtureLoraConfig  # noqa: E402


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


def test_external_entry_class_overrides_the_checkpoint_architecture():
    assert EntryClass.__name__ == "Qwen3ForCausalLM"


class _TupleLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))

    def forward(self, x):
        return F.linear(x, self.weight), None


def _fake_qwen_model_with_routed_qkv():
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.qkv_proj = _TupleLinear(4, 6)
    model.model.layers = nn.ModuleList([layer])
    attach_sglang_mixture_lora(
        layer.self_attn.qkv_proj,
        _config(),
        "decoder.layers.0.self_attention.linear_qkv",
        4,
        6,
    )
    return model


def test_sglang_dense_adapter_matches_training_adapter():
    config = _config()
    training_adapter = MixtureLoRAAdapter(
        config,
        "linear_qkv",
        4,
        6,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    rollout_adapter = SGLangMixtureLoRA(
        config,
        "decoder.layers.0.self_attention.linear_qkv",
        4,
        6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.no_grad():
        training_adapter.experts.lora_B.normal_(mean=0.0, std=0.2)
        rollout_adapter.experts.lora_A.copy_(training_adapter.experts.lora_A)
        rollout_adapter.experts.lora_B.copy_(training_adapter.experts.lora_B)
        rollout_adapter.router.weight.copy_(training_adapter.router.weight)
    x = torch.randn(5, 4)

    training_output, training_decision = training_adapter.forward_with_routing(x)
    rollout_output, rollout_decision = rollout_adapter.forward_with_routing(x)

    torch.testing.assert_close(rollout_output, training_output)
    torch.testing.assert_close(rollout_decision.pre_topk_probs, training_decision.pre_topk_probs)
    torch.testing.assert_close(rollout_decision.post_topk_weights, training_decision.post_topk_weights)
    assert torch.equal(rollout_decision.topk_indices, training_decision.topk_indices)


def test_sglang_router_uses_fp32_logits_with_bfloat16_parameters():
    adapter = SGLangMixtureLoRA(
        _config(),
        "decoder.layers.0.self_attention.linear_qkv",
        4,
        6,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    decision = adapter.route(torch.randn(5, 4, dtype=torch.bfloat16))

    assert decision.pre_topk_probs.dtype == torch.float32
    assert decision.post_topk_weights.dtype == torch.float32


def test_attached_adapter_preserves_base_parameter_name_and_adds_delta():
    model = _fake_qwen_model_with_routed_qkv()
    linear = model.model.layers[0].self_attn.qkv_proj
    x = torch.randn(3, 4)
    base_output = F.linear(x, linear.weight)
    with torch.no_grad():
        linear.mixture_lora.experts.lora_B.normal_(mean=0.0, std=0.2)

    output, bias = linear(x)

    assert bias is None
    assert output.shape == base_output.shape
    assert not torch.equal(output, base_output)
    parameter_names = set(dict(model.named_parameters()))
    assert "model.layers.0.self_attn.qkv_proj.weight" in parameter_names
    assert "model.layers.0.self_attn.qkv_proj.mixture_lora.router.weight" in parameter_names


def test_sglang_weight_loader_maps_training_names_and_validates_tensors():
    model = _fake_qwen_model_with_routed_qkv()
    prefix = "decoder.layers.0.self_attention.linear_qkv.mixture_lora"
    weights = {
        f"{prefix}.experts.lora_A": torch.randn(4, 2, 4),
        f"{prefix}.experts.lora_B": torch.randn(4, 6, 2),
        f"{prefix}.router.weight": torch.randn(4, 4),
    }

    loaded_names = load_sglang_mixture_lora_weights(model, weights.items())

    assert loaded_names == {
        "model.layers.0.self_attn.qkv_proj.mixture_lora.experts.lora_A",
        "model.layers.0.self_attn.qkv_proj.mixture_lora.experts.lora_B",
        "model.layers.0.self_attn.qkv_proj.mixture_lora.router.weight",
    }
    parameters = dict(model.named_parameters())
    for source_name, loaded_weight in weights.items():
        target_name = source_name.replace(
            "decoder.layers.0.self_attention.linear_qkv",
            "model.layers.0.self_attn.qkv_proj",
        )
        torch.testing.assert_close(parameters[target_name], loaded_weight)

    with pytest.raises(ValueError, match="has shape"):
        load_sglang_mixture_lora_weights(model, [(f"{prefix}.router.weight", torch.randn(4, 5))])
    with pytest.raises(TypeError, match="has dtype"):
        load_sglang_mixture_lora_weights(
            model,
            [(f"{prefix}.router.weight", torch.randn(4, 4, dtype=torch.float64))],
        )
    with pytest.raises(ValueError, match="Unknown Mixture-of-LoRA weight"):
        load_sglang_mixture_lora_weights(
            model,
            [("decoder.layers.1.self_attention.linear_qkv.mixture_lora.router.weight", torch.randn(4, 4))],
        )


def test_sglang_weight_loader_skips_layers_owned_by_another_pp_stage():
    model = _fake_qwen_model_with_routed_qkv()
    model.model.start_layer = 0
    model.model.end_layer = 1

    loaded_names = load_sglang_mixture_lora_weights(
        model,
        [("decoder.layers.1.self_attention.linear_qkv.mixture_lora.router.weight", torch.randn(4, 4))],
    )

    assert loaded_names == set()


def test_sglang_installs_adapters_only_on_layers_owned_by_pp_stage():
    model = nn.Module()
    model.mixture_lora_config = _config()
    model.model = nn.Module()
    model.model.start_layer = 1
    model.model.end_layer = 2

    missing_layer = nn.Module()
    local_layer = nn.Module()
    local_layer.self_attn = nn.Module()
    local_layer.self_attn.qkv_proj = _TupleLinear(4, 6)
    local_layer.self_attn.qkv_proj.input_size = 4
    local_layer.self_attn.qkv_proj.output_size = 6
    local_layer.self_attn.o_proj = _TupleLinear(6, 4)
    local_layer.self_attn.o_proj.input_size = 6
    local_layer.self_attn.o_proj.output_size = 4
    model.model.layers = nn.ModuleList([missing_layer, local_layer])

    EntryClass._install_mixture_lora(model)

    assert not hasattr(missing_layer, "self_attn")
    assert local_layer.self_attn.qkv_proj.mixture_lora.site_id == (
        "decoder.layers.1.self_attention.linear_qkv"
    )
    assert local_layer.self_attn.o_proj.mixture_lora.site_id == "decoder.layers.1.self_attention.linear_proj"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sglang_adapter_can_be_captured_by_cuda_graph():
    adapter = SGLangMixtureLoRA(
        _config(),
        "decoder.layers.0.self_attention.linear_qkv",
        16,
        24,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            adapter(x)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = adapter(x)
    graph.replay()

    assert captured_output.shape == (8, 24)
    assert torch.isfinite(captured_output).all()
