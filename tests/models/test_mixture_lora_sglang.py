# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F
from torch import nn


# The parity assertions need the training backend; the mixin needs SGLang.
pytest.importorskip("megatron.bridge")
pytest.importorskip("sglang.srt.distributed")

from relax.backends.megatron.mixture_lora_modules import MixtureLoRAAdapter  # noqa: E402
from relax.models.mixture_lora_sglang import (  # noqa: E402
    MixtureLoraSGLangModelMixin,
    SGLangMixtureLoRA,
    attach_sglang_mixture_lora,
    load_sglang_mixture_lora_weights,
)
from relax.utils.mixture_lora_common import MixtureLoraConfig  # noqa: E402


def _config(target_modules=("linear_qkv", "linear_proj")):
    return MixtureLoraConfig(
        num_experts=4,
        rank=2,
        top_k=2,
        temperature=0.8,
        aux_loss_coef=0.01,
        alpha=4.0,
        target_modules=target_modules,
    )


class _TupleLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        return F.linear(x, self.weight), None


class _QuantizedTupleLinear(nn.Module):
    def __init__(self, params_dtype):
        super().__init__()
        self.weight_packed = nn.Parameter(torch.ones(4, 4, dtype=torch.int32), requires_grad=False)
        self.params_dtype = params_dtype


class _RoutedAttentionModel(MixtureLoraSGLangModelMixin, nn.Module):
    """Minimal stand-in for an SGLang causal LM with routed attention."""

    def __init__(self, num_layers: int, *, start_layer: int = 0, end_layer: int | None = None):
        nn.Module.__init__(self)
        self.mixture_lora_config = _config()
        self.model = nn.Module()
        self.model.start_layer = start_layer
        self.model.end_layer = num_layers if end_layer is None else end_layer
        layers = []
        for layer_id in range(num_layers):
            layer = nn.Module()
            if start_layer <= layer_id < self.model.end_layer:
                layer.self_attn = nn.Module()
                layer.self_attn.qkv_proj = _TupleLinear(4, 6)
                layer.self_attn.o_proj = _TupleLinear(6, 4)
            layers.append(layer)
        self.model.layers = nn.ModuleList(layers)

    def mixture_lora_site_modules(self, layer_id: int) -> dict[str, nn.Module]:
        attention = self.model.layers[layer_id].self_attn
        return {"linear_qkv": attention.qkv_proj, "linear_proj": attention.o_proj}


def _fake_model_with_routed_qkv():
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
    model = _fake_model_with_routed_qkv()
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


def test_attached_adapter_uses_quantized_linear_params_dtype():
    linear = _QuantizedTupleLinear(torch.bfloat16)

    attach_sglang_mixture_lora(linear, _config(), "decoder.layers.0.self_attention.linear_qkv", 4, 6)

    assert {parameter.dtype for parameter in linear.mixture_lora.parameters()} == {torch.bfloat16}
    assert linear.mixture_lora.experts.lora_A.device == linear.weight_packed.device


def test_attached_adapter_rejects_missing_floating_point_params_dtype():
    linear = _QuantizedTupleLinear(torch.int32)

    with pytest.raises(TypeError, match="floating-point params_dtype"):
        attach_sglang_mixture_lora(linear, _config(), "decoder.layers.0.self_attention.linear_qkv", 4, 6)


def test_sglang_weight_loader_maps_training_names_and_validates_tensors():
    model = _fake_model_with_routed_qkv()
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
    model = _fake_model_with_routed_qkv()
    model.model.start_layer = 0
    model.model.end_layer = 1

    loaded_names = load_sglang_mixture_lora_weights(
        model,
        [("decoder.layers.1.self_attention.linear_qkv.mixture_lora.router.weight", torch.randn(4, 4))],
    )

    assert loaded_names == set()


def test_mixin_installs_adapters_only_on_layers_owned_by_pp_stage():
    model = _RoutedAttentionModel(2, start_layer=1, end_layer=2)

    model.install_mixture_lora()

    assert not hasattr(model.model.layers[0], "self_attn")
    local_attention = model.model.layers[1].self_attn
    assert local_attention.qkv_proj.mixture_lora.site_id == "decoder.layers.1.self_attention.linear_qkv"
    assert local_attention.o_proj.mixture_lora.site_id == "decoder.layers.1.self_attention.linear_proj"


def test_mixin_rejects_targets_the_architecture_does_not_expose():
    model = _RoutedAttentionModel(1)
    model.mixture_lora_config = _config(target_modules=("linear_qkv", "linear_fc1"))

    with pytest.raises(ValueError, match="Unsupported SGLang Mixture-of-LoRA targets"):
        model.install_mixture_lora()


def test_mixin_load_weights_splits_routed_tensors_from_base_weights():
    model = _RoutedAttentionModel(1)
    model.install_mixture_lora()
    base_weights = []

    class _Base:
        def load_weights(self, weights):
            base_weights.extend(weights)
            return "base-done"

    # Stand in for the SGLang base class the mixin cooperates with.
    model.__class__ = type("_Patched", (MixtureLoraSGLangModelMixin, _Base, nn.Module), {})
    prefix = "decoder.layers.0.self_attention.linear_qkv.mixture_lora"
    routed = ("router.weight", torch.randn(4, 4))

    result = model.load_weights(
        [
            ("model.layers.0.self_attn.qkv_proj.weight", torch.randn(6, 4)),
            (f"{prefix}.{routed[0]}", routed[1]),
        ]
    )

    assert result == "base-done"
    assert [name for name, _ in base_weights] == ["model.layers.0.self_attn.qkv_proj.weight"]
    torch.testing.assert_close(
        model.model.layers[0].self_attn.qkv_proj.mixture_lora.router.weight,
        routed[1],
    )


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
