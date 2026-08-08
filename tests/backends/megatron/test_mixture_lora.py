# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.backends.megatron.mixture_lora import (
    MixtureLoRAAdapter,
    MixtureParallelLinearAdapter,
    build_mixture_lora_peft,
)
from relax.utils.mixture_lora import MixtureLoraConfig


def _config(*, num_experts=3, top_k=2, rank=2, alpha=4.0):
    return MixtureLoraConfig(
        num_experts=num_experts,
        rank=rank,
        top_k=top_k,
        temperature=0.7,
        aux_loss_coef=0.01,
        alpha=alpha,
        target_modules=("linear_qkv", "linear_proj"),
    )


class _TupleLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, *, return_mode="standard"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size))
        self.bias = torch.nn.Parameter(torch.randn(output_size))
        self.return_mode = return_mode

    def forward(self, x):
        output = F.linear(x, self.weight)
        if self.return_mode == "standard":
            return output, self.bias
        if self.return_mode == "layernorm":
            return (output, x + 1.0), self.bias
        if self.return_mode == "three":
            return output, self.bias, x + 1.0
        raise AssertionError(f"unknown return mode: {self.return_mode}")


@pytest.mark.parametrize(("num_experts", "top_k"), [(3, 2), (3, 3)])
def test_mixture_lora_forward_matches_expert_reference(num_experts, top_k):
    torch.manual_seed(7)
    config = _config(num_experts=num_experts, top_k=top_k)
    adapter = MixtureLoRAAdapter(
        config,
        "decoder.layers.0.self_attention.linear_qkv",
        4,
        5,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.no_grad():
        adapter.experts.lora_A.copy_(torch.randn_like(adapter.experts.lora_A))
        adapter.experts.lora_B.copy_(torch.randn_like(adapter.experts.lora_B))
        adapter.router.weight.copy_(torch.randn_like(adapter.router.weight))

    x = torch.randn(2, 3, 4)
    actual, decision = adapter.forward_with_routing(x)
    x_flat = x.reshape(-1, x.shape[-1])
    dense_weights = decision.dense_weights()
    expected_tokens = []
    for token, token_weights in zip(x_flat, dense_weights, strict=True):
        expert_sum = torch.zeros(5)
        for expert_index, expert_weight in enumerate(token_weights):
            hidden = adapter.experts.lora_A[expert_index] @ token
            expert_output = adapter.experts.lora_B[expert_index] @ hidden
            expert_sum += expert_weight * expert_output
        expected_tokens.append(expert_sum * config.scale)
    expected = torch.stack(expected_tokens).reshape(2, 3, 5)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixture_lora_router_normalizes_in_fp32(dtype):
    adapter = MixtureLoRAAdapter(
        _config(),
        "linear_qkv",
        4,
        5,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    decision = adapter.route(torch.randn(2, 3, 4, dtype=dtype))

    assert decision.pre_topk_probs.dtype == torch.float32
    assert decision.post_topk_weights.dtype == torch.float32


def test_mixture_lora_parameter_layout_and_initialization():
    config = _config(num_experts=4, rank=3)
    adapter = MixtureLoRAAdapter(
        config,
        "linear_proj",
        6,
        7,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert adapter.experts.lora_A.shape == (4, 3, 6)
    assert adapter.experts.lora_B.shape == (4, 7, 3)
    assert adapter.router.weight.shape == (4, 6)
    assert torch.count_nonzero(adapter.experts.lora_A) > 0
    assert torch.count_nonzero(adapter.router.weight) > 0
    assert torch.count_nonzero(adapter.experts.lora_B) == 0
    assert list(adapter.executor.parameters()) == []
    assert adapter.executor.state_dict() == {}


def test_mixture_lora_wrapper_freezes_base_and_routes_gradients():
    torch.manual_seed(11)
    base = _TupleLinear(4, 5)
    for parameter in base.parameters():
        parameter.requires_grad = False
    wrapper = MixtureParallelLinearAdapter(base, _config(), "linear_qkv", 4, 5, dropout=0.0)
    with torch.no_grad():
        wrapper.mixture_lora.experts.lora_B.normal_(mean=0.0, std=0.2)

    output, _ = wrapper(torch.randn(3, 2, 4))
    output.square().mean().backward()

    assert all(parameter.grad is None for parameter in base.parameters())
    assert wrapper.mixture_lora.experts.lora_A.grad is not None
    assert wrapper.mixture_lora.experts.lora_B.grad is not None
    assert wrapper.mixture_lora.router.weight.grad is not None
    assert torch.count_nonzero(wrapper.mixture_lora.router.weight.grad) > 0


@pytest.mark.parametrize("return_mode", ["standard", "layernorm", "three"])
def test_mixture_lora_wrapper_preserves_linear_return_protocol(return_mode):
    base = _TupleLinear(4, 5, return_mode=return_mode)
    wrapper = MixtureParallelLinearAdapter(base, _config(), "linear_qkv", 4, 5, dropout=0.0)
    x = torch.randn(2, 3, 4)

    base_output, base_bias, _ = wrapper.base_linear_forward(x)
    output, bias = wrapper(x)

    assert output.shape == base_output.shape == (2, 3, 5)
    assert bias is base_bias
    torch.testing.assert_close(output, base_output)

    wrapper.disable_adapter_layers()
    disabled_output, disabled_bias = wrapper(x)
    torch.testing.assert_close(disabled_output, base_output)
    assert disabled_bias is base_bias


def test_mixture_lora_wrapper_keeps_base_state_keys_stable():
    wrapper = MixtureParallelLinearAdapter(_TupleLinear(4, 5), _config(), "linear_qkv", 4, 5, dropout=0.0)

    state = wrapper.state_dict()

    assert set(state) == {
        "weight",
        "bias",
        "mixture_lora.experts.lora_A",
        "mixture_lora.experts.lora_B",
        "mixture_lora.router.weight",
    }


def _install_fake_bridge(monkeypatch):
    megatron = types.ModuleType("megatron")
    bridge = types.ModuleType("megatron.bridge")
    peft_package = types.ModuleType("megatron.bridge.peft")
    base_module = types.ModuleType("megatron.bridge.peft.base")
    matcher_module = types.ModuleType("megatron.bridge.peft.module_matcher")
    utils_module = types.ModuleType("megatron.bridge.peft.utils")
    core_module = types.ModuleType("megatron.core")

    @dataclass
    class FakePEFT:
        params_to_save: set[str] = field(default_factory=set, init=False)

        def __call__(self, model, training=True):
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.linear_qkv = self.transform(model.linear_qkv, "linear_qkv", "decoder.layers.0.self_attention")
            return model

    @dataclass
    class FakeModuleMatcher:
        target_modules: list[str] = field(default_factory=list)

        def match(self, module, name=None, prefix=None):
            if name not in self.target_modules:
                return None
            return name, f"{prefix}.{name}" if prefix else name

    base_module.PEFT = FakePEFT
    matcher_module.ModuleMatcher = FakeModuleMatcher
    utils_module.get_adapter_attributes_from_linear = lambda module: SimpleNamespace(
        in_features=module.weight.shape[1], out_features=module.weight.shape[0]
    )
    core_module.parallel_state = SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 1)

    modules = {
        "megatron": megatron,
        "megatron.bridge": bridge,
        "megatron.bridge.peft": peft_package,
        "megatron.bridge.peft.base": base_module,
        "megatron.bridge.peft.module_matcher": matcher_module,
        "megatron.bridge.peft.utils": utils_module,
        "megatron.core": core_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_mixture_lora_peft_uses_bridge_matcher_and_freezes_base(monkeypatch):
    _install_fake_bridge(monkeypatch)
    config = _config()
    peft = build_mixture_lora_peft(config, dropout=0.0)
    model = torch.nn.Module()
    model.linear_qkv = _TupleLinear(4, 5)

    transformed = peft(model, training=True)

    assert isinstance(transformed.linear_qkv, MixtureParallelLinearAdapter)
    assert transformed.linear_qkv.mixture_lora.site_id == "decoder.layers.0.self_attention.linear_qkv"
    assert all(not parameter.requires_grad for parameter in transformed.linear_qkv.to_wrap.parameters())
    assert all(parameter.requires_grad for parameter in transformed.linear_qkv.mixture_lora.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the forward profiler check")
def test_mixture_lora_cuda_forward_backward_has_no_tensor_to_host_sync():
    adapter = MixtureLoRAAdapter(
        _config(),
        "linear_qkv",
        16,
        24,
        dropout=0.0,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    with torch.no_grad():
        adapter.experts.lora_B.normal_(mean=0.0, std=0.02)
    x = torch.randn(8, 4, 16, device="cuda", dtype=torch.float16)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        output = adapter(x)
        output.float().square().mean().backward()
    torch.cuda.synchronize()

    event_names = {event.key for event in profile.key_averages()}
    assert "aten::_local_scalar_dense" not in event_names
    assert output.shape == (8, 4, 24)
    assert adapter.experts.lora_A.grad is not None
    assert adapter.experts.lora_B.grad is not None
    assert adapter.router.weight.grad is not None
