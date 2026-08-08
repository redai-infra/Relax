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
    MixtureLoRARoutingContext,
    MixtureParallelLinearAdapter,
    activate_mixture_lora_routing_context,
    build_mixture_lora_peft,
    ensure_mixture_lora_recompute_inputs_grad,
    get_microbatch_objective_scale,
    get_mixture_lora_routing_context,
    install_mixture_lora_checkpoint_context,
)
from relax.utils.mixture_lora import MixtureLoraConfig, compute_routing_statistics


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


@pytest.mark.parametrize(
    ("calculate_per_token_loss", "is_dummy", "explicit_loss_scale", "expected"),
    [
        (False, False, None, 0.125),
        (False, False, 0.5, 0.125),
        (True, False, None, 1.0),
        (False, True, None, 0.0),
        (True, True, None, 0.0),
    ],
)
def test_microbatch_objective_scale_matches_training_modes(
    calculate_per_token_loss, is_dummy, explicit_loss_scale, expected
):
    scale = get_microbatch_objective_scale(
        calculate_per_token_loss=calculate_per_token_loss,
        is_dummy=is_dummy,
        explicit_loss_scale=explicit_loss_scale,
        num_microbatches=4,
        global_batch_size=32,
        data_parallel_world_size_with_cp=4,
    )

    assert scale == expected


def test_routing_context_aligns_batch_first_response_mask_and_records_site():
    adapter = MixtureLoRAAdapter(
        _config(),
        "decoder.layers.0.self_attention.linear_qkv",
        4,
        5,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    response_mask = torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.bool)
    context = MixtureLoRARoutingContext(
        optimizer_step=7,
        microbatch_id=2,
        response_mask=response_mask,
        num_microbatches=4,
        num_sites=2,
        num_samples=2,
        calculate_per_token_loss=False,
        objective_scale=0.25,
        main_loss_backward_scale=torch.ones(1),
    )
    x = torch.randn(3, 2, 4)

    with activate_mixture_lora_routing_context(context):
        _, decision = adapter.forward_with_routing(x)

    record = context.records[adapter.site_id]
    expected_statistics = compute_routing_statistics(decision, response_mask.transpose(0, 1).reshape(-1))
    assert record.key == (7, 2, adapter.site_id)
    torch.testing.assert_close(record.statistics.valid_token_count, torch.tensor(3.0))
    torch.testing.assert_close(record.balance_loss, expected_statistics.balance_loss)
    torch.testing.assert_close(
        record.aux_loss,
        expected_statistics.balance_loss * adapter.config.aux_loss_coef / 2 * 2 * 0.25,
    )
    assert get_mixture_lora_routing_context() is None


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_routing_context_attaches_expected_router_gradient(calculate_per_token_loss):
    adapter = MixtureLoRAAdapter(
        _config(),
        "linear_qkv",
        4,
        5,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    response_mask = torch.tensor([[0, 1, 1], [1, 1, 0]], dtype=torch.bool)
    objective_scale = 1.0 if calculate_per_token_loss else 0.25
    context = MixtureLoRARoutingContext(
        optimizer_step=0,
        microbatch_id=0,
        response_mask=response_mask,
        num_microbatches=2,
        num_sites=1,
        num_samples=2,
        calculate_per_token_loss=calculate_per_token_loss,
        objective_scale=objective_scale,
        main_loss_backward_scale=torch.ones(1),
    )
    x = torch.randn(3, 2, 4)

    with activate_mixture_lora_routing_context(context):
        output, decision = adapter.forward_with_routing(x)
    statistics = compute_routing_statistics(decision, response_mask.transpose(0, 1).reshape(-1))
    count_multiplier = statistics.valid_token_count if calculate_per_token_loss else context.num_samples
    expected_objective = statistics.balance_loss * adapter.config.aux_loss_coef * count_multiplier * objective_scale
    expected_router_grad = torch.autograd.grad(expected_objective, adapter.router.weight, retain_graph=True)[0]

    output.sum().backward()

    torch.testing.assert_close(adapter.router.weight.grad, expected_router_grad)


def test_dummy_routing_context_records_zero_aux_loss():
    adapter = MixtureLoRAAdapter(
        _config(),
        "linear_qkv",
        4,
        5,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    context = MixtureLoRARoutingContext(
        optimizer_step=0,
        microbatch_id=1,
        response_mask=torch.ones(1, 3),
        num_microbatches=2,
        num_sites=1,
        num_samples=1,
        calculate_per_token_loss=False,
        objective_scale=0.0,
        main_loss_backward_scale=torch.ones(1),
        is_dummy=True,
    )

    with activate_mixture_lora_routing_context(context):
        adapter(torch.randn(3, 1, 4))

    record = context.records[adapter.site_id]
    torch.testing.assert_close(record.statistics.valid_token_count, torch.tensor(0.0))
    torch.testing.assert_close(record.balance_loss, torch.tensor(0.0))
    torch.testing.assert_close(record.aux_loss, torch.tensor(0.0))


def test_checkpoint_wrapper_restores_captured_routing_context(monkeypatch):
    megatron_module = types.ModuleType("megatron")
    core_module = types.ModuleType("megatron.core")
    tensor_parallel_module = types.ModuleType("megatron.core.tensor_parallel")
    captured_functions = []

    def checkpoint(function, distribute_saved_activations, *args):
        captured_functions.append(function)
        return function(*args)

    tensor_parallel_module.checkpoint = checkpoint
    core_module.tensor_parallel = tensor_parallel_module
    monkeypatch.setitem(sys.modules, "megatron", megatron_module)
    monkeypatch.setitem(sys.modules, "megatron.core", core_module)
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tensor_parallel_module)
    install_mixture_lora_checkpoint_context()
    context = MixtureLoRARoutingContext(
        optimizer_step=0,
        microbatch_id=0,
        response_mask=torch.ones(1, 1),
        num_microbatches=1,
        num_sites=1,
        num_samples=1,
        calculate_per_token_loss=False,
        objective_scale=1.0,
        main_loss_backward_scale=torch.ones(1),
    )
    observed_contexts = []

    def run(x):
        observed_contexts.append(get_mixture_lora_routing_context())
        return x

    with activate_mixture_lora_routing_context(context):
        tensor_parallel_module.checkpoint(run, False, torch.ones(1))
    captured_function = captured_functions[0]
    observed_contexts.clear()

    captured_function(torch.ones(1))

    assert observed_contexts == [context]


def test_recompute_input_grad_patch_recognizes_mixture_adapter(monkeypatch):
    transformer_block_module = types.ModuleType("megatron.core.transformer.transformer_block")
    utils_module = types.ModuleType("megatron.core.utils")

    class FakeTransformerBlock(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    transformer_block_module.TransformerBlock = FakeTransformerBlock
    utils_module.unwrap_model = lambda model: model
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", types.ModuleType("megatron.core.transformer"))
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.transformer_block", transformer_block_module)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils_module)
    model = torch.nn.Module()
    model.config = SimpleNamespace(recompute_method="uniform")
    model.block = FakeTransformerBlock()

    ensure_mixture_lora_recompute_inputs_grad(model)
    patched_forward = model.block.forward
    output = model.block(torch.ones(2, 3))
    ensure_mixture_lora_recompute_inputs_grad(model)

    assert output.requires_grad
    assert model.block.forward is patched_forward


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


def test_mixture_lora_peft_instantiates_with_real_bridge_when_available():
    pytest.importorskip("megatron.bridge.peft.base")

    peft = build_mixture_lora_peft(_config(), dropout=0.0)

    assert peft.target_modules == ["linear_qkv", "linear_proj"]
    assert peft.mixture_config == _config()


def test_mixture_lora_peft_wraps_real_column_parallel_linear_when_available(tmp_path):
    pytest.importorskip("megatron.bridge.peft.base")
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig

    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the single-process distributed state")
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'distributed_init'}",
        rank=0,
        world_size=1,
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=4,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )
        model = torch.nn.Module()
        model.linear_qkv = ColumnParallelLinear(
            4,
            5,
            config=transformer_config,
            init_method=lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=0.02),
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )

        transformed = build_mixture_lora_peft(_config(), dropout=0.0)(model, training=True)
        routing_context = MixtureLoRARoutingContext(
            optimizer_step=0,
            microbatch_id=0,
            response_mask=torch.tensor([[0, 1, 1]], dtype=torch.bool),
            num_microbatches=1,
            num_sites=1,
            num_samples=1,
            calculate_per_token_loss=False,
            objective_scale=1.0,
            main_loss_backward_scale=torch.ones(1),
        )
        with activate_mixture_lora_routing_context(routing_context):
            output, bias = transformed.linear_qkv(torch.randn(3, 1, 4))
        output.sum().backward()

        assert isinstance(transformed.linear_qkv, MixtureParallelLinearAdapter)
        assert output.shape == (3, 1, 5)
        assert bias is None
        assert all(not parameter.requires_grad for parameter in transformed.linear_qkv.to_wrap.parameters())
        assert all(parameter.requires_grad for parameter in transformed.linear_qkv.mixture_lora.parameters())
        assert all(parameter.grad is None for parameter in transformed.linear_qkv.to_wrap.parameters())
        assert torch.count_nonzero(transformed.linear_qkv.mixture_lora.router.weight.grad) > 0
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


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
