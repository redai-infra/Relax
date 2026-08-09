# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math

import pytest
import torch

from relax.utils.mixture_lora import (
    MIXTURE_LORA_SCHEMA_VERSION,
    DenseRoutedLoRAExecutor,
    MixtureLoraConfig,
    MixtureLoraStateSpec,
    RoutedLoRAExecutor,
    RoutedLoRAParallelContext,
    RoutingDecision,
    TransportTensorSpec,
    build_mixture_lora_state_specs,
    compute_routing_statistics,
    deserialize_mixture_lora_config,
    mean_routing_balance_loss,
    megatron_mixture_lora_name_to_sglang,
    route_topk,
    serialize_mixture_lora_config,
)


def _normalized_entropy(values):
    if len(values) <= 1:
        return 0.0
    return -sum(value * math.log(value) for value in values if value > 0) / math.log(len(values))


def _config(**overrides):
    values = {
        "num_experts": 4,
        "rank": 2,
        "top_k": 2,
        "temperature": 1.0,
        "aux_loss_coef": 0.01,
        "alpha": 4.0,
        "target_modules": ("linear_qkv", "linear_proj"),
    }
    values.update(overrides)
    return MixtureLoraConfig(**values)


def test_runtime_config_json_round_trip_is_canonical():
    config = _config()

    serialized = serialize_mixture_lora_config(config)

    assert deserialize_mixture_lora_config(serialized) == config
    assert serialize_mixture_lora_config(deserialize_mixture_lora_config(serialized)) == serialized


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "non-empty JSON string"),
        ("[]", "must contain an object"),
        ('{"num_experts":4}', "fields do not match schema"),
        ("not-json", "Invalid Mixture-of-LoRA runtime configuration JSON"),
    ],
)
def test_runtime_config_json_rejects_incomplete_or_invalid_input(raw, message):
    with pytest.raises(ValueError, match=message):
        deserialize_mixture_lora_config(raw)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "decoder.layers.0.self_attention.linear_qkv.mixture_lora.experts.lora_A",
            "model.layers.0.self_attn.qkv_proj.mixture_lora.experts.lora_A",
        ),
        (
            "decoder.layers.17.self_attention.linear_qkv.mixture_lora.experts.lora_B",
            "model.layers.17.self_attn.qkv_proj.mixture_lora.experts.lora_B",
        ),
        (
            "decoder.layers.2.self_attention.linear_proj.mixture_lora.router.weight",
            "model.layers.2.self_attn.o_proj.mixture_lora.router.weight",
        ),
    ],
)
def test_megatron_mixture_parameter_name_maps_to_sglang(source, expected):
    assert megatron_mixture_lora_name_to_sglang(source) == expected


@pytest.mark.parametrize(
    "name",
    [
        "decoder.layers.x.self_attention.linear_qkv.mixture_lora.router.weight",
        "decoder.layers.0.mlp.linear_fc1.mixture_lora.router.weight",
        "decoder.layers.0.self_attention.linear_qkv.mixture_lora.unknown",
    ],
)
def test_megatron_mixture_parameter_name_rejects_unsupported_schema(name):
    with pytest.raises(ValueError):
        megatron_mixture_lora_name_to_sglang(name)


def _reference_routed_lora(x, lora_a, lora_b, decision, scale):
    outputs = []
    for token, indices, weights in zip(x.reshape(-1, x.shape[-1]), decision.topk_indices, decision.post_topk_weights):
        delta = torch.zeros(lora_b.shape[1], dtype=x.dtype)
        for expert_index, weight in zip(indices, weights):
            expert_output = lora_b[expert_index] @ (lora_a[expert_index] @ token)
            delta = delta + weight.to(x.dtype) * expert_output
        outputs.append(delta * scale)
    return torch.stack(outputs).reshape(*x.shape[:-1], lora_b.shape[1])


def test_mixture_lora_config_exposes_scale_and_normalizes_targets():
    config = _config(target_modules=["linear_qkv", "linear_proj"])

    assert config.schema_version == MIXTURE_LORA_SCHEMA_VERSION
    assert config.target_modules == ("linear_qkv", "linear_proj")
    assert config.scale == 2.0


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"num_experts": 1}, ValueError),
        ({"rank": 0}, ValueError),
        ({"top_k": 0}, ValueError),
        ({"top_k": 5}, ValueError),
        ({"temperature": 0.0}, ValueError),
        ({"aux_loss_coef": -0.1}, ValueError),
        ({"alpha": float("inf")}, ValueError),
        ({"target_modules": ()}, ValueError),
        ({"target_modules": "linear_qkv"}, TypeError),
        ({"target_modules": ("linear_qkv", "linear_qkv")}, ValueError),
    ],
)
def test_mixture_lora_config_rejects_invalid_values(overrides, error):
    with pytest.raises(error):
        _config(**overrides)


def test_state_specs_fix_parameter_names_and_global_shapes():
    site_id = "decoder.layers.0.self_attention.linear_qkv"

    specs = build_mixture_lora_state_specs(_config(), site_id, input_size=8, output_size=12, dtype=torch.bfloat16)

    assert [spec.parameter_name for spec in specs] == [
        f"{site_id}.mixture_lora.experts.lora_A",
        f"{site_id}.mixture_lora.experts.lora_B",
        f"{site_id}.mixture_lora.router.weight",
    ]
    assert [spec.global_shape for spec in specs] == [(4, 2, 8), (4, 12, 2), (4, 8)]
    assert all(spec.dtype == torch.bfloat16 for spec in specs)

    transport = TransportTensorSpec(specs[1], tp_shard_dim=1, tp_rank=1, tp_world_size=2)
    assert transport.parameter_name == specs[1].parameter_name
    assert transport.schema_version == MIXTURE_LORA_SCHEMA_VERSION
    assert transport.site_id == site_id
    assert transport.parameter_kind == "experts.lora_B"
    assert transport.global_shape == (4, 12, 2)
    assert transport.dtype == torch.bfloat16


def test_state_and_transport_specs_reject_invalid_layouts():
    with pytest.raises(ValueError, match="global_shape"):
        MixtureLoraStateSpec(1, "site", "router.weight", (4, 8, 2), torch.float32)
    with pytest.raises(TypeError, match="floating-point"):
        MixtureLoraStateSpec(1, "site", "router.weight", (4, 8), torch.int64)

    state = MixtureLoraStateSpec(1, "site", "router.weight", (4, 8), torch.float32)
    with pytest.raises(ValueError, match="tp_rank"):
        TransportTensorSpec(state, tp_shard_dim=None, tp_rank=2, tp_world_size=2)
    with pytest.raises(ValueError, match="tp_shard_dim"):
        TransportTensorSpec(state, tp_shard_dim=2, tp_rank=0, tp_world_size=2)


def test_route_topk_returns_fp32_normalized_weights():
    logits = torch.tensor([[4.0, 1.0, 3.0, 2.0], [0.0, 5.0, 2.0, 1.0]], dtype=torch.bfloat16)

    decision = route_topk(logits, top_k=2, temperature=1.0)

    assert decision.pre_topk_probs.shape == (2, 4)
    assert decision.topk_indices.shape == (2, 2)
    assert decision.post_topk_weights.shape == (2, 2)
    assert decision.pre_topk_probs.dtype == torch.float32
    assert decision.post_topk_weights.dtype == torch.float32
    assert decision.topk_indices.dtype == torch.long
    assert torch.equal(decision.topk_indices, torch.tensor([[0, 2], [1, 2]]))
    assert torch.allclose(decision.post_topk_weights.sum(dim=-1), torch.ones(2))

    dense_weights = decision.dense_weights()
    assert torch.allclose(dense_weights.sum(dim=-1), torch.ones(2))
    assert torch.equal(dense_weights == 0, torch.tensor([[False, True, False, True], [True, False, False, True]]))


def test_route_topk_temperature_changes_distribution():
    logits = torch.tensor([[3.0, 2.0, 1.0]])

    cold = route_topk(logits, top_k=2, temperature=0.5)
    warm = route_topk(logits, top_k=2, temperature=2.0)

    assert cold.pre_topk_probs.max() > warm.pre_topk_probs.max()
    assert cold.post_topk_weights.max() > warm.post_topk_weights.max()


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_balance_loss_uniform_baseline_does_not_depend_on_top_k(top_k):
    decision = route_topk(torch.zeros(8, 4), top_k=top_k, temperature=1.0)
    stats = compute_routing_statistics(decision, torch.ones(8, dtype=torch.bool))

    assert stats.balance_loss == pytest.approx(1.0)
    assert stats.selection_share.sum() == pytest.approx(1.0)


def test_routing_statistics_use_only_response_tokens():
    pre_topk_probs = torch.tensor(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
        ]
    )
    topk_indices = torch.tensor([[0, 1], [1, 2], [2, 1]])
    post_topk_weights = torch.tensor([[2 / 3, 1 / 3], [5 / 8, 3 / 8], [7 / 9, 2 / 9]])
    decision = RoutingDecision(pre_topk_probs, topk_indices, post_topk_weights)

    stats = compute_routing_statistics(decision, torch.tensor([1, 0, 1], dtype=torch.bool))

    assert stats.valid_token_count == 2
    assert torch.allclose(stats.pre_topk_mean_prob, torch.tensor([0.35, 0.25, 0.40]))
    assert torch.allclose(stats.post_topk_mean_weight, torch.tensor([1 / 3, 5 / 18, 7 / 18]))
    assert torch.allclose(stats.selection_share, torch.tensor([0.25, 0.50, 0.25]))
    assert torch.allclose(stats.top1_fraction, torch.tensor([0.50, 0.0, 0.50]))
    assert stats.balance_loss == pytest.approx(0.9375)

    expected_pre_entropy = (_normalized_entropy([0.6, 0.3, 0.1]) + _normalized_entropy([0.1, 0.2, 0.7])) / 2
    expected_post_entropy = (_normalized_entropy([2 / 3, 1 / 3]) + _normalized_entropy([7 / 9, 2 / 9])) / 2
    assert stats.pre_topk_normalized_entropy == pytest.approx(expected_pre_entropy)
    assert stats.post_topk_normalized_entropy == pytest.approx(expected_post_entropy)


def test_balance_loss_backpropagates_through_pre_topk_probabilities():
    logits = torch.tensor([[3.0, 2.0, 1.0], [2.0, 0.0, 1.0]], requires_grad=True)
    decision = route_topk(logits, top_k=2, temperature=1.0)

    compute_routing_statistics(decision, torch.ones(2, dtype=torch.bool)).balance_loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_balance_loss_is_averaged_across_sites_after_site_local_statistics():
    first_logits = torch.tensor([[3.0, 2.0, 1.0], [2.0, 0.0, 1.0]], requires_grad=True)
    second_logits = torch.tensor([[1.0, 3.0, 2.0], [0.0, 2.0, 3.0]], requires_grad=True)
    response_mask = torch.ones(2, dtype=torch.bool)
    first = compute_routing_statistics(route_topk(first_logits, 2, 1.0), response_mask)
    second = compute_routing_statistics(route_topk(second_logits, 2, 1.0), response_mask)

    loss = mean_routing_balance_loss((first, second))

    torch.testing.assert_close(loss, (first.balance_loss + second.balance_loss) / 2)
    loss.backward()
    assert torch.count_nonzero(first_logits.grad) > 0
    assert torch.count_nonzero(second_logits.grad) > 0


def test_balance_loss_requires_at_least_one_site():
    with pytest.raises(ValueError, match="at least one"):
        mean_routing_balance_loss(())


def test_k_one_has_zero_post_topk_entropy():
    decision = route_topk(torch.tensor([[2.0, 1.0], [0.0, 3.0]]), top_k=1, temperature=1.0)
    stats = compute_routing_statistics(decision, torch.ones(2, dtype=torch.bool))

    assert stats.post_topk_normalized_entropy == 0


def test_no_response_tokens_returns_zero_statistics_and_loss():
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0]], requires_grad=True)
    decision = route_topk(logits, top_k=1, temperature=1.0)
    stats = compute_routing_statistics(decision, torch.zeros(2, dtype=torch.bool))

    assert stats.valid_token_count == 0
    assert torch.count_nonzero(stats.pre_topk_mean_prob) == 0
    assert torch.count_nonzero(stats.post_topk_mean_weight) == 0
    assert stats.balance_loss == 0

    stats.balance_loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 1e-6), (torch.float16, 1e-2), (torch.bfloat16, 2e-2)],
)
def test_dense_executor_matches_independent_expert_loop(dtype, atol):
    torch.manual_seed(7)
    x = torch.randn(2, 3, 5, dtype=dtype)
    lora_a = torch.randn(4, 2, 5, dtype=dtype)
    lora_b = torch.randn(4, 7, 2, dtype=dtype)
    decision = route_topk(torch.randn(6, 4), top_k=2, temperature=0.7)
    original_indices = decision.topk_indices.clone()
    original_weights = decision.post_topk_weights.clone()
    executor = DenseRoutedLoRAExecutor()
    context = RoutedLoRAParallelContext(target_module="linear_qkv", sequence_parallel=False)

    actual = executor.execute(x, lora_a, lora_b, decision, scale=0.5, parallel_context=context)
    expected = _reference_routed_lora(x, lora_a, lora_b, decision, scale=0.5)

    assert isinstance(executor, RoutedLoRAExecutor)
    assert executor.state_dict() == {}
    assert list(executor.parameters()) == []
    assert actual.shape == (2, 3, 7)
    assert actual.dtype == dtype
    assert torch.allclose(actual.float(), expected.float(), atol=atol)
    assert torch.equal(decision.topk_indices, original_indices)
    assert torch.equal(decision.post_topk_weights, original_weights)


def test_dense_executor_backpropagates_to_experts_input_and_router():
    torch.manual_seed(11)
    x = torch.randn(4, 5, requires_grad=True)
    lora_a = torch.randn(3, 2, 5, requires_grad=True)
    lora_b = torch.randn(3, 7, 2, requires_grad=True)
    logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [2.0, 1.0, 3.0], [3.0, 1.0, 2.0]], requires_grad=True)
    decision = route_topk(logits, top_k=2, temperature=1.0)

    DenseRoutedLoRAExecutor()(x, lora_a, lora_b, decision, scale=1.0).square().mean().backward()

    for tensor in (x, lora_a, lora_b, logits):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


def test_dense_executor_is_repeatable_with_fixed_seed():
    def run_once():
        torch.manual_seed(19)
        x = torch.randn(3, 4)
        lora_a = torch.randn(4, 2, 4)
        lora_b = torch.randn(4, 6, 2)
        decision = route_topk(torch.randn(3, 4), top_k=2, temperature=1.0)
        return DenseRoutedLoRAExecutor()(x, lora_a, lora_b, decision, scale=0.5)

    assert torch.equal(run_once(), run_once())


def test_dense_executor_rejects_inconsistent_inputs_and_implicit_collectives():
    x = torch.randn(2, 5)
    lora_a = torch.randn(4, 2, 5)
    lora_b = torch.randn(4, 7, 2)
    decision = route_topk(torch.randn(2, 4), top_k=2, temperature=1.0)
    executor = DenseRoutedLoRAExecutor()

    with pytest.raises(ValueError, match="hidden size"):
        executor(x[:, :4], lora_a, lora_b, decision, scale=1.0)
    with pytest.raises(ValueError, match="tokens"):
        executor(x.repeat(2, 1), lora_a, lora_b, decision, scale=1.0)
    with pytest.raises(NotImplementedError, match="tensor-parallel"):
        executor(
            x,
            lora_a,
            lora_b,
            decision,
            scale=1.0,
            parallel_context=RoutedLoRAParallelContext("linear_proj", tensor_parallel_group=object()),
        )


@pytest.mark.parametrize(
    ("top_k", "temperature", "error"),
    [
        (0, 1.0, ValueError),
        (4, 1.0, ValueError),
        (1, 0.0, ValueError),
        (1, float("inf"), ValueError),
        (True, 1.0, ValueError),
        (1, True, TypeError),
    ],
)
def test_route_topk_rejects_invalid_configuration(top_k, temperature, error):
    with pytest.raises(error):
        route_topk(torch.ones(2, 3), top_k=top_k, temperature=temperature)


def test_route_topk_rejects_invalid_logits():
    with pytest.raises(ValueError, match="shape"):
        route_topk(torch.ones(2, 3, 4), top_k=2, temperature=1.0)
    with pytest.raises(TypeError, match="floating point"):
        route_topk(torch.ones(2, 3, dtype=torch.long), top_k=2, temperature=1.0)


def test_statistics_reject_mask_with_wrong_shape():
    decision = route_topk(torch.ones(2, 3), top_k=2, temperature=1.0)

    with pytest.raises(ValueError, match="response_mask"):
        compute_routing_statistics(decision, torch.ones(2, 1))
