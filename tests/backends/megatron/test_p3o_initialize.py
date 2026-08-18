# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch

from relax.backends.megatron import initialize


def _args(**overrides: object) -> Namespace:
    values = {
        "advantage_estimator": "p3o",
        "batch_invariant_mode": True,
        "deterministic_mode": True,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"deterministic_mode": False}, "deterministic-mode"),
        ({"batch_invariant_mode": False}, "batch-invariant-mode"),
    ],
)
def test_p3o_partition_modes_fail_closed(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        initialize._configure_p3o_partition_invariance(_args(**overrides))


def test_p3o_partition_modes_enable_batch_invariant_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(initialize, "enable_batch_invariant_mode", lambda: calls.append("enabled"))

    initialize._configure_p3o_partition_invariance(_args())

    assert calls == ["enabled"]


def test_p3o_partition_modes_disable_fused_wgrad_accumulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """BIK's TE wrapper must not overwrite ``main_grad`` between micro-
    batches."""
    monkeypatch.setattr(initialize, "enable_batch_invariant_mode", lambda: None)
    args = _args(gradient_accumulation_fusion=True)

    initialize._configure_p3o_partition_invariance(args)

    assert args.gradient_accumulation_fusion is False


def test_p3o_partition_modes_disable_jit_fuser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(initialize, "enable_batch_invariant_mode", lambda: None)
    calls = []
    monkeypatch.setattr(initialize, "_disable_p3o_compiled_fused_cross_entropy", lambda: calls.append("disabled"))
    args = _args(disable_jit_fuser=False)

    initialize._configure_p3o_partition_invariance(args)

    assert args.disable_jit_fuser is True
    assert calls == ["disabled"]


def test_p3o_unwraps_fused_cross_entropy_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    from megatron.core.fusions import fused_cross_entropy

    def original(*args, **kwargs):
        return args, kwargs

    def compiled(*args, **kwargs):
        raise AssertionError((args, kwargs))

    compiled._torchdynamo_orig_callable = original
    for name in (
        "calculate_logits_max",
        "calculate_predicted_logits",
        "calculate_cross_entropy_loss",
        "calculate_gradients",
    ):
        monkeypatch.setattr(fused_cross_entropy, name, compiled)

    initialize._disable_p3o_compiled_fused_cross_entropy()

    assert all(
        getattr(fused_cross_entropy, name) is original
        for name in (
            "calculate_logits_max",
            "calculate_predicted_logits",
            "calculate_cross_entropy_loss",
            "calculate_gradients",
        )
    )


def test_non_p3o_does_not_change_partition_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(initialize, "enable_batch_invariant_mode", lambda: calls.append("enabled"))

    initialize._configure_p3o_partition_invariance(
        _args(advantage_estimator="grpo", batch_invariant_mode=False, deterministic_mode=False)
    )

    assert calls == []


def test_non_p3o_honors_explicit_batch_invariant_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(initialize, "enable_batch_invariant_mode", lambda: calls.append("enabled"))

    initialize._configure_p3o_partition_invariance(
        _args(advantage_estimator="grpo", batch_invariant_mode=True, deterministic_mode=True)
    )

    assert calls == ["enabled"]


def _logical_dp_gradient(data_parallel_size: int) -> torch.Tensor:
    torch.manual_seed(11)
    weight = torch.randn(7, 5, dtype=torch.bfloat16)
    inputs = [torch.randn(length, 5, dtype=torch.bfloat16) for length in (3, 5, 4, 6, 2, 7, 4, 5)]
    targets = [torch.randn(value.shape[0], 7, dtype=torch.bfloat16) for value in inputs]
    rank_gradients = []
    for indices in torch.tensor_split(torch.arange(len(inputs)), data_parallel_size):
        main_grad = torch.zeros_like(weight, dtype=torch.float32)
        for index in indices.tolist():
            parameter = weight.clone().requires_grad_(True)
            output = torch.nn.functional.linear(inputs[index], parameter)
            ((output - targets[index]) ** 2).sum().backward()
            main_grad += parameter.grad.float()
        rank_gradients.append(main_grad)
    return sum(rank_gradients)


def test_p3o_bf16_mbs1_parameter_gradient_matches_across_dp_partitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(initialize, "enable_batch_invariant_mode", lambda: None)
    initialize._configure_p3o_partition_invariance(_args())

    reference = _logical_dp_gradient(1)
    for data_parallel_size in (2, 4):
        candidate = _logical_dp_gradient(data_parallel_size)
        reference64 = reference.double().flatten()
        candidate64 = candidate.double().flatten()
        relative_l2 = torch.linalg.vector_norm(candidate64 - reference64) / torch.linalg.vector_norm(reference64)
        cosine = torch.dot(reference64, candidate64) / (
            torch.linalg.vector_norm(reference64) * torch.linalg.vector_norm(candidate64)
        )
        assert relative_l2 <= 1e-6
        assert cosine >= 1 - 1e-9
