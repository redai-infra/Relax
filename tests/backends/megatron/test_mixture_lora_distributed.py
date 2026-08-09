# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from relax.backends.megatron.mixture_lora import (
    MixtureLoRAAdapter,
    MixtureLoRARoutingContext,
    activate_mixture_lora_routing_context,
    mixture_lora_metrics_from_packed_records,
    pack_mixture_lora_routing_records,
)
from relax.utils.mixture_lora import DenseRoutedLoRAExecutor, MixtureLoraConfig, route_topk


def _config() -> MixtureLoraConfig:
    return MixtureLoraConfig(
        num_experts=3,
        rank=2,
        top_k=2,
        temperature=1.0,
        aux_loss_coef=0.01,
        alpha=4.0,
        target_modules=("linear_qkv",),
    )


def _routing_context(site_id: str, mask: torch.Tensor, *, num_sites: int, objective_scale: float):
    config = _config()
    adapter = MixtureLoRAAdapter(
        config,
        site_id,
        4,
        5,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.no_grad():
        adapter.router.weight.zero_()
    context = MixtureLoRARoutingContext(
        optimizer_step=0,
        microbatch_id=0,
        response_mask=mask,
        num_microbatches=1,
        num_sites=num_sites,
        num_samples=1,
        calculate_per_token_loss=False,
        objective_scale=objective_scale,
        main_loss_backward_scale=torch.ones(1),
    )
    with activate_mixture_lora_routing_context(context):
        adapter(torch.ones(mask.shape[1], mask.shape[0], 4))
    return context


def _assert_uniform_metrics(metrics: dict[str, torch.Tensor], prefix: str) -> None:
    pre_topk_sum = sum(metrics[f"{prefix}/expert_{expert_id}_pre_topk_mean_prob"] for expert_id in range(3))
    post_topk_sum = sum(metrics[f"{prefix}/expert_{expert_id}_post_topk_mean_weight"] for expert_id in range(3))
    selection_sum = sum(metrics[f"{prefix}/expert_{expert_id}_selection_share"] for expert_id in range(3))
    top1_sum = sum(metrics[f"{prefix}/expert_{expert_id}_top1_fraction"] for expert_id in range(3))
    torch.testing.assert_close(pre_topk_sum, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(post_topk_sum, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(selection_sum, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(top1_sum, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(
        metrics[f"{prefix}/pre_topk_normalized_entropy"], torch.tensor(1.0, dtype=torch.float64)
    )


def _distributed_routing_metrics_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        # DP/CP-style reduction: both ranks contribute tokens for the same site.
        local_mask = (
            torch.tensor([[1, 1]], dtype=torch.bool) if rank == 0 else torch.tensor([[1, 0]], dtype=torch.bool)
        )
        context = _routing_context("layers.0.linear_qkv", local_mask, num_sites=1, objective_scale=0.5)
        packed = pack_mixture_lora_routing_records(
            [context],
            ("layers.0.linear_qkv",),
            num_experts=3,
            top_k=2,
            device=torch.device("cpu"),
        )
        dist.all_reduce(packed)
        metrics = mixture_lora_metrics_from_packed_records(
            packed,
            ("layers.0.linear_qkv",),
            num_experts=3,
            top_k=2,
            calculate_per_token_loss=False,
            data_parallel_world_size_with_cp=world_size,
        )
        _assert_uniform_metrics(metrics, "molora/layers.0.linear_qkv")
        torch.testing.assert_close(
            metrics["molora/layers.0.linear_qkv/balance_loss"], torch.tensor(1.0, dtype=torch.float64)
        )
        torch.testing.assert_close(metrics["molora/aux_loss"], torch.tensor(0.005, dtype=torch.float64))

        # PP-style reduction: each rank owns a different site and leaves the
        # other row at zero before the step-end collective.
        site_ids = ("layers.0.linear_qkv", "layers.1.linear_qkv")
        local_site_id = site_ids[rank]
        context = _routing_context(local_site_id, torch.ones(1, 2, dtype=torch.bool), num_sites=2, objective_scale=1.0)
        packed = pack_mixture_lora_routing_records(
            [context],
            site_ids,
            num_experts=3,
            top_k=2,
            device=torch.device("cpu"),
        )
        dist.all_reduce(packed)
        metrics = mixture_lora_metrics_from_packed_records(
            packed,
            site_ids,
            num_experts=3,
            top_k=2,
            calculate_per_token_loss=False,
            data_parallel_world_size_with_cp=1,
        )
        for site_id in site_ids:
            _assert_uniform_metrics(metrics, f"molora/{site_id}")
            torch.testing.assert_close(
                metrics[f"molora/{site_id}/balance_loss"], torch.tensor(1.0, dtype=torch.float64)
            )
        _assert_uniform_metrics(metrics, "molora/global")
        torch.testing.assert_close(metrics["molora/aux_loss"], torch.tensor(0.01, dtype=torch.float64))
    finally:
        dist.destroy_process_group()


def test_routing_metrics_reduce_across_two_real_processes(tmp_path):
    init_method = f"file://{tmp_path / 'mixture-lora-gloo-init'}"
    mp.spawn(_distributed_routing_metrics_worker, args=(2, init_method), nprocs=2, join=True)


def _reference_forward_and_backward(
    config: MixtureLoraConfig,
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    router_weight: torch.Tensor,
):
    reference_x = x.detach().clone().requires_grad_(True)
    reference_a = lora_a.detach().clone().requires_grad_(True)
    reference_b = lora_b.detach().clone().requires_grad_(True)
    reference_router = router_weight.detach().clone().requires_grad_(True)
    logits = F.linear(reference_x.reshape(-1, reference_x.shape[-1]).float(), reference_router.float())
    decision = route_topk(logits, config.top_k, config.temperature)
    output = DenseRoutedLoRAExecutor()(reference_x, reference_a, reference_b, decision, config.scale)
    output.square().sum().backward()
    return reference_x, reference_a, reference_b, reference_router, output.detach()


def _assert_tp_adapter_matches_reference(rank: int, *, input_is_parallel: bool, sequence_parallel: bool) -> None:
    config = MixtureLoraConfig(
        num_experts=4,
        rank=4,
        top_k=2,
        temperature=0.8,
        aux_loss_coef=0.01,
        alpha=4.0,
        target_modules=("linear_proj" if input_is_parallel else "linear_qkv",),
    )
    torch.manual_seed(1234 + int(input_is_parallel) * 10 + int(sequence_parallel))
    device = torch.device("cpu")
    sequence_length, batch_size, input_size, output_size = 4, 2, 6, 8
    full_x = torch.randn(sequence_length, batch_size, input_size, device=device)
    full_a = torch.randn(config.num_experts, config.rank, input_size, device=device)
    full_b = torch.randn(config.num_experts, output_size, config.rank, device=device)
    full_router = torch.randn(config.num_experts, input_size, device=device)
    reference_x, reference_a, reference_b, reference_router, reference_output = _reference_forward_and_backward(
        config,
        full_x,
        full_a,
        full_b,
        full_router,
    )

    adapter = MixtureLoRAAdapter(
        config,
        "linear_proj" if input_is_parallel else "linear_qkv",
        input_size,
        output_size,
        dropout=0.0,
        device=device,
        dtype=torch.float32,
        input_is_parallel=input_is_parallel,
        sequence_parallel=sequence_parallel,
        tp_group=dist.group.WORLD,
        tp_rank=rank,
        tp_world_size=2,
    )
    input_slice = slice(rank * input_size // 2, (rank + 1) * input_size // 2)
    output_slice = slice(rank * output_size // 2, (rank + 1) * output_size // 2)
    rank_slice = slice(rank * config.rank // 2, (rank + 1) * config.rank // 2)
    with torch.no_grad():
        if input_is_parallel:
            adapter.experts.lora_A.copy_(full_a[:, :, input_slice])
            adapter.router.weight.copy_(full_router[:, input_slice])
        else:
            adapter.experts.lora_A.copy_(full_a[:, rank_slice, :])
            adapter.router.weight.copy_(full_router)
        adapter.experts.lora_B.copy_(full_b[:, output_slice, :])

    if input_is_parallel:
        local_x = full_x[:, :, input_slice].detach().clone().requires_grad_(True)
    elif sequence_parallel:
        sequence_slice = slice(rank * sequence_length // 2, (rank + 1) * sequence_length // 2)
        local_x = full_x[sequence_slice].detach().clone().requires_grad_(True)
    else:
        local_x = full_x.detach().clone().requires_grad_(True)

    output = adapter(local_x)
    if input_is_parallel:
        if sequence_parallel:
            sequence_slice = slice(rank * sequence_length // 2, (rank + 1) * sequence_length // 2)
            expected_output = reference_output[sequence_slice]
        else:
            expected_output = reference_output
    else:
        expected_output = reference_output[:, :, output_slice]
    torch.testing.assert_close(output, expected_output, atol=2e-5, rtol=2e-5)
    output.square().sum().backward()

    if input_is_parallel:
        expected_input_grad = reference_x.grad[:, :, input_slice]
        expected_a_grad = reference_a.grad[:, :, input_slice]
        expected_router_grad = reference_router.grad[:, input_slice]
    else:
        expected_input_grad = reference_x.grad[sequence_slice] if sequence_parallel else reference_x.grad
        expected_a_grad = reference_a.grad[:, rank_slice, :]
        expected_router_grad = reference_router.grad
    torch.testing.assert_close(local_x.grad, expected_input_grad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(adapter.experts.lora_A.grad, expected_a_grad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        adapter.experts.lora_B.grad,
        reference_b.grad[:, output_slice, :],
        atol=3e-5,
        rtol=3e-5,
    )
    torch.testing.assert_close(adapter.router.weight.grad, expected_router_grad, atol=3e-5, rtol=3e-5)
    dist.barrier()


def _assert_tp_aux_loss_matches_reference(rank: int, *, input_is_parallel: bool, sequence_parallel: bool) -> None:
    config = MixtureLoraConfig(
        num_experts=4,
        rank=4,
        top_k=2,
        temperature=0.8,
        aux_loss_coef=0.03,
        alpha=4.0,
        target_modules=("linear_proj" if input_is_parallel else "linear_qkv",),
    )
    torch.manual_seed(4321 + int(input_is_parallel))
    input_size, output_size = 6, 8
    full_x = torch.randn(4, 2, input_size)
    full_a = torch.randn(config.num_experts, config.rank, input_size)
    full_b = torch.randn(config.num_experts, output_size, config.rank)
    full_router = torch.randn(config.num_experts, input_size)
    response_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    reference = MixtureLoRAAdapter(
        config,
        "linear_proj" if input_is_parallel else "linear_qkv",
        input_size,
        output_size,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        input_is_parallel=input_is_parallel,
    )
    with torch.no_grad():
        reference.experts.lora_A.copy_(full_a)
        reference.experts.lora_B.copy_(full_b)
        reference.router.weight.copy_(full_router)
    reference_context = MixtureLoRARoutingContext(
        optimizer_step=0,
        microbatch_id=0,
        response_mask=response_mask,
        num_microbatches=1,
        num_sites=1,
        num_samples=2,
        calculate_per_token_loss=False,
        objective_scale=1.0,
        main_loss_backward_scale=torch.ones(1),
    )
    with activate_mixture_lora_routing_context(reference_context):
        reference_output = reference(full_x)
    (reference_output.sum() * 0.0).backward()

    adapter = MixtureLoRAAdapter(
        config,
        "linear_proj" if input_is_parallel else "linear_qkv",
        input_size,
        output_size,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        input_is_parallel=input_is_parallel,
        sequence_parallel=sequence_parallel,
        tp_group=dist.group.WORLD,
        tp_rank=rank,
        tp_world_size=2,
    )
    input_slice = slice(rank * input_size // 2, (rank + 1) * input_size // 2)
    output_slice = slice(rank * output_size // 2, (rank + 1) * output_size // 2)
    rank_slice = slice(rank * config.rank // 2, (rank + 1) * config.rank // 2)
    with torch.no_grad():
        if input_is_parallel:
            adapter.experts.lora_A.copy_(full_a[:, :, input_slice])
            adapter.router.weight.copy_(full_router[:, input_slice])
        else:
            adapter.experts.lora_A.copy_(full_a[:, rank_slice, :])
            adapter.router.weight.copy_(full_router)
        adapter.experts.lora_B.copy_(full_b[:, output_slice, :])
    if input_is_parallel:
        local_x = full_x[:, :, input_slice]
    elif sequence_parallel:
        sequence_slice = slice(rank * full_x.shape[0] // 2, (rank + 1) * full_x.shape[0] // 2)
        local_x = full_x[sequence_slice]
    else:
        local_x = full_x
    context = MixtureLoRARoutingContext(
        optimizer_step=0,
        microbatch_id=0,
        response_mask=response_mask,
        num_microbatches=1,
        num_sites=1,
        num_samples=2,
        calculate_per_token_loss=False,
        objective_scale=1.0,
        main_loss_backward_scale=torch.ones(1),
    )
    with activate_mixture_lora_routing_context(context):
        output = adapter(local_x)
    (output.sum() * 0.0).backward()

    expected_router_grad = (
        reference.router.weight.grad[:, input_slice] if input_is_parallel else reference.router.weight.grad
    )
    torch.testing.assert_close(adapter.router.weight.grad, expected_router_grad, atol=3e-5, rtol=3e-5)
    assert (adapter.site_id in context.records) == (rank == 0)
    dist.barrier()


def _tensor_parallel_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        for input_is_parallel in (False, True):
            for sequence_parallel in (False, True):
                _assert_tp_adapter_matches_reference(
                    rank,
                    input_is_parallel=input_is_parallel,
                    sequence_parallel=sequence_parallel,
                )
                _assert_tp_aux_loss_matches_reference(
                    rank,
                    input_is_parallel=input_is_parallel,
                    sequence_parallel=sequence_parallel,
                )
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_forward_and_backward_match_single_rank_reference(tmp_path):
    init_method = f"file://{tmp_path / 'mixture-lora-tp-gloo-init'}"
    mp.spawn(_tensor_parallel_worker, args=(2, init_method), nprocs=2, join=True)
