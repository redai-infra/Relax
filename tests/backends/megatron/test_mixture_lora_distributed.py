# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from relax.backends.megatron.mixture_lora import (
    MixtureLoRAAdapter,
    MixtureLoRARoutingContext,
    activate_mixture_lora_routing_context,
    mixture_lora_metrics_from_packed_records,
    pack_mixture_lora_routing_records,
)
from relax.utils.mixture_lora import (
    DenseRoutedLoRAExecutor,
    MixtureLoraConfig,
    compute_routing_statistics,
    route_topk,
)


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
        activation_layout="bshd",
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


def _tensor_parallel_routing_metrics_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    from megatron.core import parallel_state

    from relax.backends.megatron.model import _reduce_mixture_lora_routing_metrics

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        site_ids = ("layers.0.linear_qkv",)
        contexts = (
            [_routing_context(site_ids[0], torch.ones(1, 2, dtype=torch.bool), num_sites=1, objective_scale=1.0)]
            if rank == 0
            else []
        )
        metrics = _reduce_mixture_lora_routing_metrics(
            SimpleNamespace(calculate_per_token_loss=False),
            contexts,
            (site_ids, 3, 2),
            torch.device("cpu"),
        )
        _assert_uniform_metrics(metrics, f"molora/{site_ids[0]}")
        torch.testing.assert_close(metrics["molora/aux_loss"], torch.tensor(0.01, dtype=torch.float64))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_routing_metrics_are_available_on_every_tensor_parallel_rank(tmp_path):
    # The spawned workers need the real Megatron tensor-parallel state, which the
    # default CPU CI does not install. This case runs in the official Relax image.
    pytest.importorskip("megatron.core")
    pytest.importorskip("megatron.training")
    init_method = f"file://{tmp_path / 'mixture-lora-tp-metrics-gloo-init'}"
    mp.spawn(_tensor_parallel_routing_metrics_worker, args=(2, init_method), nprocs=2, join=True)


def _new_distributed_adapter(config: MixtureLoraConfig, site_id: str) -> MixtureLoRAAdapter:
    return MixtureLoRAAdapter(
        config,
        site_id,
        4,
        4,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _set_distributed_adapter_weights(adapter: MixtureLoRAAdapter, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        adapter.experts.lora_A.copy_(torch.randn(adapter.experts.lora_A.shape, generator=generator))
        adapter.experts.lora_B.copy_(torch.randn(adapter.experts.lora_B.shape, generator=generator))
        adapter.router.weight.copy_(torch.randn(adapter.router.weight.shape, generator=generator))


def _distributed_microbatch(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(6100 + rank)
    x = torch.randn(3, 1, 4, generator=generator)
    mask = torch.tensor([[1, 1, rank == 0]], dtype=torch.bool)
    return x, mask


def _new_routing_context(
    site_ids: tuple[str, ...],
    mask: torch.Tensor,
    *,
    microbatch_id: int,
    main_loss_backward_scale: float,
) -> MixtureLoRARoutingContext:
    return MixtureLoRARoutingContext(
        optimizer_step=0,
        microbatch_id=microbatch_id,
        response_mask=mask,
        num_microbatches=1,
        num_sites=len(site_ids),
        num_samples=mask.shape[0],
        calculate_per_token_loss=False,
        objective_scale=1.0,
        main_loss_backward_scale=torch.tensor([main_loss_backward_scale]),
        activation_layout="bshd",
    )


def _routing_metrics_from_contexts(
    contexts: list[MixtureLoRARoutingContext],
    site_ids: tuple[str, ...],
    config: MixtureLoraConfig,
    *,
    data_parallel_world_size: int,
) -> dict[str, torch.Tensor]:
    packed = pack_mixture_lora_routing_records(
        contexts,
        site_ids,
        num_experts=config.num_experts,
        top_k=config.top_k,
        device=torch.device("cpu"),
    )
    return mixture_lora_metrics_from_packed_records(
        packed,
        site_ids,
        num_experts=config.num_experts,
        top_k=config.top_k,
        calculate_per_token_loss=False,
        data_parallel_world_size_with_cp=data_parallel_world_size,
    )


def _assert_metric_dict_close(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-8, rtol=1e-8, msg=name)


def _data_parallel_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        config = _config()
        site_ids = ("layers.0.linear_qkv",)
        adapter = _new_distributed_adapter(config, site_ids[0])
        _set_distributed_adapter_weights(adapter, 6200)
        distributed_adapter = DistributedDataParallel(adapter)

        local_x, local_mask = _distributed_microbatch(rank)
        local_context = _new_routing_context(
            site_ids,
            local_mask,
            microbatch_id=rank,
            main_loss_backward_scale=1.0,
        )
        with activate_mixture_lora_routing_context(local_context):
            local_output = distributed_adapter(local_x)
        local_output.square().mean().backward()

        reference = _new_distributed_adapter(config, site_ids[0])
        _set_distributed_adapter_weights(reference, 6200)
        reference_contexts = []
        for microbatch_rank in range(world_size):
            reference_x, reference_mask = _distributed_microbatch(microbatch_rank)
            reference_context = _new_routing_context(
                site_ids,
                reference_mask,
                microbatch_id=microbatch_rank,
                main_loss_backward_scale=1.0 / world_size,
            )
            with activate_mixture_lora_routing_context(reference_context):
                reference_output = reference(reference_x)
            (reference_output.square().mean() / world_size).backward()
            reference_contexts.append(reference_context)

        for distributed_param, reference_param in zip(
            distributed_adapter.module.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(distributed_param.grad, reference_param.grad, atol=2e-6, rtol=2e-6)

        packed = pack_mixture_lora_routing_records(
            [local_context],
            site_ids,
            num_experts=config.num_experts,
            top_k=config.top_k,
            device=torch.device("cpu"),
        )
        dist.all_reduce(packed)
        actual_metrics = mixture_lora_metrics_from_packed_records(
            packed,
            site_ids,
            num_experts=config.num_experts,
            top_k=config.top_k,
            calculate_per_token_loss=False,
            data_parallel_world_size_with_cp=world_size,
        )
        expected_metrics = _routing_metrics_from_contexts(
            reference_contexts,
            site_ids,
            config,
            data_parallel_world_size=world_size,
        )
        _assert_metric_dict_close(actual_metrics, expected_metrics)
    finally:
        dist.destroy_process_group()


def test_data_parallel_gradients_and_metrics_match_microbatch_reference(tmp_path):
    init_method = f"file://{tmp_path / 'mixture-lora-dp-gloo-init'}"
    mp.spawn(_data_parallel_worker, args=(2, init_method), nprocs=2, join=True)


def _pipeline_parallel_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        config = _config()
        site_ids = ("layers.0.linear_qkv", "layers.1.linear_qkv")
        x, mask = _distributed_microbatch(0)

        reference_adapters = [_new_distributed_adapter(config, site_id) for site_id in site_ids]
        for stage, reference_adapter in enumerate(reference_adapters):
            _set_distributed_adapter_weights(reference_adapter, 6300 + stage)
        reference_context = _new_routing_context(
            site_ids,
            mask,
            microbatch_id=0,
            main_loss_backward_scale=1.0,
        )
        with activate_mixture_lora_routing_context(reference_context):
            reference_hidden = reference_adapters[0](x)
            reference_output = reference_adapters[1](reference_hidden)
        reference_output.square().mean().backward()

        local_adapter = _new_distributed_adapter(config, site_ids[rank])
        _set_distributed_adapter_weights(local_adapter, 6300 + rank)
        local_context = _new_routing_context(
            site_ids,
            mask,
            microbatch_id=0,
            main_loss_backward_scale=1.0,
        )
        if rank == 0:
            with activate_mixture_lora_routing_context(local_context):
                local_hidden = local_adapter(x)
            torch.testing.assert_close(local_hidden, reference_hidden.detach())
            dist.send(local_hidden.detach(), dst=1)
            hidden_gradient = torch.empty_like(local_hidden)
            dist.recv(hidden_gradient, src=1)
            local_hidden.backward(hidden_gradient)
        else:
            local_hidden = torch.empty_like(reference_hidden)
            dist.recv(local_hidden, src=0)
            local_hidden.requires_grad_(True)
            with activate_mixture_lora_routing_context(local_context):
                local_output = local_adapter(local_hidden)
            torch.testing.assert_close(local_output, reference_output.detach())
            local_output.square().mean().backward()
            dist.send(local_hidden.grad, dst=0)

        for local_param, reference_param in zip(
            local_adapter.parameters(), reference_adapters[rank].parameters(), strict=True
        ):
            torch.testing.assert_close(local_param.grad, reference_param.grad, atol=2e-6, rtol=2e-6)
        assert local_adapter.router.weight.grad.norm() > 0

        packed = pack_mixture_lora_routing_records(
            [local_context],
            site_ids,
            num_experts=config.num_experts,
            top_k=config.top_k,
            device=torch.device("cpu"),
        )
        dist.all_reduce(packed)
        actual_metrics = mixture_lora_metrics_from_packed_records(
            packed,
            site_ids,
            num_experts=config.num_experts,
            top_k=config.top_k,
            calculate_per_token_loss=False,
            data_parallel_world_size_with_cp=1,
        )
        expected_metrics = _routing_metrics_from_contexts(
            [reference_context],
            site_ids,
            config,
            data_parallel_world_size=1,
        )
        _assert_metric_dict_close(actual_metrics, expected_metrics)
    finally:
        dist.destroy_process_group()


def test_pipeline_parallel_stage_gradients_and_metrics_match_reference(tmp_path):
    init_method = f"file://{tmp_path / 'mixture-lora-pp-gloo-init'}"
    mp.spawn(_pipeline_parallel_worker, args=(2, init_method), nprocs=2, join=True)


def _context_parallel_balance_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        config = _config()
        full_logits = torch.tensor(
            [
                [2.0, 0.5, -1.0],
                [0.2, 1.7, -0.4],
                [-0.3, 0.1, 2.1],
                [1.2, -0.7, 0.4],
            ]
        )
        for calculate_per_token_loss in (False, True):
            for empty_second_rank in (False, True):
                full_mask = torch.tensor(
                    [1, 1, 0, 0] if empty_second_rank else [1, 1, 1, 0],
                    dtype=torch.bool,
                )
                local_logits = full_logits.chunk(world_size)[rank].clone().requires_grad_(True)
                local_mask = full_mask.chunk(world_size)[rank]
                decision = route_topk(local_logits, config.top_k, config.temperature)
                objective_scale = 1.0 if calculate_per_token_loss else float(world_size)
                context = MixtureLoRARoutingContext(
                    optimizer_step=0,
                    microbatch_id=2 * int(calculate_per_token_loss) + int(empty_second_rank),
                    response_mask=local_mask.unsqueeze(0),
                    num_microbatches=1,
                    num_sites=1,
                    num_samples=1,
                    calculate_per_token_loss=calculate_per_token_loss,
                    objective_scale=objective_scale,
                    main_loss_backward_scale=torch.ones(1),
                    activation_layout="bshd",
                    context_parallel_group=dist.group.WORLD,
                    context_parallel_world_size=world_size,
                )
                output = local_logits.sum() * 0.0 + torch.zeros(2, 1, 1)
                attached = context.attach_aux_loss(
                    output,
                    torch.zeros(2, 1, 1),
                    "layers.0.linear_qkv",
                    config,
                    decision,
                )
                attached.sum().backward()

                reference_logits = full_logits.clone().requires_grad_(True)
                reference_decision = route_topk(reference_logits, config.top_k, config.temperature)
                reference_balance_loss = compute_routing_statistics(reference_decision, full_mask).balance_loss
                objective_count = full_mask.sum() if calculate_per_token_loss else 1
                (reference_balance_loss * config.aux_loss_coef * objective_count * objective_scale).backward()
                expected_gradient = reference_logits.grad.chunk(world_size)[rank]
                torch.testing.assert_close(local_logits.grad, expected_gradient, atol=1e-6, rtol=1e-6)

                record = context.records["layers.0.linear_qkv"]
                torch.testing.assert_close(record.balance_loss, reference_balance_loss.detach())
                torch.testing.assert_close(
                    record.aux_loss,
                    reference_balance_loss.detach()
                    * config.aux_loss_coef
                    * objective_count
                    * objective_scale
                    / world_size,
                )
                packed = pack_mixture_lora_routing_records(
                    [context],
                    ("layers.0.linear_qkv",),
                    num_experts=config.num_experts,
                    top_k=config.top_k,
                    device=torch.device("cpu"),
                )
                dist.all_reduce(packed)
                metrics = mixture_lora_metrics_from_packed_records(
                    packed,
                    ("layers.0.linear_qkv",),
                    num_experts=config.num_experts,
                    top_k=config.top_k,
                    calculate_per_token_loss=calculate_per_token_loss,
                    data_parallel_world_size_with_cp=world_size,
                )
                torch.testing.assert_close(
                    metrics["molora/aux_loss"],
                    (reference_balance_loss.detach() * config.aux_loss_coef).to(torch.float64),
                )
                dist.barrier()
    finally:
        dist.destroy_process_group()


def test_context_parallel_balance_loss_matches_full_sequence_reference(tmp_path):
    init_method = f"file://{tmp_path / 'mixture-lora-cp-gloo-init'}"
    mp.spawn(_context_parallel_balance_worker, args=(2, init_method), nprocs=2, join=True)


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
        activation_layout="bshd",
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
        activation_layout="bshd",
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
    pytest.importorskip("megatron.core.tensor_parallel.layers")
    init_method = f"file://{tmp_path / 'mixture-lora-tp-gloo-init'}"
    mp.spawn(_tensor_parallel_worker, args=(2, init_method), nprocs=2, join=True)


def _tensor_parallel_initialization_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(1234)
        config = MixtureLoraConfig(
            num_experts=2,
            rank=4,
            top_k=1,
            temperature=1.0,
            aux_loss_coef=0.01,
            alpha=4.0,
            target_modules=("linear_qkv",),
        )
        adapter = MixtureLoRAAdapter(
            config,
            "linear_qkv",
            input_size=8,
            output_size=8,
            dropout=0.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
            tp_group=dist.group.WORLD,
            tp_rank=rank,
            tp_world_size=world_size,
            use_cpu_initialization=True,
        )
        shards = [torch.empty_like(adapter.experts.lora_A) for _ in range(world_size)]
        dist.all_gather(shards, adapter.experts.lora_A)
        full_a = torch.cat(shards, dim=1)
        assert full_a.shape == (config.num_experts, config.rank, 8)
        assert not torch.equal(shards[0], shards[1])
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_lora_a_initialization_uses_distinct_rank_blocks(tmp_path):
    pytest.importorskip("megatron.core.tensor_parallel.layers")
    init_method = f"file://{tmp_path / 'mixture-lora-tp-init-gloo'}"
    mp.spawn(_tensor_parallel_initialization_worker, args=(2, init_method), nprocs=2, join=True)


def _tensor_parallel_cuda_initialization_worker(rank: int, world_size: int, init_method: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    try:
        model_parallel_cuda_manual_seed(
            1234,
            tp_rank=rank,
            ep_rank=0,
            etp_rank=0,
            force_reset_rng=True,
        )
        config = MixtureLoraConfig(
            num_experts=2,
            rank=4,
            top_k=1,
            temperature=1.0,
            aux_loss_coef=0.01,
            alpha=4.0,
            target_modules=("linear_qkv",),
        )
        adapter = MixtureLoRAAdapter(
            config,
            "linear_qkv",
            input_size=8,
            output_size=8,
            dropout=0.0,
            device=torch.device("cuda", rank),
            dtype=torch.float32,
            tp_group=dist.group.WORLD,
            tp_rank=rank,
            tp_world_size=world_size,
            use_cpu_initialization=False,
        )
        shards = [torch.empty_like(adapter.experts.lora_A) for _ in range(world_size)]
        dist.all_gather(shards, adapter.experts.lora_A)
        assert not torch.equal(shards[0], shards[1])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_tensor_parallel_cuda_lora_a_initialization_uses_model_parallel_rng(tmp_path):
    pytest.importorskip("megatron.core.tensor_parallel.layers")
    init_method = f"file://{tmp_path / 'mixture-lora-tp-init-nccl'}"
    mp.spawn(_tensor_parallel_cuda_initialization_worker, args=(2, init_method), nprocs=2, join=True)
