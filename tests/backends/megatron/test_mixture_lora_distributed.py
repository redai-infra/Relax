# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.megatron.mixture_lora import (
    MixtureLoRAAdapter,
    MixtureLoRARoutingContext,
    activate_mixture_lora_routing_context,
    mixture_lora_metrics_from_packed_records,
    pack_mixture_lora_routing_records,
)
from relax.utils.mixture_lora import MixtureLoraConfig


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
        local_mask = torch.tensor([[1, 1]], dtype=torch.bool) if rank == 0 else torch.tensor([[1, 0]], dtype=torch.bool)
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
