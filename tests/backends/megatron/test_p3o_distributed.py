# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real Gloo checks for P3O stats and objective synchronization."""

from __future__ import annotations

import math
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.megatron import p3o_step
from relax.backends.megatron.p3o_step import synchronize_p3o_stats
from relax.utils.training.p3o_utils import (
    P3OSufficientStats,
    compute_p3o_sufficient_stats,
    compute_p3o_token_terms,
    finalize_p3o_step_context,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _nonfinite_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    try:
        p3o_step.mpu.is_pipeline_last_stage = lambda ignore_virtual=True: True
        p3o_step.mpu.get_data_parallel_group = lambda with_context_parallel=True: dist.group.WORLD
        p3o_step.mpu.get_pipeline_model_parallel_world_size = lambda: 1

        stats = (
            P3OSufficientStats.zeros()
            if rank == 0
            else P3OSufficientStats.from_vector(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
        )
        invalid_count = torch.tensor(float(rank == 0), dtype=torch.float64)

        try:
            synchronize_p3o_stats(stats, invalid_count)
        except ValueError as error:
            assert "non-finite importance ratio" in str(error)
        else:
            raise AssertionError("every rank must fail after the synchronized invalid flag")

        healthy = torch.ones((), dtype=torch.float64)
        dist.all_reduce(healthy)
        assert healthy.item() == world_size
    finally:
        dist.destroy_process_group()


def _pipeline_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    try:
        dp_groups = [dist.new_group([dp_rank]) for dp_rank in range(world_size)]
        p3o_step.mpu.is_pipeline_last_stage = lambda ignore_virtual=True: rank == world_size - 1
        p3o_step.mpu.get_data_parallel_group = lambda with_context_parallel=True: dp_groups[rank]
        p3o_step.mpu.get_pipeline_model_parallel_world_size = lambda: world_size
        p3o_step.mpu.get_pipeline_model_parallel_group = lambda: dist.group.WORLD

        expected = torch.tensor([7.5, 21.25, 4.0], dtype=torch.float64)
        stats = P3OSufficientStats.from_vector(expected) if rank == world_size - 1 else P3OSufficientStats.zeros()

        synchronized = synchronize_p3o_stats(stats, torch.zeros((), dtype=torch.float64))

        torch.testing.assert_close(synchronized.as_vector(), expected, rtol=0.0, atol=0.0)
    finally:
        dist.destroy_process_group()


def _partition_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    try:
        singleton_groups = [dist.new_group([group_rank]) for group_rank in range(world_size)]
        dp2_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        active_group = [dist.group.WORLD]
        p3o_step.mpu.is_pipeline_last_stage = lambda ignore_virtual=True: True
        p3o_step.mpu.get_data_parallel_group = lambda with_context_parallel=True: active_group[0]
        p3o_step.mpu.get_pipeline_model_parallel_world_size = lambda: 1

        behavior = torch.full((11,), -2.0)
        ratios = (1.0, 2.0, 0.5, 4.0, 0.8, 1.4, 0.25, 3.0, 1.1, 0.6, 2.5)
        log_probs_value = behavior + torch.tensor([math.log(value) for value in ratios])
        advantages = torch.tensor([1.0, -1.0, 0.5, 2.0, -0.2, 0.7, -1.5, 1.2, 0.4, -0.8, 1.8])
        valid_mask = torch.tensor([True, True, False, True, True, False, True, True, True, False, True])
        all_indices = torch.arange(log_probs_value.numel())

        oracle_context = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs_value, behavior, valid_mask))
        oracle_log_probs = log_probs_value.clone().requires_grad_(True)
        oracle_terms = compute_p3o_token_terms(
            oracle_log_probs,
            behavior,
            advantages,
            valid_mask,
            oracle_context,
        )
        oracle_loss = (oracle_terms.score_loss + oracle_terms.adaptive_kl_loss).sum()
        oracle_loss = oracle_loss / oracle_context.valid_token_count
        oracle_loss.backward()
        oracle_gradient = oracle_log_probs.grad.detach()

        def assert_partition(shards: list[torch.Tensor], process_group) -> None:
            active_group[0] = process_group
            local_stats = P3OSufficientStats.zeros()
            for shard in shards:
                if shard.numel() == 0:
                    local_stats = local_stats + P3OSufficientStats.zeros()
                else:
                    local_stats = local_stats + compute_p3o_sufficient_stats(
                        log_probs_value[shard],
                        behavior[shard],
                        valid_mask[shard],
                    )
            synchronized = synchronize_p3o_stats(local_stats, torch.zeros((), dtype=torch.float64))
            context = finalize_p3o_step_context(synchronized)
            torch.testing.assert_close(context.normalized_ess, oracle_context.normalized_ess)

            local_log_probs = log_probs_value.clone().requires_grad_(True)
            local_total = 0.0 * local_log_probs.sum()
            for shard in shards:
                if shard.numel() == 0:
                    continue
                terms = compute_p3o_token_terms(
                    local_log_probs[shard],
                    behavior[shard],
                    advantages[shard],
                    valid_mask[shard],
                    context,
                )
                local_total = local_total + terms.score_loss.sum() + terms.adaptive_kl_loss.sum()
            local_loss = local_total / context.valid_token_count
            local_loss.backward()

            reduced_loss = local_loss.detach().clone()
            reduced_gradient = local_log_probs.grad.detach().clone()
            dist.all_reduce(reduced_loss, group=process_group)
            dist.all_reduce(reduced_gradient, group=process_group)
            torch.testing.assert_close(reduced_loss, oracle_loss.detach())
            torch.testing.assert_close(reduced_gradient, oracle_gradient)

        assert_partition([all_indices], singleton_groups[rank])
        assert_partition(list(torch.tensor_split(all_indices, 2))[rank % 2 : rank % 2 + 1], dp2_groups[rank // 2])
        assert_partition([torch.tensor_split(all_indices, world_size)[rank]], dist.group.WORLD)

        static_dp2_cp2 = [
            torch.tensor([0, 7, 8]),
            torch.tensor([1, 6, 9]),
            torch.tensor([2, 5, 10]),
            torch.tensor([3, 4]),
        ]
        assert_partition([static_dp2_cp2[rank]], dist.group.WORLD)

        dynamic_cp = [
            [torch.tensor([0, 1]), torch.tensor([6])],
            [torch.tensor([2]), torch.tensor([5, 7, 9])],
            [torch.tensor([3, 4]), torch.tensor([8, 10])],
            [torch.empty(0, dtype=torch.long)],
        ]
        assert_partition(dynamic_cp[rank], dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def test_p3o_distributed_nonfinite_fails_synchronously():
    world_size = 2
    mp.spawn(_nonfinite_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


def test_p3o_distributed_pipeline_broadcasts_last_stage_stats():
    world_size = 2
    mp.spawn(_pipeline_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


def test_p3o_distributed_partition_and_objective_invariance():
    world_size = 4
    mp.spawn(_partition_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
