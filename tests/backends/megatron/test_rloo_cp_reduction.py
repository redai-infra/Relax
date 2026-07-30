# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""RLOO CP reduction invariance test using real torch.distributed gloo.

Verifies that the loss reduction (sum_of_sample_mean and sum_of_token) is
CP-invariant: the all-reduced result of per-rank partial reductions equals
the CP=1 (full-sequence) result, and each response token is partitioned
exactly once across CP ranks. No GPU required — runs on CPU via gloo.
"""

from __future__ import annotations

import os

import pytest


torch = pytest.importorskip("torch")


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rloo_cp_worker(rank, world_size, port, calculate_per_token_loss):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        cp_size = world_size
        torch.manual_seed(0)

        # 3 samples with different response lengths, each divisible by 2*cp
        # so contiguous splitting produces exact chunks.
        chunk = 4
        base = 2 * cp_size * chunk
        resp_lens = [base * (i + 1) for i in range(3)]
        rewards = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)

        # LOO advantages (same on every rank)
        from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

        advantages = compute_rloo_leave_one_out_rewards(rewards)

        # Full log_probs (same seed → same on every rank)
        full_log_probs = [torch.randn(L, dtype=torch.float64) for L in resp_lens]
        full_loss_masks = [torch.ones(L, dtype=torch.float64) for L in resp_lens]

        # --- CP=1 reference (computed on every rank for comparison) ---
        full_pg = []
        for i, L in enumerate(resp_lens):
            adv_broadcast = advantages[i].expand(L)
            pg = -(adv_broadcast * full_log_probs[i])
            full_pg.append(pg)

        if calculate_per_token_loss:
            ref = sum((pg * m).sum() for pg, m in zip(full_pg, full_loss_masks, strict=True))
        else:
            ref = sum(
                (pg * m).sum() / torch.clamp_min(m.sum(), 1)
                for pg, m in zip(full_pg, full_loss_masks, strict=True)
            )

        # --- CP=world_size: each rank holds a contiguous chunk of each sample ---
        local_pg_chunks = []
        local_mask_chunks = []
        local_full_masks = []  # full mask (for denominator)
        chunk_lengths = []
        for i, L in enumerate(resp_lens):
            tokens = full_log_probs[i]
            mask = full_loss_masks[i]
            # Contiguous split: rank r gets tokens [r*L/cp : (r+1)*L/cp]
            chunk_len = L // cp_size
            start = rank * chunk_len
            end = start + chunk_len
            local_chunk = tokens[start:end]
            local_mask = mask[start:end]
            local_pg = -(advantages[i].expand(chunk_len) * local_chunk)
            local_pg_chunks.append(local_pg)
            local_mask_chunks.append(local_mask)
            local_full_masks.append(mask)  # full mask for denominator
            chunk_lengths.append(chunk_len)

        # Local partial reduction (mirrors get_sum_of_sample_mean / sum_of_token)
        if calculate_per_token_loss:
            local_sum = sum(
                (pg * m).sum()
                for pg, m in zip(local_pg_chunks, local_mask_chunks, strict=True)
            )
        else:
            local_sum = sum(
                (pg * local_mask).sum() / torch.clamp_min(full_mask.sum(), 1)
                for pg, local_mask, full_mask in zip(
                    local_pg_chunks, local_mask_chunks, local_full_masks, strict=True
                )
            )

        # All-reduce across CP ranks
        reduced = local_sum.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

        # Verify: reduced == CP=1 reference
        assert torch.allclose(reduced, ref, atol=1e-10), (
            f"CP={cp_size} rank={rank} per_token={calculate_per_token_loss}: "
            f"reduced={reduced.item()} != ref={ref.item()}"
        )

        # Verify partition: each token appears exactly once across ranks.
        # Gather all chunk lengths and verify they sum to the full length.
        chunk_lens_tensor = torch.tensor([sum(chunk_lengths)], dtype=torch.int64)
        dist.all_reduce(chunk_lens_tensor, op=dist.ReduceOp.SUM)
        total_tokens = sum(resp_lens)
        assert chunk_lens_tensor.item() == total_tokens, (
            f"CP={cp_size}: partitioned tokens {chunk_lens_tensor.item()} != total {total_tokens}"
        )

    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_rloo_cp_partial_reductions_sum_to_unsplit(world_size, calculate_per_token_loss):
    import torch.multiprocessing as mp

    port = _free_port()
    mp.spawn(
        _rloo_cp_worker,
        args=(world_size, port, calculate_per_token_loss),
        nprocs=world_size,
        join=True,
    )
