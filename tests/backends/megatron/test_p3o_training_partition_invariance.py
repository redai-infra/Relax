# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""End-to-end regression for P3O's strict DP/CP training-step contract."""

from __future__ import annotations

import os
import socket
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from relax.backends.megatron import initialize
from relax.backends.megatron.cp_utils import p3o_cp_replicated_slice


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 12, dtype=torch.bfloat16)
        self.linear_in = torch.nn.Linear(12, 24, bias=False, dtype=torch.bfloat16)
        self.linear_out = torch.nn.Linear(24, 12, bias=False, dtype=torch.bfloat16)
        self.norm = torch.nn.LayerNorm(12, dtype=torch.bfloat16)
        self.lm_head = torch.nn.Linear(12, 32, bias=False, dtype=torch.bfloat16)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        residual = hidden
        hidden = self.linear_out(F.silu(self.linear_in(hidden)))
        return self.lm_head(self.norm(hidden + residual)).float()


def _fixture() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    samples = []
    for index, length in enumerate((4, 8, 4, 8, 8, 12, 4, 8)):
        tokens = (torch.arange(length, dtype=torch.long) * (index + 3) + index + 1) % 31
        behavior = -3.6 + 0.007 * torch.arange(length, dtype=torch.float32) + 0.003 * index
        advantages = torch.sin(torch.arange(length, dtype=torch.float32) * 0.7 + index) * (index + 1)
        samples.append((tokens, behavior, advantages))
    return samples


def _current_log_probs(model: _TinyPolicy, tokens: torch.Tensor) -> torch.Tensor:
    logits = model(tokens)
    labels = torch.roll(tokens, shifts=-1)
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _cp_token_indices(length: int, cp_size: int, cp_rank: int) -> torch.Tensor:
    chunk_size = length // (2 * cp_size)
    first = torch.arange(cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    second_rank = 2 * cp_size - cp_rank - 1
    second = torch.arange(second_rank * chunk_size, (second_rank + 1) * chunk_size)
    return torch.cat((first, second))


def _context(model: _TinyPolicy, samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
    ratios = []
    with torch.no_grad():
        for tokens, behavior, _ in samples:
            ratios.append(torch.exp(_current_log_probs(model, tokens) - behavior))
    ratio = torch.cat(ratios).double()
    return float((ratio.sum().square() / (ratio.square().sum() * ratio.numel())).clamp(max=1.0))


def _worker(
    rank: int,
    world_size: int,
    port: int,
    dp_size: int,
    cp_size: int,
    scope: str,
    output_path: str,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(20260817)
        model = _TinyPolicy()
        initialize.enable_batch_invariant_mode = lambda: None
        args = Namespace(
            advantage_estimator="p3o",
            batch_invariant_mode=True,
            deterministic_mode=True,
            gradient_accumulation_fusion=True,
        )
        initialize._configure_p3o_partition_invariance(args)

        samples = _fixture()
        dp_rank = rank // cp_size
        cp_rank = rank % cp_size
        cp_group = None
        if cp_size > 1:
            for group_dp_rank in range(dp_size):
                group = dist.new_group(ranks=list(range(group_dp_rank * cp_size, (group_dp_rank + 1) * cp_size)))
                if group_dp_rank == dp_rank:
                    cp_group = group
        sample_indices = torch.tensor_split(torch.arange(len(samples)), dp_size)[dp_rank].tolist()
        step_cap = _context(model, samples)
        main_grads = {
            name: torch.zeros_like(parameter, dtype=torch.float32) for name, parameter in model.named_parameters()
        }
        local_loss = torch.zeros((), dtype=torch.float64)
        local_tokens = torch.zeros((), dtype=torch.float64)

        for sample_index in sample_indices:
            tokens, behavior, advantages = samples[sample_index]
            logits = model(tokens)
            labels = torch.roll(tokens, shifts=-1)
            if scope == "micro-batch":
                full_current = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                full_ratio = torch.exp(full_current.detach() - behavior)
                cap = float(
                    (
                        full_ratio.double().sum().square() / (full_ratio.double().square().sum() * full_ratio.numel())
                    ).clamp(max=1.0)
                )
            else:
                cap = step_cap
            if cp_size > 1:
                logits = p3o_cp_replicated_slice(logits, [0, tokens.numel()], cp_size, cp_rank, cp_group)
                token_indices = _cp_token_indices(tokens.numel(), cp_size, cp_rank)
                labels = labels[token_indices]
                behavior = behavior[token_indices]
                advantages = advantages[token_indices]
            current = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(current.detach() - behavior)
            coefficients = torch.minimum(ratio, ratio.new_tensor(cap)) * advantages
            loss = -(coefficients.detach().double() * current.double()).sum()
            loss.backward()
            local_loss += loss.detach().double()
            local_tokens += current.numel()

            for name, parameter in model.named_parameters():
                gradient = parameter.grad.detach().float()
                if args.gradient_accumulation_fusion:
                    # This is the contract of TE's fused ``out=main_grad`` wgrad
                    # path. MCore BIK ignored ``accumulate=True`` and overwrote.
                    main_grads[name].copy_(gradient)
                else:
                    main_grads[name].add_(gradient)
                parameter.grad = None

        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_tokens, op=dist.ReduceOp.SUM)
        flattened = []
        for gradient in main_grads.values():
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(local_tokens)
            flattened.append(gradient.reshape(-1))
        if rank == 0:
            torch.save(
                {
                    "gradient": torch.cat(flattened),
                    "loss": local_loss / local_tokens,
                    "valid_tokens": local_tokens,
                },
                output_path,
            )
    finally:
        dist.destroy_process_group()


def _run_topology(tmp_path: Path, name: str, dp_size: int, cp_size: int, scope: str) -> dict[str, torch.Tensor]:
    output_path = tmp_path / f"{scope}_{name}.pt"
    world_size = dp_size * cp_size
    mp.spawn(
        _worker,
        args=(world_size, _free_port(), dp_size, cp_size, scope, str(output_path)),
        nprocs=world_size,
        join=True,
    )
    return torch.load(output_path, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("scope", ["micro-batch", "step"])
def test_p3o_full_training_backward_matches_dp1_across_dp_and_cp2(tmp_path: Path, scope: str) -> None:
    reference = _run_topology(tmp_path, "dp1", dp_size=1, cp_size=1, scope=scope)
    for name, dp_size, cp_size in (("dp2", 2, 1), ("dp4", 4, 1), ("dp2cp2", 2, 2)):
        candidate = _run_topology(tmp_path, name, dp_size=dp_size, cp_size=cp_size, scope=scope)
        reference_gradient = reference["gradient"].double()
        candidate_gradient = candidate["gradient"].double()
        relative_l2 = torch.linalg.vector_norm(candidate_gradient - reference_gradient) / torch.linalg.vector_norm(
            reference_gradient
        )
        cosine = torch.dot(reference_gradient, candidate_gradient) / (
            torch.linalg.vector_norm(reference_gradient) * torch.linalg.vector_norm(candidate_gradient)
        )

        assert relative_l2 <= 1e-6, name
        assert cosine >= 1 - 1e-9, name
        torch.testing.assert_close(candidate["loss"], reference["loss"], rtol=1e-6, atol=1e-9)
        assert candidate["valid_tokens"].item() == reference["valid_tokens"].item()


def test_p3o_strict_dp_rollout_merge_keeps_one_loss_anchor() -> None:
    from relax.backends.megatron.data import _merge_p3o_strict_dp_rollout_batches

    batches = [
        {
            "tokens": [torch.tensor([rank + 1, rank + 2])],
            "total_lengths": [2],
            "response_lengths": [2],
            "loss_masks": [torch.ones(2, dtype=torch.int64)],
            "advantages": [torch.tensor([float(rank + 1), float(rank + 1)])],
            "rollout_mini_local_sample_counts": [1],
            "dynamic_global_batch_size": 2,
        }
        for rank in range(2)
    ]

    anchor = _merge_p3o_strict_dp_rollout_batches(batches, is_anchor=True)
    replica = _merge_p3o_strict_dp_rollout_batches(batches, is_anchor=False)

    assert anchor["total_lengths"] == [2, 2]
    assert anchor["rollout_mini_local_sample_counts"] == [2]
    assert [mask.tolist() for mask in anchor["loss_masks"]] == [[1, 1], [1, 1]]
    assert [mask.tolist() for mask in replica["loss_masks"]] == [[0, 0], [0, 0]]
    assert [value.tolist() for value in replica["advantages"]] == [[1.0, 1.0], [2.0, 2.0]]


def test_p3o_strict_dp_rollout_merge_rejects_multiple_rollout_minis() -> None:
    from relax.backends.megatron.data import _merge_p3o_strict_dp_rollout_batches

    batch = {
        "tokens": [torch.tensor([1]), torch.tensor([2])],
        "total_lengths": [1, 1],
        "response_lengths": [1, 1],
        "loss_masks": [torch.ones(1), torch.ones(1)],
        "rollout_mini_local_sample_counts": [1, 1],
    }

    with pytest.raises(RuntimeError, match="exactly one rollout mini"):
        _merge_p3o_strict_dp_rollout_batches([batch], is_anchor=True)
