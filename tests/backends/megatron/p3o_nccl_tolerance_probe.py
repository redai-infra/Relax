# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Calibrate P3O BF16/NCCL drift against a single-rank FP32 oracle.

Run this file with exactly four processes, for example::

    torchrun --standalone --nproc-per-node=4 \
        tests/backends/megatron/p3o_nccl_tolerance_probe.py \
        --output /workspace/Output/task40/p3o_nccl_tolerance.json

This is a deterministic synthetic-batch calibration of the P3O formula,
FP64 sufficient-statistic reduction, BF16 forward, gradient reduction, and
optimizer update. It is not a substitute for replaying real model rollout
data through the full Megatron backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as functional

from relax.utils.training.p3o_utils import (
    P3OSufficientStats,
    compute_p3o_sufficient_stats,
    compute_p3o_token_terms,
    finalize_p3o_step_context,
)


WORLD_SIZE = 4
OPTIMIZER_STEPS = 10
TOKEN_COUNT = 64
FEATURE_COUNT = 8

# Frozen on 2026-07-31 from the first valid four-A100 calibration at
# Relax@801abc7. The observed maxima were 5.60e-4, 1.50e-3, and 1.62e-3;
# each threshold is rounded upward before any current-HEAD training run.
ESS_RTOL = 1e-3
ESS_ATOL = 1e-3
LOSS_RTOL = 2e-3
LOSS_ATOL = 3e-3
GRAD_RELATIVE_L2_TOL = 2e-3


class TinyPolicy(torch.nn.Module):
    """One-layer score model with FP32 master parameters."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.bias = torch.nn.Parameter(bias.clone())

    def forward(self, features: torch.Tensor, *, bf16: bool) -> torch.Tensor:
        """Return one score per token, optionally using BF16 matmul inputs."""
        if bf16:
            return functional.linear(
                features.to(torch.bfloat16),
                self.weight.to(torch.bfloat16),
                self.bias.to(torch.bfloat16),
            ).float()
        return functional.linear(features, self.weight, self.bias)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_value = float(actual.detach())
    expected_value = float(expected.detach())
    denominator = max(abs(expected_value), torch.finfo(torch.float64).eps)
    return abs(actual_value - expected_value) / denominator


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected.double()).clamp_min(torch.finfo(torch.float64).eps)
    return float(torch.linalg.vector_norm(actual.double() - expected.double()) / denominator)


def _flat_gradients(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.detach().flatten() for parameter in model.parameters()])


def _flat_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


def _global_context(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, object]:
    local_stats = compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask)
    vector = local_stats.as_vector()
    dist.all_reduce(vector, op=dist.ReduceOp.SUM)
    context = finalize_p3o_step_context(P3OSufficientStats.from_vector(vector))
    return vector, context


def run_probe(output_path: Path) -> None:
    """Run the ten-step calibration and write rank zero's JSON result."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"P3O NCCL calibration requires exactly {WORLD_SIZE} ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)

    generator = torch.Generator(device="cpu").manual_seed(20260731)
    features_cpu = torch.randn(TOKEN_COUNT, FEATURE_COUNT, generator=generator, dtype=torch.float32)
    weight_cpu = torch.randn(1, FEATURE_COUNT, generator=generator, dtype=torch.float32) * 0.1
    # Keep sampled-token log-probs and the scalar loss at realistic non-zero
    # magnitudes so relative-error calibration is not dominated by a zero
    # crossing in the denominator.
    bias_cpu = torch.tensor([-2.0], dtype=torch.float32)
    advantages_cpu = 1.0 + 0.5 * torch.sin(torch.arange(TOKEN_COUNT, dtype=torch.float32) * 0.37)
    valid_mask_cpu = torch.arange(TOKEN_COUNT) % 7 != 0
    initial_scores_cpu = functional.linear(features_cpu, weight_cpu, bias_cpu).squeeze(-1)
    structured_log_ratio = 0.55 * torch.sin(torch.arange(TOKEN_COUNT, dtype=torch.float32) * 0.23)
    behavior_cpu = initial_scores_cpu - structured_log_ratio

    local_indices = torch.arange(rank, TOKEN_COUNT, world_size)
    local_features = features_cpu[local_indices].to(device)
    local_advantages = advantages_cpu[local_indices].to(device)
    local_mask = valid_mask_cpu[local_indices].to(device)
    local_behavior = behavior_cpu[local_indices].to(device)

    distributed_model = TinyPolicy(weight_cpu.to(device), bias_cpu.to(device)).to(device)
    distributed_optimizer = torch.optim.AdamW(
        distributed_model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01
    )

    reference_model = None
    reference_optimizer = None
    if rank == 0:
        reference_model = TinyPolicy(weight_cpu.to(device), bias_cpu.to(device)).to(device)
        reference_optimizer = torch.optim.AdamW(
            reference_model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01
        )
        features = features_cpu.to(device)
        advantages = advantages_cpu.to(device)
        valid_mask = valid_mask_cpu.to(device)
        behavior = behavior_cpu.to(device)

    observations: list[dict[str, float | int]] = []
    for step in range(OPTIMIZER_STEPS):
        distributed_optimizer.zero_grad(set_to_none=True)
        local_log_probs = distributed_model(local_features, bf16=True).squeeze(-1)
        _, distributed_context = _global_context(local_log_probs, local_behavior, local_mask)
        distributed_terms = compute_p3o_token_terms(
            local_log_probs,
            local_behavior,
            local_advantages,
            local_mask,
            distributed_context,
        )
        local_loss = (distributed_terms.score_loss + distributed_terms.adaptive_kl_loss).sum()
        local_loss = local_loss / distributed_context.valid_token_count
        local_loss.backward()
        for parameter in distributed_model.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        distributed_gradient = _flat_gradients(distributed_model)
        distributed_loss = local_loss.detach().clone()
        dist.all_reduce(distributed_loss, op=dist.ReduceOp.SUM)
        distributed_optimizer.step()

        if rank == 0:
            assert reference_model is not None and reference_optimizer is not None
            reference_optimizer.zero_grad(set_to_none=True)
            reference_log_probs = reference_model(features, bf16=False).squeeze(-1)
            reference_context = finalize_p3o_step_context(
                compute_p3o_sufficient_stats(reference_log_probs, behavior, valid_mask)
            )
            reference_terms = compute_p3o_token_terms(
                reference_log_probs,
                behavior,
                advantages,
                valid_mask,
                reference_context,
            )
            reference_loss = (reference_terms.score_loss + reference_terms.adaptive_kl_loss).sum()
            reference_loss = reference_loss / reference_context.valid_token_count
            reference_loss.backward()
            reference_gradient = _flat_gradients(reference_model)
            reference_optimizer.step()

            distributed_parameters = _flat_parameters(distributed_model)
            reference_parameters = _flat_parameters(reference_model)
            observations.append(
                {
                    "step": step + 1,
                    "ess_bf16_nccl": float(distributed_context.normalized_ess),
                    "ess_fp32": float(reference_context.normalized_ess),
                    "ess_abs_error": abs(
                        float(distributed_context.normalized_ess) - float(reference_context.normalized_ess)
                    ),
                    "ess_rel_error": _relative_error(
                        distributed_context.normalized_ess, reference_context.normalized_ess
                    ),
                    "loss_bf16_nccl": float(distributed_loss),
                    "loss_fp32": float(reference_loss.detach()),
                    "loss_abs_error": abs(float(distributed_loss) - float(reference_loss.detach())),
                    "loss_rel_error": _relative_error(distributed_loss, reference_loss),
                    "grad_relative_l2": _relative_l2(distributed_gradient, reference_gradient),
                    "parameter_relative_l2": _relative_l2(distributed_parameters, reference_parameters),
                }
            )

    passed = torch.ones((), dtype=torch.int32, device=device)
    if rank == 0:
        summary = {
            "scope": "deterministic synthetic fixed-token batch; not a real-model rollout replay",
            "world_size": world_size,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(world_size)],
            "steps": OPTIMIZER_STEPS,
            "tokens": TOKEN_COUNT,
            "valid_tokens": int(valid_mask_cpu.sum()),
            "max_ess_abs_error": max(item["ess_abs_error"] for item in observations),
            "max_ess_rel_error": max(item["ess_rel_error"] for item in observations),
            "max_loss_abs_error": max(item["loss_abs_error"] for item in observations),
            "max_loss_rel_error": max(item["loss_rel_error"] for item in observations),
            "max_grad_relative_l2": max(item["grad_relative_l2"] for item in observations),
            "max_parameter_relative_l2": max(item["parameter_relative_l2"] for item in observations),
            "frozen_tolerances": {
                "ess_rtol": ESS_RTOL,
                "ess_atol": ESS_ATOL,
                "loss_rtol": LOSS_RTOL,
                "loss_atol": LOSS_ATOL,
                "grad_relative_l2": GRAD_RELATIVE_L2_TOL,
            },
            "observations": observations,
        }
        if not all(math.isfinite(value) for item in observations for value in item.values()):
            raise RuntimeError("P3O NCCL calibration produced a non-finite metric")
        summary["passed"] = (
            summary["max_ess_rel_error"] <= ESS_RTOL
            and summary["max_ess_abs_error"] <= ESS_ATOL
            and summary["max_loss_rel_error"] <= LOSS_RTOL
            and summary["max_loss_abs_error"] <= LOSS_ATOL
            and summary["max_grad_relative_l2"] <= GRAD_RELATIVE_L2_TOL
        )
        passed.fill_(int(summary["passed"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))

    dist.broadcast(passed, src=0)
    dist.destroy_process_group()
    if not bool(passed):
        raise RuntimeError("P3O BF16/NCCL drift exceeded the frozen synthetic-batch tolerances")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.output)


if __name__ == "__main__":
    main()
