# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Advantage-stage replay adapters.

These reuse the pure tensor kernels from ``relax.utils.training.ppo_utils``
(``compute_approx_kl``, ``get_grpo_returns``) — single source of truth, no
production math restated here. ``ppo_utils`` imports only ``torch`` and
``torch.distributed`` at module scope, so it is safe to import offline.
"""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.logging_utils import get_logger
from relax.utils.replay.bundle import LoadedBundle
from relax.utils.replay.compare import compare_tensors, tolerance_for
from relax.utils.replay.report import StageResult, StageStatus
from relax.utils.replay.schema import StageId
from relax.utils.replay.stages import register_adapter
from relax.utils.training.ppo_utils import compute_approx_kl, get_grpo_returns


logger = get_logger(__name__)


def _sample_ids(bundle: LoadedBundle) -> list[str]:
    return [record.sample_id for record in bundle.index.samples]


def _response_lengths(bundle: LoadedBundle) -> list[int]:
    return [record.response_length for record in bundle.index.samples]


@register_adapter(StageId.ADVANTAGE_KL)
def replay_advantage_kl(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Recompute per-token approximate KL between rollout and reference log-
    probs.

    Matches production ``compute_advantages_and_returns``: the operands are the
    rollout/old-policy log-probs (``old_log_probs``) and the frozen reference-
    policy log-probs (``ref_log_probs``) — *not* current vs old.
    """
    config = bundle.index.config
    result = StageResult(stage=StageId.ADVANTAGE_KL.value, status=StageStatus.PASS)

    old_log_probs = bundle.tensors["old_log_probs"]
    ref_log_probs = bundle.tensors["ref_log_probs"]

    if config.kl_coef == 0:
        kl = torch.zeros_like(old_log_probs, dtype=torch.float32)
    else:
        kl = compute_approx_kl(old_log_probs, ref_log_probs, kl_loss_type=config.kl_loss_type)

    ctx[StageId.ADVANTAGE_KL.value] = kl

    expected = bundle.tensors.get("kl")
    if expected is None:
        result.message = "no expected kl payload recorded"
        return result

    atol, rtol = tolerance_for(bundle.manifest, StageId.ADVANTAGE_KL)
    divergences, max_err, mismatches, non_finite = compare_tensors(
        expected,
        kl,
        field="kl",
        atol=atol,
        rtol=rtol,
        sample_ids=_sample_ids(bundle),
        response_lengths=_response_lengths(bundle),
    )
    result.divergences = divergences
    result.max_abs_error = max_err
    result.mismatch_count = mismatches
    result.non_finite_count = non_finite
    if mismatches or non_finite:
        result.status = StageStatus.FAIL
        result.message = f"{mismatches} kl token(s) diverged"
    return result


@register_adapter(StageId.ADVANTAGE_ESTIMATE)
def replay_advantage_estimate(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Recompute GRPO-family returns/advantages from normalized rewards +
    KL."""
    result = StageResult(stage=StageId.ADVANTAGE_ESTIMATE.value, status=StageStatus.PASS)

    normalized_rewards = ctx.get(StageId.REWARD_POST_PROCESS.value)
    if normalized_rewards is None:
        result.status = StageStatus.FAIL
        result.message = "upstream reward.post_process output missing"
        return result

    kl = ctx.get(StageId.ADVANTAGE_KL.value)
    if kl is None:
        result.status = StageStatus.FAIL
        result.message = "upstream advantage.kl output missing"
        return result

    rewards = torch.tensor(normalized_rewards, dtype=torch.float32)
    kl_per_sample = list(torch.split(kl, _response_lengths(bundle)))
    returns = get_grpo_returns(rewards, kl_per_sample)
    advantages = torch.cat(returns)

    ctx[StageId.ADVANTAGE_ESTIMATE.value] = advantages

    expected = bundle.tensors.get("advantages")
    if expected is None:
        result.message = "no expected advantages payload recorded"
        return result

    atol, rtol = tolerance_for(bundle.manifest, StageId.ADVANTAGE_ESTIMATE)
    divergences, max_err, mismatches, non_finite = compare_tensors(
        expected,
        advantages,
        field="advantages",
        atol=atol,
        rtol=rtol,
        sample_ids=_sample_ids(bundle),
        response_lengths=_response_lengths(bundle),
    )
    result.divergences = divergences
    result.max_abs_error = max_err
    result.mismatch_count = mismatches
    result.non_finite_count = non_finite
    if mismatches or non_finite:
        result.status = StageStatus.FAIL
        result.message = f"{mismatches} advantage token(s) diverged"
    return result
