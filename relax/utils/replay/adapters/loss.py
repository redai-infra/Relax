# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy-loss replay adapter.

Reuses ``compute_policy_loss`` from ``relax.utils.training.ppo_utils`` and
restates only the CP=1 reduction semantics from
``relax.backends.megatron.cp_utils.get_sum_of_sample_mean`` (per-sample masked
mean, then summed — a plain-CPU computation with no collective). KL-loss, OPD
and TIS paths are outside the frozen V1 scope.
"""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.logging_utils import get_logger
from relax.utils.replay.bundle import LoadedBundle
from relax.utils.replay.compare import compare_scalar, scalar_error, tolerance_for
from relax.utils.replay.report import FieldDivergence, StageResult, StageStatus
from relax.utils.replay.schema import StageId
from relax.utils.replay.stages import register_adapter
from relax.utils.training.ppo_utils import compute_policy_loss


logger = get_logger(__name__)


def _sum_of_sample_mean(x: torch.Tensor, response_lengths: list[int], loss_masks: list[list[int]]) -> torch.Tensor:
    """CP=1 ``sum_of_sample_mean``: per-sample masked mean, summed over
    samples."""
    total = 0.0
    for chunk, mask in zip(torch.split(x, response_lengths), loss_masks, strict=False):
        mask_t = torch.tensor(mask, dtype=x.dtype, device=x.device)
        denominator = torch.clamp_min(mask_t.sum(), 1)
        total = total + (chunk * mask_t).sum() / denominator
    return total


@register_adapter(StageId.LOSS_POLICY)
def replay_loss_policy(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Recompute the GRPO policy loss and its reduction metrics."""
    config = bundle.index.config
    result = StageResult(stage=StageId.LOSS_POLICY.value, status=StageStatus.PASS)

    advantages = ctx.get(StageId.ADVANTAGE_ESTIMATE.value)
    if advantages is None:
        # Partial capture (loss-only): fall back to the recorded advantages
        # tensor, which is the loss stage's actual input. Full-chain replays
        # still use ctx so upstream divergence propagates (see spec §5.9.2).
        advantages = bundle.tensors.get("advantages")
    if advantages is None:
        result.status = StageStatus.FAIL
        result.message = "upstream advantage.estimate output missing"
        return result

    old_log_probs = bundle.tensors["old_log_probs"]
    log_probs = bundle.tensors["log_probs"]
    entropy = bundle.tensors["entropy"]

    response_lengths = [record.response_length for record in bundle.index.samples]
    loss_masks = [record.loss_mask for record in bundle.index.samples]

    ppo_kl = old_log_probs - log_probs
    pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, config.eps_clip, config.eps_clip_high)

    pg_loss_scalar = _sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
    clipfrac_scalar = _sum_of_sample_mean(clipfrac, response_lengths, loss_masks)
    ppo_kl_scalar = _sum_of_sample_mean(ppo_kl, response_lengths, loss_masks)
    entropy_loss = _sum_of_sample_mean(entropy, response_lengths, loss_masks)
    loss = pg_loss_scalar - config.entropy_coef * entropy_loss

    recomputed = {
        "loss": float(loss.item()),
        "pg_loss": float(pg_loss_scalar.item()),
        "entropy_loss": float(entropy_loss.item()),
        "pg_clipfrac": float(clipfrac_scalar.item()),
        "ppo_kl": float(ppo_kl_scalar.item()),
    }

    expected = bundle.expected.get(StageId.LOSS_POLICY.value)
    if expected is None:
        result.message = "no expected loss metrics recorded"
        return result

    atol, rtol = tolerance_for(bundle.manifest, StageId.LOSS_POLICY)
    for field, actual_value in recomputed.items():
        expected_value = expected.get(field)
        if expected_value is None:
            continue
        if not compare_scalar(float(expected_value), float(actual_value), atol=atol, rtol=rtol):
            result.status = StageStatus.FAIL
            result.divergences.append(
                FieldDivergence(
                    field=field,
                    expected=float(expected_value),
                    actual=float(actual_value),
                    abs_error=scalar_error(float(expected_value), float(actual_value)),
                )
            )
    if result.status == StageStatus.FAIL:
        result.message = f"loss divergence in fields {[d.field for d in result.divergences]}"
    return result
