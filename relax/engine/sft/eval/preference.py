# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Forward-only callbacks and reducers for held-out preference evaluation."""

import torch

from relax.utils.training.preference_utils import select_packed_sequence_scores


def compute_reward_model_eval_step(
    logits: torch.Tensor,
    *,
    total_lengths,
    score_positions=None,
    **_,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    if score_positions is None:
        raise ValueError("reward-model eval batch is missing score_positions")
    scores = select_packed_sequence_scores(logits, total_lengths, score_positions).detach()
    return torch.empty((0,), device=logits.device), {"scores": [scores]}


def pair_metric_sums(chosen: torch.Tensor, rejected: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
    """Return additive loss/count/score/margin/correct/tie statistics."""
    if chosen.shape != rejected.shape or chosen.shape != losses.shape:
        raise ValueError("preference eval tensors must have identical shapes")
    margin = chosen - rejected
    epsilon = 1e-6
    correct = (margin > epsilon).to(torch.float64).sum()
    ties = (margin.abs() <= epsilon).to(torch.float64).sum()
    return torch.stack(
        [
            losses.to(torch.float64).sum(),
            torch.tensor(float(losses.numel()), device=losses.device, dtype=torch.float64),
            chosen.to(torch.float64).sum(),
            rejected.to(torch.float64).sum(),
            margin.to(torch.float64).sum(),
            correct,
            ties,
        ]
    )


def finalize_pair_metrics(values: torch.Tensor, *, prefix: str) -> dict[str, float]:
    loss_sum, count, chosen_sum, rejected_sum, margin_sum, correct, ties = values.tolist()
    if count <= 0:
        raise ValueError("preference evaluator received zero pairs")
    return {
        f"eval/{prefix}_loss": loss_sum / count,
        f"eval/{prefix}_chosen": chosen_sum / count,
        f"eval/{prefix}_rejected": rejected_sum / count,
        f"eval/{prefix}_margin": margin_sum / count,
        f"eval/{prefix}_strict_accuracy": correct / count,
        f"eval/{prefix}_tie_rate": ties / count,
        f"eval/{prefix}_tie_aware_accuracy": (correct + 0.5 * ties) / count,
        f"eval/{prefix}_pairs": count,
    }


__all__ = ["compute_reward_model_eval_step", "finalize_pair_metrics", "pair_metric_sums"]
