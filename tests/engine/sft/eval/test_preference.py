# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.engine.sft.eval.preference import compute_reward_model_eval_step, finalize_pair_metrics, pair_metric_sums


def test_reward_model_eval_emits_one_score_per_branch_for_order_restoration():
    _, outputs = compute_reward_model_eval_step(
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
        total_lengths=[2, 2],
        score_positions=[1, 1],
    )

    assert len(outputs["scores"]) == 2
    assert all(score.ndim == 0 for score in outputs["scores"])
    assert torch.stack(outputs["scores"]).tolist() == [1.0, 3.0]


def test_pair_metric_sums_and_finalize_keep_ties_explicit():
    chosen = torch.tensor([2.0, 1.0, 1.0])
    rejected = torch.tensor([1.0, 2.0, 1.0])
    losses = torch.tensor([0.1, 0.2, 0.3])

    metrics = finalize_pair_metrics(pair_metric_sums(chosen, rejected, losses), prefix="rm")

    assert metrics["eval/rm_loss"] == pytest.approx(0.2)
    assert metrics["eval/rm_strict_accuracy"] == pytest.approx(1 / 3)
    assert metrics["eval/rm_tie_rate"] == pytest.approx(1 / 3)
    assert metrics["eval/rm_tie_aware_accuracy"] == pytest.approx(0.5)
    assert metrics["eval/rm_pairs"] == 3
