# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""compute_ppl_metrics turns already-reduced sums (total negative log prob,
total tokens) into the eval/loss + eval/ppl + eval/num_tokens dict."""

import math


def test_compute_ppl_metrics_aggregates_correctly():
    """Known totals → loss = sum/tokens, ppl = exp(loss)."""
    from relax.engine.sft.eval.ppl import compute_ppl_metrics

    metrics = compute_ppl_metrics(total_neg_log_prob=10.0, total_tokens=6)

    expected_loss = 10.0 / 6.0
    assert math.isclose(metrics["eval/loss"], expected_loss, rel_tol=1e-6)
    assert math.isclose(metrics["eval/ppl"], math.exp(expected_loss), rel_tol=1e-5)
    assert metrics["eval/num_tokens"] == 6


def test_compute_ppl_metrics_empty_returns_zero_safely():
    """Zero tokens → all zeros, no DivisionByZero."""
    from relax.engine.sft.eval.ppl import compute_ppl_metrics

    metrics = compute_ppl_metrics(total_neg_log_prob=0.0, total_tokens=0)
    assert metrics["eval/num_tokens"] == 0
    assert metrics["eval/loss"] == 0.0
    assert metrics["eval/ppl"] == 0.0
