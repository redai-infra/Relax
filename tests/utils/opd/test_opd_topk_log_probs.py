# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for global Top-K token ids across TP shards."""

import pytest
import torch

from relax.utils.opd.opd_utils import compute_log_probs_on_topk_token_ids  # noqa: E402


def test_topk_log_probs_keep_global_zero_and_sentinel_ids() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], requires_grad=True)
    token_ids = torch.tensor([[0, 3, -1]], dtype=torch.long)

    actual = compute_log_probs_on_topk_token_ids(logits, token_ids, process_group=None)
    expected = torch.cat([logits[:, :1], logits[:, 3:4], torch.full((1, 1), float("-inf"))], dim=-1) - torch.logsumexp(
        logits, dim=-1, keepdim=True
    )

    assert torch.allclose(actual[:, :2], expected[:, :2])
    assert torch.isneginf(actual[:, 2]).all()


def test_topk_log_probs_reject_ids_outside_global_vocabulary() -> None:
    logits = torch.zeros((1, 4))
    token_ids = torch.tensor([[4]], dtype=torch.long)

    with pytest.raises(ValueError, match="global vocabulary size"):
        compute_log_probs_on_topk_token_ids(logits, token_ids, process_group=None)
