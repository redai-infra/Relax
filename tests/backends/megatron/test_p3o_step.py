# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for optimizer-step P3O stats synchronization."""

from __future__ import annotations

import pytest
import torch

from relax.backends.megatron import p3o_step
from relax.backends.megatron.p3o_step import synchronize_p3o_stats
from relax.utils.training.p3o_utils import P3OSufficientStats


def _stats(values: tuple[float, float, float]) -> P3OSufficientStats:
    vector = torch.tensor(values, dtype=torch.float64)
    return P3OSufficientStats.from_vector(vector)


def test_p3o_step_single_pipeline_stage_preserves_stats(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    stats = _stats((7.5, 21.25, 4.0))

    synchronized = synchronize_p3o_stats(stats, torch.zeros((), dtype=torch.float64))

    torch.testing.assert_close(synchronized.as_vector(), stats.as_vector(), rtol=0.0, atol=0.0)


def test_p3o_step_non_last_stage_receives_pipeline_last_stats(monkeypatch):
    expected = torch.tensor([7.5, 21.25, 4.0, 0.0], dtype=torch.float64)
    pp_group = object()

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(p3o_step.mpu, "is_pipeline_last_stage", lambda ignore_virtual=True: False)
    monkeypatch.setattr(p3o_step.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(p3o_step.mpu, "get_pipeline_model_parallel_group", lambda: pp_group)

    def fail_if_reduced(*args, **kwargs):
        raise AssertionError("a non-last PP stage must not reduce token stats over DP x CP")

    def broadcast_from_last(vector, *, group, group_src):
        assert group is pp_group
        assert group_src == 1
        vector.copy_(expected)

    monkeypatch.setattr(torch.distributed, "all_reduce", fail_if_reduced)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast_from_last)

    synchronized = synchronize_p3o_stats(
        P3OSufficientStats.zeros(),
        torch.zeros((), dtype=torch.float64),
    )

    torch.testing.assert_close(synchronized.as_vector(), expected[:3], rtol=0.0, atol=0.0)


def test_p3o_step_raises_only_after_global_invalid_flag_is_visible(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(p3o_step.mpu, "is_pipeline_last_stage", lambda ignore_virtual=True: True)
    monkeypatch.setattr(p3o_step.mpu, "get_data_parallel_group", lambda with_context_parallel=True: object())
    monkeypatch.setattr(p3o_step.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)

    def all_reduce(vector, *, op, group):
        vector[3] = 1.0

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    with pytest.raises(ValueError, match="non-finite importance ratio"):
        synchronize_p3o_stats(_stats((1.0, 1.0, 1.0)), torch.zeros((), dtype=torch.float64))
