# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.backends.megatron import data  # noqa: E402


def test_log_rollout_data_keeps_rloo_sequence_metrics_cp_invariant(monkeypatch):
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(data.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(data.mpu, "get_data_parallel_group", lambda with_context_parallel=True: None)
    monkeypatch.setattr(data.dist, "all_reduce", lambda tensor, op, group: None)
    captured = {}

    def fake_gather_log_data(metric_name, args, rollout_id, log_dict):
        captured.update(log_dict)
        return {f"{metric_name}/{key}": value for key, value in log_dict.items()}

    monkeypatch.setattr(data, "gather_log_data", fake_gather_log_data)
    args = Namespace(
        dynamic_context_parallel=False,
        qkv_format="bshd",
        is_vl_model=False,
        uses_unsplit_forward=False,
        use_opd=False,
        ci_test=False,
        log_multi_turn=False,
        log_correct_samples=False,
    )
    rollout_data = {
        "response_lengths": [2, 2],
        "loss_masks": [torch.ones(2), torch.ones(2)],
        "total_lengths": [4, 4],
        "rloo_shaped_reward": [torch.tensor([1.0]), torch.tensor([3.0])],
        "rloo_sequence_kl": [torch.tensor([0.1]), torch.tensor([0.3])],
        "rloo_baseline": [torch.tensor([0.5]), torch.tensor([1.5])],
        "rloo_advantage": [torch.tensor([-1.0]), torch.tensor([1.0])],
        "rloo_advantage_abs": [torch.tensor([1.0]), torch.tensor([1.0])],
        "rloo_empty_response_fraction": [torch.tensor([0.0]), torch.tensor([1.0])],
        "custom_tensor_metric": [torch.tensor([1.0]), torch.tensor([3.0])],
    }

    data.log_rollout_data(0, args, rollout_data)

    assert captured["rloo_shaped_reward"] == 2.0
    assert captured["rloo_sequence_kl"] == pytest.approx(0.2)
    assert captured["rloo_baseline"] == 1.0
    assert captured["rloo_advantage"] == 0.0
    assert captured["rloo_advantage_abs"] == 1.0
    assert captured["rloo_empty_response_fraction"] == 0.5
    assert captured["custom_tensor_metric"] == 4.0
