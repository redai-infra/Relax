# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.backends.megatron import loss  # noqa: E402


def test_compute_rloo_advantages_applies_sequence_kl_before_leave_one_out(monkeypatch):
    monkeypatch.setattr(loss.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(loss.mpu, "get_data_parallel_world_size", lambda with_context_parallel=False: 1)
    monkeypatch.setattr(loss.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss.mpu, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(
        loss,
        "compute_approx_kl",
        lambda log_probs, ref_log_probs, kl_loss_type: log_probs - ref_log_probs,
    )
    gather_calls = []

    def fake_all_gather(
        tensor,
        total_length,
        response_length,
        padded_total_length=None,
        qkv_format="thd",
        max_seq_len=None,
        **kwargs,
    ):
        gather_calls.append((qkv_format, max_seq_len))
        return tensor

    monkeypatch.setattr(loss, "all_gather_with_cp", fake_all_gather)

    args = Namespace(
        advantage_estimator="rloo",
        use_rollout_logprobs=False,
        kl_coef=0.5,
        kl_loss_type="k1",
        qkv_format="bshd",
        is_vl_model=False,
        uses_unsplit_forward=False,
        n_samples_per_prompt=2,
        normalize_advantages=False,
        use_opd=False,
    )
    rollout_data = {
        "log_probs": [torch.tensor([-0.2, -0.4]), torch.tensor([-0.5])],
        "ref_log_probs": [torch.tensor([-0.3, -0.1]), torch.tensor([-0.7])],
        "rewards": [1.0, 0.0],
        "values": None,
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1.0, 0.0]), torch.tensor([1.0])],
        "total_lengths": [4, 3],
        "max_seq_lens": [8, 8],
        "group_indices": [7, 7],
    }

    loss.compute_advantages_and_returns(args, rollout_data)
    assert gather_calls == [("bshd", 8), ("bshd", 8)]

    expected_kl = torch.tensor([0.1, 0.2])
    expected_shaped_rewards = torch.tensor([0.95, -0.1])
    expected_baselines = torch.tensor([-0.1, 0.95])
    expected_advantages = torch.tensor([1.05, -1.05])

    assert torch.allclose(torch.cat(rollout_data["rloo_sequence_kl"]), expected_kl, atol=1e-6)
    assert torch.allclose(torch.cat(rollout_data["rloo_shaped_reward"]), expected_shaped_rewards, atol=1e-6)
    assert torch.allclose(torch.cat(rollout_data["rloo_baseline"]), expected_baselines, atol=1e-6)
    assert torch.allclose(torch.cat(rollout_data["rloo_advantage"]), expected_advantages, atol=1e-6)
    assert torch.allclose(rollout_data["advantages"][0], torch.full((2,), 1.05), atol=1e-6)
    assert torch.allclose(rollout_data["advantages"][1], torch.full((1,), -1.05), atol=1e-6)


def test_compute_rloo_advantages_keeps_empty_response_in_baseline(monkeypatch):
    monkeypatch.setattr(loss.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(loss.mpu, "get_data_parallel_world_size", lambda with_context_parallel=False: 1)

    args = Namespace(
        advantage_estimator="rloo",
        use_rollout_logprobs=True,
        kl_coef=0.0,
        kl_loss_type="k1",
        qkv_format="thd",
        is_vl_model=False,
        uses_unsplit_forward=False,
        n_samples_per_prompt=2,
        normalize_advantages=False,
        use_opd=False,
    )
    rollout_data = {
        "rollout_log_probs": [torch.tensor([]), torch.tensor([-0.5])],
        "ref_log_probs": None,
        "rewards": [1.0, 0.0],
        "values": None,
        "response_lengths": [0, 1],
        "loss_masks": [torch.tensor([]), torch.tensor([1.0])],
        "total_lengths": [2, 3],
        "group_indices": [3, 3],
    }

    loss.compute_advantages_and_returns(args, rollout_data)

    assert rollout_data["advantages"][0].numel() == 0
    assert torch.allclose(rollout_data["advantages"][1], torch.tensor([-1.0]))
    assert torch.equal(torch.cat(rollout_data["rloo_empty_response_fraction"]), torch.tensor([1.0, 0.0]))
