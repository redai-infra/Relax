# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from examples.mem_agent.convert import convert_samples
from relax.utils.types import Sample


def _turn(turn_index: int) -> dict:
    return {
        "turn_index": turn_index,
        "kind": "final" if turn_index == 2 else "memory",
        "tokens": [100 + turn_index, 200 + turn_index, 201 + turn_index],
        "response_length": 2,
        "loss_mask": [1, 1],
        "rollout_log_probs": [-0.1, -0.2],
        "finish_reason": Sample.Status.COMPLETED.value,
    }


def _sample(index: int, reward: float, turn_count: int) -> Sample:
    return Sample(
        index=index,
        group_index=9,
        reward={"score": reward},
        status=Sample.Status.COMPLETED,
        train_metadata={"mem_agent_turns": [_turn(turn_index) for turn_index in range(turn_count)]},
    )


def _args(credit_assignment="split", debug_train_only=True):
    return SimpleNamespace(
        reward_key="score",
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        n_samples_per_prompt=2,
        grpo_std_normalization=True,
        mem_agent_credit_assignment=credit_assignment,
        debug_train_only=debug_train_only,
    )


def test_converter_normalizes_before_expansion_and_keeps_all_turns():
    samples = [_sample(3, 0.0, 3), _sample(4, 1.0, 3)]
    data = convert_samples(_args(), samples)
    normalized = torch.tensor([0.0, 1.0])
    normalized = (normalized - normalized.mean()) / (normalized.std() + 1e-6)

    assert len(data["tokens"]) == 6
    assert data["sample_indices"] == [3, 3, 3, 4, 4, 4]
    assert data["turn_indices"] == [0, 1, 2, 0, 1, 2]
    assert data["rewards"][:3] == pytest.approx([normalized[0].item() / 3] * 3)
    assert data["rewards"][3:] == pytest.approx([normalized[1].item() / 3] * 3)
    assert data["raw_reward"] == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_converter_share_credit_and_tensordict_contract():
    samples = [_sample(3, 0.0, 2), _sample(4, 1.0, 2)]
    data = convert_samples(_args(credit_assignment="share", debug_train_only=False), samples)
    assert len(data["total_lengths"]) == 4
    assert data.batch_size[0] == 4
    assert data["rewards"][0].item() == pytest.approx(data["rewards"][1].item())


def test_converter_rejects_misaligned_or_failed_trajectory():
    sample = _sample(3, 1.0, 1)
    sample.train_metadata["mem_agent_turns"][0]["rollout_log_probs"] = [-0.1]
    with pytest.raises(ValueError, match="misaligned rollout_log_probs"):
        convert_samples(_args(), [sample, _sample(4, 0.0, 1)])

    failed = _sample(5, 0.0, 1)
    failed.status = Sample.Status.FAILED
    with pytest.raises(ValueError, match="status=failed"):
        convert_samples(_args(), [failed, _sample(6, 1.0, 1)])


def test_converter_rejects_inconsistent_group_turn_counts_and_non_divisible_rows():
    with pytest.raises(ValueError, match="inconsistent turn counts"):
        convert_samples(_args(), [_sample(3, 0.0, 1), _sample(4, 1.0, 2)])

    args = _args()
    args.mem_agent_train_rows_multiple = 4
    with pytest.raises(ValueError, match="is not divisible"):
        convert_samples(args, [_sample(3, 0.0, 3), _sample(4, 1.0, 3)])
