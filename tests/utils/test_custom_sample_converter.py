# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from examples.mem_agent.convert import convert_samples
from relax.utils.types import Sample
from relax.utils.utils import (
    CURRENT_ROLLOUT_BATCH,
    convert_samples_to_train_data_with_custom,
    get_debug_data,
    get_train_data_group_size,
    get_train_sample_expansion_factor,
    transfer_batch_to_data_system,
)


def _sample() -> Sample:
    turns = []
    for turn_index in range(3):
        turns.append(
            {
                "turn_index": turn_index,
                "kind": "final" if turn_index == 2 else "memory",
                "tokens": [10 + turn_index, 20, 21],
                "response_length": 2,
                "loss_mask": [1, 1],
                "rollout_log_probs": [-0.1, -0.2],
                "finish_reason": Sample.Status.COMPLETED.value,
            }
        )
    return Sample(
        index=7,
        group_index=0,
        reward={"score": 1.0},
        status=Sample.Status.COMPLETED,
        train_metadata={"mem_agent_turns": turns},
    )


def _args(**overrides):
    values = {
        "custom_convert_samples_to_train_data_path": "examples.mem_agent.convert.convert_samples",
        "custom_reward_post_process_path": None,
        "agentic_custom_advantage_path": None,
        "reward_key": "score",
        "advantage_estimator": "grpo",
        "rewards_normalization": False,
        "grpo_std_normalization": True,
        "n_samples_per_prompt": 1,
        "mem_agent_credit_assignment": "split",
        "mem_agent_max_memory_tokens": 1024,
        "mem_agent_max_final_tokens": 256,
        "debug_train_only": False,
        "load_debug_rollout_data_subsample": None,
        "use_dynamic_global_batch_size": False,
        "custom_train_expanded_batch": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_converter_dispatch_preserves_default_and_custom_paths(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("relax.utils.utils.convert_samples_to_train_data", lambda args, samples: sentinel)
    args = _args(custom_convert_samples_to_train_data_path=None)
    assert convert_samples_to_train_data_with_custom(args, [_sample()]) is sentinel

    custom = object()
    monkeypatch.setattr("relax.utils.utils.load_function", lambda path: lambda args, samples: custom)
    args.custom_convert_samples_to_train_data_path = "package.converter"
    assert convert_samples_to_train_data_with_custom(args, [_sample()]) is custom


def test_debug_replay_uses_custom_converter(tmp_path):
    path_template = str(tmp_path / "rollout_{rollout_id}.pt")
    torch.save({"samples": [_sample().to_dict()]}, path_template.format(rollout_id=3))
    args = _args(debug_train_only=True, load_debug_rollout_data=path_template)
    replay = get_debug_data(args, rollout_id=3, batch_size=3, dp_rank=0)
    assert replay["turn_indices"] == [0, 1, 2]
    assert len(replay["tokens"]) == 3


def test_dynamic_debug_replay_slices_expanded_rows_across_dp(tmp_path):
    first = _sample()
    second = _sample()
    second.index = 8
    second.group_index = 1
    path_template = str(tmp_path / "rollout_{rollout_id}.pt")
    torch.save({"samples": [first.to_dict(), second.to_dict()]}, path_template.format(rollout_id=4))
    args = _args(
        debug_train_only=True,
        load_debug_rollout_data=path_template,
        custom_train_expanded_batch=True,
    )
    replay = get_debug_data(args, rollout_id=4, batch_size=None, dp_rank=1, dp_size=2)
    assert len(replay["tokens"]) == 3
    assert replay["sample_indices"] == [8, 8, 8]


@pytest.mark.asyncio
async def test_online_transfer_queues_every_expanded_turn():
    class Client:
        payload = None

        async def async_put(self, **kwargs):
            self.payload = kwargs

    client = Client()
    CURRENT_ROLLOUT_BATCH.clear()
    destination_rollout_id, transferred_rows = await transfer_batch_to_data_system(
        _args(), [[_sample()]], 1, 5, client, is_last=True
    )
    assert destination_rollout_id == 5
    assert transferred_rows == 3
    assert len(client.payload["data"]["total_lengths"]) == 3
    assert len(client.payload["custom_meta"]) == 3
    assert client.payload["partition_id"] == "train_5"
    assert client.payload["is_last"] is True


def test_expansion_capacity_and_group_defaults_are_backward_compatible():
    default = SimpleNamespace(n_samples_per_prompt=8)
    expanded = SimpleNamespace(
        n_samples_per_prompt=8,
        custom_train_sample_expansion_factor=65,
        custom_train_data_group_size=1,
    )
    assert get_train_sample_expansion_factor(default) == 1
    assert get_train_data_group_size(default) == 8
    assert get_train_sample_expansion_factor(expanded) == 65
    assert get_train_data_group_size(expanded) == 1


def test_custom_converter_function_returns_all_rows_without_tail_trim():
    data = convert_samples(_args(debug_train_only=True), [_sample()])
    assert len(data["tokens"]) == 3
