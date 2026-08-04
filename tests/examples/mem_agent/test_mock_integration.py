# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU-only integration of rollout, reward, expansion, and queue transfer."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from examples.mem_agent.reward import reward_func
from examples.mem_agent.rollout import generate_trajectory
from relax.utils.types import Sample
from relax.utils.utils import CURRENT_ROLLOUT_BATCH, transfer_batch_to_data_system


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return f"<chat>{messages[0]['content']}</chat>"


def _args():
    return SimpleNamespace(
        mem_agent_chunk_tokens=3,
        mem_agent_max_memory_tokens=16,
        mem_agent_max_final_tokens=16,
        mem_agent_max_chunks=8,
        mem_agent_credit_assignment="split",
        custom_convert_samples_to_train_data_path="examples.mem_agent.convert.convert_samples",
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
        debug_train_only=False,
    )


@pytest.mark.asyncio
async def test_mock_mem_agent_pipeline_transfers_every_turn_once():
    tokenizer = FakeTokenizer()
    turn_positions = defaultdict(int)
    observed_prompts = defaultdict(list)

    async def fake_generate(args, turn, sampling_params, evaluation):
        del args, sampling_params, evaluation
        position = turn_positions[turn.index]
        turn_positions[turn.index] += 1
        observed_prompts[turn.index].append(turn.prompt)
        if position < 2:
            response = f"M{position + 1}"
        else:
            response = r"\boxed{x}" if turn.index == 0 else r"\boxed{wrong}"
        response_ids = tokenizer.encode(response)
        turn.response = response
        turn.tokens = tokenizer.encode(turn.prompt) + response_ids
        turn.rollout_tokens = list(turn.tokens)
        turn.response_length = len(response_ids)
        turn.loss_mask = [1] * len(response_ids)
        turn.rollout_log_probs = [-0.2] * len(response_ids)
        turn.status = Sample.Status.COMPLETED
        return turn

    args = _args()
    samples = []
    for index in range(2):
        sample = Sample(
            index=index,
            group_index=5,
            prompt="Question?",
            metadata={"context": "abcdef", "ground_truth": ["x"]},
        )
        sample = await generate_trajectory(args, sample, {}, tokenizer, generator=fake_generate)
        sample.reward = await reward_func(args, sample)
        samples.append(sample)

    class Client:
        payload = None

        async def async_put(self, **kwargs):
            self.payload = kwargs

    client = Client()
    CURRENT_ROLLOUT_BATCH.clear()
    destination_rollout_id, transferred_rows = await transfer_batch_to_data_system(
        args, [samples], 1, 0, client, is_last=True
    )
    train_data = client.payload["data"]

    assert destination_rollout_id == 0
    assert transferred_rows == 6
    assert len(train_data["tokens"]) == 6
    assert train_data["sample_indices"].tolist() == [0, 0, 0, 1, 1, 1]
    assert train_data["turn_indices"].tolist() == [0, 1, 2, 0, 1, 2]
    assert train_data["raw_reward"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert len(client.payload["custom_meta"]) == 6
    assert all("<section>" not in prompts[-1] for prompts in observed_prompts.values())
    assert all("abc" not in prompts[-1] and "def" not in prompts[-1] for prompts in observed_prompts.values())
