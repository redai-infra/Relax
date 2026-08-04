# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from examples.mem_agent.prompts import NO_MEMORY, truncate_text_to_tokens
from examples.mem_agent.rollout import chunk_context, generate, generate_trajectory
from relax.utils.types import Sample


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<chat>{messages[0]['content']}</chat>"


def _args():
    return SimpleNamespace(
        mem_agent_chunk_tokens=3,
        mem_agent_max_memory_tokens=64,
        mem_agent_max_final_tokens=32,
        mem_agent_max_chunks=8,
    )


@pytest.mark.asyncio
async def test_generate_trajectory_overwrites_memory_and_expands_every_turn():
    tokenizer = FakeTokenizer()
    responses = ["M1<|im_end|>", "M2", r"The answer is \boxed{x}."]
    prompts = []
    max_new_tokens = []

    async def fake_generate(args, turn, sampling_params, evaluation):
        del args
        prompts.append(turn.prompt)
        max_new_tokens.append(sampling_params["max_new_tokens"])
        response = responses.pop(0)
        prompt_ids = tokenizer.encode(turn.prompt)
        response_ids = tokenizer.encode(response)
        turn.tokens = prompt_ids + response_ids
        turn.rollout_tokens = list(turn.tokens)
        turn.response = response
        turn.response_length = len(response_ids)
        turn.loss_mask = [1] * len(response_ids)
        turn.rollout_log_probs = [] if evaluation else [-0.25] * len(response_ids)
        turn.status = Sample.Status.COMPLETED
        return turn

    sample = Sample(index=11, group_index=2, prompt="Where?", metadata={"context": "abcdef"}, session_id="s")
    result = await generate_trajectory(
        _args(), sample, {"temperature": 1.0}, tokenizer, generator=fake_generate, evaluation=False
    )

    assert result.status == Sample.Status.COMPLETED
    assert result.response == r"The answer is \boxed{x}."
    assert result.metadata["num_chunks"] == 2
    assert result.metadata["context_truncated"] is False
    assert result.metadata["memory_token_lengths"] == [2, 2]
    assert len(result.train_metadata["mem_agent_turns"]) == 3
    assert NO_MEMORY in prompts[0]
    assert "abc" in prompts[0]
    assert "<memory>\nM1\n</memory>" in prompts[1]
    assert "abc" not in prompts[1]
    assert "def" in prompts[1]
    assert "<memory>\nM2\n</memory>" in prompts[2]
    assert "<section>" not in prompts[2]
    assert "abc" not in prompts[2] and "def" not in prompts[2]
    assert max_new_tokens == [64, 64, 32]

    for turn in result.train_metadata["mem_agent_turns"]:
        assert len(turn["loss_mask"]) == turn["response_length"]
        assert len(turn["rollout_log_probs"]) == turn["response_length"]
        assert len(turn["tokens"]) >= turn["response_length"]


def test_chunk_context_preserves_boundaries_and_marks_truncation():
    tokenizer = FakeTokenizer()
    chunks, truncated = chunk_context(tokenizer, "abcdefgh", chunk_tokens=3, max_chunks=2)
    assert [tokenizer.decode(chunk) for chunk in chunks] == ["abc", "def"]
    assert truncated is True


def test_truncate_text_to_tokens_retokenizes_to_the_hard_limit():
    text, length = truncate_text_to_tokens(FakeTokenizer(), "MEMORY", 3)
    assert text == "MEM"
    assert length == 3

    class ExpandingTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens)
            return ids + ([0] if text.endswith("!") else [])

    text, length = truncate_text_to_tokens(ExpandingTokenizer(), "AB!", 3)
    assert text == "AB"
    assert length == 2


@pytest.mark.asyncio
async def test_generate_trajectory_aborts_without_context():
    sample = Sample(index=0, prompt="question", metadata={})
    result = await generate_trajectory(_args(), sample, {}, FakeTokenizer())
    assert result.status == Sample.Status.ABORTED
    assert result.train_metadata is None


@pytest.mark.asyncio
async def test_generate_trajectory_bounds_memory_before_next_turn():
    tokenizer = FakeTokenizer()
    responses = ["MEMORY", r"\boxed{x}"]
    prompts = []

    async def fake_generate(args, turn, sampling_params, evaluation):
        del args, evaluation
        prompts.append(turn.prompt)
        response = responses.pop(0)
        response_ids = tokenizer.encode(response)[: sampling_params["max_new_tokens"]]
        turn.response = tokenizer.decode(response_ids)
        turn.tokens = tokenizer.encode(turn.prompt) + response_ids
        turn.rollout_tokens = list(turn.tokens)
        turn.response_length = len(response_ids)
        turn.loss_mask = [1] * len(response_ids)
        turn.rollout_log_probs = [-0.1] * len(response_ids)
        turn.status = Sample.Status.TRUNCATED if response == "MEMORY" else Sample.Status.COMPLETED
        return turn

    args = _args()
    args.mem_agent_max_memory_tokens = 3
    sample = Sample(index=0, group_index=0, prompt="Q", metadata={"context": "abc"})
    result = await generate_trajectory(args, sample, {}, tokenizer, generator=fake_generate)
    assert result.metadata["memory_token_lengths"] == [3]
    assert "<memory>\nMEM\n</memory>" in prompts[-1]
    assert "MEMORY" not in prompts[-1]


@pytest.mark.asyncio
async def test_generate_trajectory_empty_update_does_not_reuse_old_memory():
    tokenizer = FakeTokenizer()
    responses = ["M1", "<|im_end|>", r"\boxed{x}"]
    prompts = []

    async def fake_generate(args, turn, sampling_params, evaluation):
        del args, sampling_params, evaluation
        prompts.append(turn.prompt)
        response = responses.pop(0)
        response_ids = tokenizer.encode(response)
        turn.response = response
        turn.tokens = tokenizer.encode(turn.prompt) + response_ids
        turn.rollout_tokens = list(turn.tokens)
        turn.response_length = len(response_ids)
        turn.loss_mask = [1] * len(response_ids)
        turn.rollout_log_probs = [-0.1] * len(response_ids)
        turn.status = Sample.Status.COMPLETED
        return turn

    sample = Sample(index=0, group_index=0, prompt="Q", metadata={"context": "abcdef"})
    result = await generate_trajectory(_args(), sample, {}, tokenizer, generator=fake_generate)

    assert result.metadata["memory_token_lengths"] == [2, 0]
    assert "<memory>\n\n</memory>" in prompts[-1]
    assert "<memory>\nM1\n</memory>" not in prompts[-1]


@pytest.mark.asyncio
async def test_public_generate_entry_uses_sglang_state_and_turn_generator(monkeypatch):
    """Exercise the exact callable wired by --custom-generate-function-path."""
    tokenizer = FakeTokenizer()
    responses = iter(["MEM", r"\boxed{x}"])

    class FakeGenerateState:
        def __init__(self, args):
            del args
            self.tokenizer = tokenizer

    async def fake_sglang_generate(args, turn, sampling_params, evaluation):
        del args, sampling_params, evaluation
        response = next(responses)
        response_ids = tokenizer.encode(response)
        turn.response = response
        turn.tokens = tokenizer.encode(turn.prompt) + response_ids
        turn.rollout_tokens = list(turn.tokens)
        turn.response_length = len(response_ids)
        turn.loss_mask = [1] * len(response_ids)
        turn.rollout_log_probs = [-0.1] * len(response_ids)
        turn.status = Sample.Status.COMPLETED
        return turn

    fake_module = ModuleType("relax.engine.rollout.sglang_rollout")
    fake_module.GenerateState = FakeGenerateState
    fake_module.generate = fake_sglang_generate
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    sample = Sample(index=3, group_index=0, prompt="Q", metadata={"context": "abc"})
    result = await generate(_args(), sample, {"temperature": 1.0})

    assert result.status == Sample.Status.COMPLETED
    assert result.response == r"\boxed{x}"
    assert [turn["kind"] for turn in result.train_metadata["mem_agent_turns"]] == ["memory", "final"]
