# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Chunk-by-chunk recurrent memory rollout for MemAgent."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Awaitable, Callable
from typing import Any

from examples.mem_agent.contracts import require_strict_alignment
from examples.mem_agent.prompts import (
    NO_MEMORY,
    final_instruction,
    memory_instruction,
    render_chat_prompt,
    strip_stop_tokens,
    truncate_text_to_tokens,
)
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)
TurnGenerator = Callable[[Namespace, Sample, dict[str, Any], bool], Awaitable[Sample]]


def chunk_context(tokenizer: Any, context: str, chunk_tokens: int, max_chunks: int) -> tuple[list[list[int]], bool]:
    """Split context on tokenizer boundaries without overlap or token loss."""
    if chunk_tokens <= 0:
        raise ValueError("mem_agent_chunk_tokens must be positive.")
    if max_chunks <= 0:
        raise ValueError("mem_agent_max_chunks must be positive.")
    token_ids = tokenizer.encode(context, add_special_tokens=False)
    chunks = [token_ids[offset : offset + chunk_tokens] for offset in range(0, len(token_ids), chunk_tokens)]
    return chunks[:max_chunks], len(chunks) > max_chunks


def _question_from_sample(sample: Sample) -> str:
    question = sample.metadata.get("question") if isinstance(sample.metadata, dict) else None
    if question:
        return str(question)
    if isinstance(sample.prompt, str):
        return sample.prompt
    if sample.prompt:
        return str(sample.prompt[-1].get("content", ""))
    return ""


def _validate_turn(turn: dict[str, Any], require_log_probs: bool, max_response_tokens: int) -> None:
    if turn["finish_reason"] not in (Sample.Status.COMPLETED.value, Sample.Status.TRUNCATED.value):
        raise ValueError(f"MemAgent turn has invalid finish status: {turn['finish_reason']}.")
    response_length = turn["response_length"]
    if response_length <= 0:
        raise ValueError("MemAgent turn returned no trainable response tokens.")
    if response_length > max_response_tokens:
        raise ValueError(
            f"MemAgent turn returned {response_length} response tokens, exceeding its limit {max_response_tokens}."
        )
    if len(turn["tokens"]) < response_length:
        raise ValueError("MemAgent turn response_length exceeds total token count.")
    if len(turn["loss_mask"]) != response_length:
        raise ValueError("MemAgent turn loss_mask is not aligned with response tokens.")
    log_probs = turn["rollout_log_probs"]
    if require_log_probs and len(log_probs) != response_length:
        raise ValueError("MemAgent turn rollout_log_probs are not aligned with response tokens.")


def _turn_record(
    turn_sample: Sample,
    turn_index: int,
    kind: str,
    evaluation: bool,
    max_response_tokens: int,
) -> dict[str, Any]:
    record = {
        "turn_index": turn_index,
        "kind": kind,
        "tokens": list(turn_sample.tokens),
        "response_length": turn_sample.response_length,
        "loss_mask": list(turn_sample.loss_mask or [1] * turn_sample.response_length),
        "rollout_log_probs": list(turn_sample.rollout_log_probs or []),
        "finish_reason": turn_sample.status.value,
    }
    _validate_turn(record, require_log_probs=not evaluation, max_response_tokens=max_response_tokens)
    return record


async def _run_turn(
    args: Namespace,
    parent: Sample,
    prompt: str,
    sampling_params: dict[str, Any],
    max_new_tokens: int,
    evaluation: bool,
    generator: TurnGenerator,
) -> Sample:
    turn = Sample(
        group_index=parent.group_index,
        index=parent.index,
        prompt=prompt,
        metadata={},
        session_id=parent.session_id,
    )
    params = {**sampling_params, "max_new_tokens": max_new_tokens}
    return await generator(args, turn, params, evaluation)


async def generate_trajectory(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    tokenizer: Any,
    generator: TurnGenerator | None = None,
    evaluation: bool = False,
) -> Sample:
    """Generate all independent memory turns and the final-answer turn."""
    sample.metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    context = str(sample.metadata.get("context", ""))
    question = _question_from_sample(sample).strip()
    if not context or not question:
        sample.status = Sample.Status.ABORTED
        sample.rollout_log_probs = []
        return sample

    chunk_tokens = int(getattr(args, "mem_agent_chunk_tokens", 2048))
    max_memory_tokens = int(getattr(args, "mem_agent_max_memory_tokens", 1024))
    max_final_tokens = int(getattr(args, "mem_agent_max_final_tokens", 256))
    max_chunks = int(getattr(args, "mem_agent_max_chunks", 64))
    enable_thinking = getattr(args, "mem_agent_enable_thinking", None)
    require_strict_alignment(args)
    if max_memory_tokens <= 0 or max_final_tokens <= 0:
        raise ValueError("MemAgent memory and final response limits must be positive.")

    chunks, context_truncated = chunk_context(tokenizer, context, chunk_tokens, max_chunks)
    if not chunks:
        sample.status = Sample.Status.ABORTED
        sample.rollout_log_probs = []
        return sample
    if generator is None:
        raise ValueError("A turn generator is required for a non-empty MemAgent trajectory.")

    turns: list[dict[str, Any]] = []
    memory = NO_MEMORY
    memory_token_lengths: list[int] = []
    any_turn_truncated = False

    # Every chunk starts an independent conversation. Only the generated
    # memory text survives; prior prompts and token history are not appended.
    for chunk_ids in chunks:
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        prompt = render_chat_prompt(
            tokenizer,
            memory_instruction(question, memory, chunk, evaluation=evaluation),
            enable_thinking=enable_thinking,
        )
        turn_sample = await _run_turn(
            args,
            sample,
            prompt,
            sampling_params,
            max_memory_tokens,
            evaluation,
            generator,
        )
        if turn_sample.status in (Sample.Status.ABORTED, Sample.Status.FAILED):
            # ReLax retries ABORTED groups. Never return a partially populated
            # FAILED trajectory that the transfer path could accidentally see.
            sample.status = Sample.Status.ABORTED
            sample.rollout_log_probs = []
            return sample
        turns.append(_turn_record(turn_sample, len(turns), "memory", evaluation, max_memory_tokens))
        # Overwrite on every turn, including an empty post-processed response;
        # carrying the old value forward would silently change the recurrence.
        memory, memory_length = truncate_text_to_tokens(
            tokenizer,
            strip_stop_tokens(turn_sample.response),
            max_memory_tokens,
        )
        memory_token_lengths.append(memory_length)
        any_turn_truncated = any_turn_truncated or turn_sample.status == Sample.Status.TRUNCATED

    # The final request deliberately excludes context/chunks. This enforces
    # the question + latest-memory information boundary from the task spec.
    final_prompt = render_chat_prompt(
        tokenizer,
        final_instruction(question, memory, evaluation=evaluation),
        enable_thinking=enable_thinking,
    )
    final_sample = await _run_turn(
        args,
        sample,
        final_prompt,
        sampling_params,
        max_final_tokens,
        evaluation,
        generator,
    )
    if final_sample.status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        sample.status = Sample.Status.ABORTED
        sample.rollout_log_probs = []
        return sample
    turns.append(_turn_record(final_sample, len(turns), "final", evaluation, max_final_tokens))

    final_output = strip_stop_tokens(final_sample.response)
    # Preserve a valid top-level Sample for logs and failure recovery. The
    # custom converter trains from all turn records below, not this final view.
    sample.prompt = final_prompt
    sample.response = final_output
    sample.tokens = list(final_sample.tokens)
    sample.rollout_tokens = list(final_sample.rollout_tokens)
    sample.response_length = final_sample.response_length
    sample.loss_mask = list(final_sample.loss_mask or [1] * final_sample.response_length)
    sample.rollout_log_probs = list(final_sample.rollout_log_probs or [])
    sample.metadata.update(
        {
            "question": question,
            "final_output": final_output,
            "num_chunks": len(chunks),
            "context_truncated": context_truncated,
            "any_turn_truncated": any_turn_truncated or final_sample.status == Sample.Status.TRUNCATED,
            "memory_token_lengths": memory_token_lengths,
            "final_memory_tokens": memory_token_lengths[-1],
        }
    )
    sample.train_metadata = {"mem_agent_turns": turns}
    sample.status = final_sample.status
    return sample


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """ReLax custom-generate entry point."""
    # Keep SGLang imports lazy so pure chunk/prompt tests can run in a
    # lightweight CPU environment without initializing serving dependencies.
    from relax.engine.rollout.sglang_rollout import GenerateState
    from relax.engine.rollout.sglang_rollout import generate as sglang_generate

    tokenizer = GenerateState(args).tokenizer
    try:
        return await generate_trajectory(
            args,
            sample,
            sampling_params,
            tokenizer,
            generator=sglang_generate,
            evaluation=evaluation,
        )
    except Exception as exc:
        logger.error(f"MemAgent rollout failed for sample index={sample.index}: {exc}")
        sample.response = ""
        sample.rollout_log_probs = []
        sample.train_metadata = None
        sample.status = Sample.Status.ABORTED
        return sample
