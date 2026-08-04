# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prompt templates shared by MemAgent rollout and evaluation."""

from __future__ import annotations

from typing import Any


NO_MEMORY = "No previous memory"
STOP_TOKEN_STRINGS = ("<|im_end|>", "<|endoftext|>")

MEMORY_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>{problem_tag_suffix}
{question}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

FINAL_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>{problem_tag_suffix}
{question}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""


def strip_stop_tokens(text: str) -> str:
    """Remove model terminators retained by some serving backends."""
    for token in STOP_TOKEN_STRINGS:
        text = text.replace(token, "")
    return text.strip()


def truncate_text_to_tokens(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, int]:
    """Clamp generated memory after re-tokenization and report its length."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    bounded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    bounded_ids = tokenizer.encode(bounded_text, add_special_tokens=False)
    # Decode/encode is not guaranteed to be token-id preserving for every
    # tokenizer. Trim characters only in that rare case so the text inserted
    # into the next prompt has a verifiable hard token bound.
    while len(bounded_ids) > max_tokens and bounded_text:
        bounded_text = bounded_text[:-1]
        bounded_ids = tokenizer.encode(bounded_text, add_special_tokens=False)
    return bounded_text, len(bounded_ids)


def _problem_tag_suffix(evaluation: bool) -> str:
    """Preserve the one-character prompt difference in fixed VIME code.

    VIME training has ``<problem> `` while its evaluator has ``<problem>``.
    Keeping the variants explicit makes both reproduction paths byte-aligned
    instead of silently choosing one template for both.
    """
    return "" if evaluation else " "


def memory_instruction(question: str, memory: str, chunk: str, *, evaluation: bool = False) -> str:
    return MEMORY_TEMPLATE.format(
        question=question,
        memory=memory,
        chunk=chunk,
        problem_tag_suffix=_problem_tag_suffix(evaluation),
    )


def final_instruction(question: str, memory: str, *, evaluation: bool = False) -> str:
    return FINAL_TEMPLATE.format(
        question=question,
        memory=memory,
        problem_tag_suffix=_problem_tag_suffix(evaluation),
    )


def render_chat_prompt(tokenizer: Any, instruction: str, *, enable_thinking: bool | None = None) -> str:
    """Render one independent user turn with the model's chat template.

    The formal Qwen3-4B reproduction leaves ``enable_thinking`` unspecified to
    match fixed VIME. The resource-constrained Qwen3-0.6B pilot can disable it
    explicitly so a short response budget contains memory/answer text instead
    of only a truncated reasoning trace.
    """
    messages = [{"role": "user", "content": instruction}]
    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
