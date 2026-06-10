# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Render CanonicalSample → (input_ids, loss_mask) tensors via tokenizer chat
template.

Two paths (spec §7.5):
1. Preferred: `apply_chat_template(..., return_assistant_tokens_mask=True)` — relies
   on the model's official jinja template containing `{% generation %}` tags.
2. Fallback: per-message tokenize and concatenate, using `CanonicalMessage.learn`
   to build the mask. Used when the template lacks `{% generation %}`.
"""

import re
from collections.abc import Mapping
from typing import Any

import torch

from relax.engine.sft.dataset.sample import CanonicalSample
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_GENERATION_MARKER_RE = re.compile(r"{%\s*generation\s*%}")
_FALLBACK_WARNED: set[int] = set()  # tokenizer id → warned once


def HAS_GENERATION_MARKER(template_str: str | None) -> bool:  # noqa: N802
    if not template_str:
        return False
    return bool(_GENERATION_MARKER_RE.search(template_str))


def _to_chat_messages(sample: CanonicalSample) -> list[dict[str, Any]]:
    """Convert CanonicalMessage list to dict format expected by
    apply_chat_template."""
    out = []
    for m in sample.messages:
        out.append({"role": m.role, "content": m.content})
    return out


def _render_with_assistant_mask(sample: CanonicalSample, *, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Path 1: ask tokenizer for the assistant-only mask directly."""
    result = tokenizer.apply_chat_template(
        _to_chat_messages(sample),
        tools=sample.tools,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = result["input_ids"]
    masks = result["assistant_masks"]
    if isinstance(masks, list):
        masks = torch.tensor(masks)
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if masks.dim() == 2:
        masks = masks.squeeze(0)
    return input_ids.long(), masks.long()


_THINK_OPEN = "<think>\n"
_IM_END = "<|im_end|>"


def _render_per_message_fallback(sample: CanonicalSample, *, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Path 2: single full render + char-level mask projected back through
    `offset_mapping`.

    Approach mirrors slime PR THUDM/slime#1742: rendering messages one at a
    time breaks on templates that validate the message list as a whole
    (e.g. Qwen3.5-VL aborts with "No user query found in messages." on an
    assistant-only list) and also on templates that re-render past turns
    based on the full message sequence (Qwen3.5-VL drops `<think>` blocks
    from prior assistants once a new user turn appears, so any
    chunked-prefix length delta is wrong on multi-turn). Instead:

      1. ``apply_chat_template(messages, tokenize=False)`` → ``rendered_text``
      2. fast-tokenize that text with ``return_offsets_mapping=True``
      3. sanity-check the re-tokenize matches the direct tokenize
      4. scan the text for ChatML ``<|im_start|>{role}\\n…<|im_end|>`` spans
         in declaration order, marking chars 1 for messages where
         ``learn=True`` (skipping the leading ``<think>\\n`` opener inside
         an assistant turn so the tag itself stays out of the loss)
      5. project char-mask → token-mask via a prefix-sum on ``offset_mapping``

    Requires a fast tokenizer; raises ``ValueError`` otherwise. Assumes
    Qwen-style ChatML wrapping — non-ChatML templates should expose
    ``{% generation %}`` markers so Path 1 handles them natively.
    """
    msgs = [{"role": m.role, "content": m.content} for m in sample.messages]
    rendered_text = tokenizer.apply_chat_template(msgs, tools=sample.tools, tokenize=False)

    tokenized = tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = tokenized["input_ids"]
    offset_mapping = tokenized.get("offset_mapping")
    if offset_mapping is None:
        raise ValueError(
            "SFT loss-mask fallback requires a fast tokenizer with "
            "`return_offsets_mapping` support; got a slow tokenizer."
        )

    expected = tokenizer.apply_chat_template(msgs, tools=sample.tools, tokenize=True)
    if isinstance(expected, Mapping):
        expected = expected["input_ids"]
    if isinstance(expected, torch.Tensor):
        if expected.dim() > 1:
            expected = expected[0]
        expected = expected.tolist()
    elif len(expected) > 0 and isinstance(expected[0], list):
        expected = expected[0]
    if list(token_ids) != list(expected):
        raise RuntimeError(
            "Rendered-text re-tokenization does not match direct "
            "`apply_chat_template(..., tokenize=True)` output; mask projection "
            "via offset_mapping would be unreliable."
        )

    char_mask = bytearray(len(rendered_text))  # zeros
    cursor = 0
    for msg in sample.messages:
        header = f"<|im_start|>{msg.role}\n"
        header_pos = rendered_text.find(header, cursor)
        if header_pos < 0:
            raise RuntimeError(
                f"could not locate {msg.role!r} message after cursor {cursor} in rendered chat template output"
            )
        content_start = header_pos + len(header)
        end_pos = rendered_text.find(_IM_END, content_start)
        if end_pos < 0:
            raise RuntimeError(f"could not locate <|im_end|> for {msg.role!r} message")
        span_end = end_pos + len(_IM_END)
        if span_end < len(rendered_text) and rendered_text[span_end] == "\n":
            span_end += 1
        cursor = span_end

        if not msg.learn:
            continue

        mask_start = content_start
        if msg.role == "assistant" and rendered_text[content_start : content_start + len(_THINK_OPEN)] == _THINK_OPEN:
            mask_start += len(_THINK_OPEN)
        for pos in range(mask_start, span_end):
            char_mask[pos] = 1

    psum = [0] * (len(char_mask) + 1)
    for i, c in enumerate(char_mask):
        psum[i + 1] = psum[i] + c

    loss_mask = [0] * len(token_ids)
    for i, (s, e) in enumerate(offset_mapping):
        if e > s and psum[e] - psum[s] > 0:
            loss_mask[i] = 1

    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(loss_mask, dtype=torch.long)


def render_with_loss_mask(sample: CanonicalSample, *, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a single sample.

    Returns 1D `(input_ids, loss_mask)` int64 tensors.
    """
    template = getattr(tokenizer, "chat_template", None)
    if HAS_GENERATION_MARKER(template):
        return _render_with_assistant_mask(sample, tokenizer=tokenizer)

    tok_id = id(tokenizer)
    if tok_id not in _FALLBACK_WARNED:
        logger.warning(
            "Tokenizer chat_template does not contain {%% generation %%} tag — "
            "falling back to per-message tokenization for SFT loss_mask. "
            "Mask boundaries may differ slightly from the template-aware path. "
            "(This warning is shown once per tokenizer instance.)"
        )
        _FALLBACK_WARNED.add(tok_id)
    return _render_per_message_fallback(sample, tokenizer=tokenizer)


def render_to_text(sample: CanonicalSample, *, tokenizer) -> str:
    """Render a sample to the chat-template text WITHOUT tokenizing.

    Used by the multimodal path: the text (containing un-expanded
    ``<|image_pad|>`` etc. placeholders) is fed into the HF processor, which
    expands those placeholders to per-image-grid token runs and produces the
    ``input_ids`` the model actually consumes.
    """
    return tokenizer.apply_chat_template(
        _to_chat_messages(sample),
        tools=sample.tools,
        tokenize=False,
    )
