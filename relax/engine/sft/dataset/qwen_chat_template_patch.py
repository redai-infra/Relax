# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen chat-template compatibility patches for SFT."""

import hashlib
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from relax.engine.sft.dataset.chat_template_patch import TemplatePatchResult
from relax.engine.sft.dataset.sample import CanonicalSample


_QWEN_HISTORY_GATE = "{%- if loop.index0 > ns.last_query_index %}"
_QWEN_PRESERVE_HISTORY_GATE = (
    "{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}"
)
# Qwen3.8 (model_type qwen3_5) ships a third gate form that defaults to
# preserving historical thinking (undefined/true keeps it, explicit false drops
# it). This is already the behavior our auto-preserve resolution wants, so it
# needs no template rewrite — only recognition so the RuntimeError guard below
# does not trip on it.
_QWEN38_PRESERVE_HISTORY_GATE = (
    "{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}"
)
_PATCH_NAME = "qwen_history_thinking"


def _content_as_text(content: str | list[dict] | None) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        item.get("text", "") for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)
    )


def _is_tool_response_user_message(content: str | list[dict] | None) -> bool:
    if not isinstance(content, str):
        return False
    text = content.strip()
    return text.startswith("<tool_response>") and text.endswith("</tool_response>")


def _has_learnable_historical_thinking(sample: CanonicalSample) -> bool:
    """Whether the last user turn makes supervised assistant thinking
    historical."""
    last_user_index = max(
        (
            index
            for index, message in enumerate(sample.messages)
            if message.role == "user" and not _is_tool_response_user_message(message.content)
        ),
        default=-1,
    )
    if last_user_index < 0:
        return False
    return any(
        index < last_user_index
        and message.role == "assistant"
        and message.learn
        and "</think>" in _content_as_text(message.content)
        for index, message in enumerate(sample.messages)
    )


@lru_cache(maxsize=32)
def _patch_qwen_history_gate(template: str) -> tuple[str, bool] | None:
    """Backport Qwen3.6's preserve gate to the exact Qwen3.5 gate."""
    old_count = template.count(_QWEN_HISTORY_GATE)
    native_count = template.count(_QWEN_PRESERVE_HISTORY_GATE)
    qwen38_count = template.count(_QWEN38_PRESERVE_HISTORY_GATE)
    looks_like_qwen_history = "ns.last_query_index" in template and "reasoning_content" in template
    if old_count == 0 and native_count == 0 and qwen38_count == 0 and not looks_like_qwen_history:
        return None
    if old_count == 1 and native_count == 0 and qwen38_count == 0:
        return template.replace(_QWEN_HISTORY_GATE, _QWEN_PRESERVE_HISTORY_GATE, 1), True
    if old_count == 0 and native_count == 1 and qwen38_count == 0:
        return template, False
    if old_count == 0 and native_count == 0 and qwen38_count == 1:
        # Qwen3.8 gate already preserves by default; use it as-is.
        return template, False

    template_hash = hashlib.sha256(template.encode()).hexdigest()[:16]
    raise RuntimeError(
        "Cannot safely patch the Qwen history-thinking gate: "
        f"expected one old gate or one native gate, found old={old_count}, native={native_count}, "
        f"and qwen38={qwen38_count} (template sha256={template_hash})."
    )


def try_patch_qwen_chat_template(
    sample: CanonicalSample,
    template: str | None,
    kwargs: Mapping[str, Any],
) -> TemplatePatchResult | None:
    """Patch recognized Qwen history templates and resolve preserve policy."""
    if not isinstance(template, str):
        return None
    patched = _patch_qwen_history_gate(template)
    if patched is None:
        return None

    patched_template, changed = patched
    resolved_kwargs = dict(kwargs)
    preserve_thinking = resolved_kwargs.get("preserve_thinking")
    if preserve_thinking is not None and not isinstance(preserve_thinking, bool):
        raise ValueError("apply_chat_template_kwargs.preserve_thinking must be true, false, or null")

    # Explicit booleans are hard overrides; missing/null enables sample-aware auto-preserve.
    if preserve_thinking is None and _has_learnable_historical_thinking(sample):
        resolved_kwargs["preserve_thinking"] = True

    return TemplatePatchResult(
        template=patched_template,
        kwargs=resolved_kwargs,
        patch_name=_PATCH_NAME,
        changed=changed,
    )
