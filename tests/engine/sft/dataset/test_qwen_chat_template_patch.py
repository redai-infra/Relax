# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen history-thinking template patch tests."""

import re

import pytest
import torch

from relax.engine.sft.dataset.chat_template import render_to_text, render_with_loss_mask
from relax.engine.sft.dataset.qwen_chat_template_patch import (
    _QWEN38_PRESERVE_HISTORY_GATE,
    _QWEN_HISTORY_GATE,
    _QWEN_PRESERVE_HISTORY_GATE,
    try_patch_qwen_chat_template,
)
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample


_QWEN35_TEMPLATE = "\n".join(("template-start", _QWEN_HISTORY_GATE, "template-end"))
_QWEN36_TEMPLATE = _QWEN35_TEMPLATE.replace(_QWEN_HISTORY_GATE, _QWEN_PRESERVE_HISTORY_GATE)
# Qwen3.8 ships a preserve-by-default gate that also references reasoning_content.
_QWEN38_TEMPLATE = "\n".join(("template-start reasoning_content", _QWEN38_PRESERVE_HISTORY_GATE, "template-end"))


def _make_sample(*, historical_learn: bool = True) -> CanonicalSample:
    return CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="plan a trip", learn=False),
            CanonicalMessage(
                role="assistant",
                content="<think>\nNEED_SKILL\n</think>\n\n",
                learn=historical_learn,
                tool_calls=[
                    {
                        "type": "function",
                        "function": {
                            "name": "activate_skill",
                            "arguments": {"skill_name": "daily-overview"},
                        },
                    }
                ],
            ),
            CanonicalMessage(role="tool", content="skill loaded", learn=False),
            CanonicalMessage(role="user", content="# daily-overview skill", learn=False),
            CanonicalMessage(
                role="assistant",
                content="<think>\nFINAL_REASON\n</think>\n\nfinal answer",
                learn=True,
            ),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )


class _FakeQwenHistoryTokenizer:
    """Execute only the Qwen history behavior relevant to this regression."""

    chat_template = _QWEN35_TEMPLATE

    def __init__(self, chat_template: str | None = None):
        if chat_template is not None:
            self.chat_template = chat_template
        self.last_template = self.chat_template

    @staticmethod
    def _tokenize(text):
        return [ord(char) for char in text], [(index, index + 1) for index in range(len(text))]

    @staticmethod
    def _render_tool_calls(tool_calls):
        text = ""
        for tool_call in tool_calls or []:
            function = tool_call.get("function", tool_call)
            text += f"\n<tool_call>\n<function={function['name']}>"
            for name, value in (function.get("arguments") or {}).items():
                text += f"\n<parameter={name}>\n{value}\n</parameter>"
            text += "\n</function>\n</tool_call>"
        return text

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, **kwargs):  # noqa: ARG002
        self.last_template = kwargs.get("chat_template", self.chat_template)
        preserve = kwargs.get("preserve_thinking") is True and _QWEN_PRESERVE_HISTORY_GATE in self.last_template
        last_user_index = max(
            (index for index, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )

        rendered = ""
        previous_role = None
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content") or ""
            if role == "tool":
                if previous_role != "tool":
                    rendered += "<|im_start|>user"
                rendered += f"\n<tool_response>\n{content}\n</tool_response>"
                next_role = messages[index + 1]["role"] if index + 1 < len(messages) else None
                if next_role != "tool":
                    rendered += "<|im_end|>\n"
            else:
                if role == "assistant" and "</think>" in content:
                    thinking, _, answer = content.partition("</think>")
                    reasoning = thinking.rsplit("<think>", 1)[-1].strip("\n")
                    content = answer.lstrip("\n")
                    if preserve or index > last_user_index:
                        content = f"<think>\n{reasoning}\n</think>\n\n{content}"
                rendered += f"<|im_start|>{role}\n{content}"
                if role == "assistant":
                    rendered += self._render_tool_calls(message.get("tool_calls"))
                rendered += "<|im_end|>\n"
            previous_role = role

        if not tokenize:
            return rendered
        ids, _ = self._tokenize(rendered)
        return ids

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **kwargs):  # noqa: ARG002
        ids, offsets = self._tokenize(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def _learned_text(input_ids: torch.Tensor, loss_mask: torch.Tensor) -> str:
    return "".join(chr(int(char)) for char, mask in zip(input_ids.tolist(), loss_mask.tolist()) if mask == 1)


def test_qwen_patch_unknown_template_is_not_applicable():
    assert try_patch_qwen_chat_template(_make_sample(), "plain", {}) is None


def test_qwen35_patch_backports_gate_and_auto_preserves_history():
    result = try_patch_qwen_chat_template(_make_sample(), _QWEN35_TEMPLATE, {})
    assert result is not None
    assert result.changed
    assert result.template == _QWEN36_TEMPLATE
    assert "chat_template" not in result.kwargs
    assert result.kwargs["preserve_thinking"] is True


def test_qwen36_native_gate_is_idempotent():
    result = try_patch_qwen_chat_template(_make_sample(), _QWEN36_TEMPLATE, {})
    assert result is not None
    assert not result.changed
    assert result.template == _QWEN36_TEMPLATE
    assert "chat_template" not in result.kwargs
    assert result.kwargs["preserve_thinking"] is True


def test_qwen38_preserve_gate_is_recognized_without_rewrite():
    result = try_patch_qwen_chat_template(_make_sample(), _QWEN38_TEMPLATE, {})
    assert result is not None
    assert not result.changed
    assert result.template == _QWEN38_TEMPLATE
    assert "chat_template" not in result.kwargs
    assert result.kwargs["preserve_thinking"] is True


def test_qwen38_preserve_gate_explicit_false_disables_auto_preserve():
    result = try_patch_qwen_chat_template(
        _make_sample(),
        _QWEN38_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert not result.changed
    assert result.kwargs["preserve_thinking"] is False


def test_qwen_patch_rejects_ambiguous_gate():
    with pytest.raises(RuntimeError, match="old=2"):
        try_patch_qwen_chat_template(_make_sample(), _QWEN35_TEMPLATE + _QWEN_HISTORY_GATE, {})


def test_qwen_patch_fails_fast_on_recognized_template_drift():
    drifted = "reasoning_content\n{%- if loop.index0 >= ns.last_query_index %}"
    with pytest.raises(RuntimeError, match="old=0.*native=0"):
        try_patch_qwen_chat_template(_make_sample(), drifted, {})


def test_qwen_patch_explicit_false_disables_auto_preserve():
    result = try_patch_qwen_chat_template(
        _make_sample(),
        _QWEN35_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is False


def test_qwen_patch_explicit_null_uses_auto_preserve():
    result = try_patch_qwen_chat_template(
        _make_sample(),
        _QWEN35_TEMPLATE,
        {"preserve_thinking": None},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is True


def test_qwen_patch_rejects_non_boolean_preserve_thinking():
    with pytest.raises(ValueError, match="must be true, false, or null"):
        try_patch_qwen_chat_template(
            _make_sample(),
            _QWEN35_TEMPLATE,
            {"preserve_thinking": "true"},
        )


def test_qwen_patch_allows_compression_of_unlearned_history():
    result = try_patch_qwen_chat_template(
        _make_sample(historical_learn=False),
        _QWEN35_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is False


def test_qwen_patch_explicit_true_preserves_unlearned_history():
    result = try_patch_qwen_chat_template(
        _make_sample(historical_learn=False),
        _QWEN35_TEMPLATE,
        {"preserve_thinking": True},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is True


def test_qwen_patch_excludes_wrapped_tool_response_from_last_user_boundary():
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="query", learn=False),
            CanonicalMessage(role="assistant", content="<think>reason</think>", learn=True),
            CanonicalMessage(
                role="user",
                content="<tool_response>result</tool_response>",
                learn=False,
            ),
            CanonicalMessage(role="assistant", content="answer", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    result = try_patch_qwen_chat_template(
        sample,
        _QWEN35_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is False


def test_qwen35_render_with_loss_mask_preserves_think_before_tool_call():
    tokenizer = _FakeQwenHistoryTokenizer()
    input_ids, loss_mask = render_with_loss_mask(_make_sample(), tokenizer=tokenizer)
    learned = _learned_text(input_ids, loss_mask)

    assert tokenizer.last_template == _QWEN36_TEMPLATE
    assert tokenizer.chat_template == _QWEN35_TEMPLATE
    assert re.search(r"NEED_SKILL\n</think>\n+<tool_call>", learned)
    assert "activate_skill" in learned
    assert "skill loaded" not in learned
    assert "# daily-overview skill" not in learned


def test_qwen35_render_with_loss_mask_explicit_false_compresses_history():
    tokenizer = _FakeQwenHistoryTokenizer()
    input_ids, loss_mask = render_with_loss_mask(
        _make_sample(),
        tokenizer=tokenizer,
        apply_chat_template_kwargs={"preserve_thinking": False},
    )
    learned = _learned_text(input_ids, loss_mask)

    assert tokenizer.last_template == _QWEN36_TEMPLATE
    assert "NEED_SKILL" not in learned
    assert "activate_skill" in learned


def test_qwen35_render_to_text_uses_same_patch_dispatcher():
    tokenizer = _FakeQwenHistoryTokenizer()
    text = render_to_text(_make_sample(), tokenizer=tokenizer)
    first_assistant = text.split("<|im_start|>assistant\n", 1)[1].split("<|im_end|>", 1)[0]
    assert "<think>\nNEED_SKILL\n</think>" in first_assistant
    assert tokenizer.last_template == _QWEN36_TEMPLATE
