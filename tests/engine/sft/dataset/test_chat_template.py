# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for chat_template.render_with_loss_mask."""

from unittest.mock import MagicMock

import torch

from relax.engine.sft.dataset.chat_template import (
    HAS_GENERATION_MARKER,
    render_with_loss_mask,
)
from relax.engine.sft.dataset.sample import (
    CanonicalMessage,
    CanonicalSample,
)


def _make_sample():
    return CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(role="assistant", content="A", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )


def _mock_tokenizer_with_generation_marker(template_str: str = "{% generation %}assistant{% endgeneration %}"):
    """Mock that simulates a chat template containing {% generation %}."""
    tok = MagicMock()
    tok.chat_template = template_str

    # apply_chat_template returns dict with input_ids + assistant_masks
    def _apply(
        messages,
        *,
        tools=None,
        tokenize=True,
        return_tensors=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        if return_assistant_tokens_mask:
            return {"input_ids": ids, "assistant_masks": [[0, 0, 0, 1, 1]]}
        return ids

    tok.apply_chat_template.side_effect = _apply
    return tok


class _FakeFastTokenizerNoGenerationMarker:
    """Fake fast tokenizer that wraps messages in Qwen-style ChatML and
    tokenizes char-by-char with offset_mapping support — the minimum surface
    needed by the offset-mapping fallback in chat_template.py."""

    chat_template = "{{ messages[0]['content'] }}"  # no {% generation %}

    @staticmethod
    def _render(messages):
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return out

    @staticmethod
    def _tokenize(text):
        ids = [ord(c) for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return ids, offsets

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, **kwargs):  # noqa: ARG002
        text = self._render(messages)
        if not tokenize:
            return text
        ids, _ = self._tokenize(text)
        return ids

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **kwargs):  # noqa: ARG002
        ids, offsets = self._tokenize(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def _mock_tokenizer_without_generation_marker():
    return _FakeFastTokenizerNoGenerationMarker()


def test_has_generation_marker_detects_correctly():
    assert HAS_GENERATION_MARKER("foo {% generation %} bar")
    assert HAS_GENERATION_MARKER("{%generation%}")  # no spaces
    assert not HAS_GENERATION_MARKER("plain template")
    assert not HAS_GENERATION_MARKER("")


def test_render_uses_assistant_mask_when_template_supports_it():
    tok = _mock_tokenizer_with_generation_marker()
    sample = _make_sample()
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(loss_mask, torch.Tensor)
    assert input_ids.shape == loss_mask.shape
    assert loss_mask.tolist() == [0, 0, 0, 1, 1]
    # Verify return_assistant_tokens_mask was requested
    call_kwargs = tok.apply_chat_template.call_args.kwargs
    assert call_kwargs.get("return_assistant_tokens_mask") is True


def test_render_falls_back_when_no_generation_marker(capsys):
    tok = _mock_tokenizer_without_generation_marker()
    sample = _make_sample()
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    # Per-message fallback: user contributes 1 token (len('Q')), assistant 1 (len('A'))
    assert input_ids.shape == loss_mask.shape
    # Only assistant turn participates
    assert loss_mask.sum().item() >= 1
    # The first message (user) contribution must be 0
    assert loss_mask[0].item() == 0


def test_render_passes_tools_through_to_tokenizer():
    tok = _mock_tokenizer_with_generation_marker()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="x", learn=False),
            CanonicalMessage(role="assistant", content="y", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
        tools=[{"type": "function", "function": {"name": "add"}}],
    )
    render_with_loss_mask(sample, tokenizer=tok)
    call_kwargs = tok.apply_chat_template.call_args.kwargs
    assert call_kwargs["tools"] == sample.tools


def test_render_returns_int_tensors_no_padding():
    tok = _mock_tokenizer_with_generation_marker()
    sample = _make_sample()
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    assert input_ids.dtype in (torch.long, torch.int32, torch.int64)
    assert loss_mask.dtype in (torch.long, torch.int32, torch.int64, torch.bool)
    # 1D after squeeze
    assert input_ids.dim() == 1
    assert loss_mask.dim() == 1


def test_fallback_loss_mask_only_on_learn_messages():
    tok = _mock_tokenizer_without_generation_marker()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="system", content="sys", learn=False),
            CanonicalMessage(role="user", content="ab", learn=False),
            CanonicalMessage(role="assistant", content="cde", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    # New fallback marks the entire assistant content span up to (and including)
    # "<|im_end|>\n", so 3 chars of "cde" + 10 chars of "<|im_end|>" + 1 newline = 14.
    learn_span = len("cde") + len("<|im_end|>") + 1
    assert loss_mask.sum().item() == learn_span
    # The trailing learn_span positions should all be 1 (mock tokenizes char-by-char,
    # so the assistant span lands at the tail of the rendered ChatML).
    assert loss_mask[-learn_span:].tolist() == [1] * learn_span
    # All earlier positions (system + user turns) should be 0.
    assert loss_mask[:-learn_span].sum().item() == 0
    assert input_ids.shape == loss_mask.shape
