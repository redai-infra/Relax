# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.mem_agent.eval_ruler_hqa import base_infer


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.mark.asyncio
async def test_base_infer_truncates_context_without_dropping_question(monkeypatch):
    captured = {}

    async def fake_chat_once(session, base_url, api_key, model, instruction, temperature, top_p, max_tokens):
        del session, base_url, api_key, model, temperature, top_p, max_tokens
        captured["instruction"] = instruction
        return r"\boxed{x}"

    monkeypatch.setattr("examples.mem_agent.eval_ruler_hqa._chat_once", fake_chat_once)
    args = SimpleNamespace(
        max_input_tokens=100,
        base_url="http://unused",
        api_key="EMPTY",
        model="model",
        temperature=0.0,
        top_p=1.0,
        max_final_tokens=16,
    )
    _, diagnostics = await base_infer(
        {"context": "c" * 500, "input": "Which answer?"}, args, CharacterTokenizer(), object()
    )

    assert diagnostics["context_truncated"] is True
    assert "Question: Which answer?" in captured["instruction"]
    assert len(captured["instruction"]) <= args.max_input_tokens
