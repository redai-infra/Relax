# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.mem_agent.eval_ruler_hqa import base_infer, run_evaluation


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


@pytest.mark.asyncio
async def test_evaluation_error_keeps_ground_truth_and_counts_as_zero(monkeypatch, tmp_path):
    async def failed_infer(item, args, tokenizer, session):
        del item, args, tokenizer, session
        raise RuntimeError("server unavailable")

    monkeypatch.setattr("examples.mem_agent.eval_ruler_hqa.recurrent_infer", failed_infer)
    data_file = tmp_path / "eval.jsonl"
    data_file.write_text('{"input":"Question","context":"Context","answers":["A"]}\n', encoding="utf-8")
    args = SimpleNamespace(
        mode="recurrent",
        concurrency=1,
        timeout=1,
        model="model",
        tokenizer="tokenizer",
        data_file=data_file,
        temperature=0.7,
        top_p=0.95,
        chunk_tokens=2048,
        max_memory_tokens=1024,
        max_final_tokens=256,
        max_chunks=64,
        max_input_tokens=7936,
        server_max_model_len=8192,
    )
    records, summary = await run_evaluation(
        [{"_id": "q1", "input": "Question", "context": "Context", "answers": ["A", "Alias"]}],
        args,
        CharacterTokenizer(),
    )

    assert records[0]["answers"] == ["A", "Alias"]
    assert records[0]["pred"] == ""
    assert records[0]["judge_boxed_em"] == 0.0
    assert "server unavailable" in records[0]["error"]
    assert summary["total"] == 1
    assert summary["errors"] == 1
    assert summary["sub_em_pct"] == 0.0
    assert len(summary["data_sha256"]) == 64
    assert summary["evaluator_schema_version"] == "mem-agent-vime-eval-v1"
