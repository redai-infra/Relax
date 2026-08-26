# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Evaluate recurrent MemAgent or a single-context baseline on HotpotQA/RULER-
HQA."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp

from examples.mem_agent.metrics import aggregate, exact_match, f1_score, sub_exact_match
from examples.mem_agent.prompts import (
    NO_MEMORY,
    final_instruction,
    memory_instruction,
    strip_stop_tokens,
    truncate_text_to_tokens,
)
from examples.mem_agent.reward import exact_match_any, extract_last_boxed


EVALUATOR_SCHEMA_VERSION = "mem-agent-vime-eval-v1"


def sha256_file(path: Path) -> str:
    """Hash the exact normalized dataset consumed by one evaluation run."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            raw = [json.loads(line) for line in source if line.strip()]
    else:
        with path.open(encoding="utf-8") as source:
            raw = json.load(source)
        if isinstance(raw, dict):
            raw = list(raw.values())

    data = []
    for index, item in enumerate(raw):
        if "input" in item:
            normalized = dict(item)
            normalized.setdefault("_id", index)
        else:
            metadata = item.get("metadata") or {}
            normalized = {
                "_id": item.get("_id", index),
                "input": item.get("prompt", metadata.get("question", "")),
                "answers": metadata.get("ground_truth", [item.get("label", "")]),
                "context": metadata.get("context", ""),
                "num_docs": metadata.get("num_docs", 0),
            }
        if normalized.get("input") and normalized.get("context") and normalized.get("answers"):
            data.append(normalized)
    return data


async def _chat_once(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    instruction: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None = None,
    enable_thinking: bool | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": instruction}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if enable_thinking is not None:
        # vLLM and SGLang expose Qwen's chat-template controls through this
        # OpenAI-compatible extension. Formal VIME evaluation leaves it unset;
        # the short-response 0.6B pilot disables thinking explicitly.
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    async with session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    ) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:300]}")
        result = json.loads(body)
    return str(result["choices"][0]["message"]["content"])


async def recurrent_infer(
    item: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: Any,
    session: aiohttp.ClientSession,
    sample_index: int = 0,
) -> tuple[str, dict[str, Any]]:
    question = str(item["input"]).strip()
    context_ids = tokenizer.encode(str(item["context"]).strip(), add_special_tokens=False)
    all_chunks = [
        context_ids[offset : offset + args.chunk_tokens] for offset in range(0, len(context_ids), args.chunk_tokens)
    ]
    chunks = all_chunks[: args.max_chunks]
    memory = NO_MEMORY
    memory_lengths = []
    for chunk_index, chunk_ids in enumerate(chunks):
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        generated_memory = strip_stop_tokens(
            await _chat_once(
                session,
                args.base_url,
                args.api_key,
                args.model,
                memory_instruction(question, memory, chunk, evaluation=True),
                args.temperature,
                args.top_p,
                args.max_memory_tokens,
                _request_seed(args, item["_id"], sample_index, "memory", chunk_index),
                args.enable_thinking,
            )
        )
        # Match training exactly: the next turn only sees the re-tokenized,
        # bounded output from the immediately preceding memory update.
        memory, memory_length = truncate_text_to_tokens(tokenizer, generated_memory, args.max_memory_tokens)
        memory_lengths.append(memory_length)
    answer = await _chat_once(
        session,
        args.base_url,
        args.api_key,
        args.model,
        final_instruction(question, memory, evaluation=True),
        args.temperature,
        args.top_p,
        args.max_final_tokens,
        _request_seed(args, item["_id"], sample_index, "final", len(chunks)),
        args.enable_thinking,
    )
    return answer, {
        "num_chunks": len(chunks),
        "context_truncated": len(all_chunks) > args.max_chunks,
        "memory_token_lengths": memory_lengths,
    }


async def base_infer(
    item: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: Any,
    session: aiohttp.ClientSession,
    sample_index: int = 0,
) -> tuple[str, dict[str, Any]]:
    suffix = f"\n\nQuestion: {item['input']}\nPlease answer the question and put the answer in \\boxed{{}}."
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if len(suffix_ids) > args.max_input_tokens:
        raise ValueError("Question and answer instruction exceed max_input_tokens; context cannot be retained.")
    context_ids = tokenizer.encode(str(item["context"]), add_special_tokens=False)
    context_budget = args.max_input_tokens - len(suffix_ids)
    retained_context_ids = context_ids[:context_budget]
    instruction = tokenizer.decode(retained_context_ids, skip_special_tokens=True) + suffix
    # Decoding then concatenating can change a BPE boundary by a token. Trim
    # context only, never the question suffix, until the served prompt fits.
    while len(tokenizer.encode(instruction, add_special_tokens=False)) > args.max_input_tokens:
        retained_context_ids = retained_context_ids[:-1]
        instruction = tokenizer.decode(retained_context_ids, skip_special_tokens=True) + suffix
    context_truncated = len(retained_context_ids) < len(context_ids)
    answer = await _chat_once(
        session,
        args.base_url,
        args.api_key,
        args.model,
        instruction,
        args.temperature,
        args.top_p,
        args.max_final_tokens,
        _request_seed(args, item["_id"], sample_index, "base", 0),
        args.enable_thinking,
    )
    return answer, {"num_chunks": 1, "context_truncated": context_truncated, "memory_token_lengths": []}


def _request_seed(
    args: argparse.Namespace,
    item_id: Any,
    sample_index: int,
    stage: str,
    turn_index: int,
) -> int | None:
    """Derive a stable per-request seed without coupling concurrent tasks."""
    base_seed = getattr(args, "seed", None)
    if base_seed is None:
        return None
    material = f"{base_seed}|{item_id}|{sample_index}|{stage}|{turn_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def summarize_pass_at_n(records: list[dict[str, Any]], samples_per_item: int) -> dict[str, float | int]:
    """Summarize boxed-reward Pass@N and GRPO-useful group variance."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["_id"])].append(record)

    complete_groups = [group for group in grouped.values() if len(group) == samples_per_item]
    success_counts = [
        sum(float(record.get("judge_boxed_em", 0.0)) == 1.0 for record in group) for group in complete_groups
    ]
    pass_count = sum(count > 0 for count in success_counts)
    variance_count = sum(0 < count < samples_per_item for count in success_counts)
    denominator = len(complete_groups)
    return {
        "pass_at_n": pass_count / denominator if denominator else 0.0,
        "pass_at_n_pct": 100 * pass_count / denominator if denominator else 0.0,
        "complete_prompt_groups": denominator,
        "reward_variance_groups": variance_count,
        "reward_variance_group_pct": 100 * variance_count / denominator if denominator else 0.0,
        "mean_successes_per_prompt": sum(success_counts) / denominator if denominator else 0.0,
    }


async def run_evaluation(
    data: list[dict[str, Any]], args: argparse.Namespace, tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Capture the digest before any long-running inference. A later overwrite
    # of the path must not make the summary claim that different bytes were
    # evaluated.
    data_sha256 = sha256_file(args.data_file)
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def evaluate_one(item: dict[str, Any], sample_index: int) -> dict[str, Any]:
            answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]
            answers = [str(answer) for answer in answers]
            ground_truth = answers[0]
            try:
                async with semaphore:
                    if args.mode == "recurrent":
                        response, diagnostics = await recurrent_infer(
                            item, args, tokenizer, session, sample_index=sample_index
                        )
                    else:
                        response, diagnostics = await base_infer(
                            item, args, tokenizer, session, sample_index=sample_index
                        )
                # RULER-HQA's VIME-compatible metrics score the first reference.
                # boxed_em additionally mirrors the HotpotQA training reward and
                # accepts any annotated answer.
                prediction = extract_last_boxed(response[-300:])
                return {
                    "_id": item["_id"],
                    "sample_index": sample_index,
                    "answer": ground_truth,
                    "answers": answers,
                    "pred": prediction,
                    "judge_f1": f1_score(prediction, ground_truth),
                    "judge_em": exact_match(prediction, ground_truth),
                    "judge_sub_em": sub_exact_match(prediction, ground_truth),
                    "judge_boxed_em": float(bool(prediction) and exact_match_any(prediction, answers)),
                    "response": response,
                    **diagnostics,
                }
            except Exception as exc:
                # Preserve the target and explicit zero scores in raw output.
                # Reviewers can therefore audit every input row even when the
                # serving request itself failed.
                return {
                    "_id": item["_id"],
                    "sample_index": sample_index,
                    "answer": ground_truth,
                    "answers": answers,
                    "pred": "",
                    "judge_f1": 0.0,
                    "judge_em": 0.0,
                    "judge_sub_em": 0.0,
                    "judge_boxed_em": 0.0,
                    "response": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        records = await asyncio.gather(
            *(evaluate_one(item, sample_index) for item in data for sample_index in range(args.samples_per_item))
        )

    summary = {
        **aggregate(records),
        **summarize_pass_at_n(records, args.samples_per_item),
        "mode": args.mode,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "data_file": str(args.data_file),
        # Path equality alone is not proof that sequential checkpoint runs saw
        # identical bytes. The comparator requires this digest and evaluator
        # schema before it accepts a controlled comparison.
        "data_sha256": data_sha256,
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sampling_count": args.samples_per_item,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "chunk_tokens": args.chunk_tokens,
        "max_memory_tokens": args.max_memory_tokens,
        "max_final_tokens": args.max_final_tokens,
        "max_chunks": args.max_chunks,
        "max_input_tokens": args.max_input_tokens,
        "server_max_model_len": args.server_max_model_len,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("recurrent", "base"), default="recurrent")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--samples-per-item", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--max-memory-tokens", type=int, default=1024)
    parser.add_argument("--max-final-tokens", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=7936)
    parser.add_argument("--server-max-model-len", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=86400)
    args = parser.parse_args()
    if args.samples_per_item <= 0:
        parser.error("--samples-per-item must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    data = load_data(args.data_file)
    records, summary = asyncio.run(run_evaluation(data, args, tokenizer))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / f"{args.run_name}.jsonl"
    summary_path = args.output_dir / f"{args.run_name}.summary.json"
    with records_path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
