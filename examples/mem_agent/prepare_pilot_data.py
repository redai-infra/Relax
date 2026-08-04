# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build a short-context, non-degenerate Qwen3-0.6B MemAgent pilot set.

The workflow is deliberately two-stage. ``candidates`` filters only immutable
input properties such as token length. After an untrained recurrent Pass@N
run, ``select`` keeps prompts with both successes and failures. This gives GRPO
useful within-group reward variance without pretending the screened pilot is a
formal, unbiased HotpotQA benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _answers(row: dict[str, Any]) -> list[str]:
    metadata = _metadata(row)
    answers = metadata.get("ground_truth", [row.get("label", "")])
    if isinstance(answers, str):
        answers = [answers]
    return [str(answer).strip() for answer in answers if str(answer).strip()]


def _contains_subsequence(sequence: list[int], subsequence: list[int]) -> bool:
    if not subsequence or len(subsequence) > len(sequence):
        return False
    width = len(subsequence)
    return any(sequence[offset : offset + width] == subsequence for offset in range(len(sequence) - width + 1))


def build_candidates(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    chunk_tokens: int,
    min_chunks: int,
    max_chunks: int,
    candidate_count: int,
    seed: int,
    require_answer_in_context: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter by immutable length/content properties and sample deterministically."""
    if not 0 < min_chunks <= max_chunks:
        raise ValueError("Expected 0 < min_chunks <= max_chunks.")
    if chunk_tokens <= 0 or candidate_count <= 0:
        raise ValueError("chunk_tokens and candidate_count must be positive.")

    eligible = []
    rejected = Counter()
    chunk_histogram = Counter()
    for source_index, row in enumerate(rows):
        metadata = _metadata(row)
        context = str(metadata.get("context", "")).strip()
        answers = _answers(row)
        if not context or not answers:
            rejected["missing_context_or_answer"] += 1
            continue
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        num_chunks = math.ceil(len(context_ids) / chunk_tokens)
        if num_chunks < min_chunks:
            rejected["too_short"] += 1
            continue
        if num_chunks > max_chunks:
            rejected["too_long"] += 1
            continue
        answer_visible = any(
            _contains_subsequence(context_ids, tokenizer.encode(answer, add_special_tokens=False))
            for answer in answers
        )
        if require_answer_in_context and not answer_visible:
            rejected["answer_not_in_context"] += 1
            continue

        candidate = copy.deepcopy(row)
        candidate_id = str(candidate.get("_id", f"train-{source_index:06d}"))
        candidate["_id"] = candidate_id
        candidate_metadata = candidate.setdefault("metadata", {})
        candidate_metadata["pilot"] = {
            "source_index": source_index,
            "context_tokens": len(context_ids),
            "num_chunks": num_chunks,
            "answer_in_context": answer_visible,
        }
        eligible.append(candidate)
        chunk_histogram[num_chunks] += 1

    random.Random(seed).shuffle(eligible)
    selected = eligible[:candidate_count]
    if len(selected) < candidate_count:
        raise ValueError(f"Only {len(selected)} rows satisfy pilot filters; requested {candidate_count}.")
    return selected, {
        "input_rows": len(rows),
        "eligible_rows": len(eligible),
        "candidate_rows": len(selected),
        "rejected": dict(sorted(rejected.items())),
        "eligible_chunk_histogram": {str(key): value for key, value in sorted(chunk_histogram.items())},
        "chunk_tokens": chunk_tokens,
        "min_chunks": min_chunks,
        "max_chunks": max_chunks,
        "candidate_count": candidate_count,
        "seed": seed,
        "require_answer_in_context": require_answer_in_context,
    }


def _screening_groups(
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    samples_per_item: int,
    min_successes: int,
    max_successes: int,
    preferred_min_successes: int,
    preferred_max_successes: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["_id"])].append(record)

    screening = []
    for candidate in candidates:
        candidate_id = str(candidate["_id"])
        group = grouped.get(candidate_id, [])
        errors = sum(bool(record.get("error")) for record in group)
        sample_indices = {int(record.get("sample_index", -1)) for record in group}
        successes = sum(float(record.get("judge_boxed_em", 0.0)) == 1.0 for record in group)
        complete = len(group) == samples_per_item and sample_indices == set(range(samples_per_item))
        if not complete:
            status = "incomplete"
        elif errors:
            status = "request_error"
        elif preferred_min_successes <= successes <= preferred_max_successes:
            status = "preferred"
        elif min_successes <= successes <= max_successes:
            status = "eligible_boundary"
        else:
            status = "no_reward_variance"
        screening.append(
            {
                "_id": candidate_id,
                "samples": len(group),
                "successes": successes,
                "failures": len(group) - successes,
                "errors": errors,
                "status": status,
                "selected_split": None,
            }
        )
    return screening


def select_pilot_sets(
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    samples_per_item: int,
    train_count: int,
    eval_count: int,
    seed: int,
    min_successes: int = 1,
    max_successes: int | None = None,
    preferred_min_successes: int = 2,
    preferred_max_successes: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select disjoint train/eval diagnostics with Pass@N and reward variance."""
    if samples_per_item <= 1:
        raise ValueError("Pass@N screening requires samples_per_item > 1.")
    max_successes = samples_per_item - 1 if max_successes is None else max_successes
    preferred_max_successes = samples_per_item - 2 if preferred_max_successes is None else preferred_max_successes
    if not 1 <= min_successes <= max_successes < samples_per_item:
        raise ValueError("Success bounds must guarantee at least one success and one failure.")

    screening = _screening_groups(
        candidates,
        records,
        samples_per_item=samples_per_item,
        min_successes=min_successes,
        max_successes=max_successes,
        preferred_min_successes=preferred_min_successes,
        preferred_max_successes=preferred_max_successes,
    )
    by_id = {str(row["_id"]): row for row in candidates}
    eligible = [entry for entry in screening if entry["status"] in ("preferred", "eligible_boundary")]
    tie_break = {entry["_id"]: random.Random(f"{seed}:{entry['_id']}").random() for entry in eligible}
    eligible.sort(
        key=lambda entry: (
            entry["status"] != "preferred",
            abs(entry["successes"] - samples_per_item / 2),
            tie_break[entry["_id"]],
        )
    )
    required = train_count + eval_count
    if len(eligible) < required:
        raise ValueError(
            f"Only {len(eligible)} complete prompts have non-degenerate Pass@{samples_per_item}; "
            f"need {required}. Run another candidate shard before training."
        )

    chosen = eligible[:required]
    random.Random(seed).shuffle(chosen)
    eval_entries = chosen[:eval_count]
    train_entries = chosen[eval_count:]
    selected_lookup = {entry["_id"]: entry for entry in chosen}
    for entry in screening:
        if entry["_id"] in {item["_id"] for item in train_entries}:
            entry["selected_split"] = "train"
        elif entry["_id"] in {item["_id"] for item in eval_entries}:
            entry["selected_split"] = "eval"

    def materialize(entries: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        output = []
        for entry in entries:
            row = copy.deepcopy(by_id[entry["_id"]])
            pilot = row.setdefault("metadata", {}).setdefault("pilot", {})
            pilot.update(
                {
                    "split": split,
                    "baseline_samples": samples_per_item,
                    "baseline_successes": entry["successes"],
                    "baseline_failures": entry["failures"],
                    "baseline_pass_at_n": True,
                    "baseline_reward_variance": True,
                }
            )
            output.append(row)
        return output

    train_rows = materialize(train_entries, "train")
    eval_rows = materialize(eval_entries, "eval")
    assert not ({row["_id"] for row in train_rows} & {row["_id"] for row in eval_rows})
    return (
        train_rows,
        eval_rows,
        {
            "samples_per_item": samples_per_item,
            "train_count": train_count,
            "eval_count": eval_count,
            "seed": seed,
            "min_successes": min_successes,
            "max_successes": max_successes,
            "preferred_min_successes": preferred_min_successes,
            "preferred_max_successes": preferred_max_successes,
            "screening": screening,
            "selected_ids": {
                "train": [row["_id"] for row in train_rows],
                "eval": [row["_id"] for row in eval_rows],
            },
            "selected_successes": {
                entry_id: selected_lookup[entry_id]["successes"] for entry_id in sorted(selected_lookup)
            },
        },
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("--input", type=Path, required=True)
    candidates.add_argument("--tokenizer", required=True)
    candidates.add_argument("--output", type=Path, required=True)
    candidates.add_argument("--manifest", type=Path, required=True)
    candidates.add_argument("--chunk-tokens", type=int, default=512)
    candidates.add_argument("--min-chunks", type=int, default=2)
    candidates.add_argument("--max-chunks", type=int, default=4)
    candidates.add_argument("--candidate-count", type=int, default=24)
    candidates.add_argument("--seed", type=int, default=42)
    candidates.add_argument("--allow-answer-not-in-context", action="store_true")

    select = subparsers.add_parser("select")
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--baseline-records", type=Path, required=True)
    select.add_argument("--train-output", type=Path, required=True)
    select.add_argument("--eval-output", type=Path, required=True)
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--samples-per-item", type=int, default=8)
    select.add_argument("--train-count", type=int, default=8)
    select.add_argument("--eval-count", type=int, default=4)
    select.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "candidates":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        output_rows, manifest = build_candidates(
            read_jsonl(args.input),
            tokenizer,
            chunk_tokens=args.chunk_tokens,
            min_chunks=args.min_chunks,
            max_chunks=args.max_chunks,
            candidate_count=args.candidate_count,
            seed=args.seed,
            require_answer_in_context=not args.allow_answer_not_in_context,
        )
        write_jsonl(args.output, output_rows)
        manifest.update(
            {
                "source_file": str(args.input),
                "source_sha256": sha256_file(args.input),
                "output_file": str(args.output),
                "output_sha256": sha256_file(args.output),
                "tokenizer": args.tokenizer,
            }
        )
        _write_manifest(args.manifest, manifest)
        return

    candidate_rows = read_jsonl(args.candidates)
    train_rows, eval_rows, manifest = select_pilot_sets(
        candidate_rows,
        read_jsonl(args.baseline_records),
        samples_per_item=args.samples_per_item,
        train_count=args.train_count,
        eval_count=args.eval_count,
        seed=args.seed,
    )
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    manifest.update(
        {
            "candidate_file": str(args.candidates),
            "candidate_sha256": sha256_file(args.candidates),
            "baseline_records_file": str(args.baseline_records),
            "baseline_records_sha256": sha256_file(args.baseline_records),
            "train_file": str(args.train_output),
            "train_sha256": sha256_file(args.train_output),
            "eval_file": str(args.eval_output),
            "eval_sha256": sha256_file(args.eval_output),
        }
    )
    _write_manifest(args.manifest, manifest)


if __name__ == "__main__":
    main()
