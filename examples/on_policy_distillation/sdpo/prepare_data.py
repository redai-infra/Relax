# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare the reference SDPO SciKnowEval and ToolAlpaca data for Relax."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)
TARGET_DOMAINS = frozenset({"Chemistry", "Physics", "Biology", "Materials"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        if path.suffix.lower() == ".jsonl":
            return _read_jsonl(path)
        text = path.read_text(encoding="utf-8").strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return _read_jsonl(path)
        return value if isinstance(value, list) else [value]
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as parquet

        return parquet.read_table(path).to_pylist()
    raise ValueError(f"Unsupported input format: {path}")


def _canonical_domain(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized == "material":
        return "Materials"
    return normalized.capitalize()


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_sciknoweval_row(
    row: dict[str, Any],
    *,
    source_split: str,
    domain: str | None,
) -> dict[str, Any] | None:
    if row.get("dataset") == "sciknoweval" and isinstance(row.get("prompt"), str) and "answer" in row:
        normalized_domain = _canonical_domain(domain)
        if normalized_domain not in {"Chemistry", "Physics", "Biology", "Materials"}:
            return None
        prompt = str(row["prompt"]).strip()
        system = str(row.get("system") or "").strip()
        if system:
            prompt = f"{system}\n\n{prompt}"
        answer = row.get("answer", "")
        metadata = {
            "data_source": "sciknoweval",
            "source_split": source_split,
            "domain": normalized_domain,
            "task_type": str(row.get("kind", "mcq")),
            "answer_key": answer,
            "source_index": row.get("idx"),
        }
        return {"prompt": prompt, "label": _json_text(answer), "metadata": metadata}

    details = row.get("details") or {}
    if not isinstance(details, dict) or str(details.get("level", "")).upper() != "L3":
        return None

    source_domain = _canonical_domain(row.get("domain"))
    if source_domain not in TARGET_DOMAINS:
        return None

    choices = row.get("choices") or {}
    choice_lines = [
        f"{label}: {text}" for label, text in zip(choices.get("label") or [], choices.get("text") or [], strict=False)
    ]
    prompt_value = row.get("prompt", {})
    prompt_default = prompt_value.get("default", "") if isinstance(prompt_value, dict) else prompt_value
    question = str(row.get("question") or prompt_default).strip()
    prompt = question
    if choice_lines:
        prompt = f"{question}\n\n" + "\n".join(choice_lines)
    prompt += "\n\nReason carefully and provide the final answer."

    normalized_domain = source_domain
    answer = row.get("answerKey") or row.get("answer", "")
    metadata = {
        "data_source": "sciknoweval",
        "source_split": source_split,
        "domain": normalized_domain,
        "task_type": str(row.get("type", "unknown")),
        "answer_key": answer,
    }
    return {"prompt": prompt, "label": _json_text(answer), "metadata": metadata}


def _normalize_tool_row(row: dict[str, Any], *, source_split: str, dataset: str) -> dict[str, Any] | None:
    if row.get("dataset") == "tooluse" and isinstance(row.get("prompt"), str) and "answer" in row:
        answer = row.get("answer", "")
        try:
            golden_answer = json.loads(answer) if isinstance(answer, str) else answer
        except json.JSONDecodeError:
            golden_answer = answer
        prompt = str(row.get("prompt", "")).strip()
        metadata = {
            "data_source": "tooluse",
            "source_split": source_split,
            "task_type": str(row.get("kind", "tooluse")),
            "golden_answer": golden_answer,
            "source_index": row.get("idx"),
        }
        return {"prompt": prompt, "label": _json_text(answer), "metadata": metadata}

    if dataset != "toolalpaca" or "golden_answer" not in row:
        return None

    name = str(row.get("name", "")).strip()
    description = str(row.get("description", "")).strip()
    documentation = str(row.get("nl_documentation", "")).strip()
    instruction = str(row.get("instruction", row.get("prompt", ""))).strip()
    prompt = (
        "You are given an API specification and a user request. Select the correct tool and "
        "emit the tool call using exactly:\n"
        "Action: <tool name>\nAction Input: <JSON object>\n\n"
        f"Tool name: {name}\n"
        f"Tool description: {description}\n"
        f"Tool documentation:\n{documentation}\n\n"
        f"User request:\n{instruction}"
    )
    golden_answer = row.get("golden_answer") or []
    metadata = {
        "data_source": "toolalpaca",
        "source_split": source_split,
        "task_type": "tool_call",
        "golden_answer": golden_answer,
    }
    return {"prompt": prompt, "label": _json_text(golden_answer), "metadata": metadata}


def _normalize_relax_row(row: dict[str, Any], *, source_split: str) -> dict[str, Any] | None:
    if not isinstance(row.get("prompt"), str) or "label" not in row:
        return None
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata.setdefault("source_split", source_split)
    return {"prompt": row["prompt"], "label": row["label"], "metadata": metadata}


def normalize_rows(
    dataset: str,
    rows: Iterable[dict[str, Any]],
    *,
    source_split: str,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Convert one supported source schema into Relax's prompt-data schema."""
    if dataset not in {"sciknoweval", "toolalpaca", "tooluse"}:
        raise ValueError(f"Unsupported SDPO dataset {dataset!r}")
    if source_split not in {"train", "test"}:
        raise ValueError(f"Unsupported source split {source_split!r}; expected 'train' or 'test'")

    normalized_rows = []
    for row in rows:
        normalized = _normalize_relax_row(row, source_split=source_split)
        if normalized is None:
            normalized = (
                _normalize_sciknoweval_row(row, source_split=source_split, domain=domain)
                if dataset == "sciknoweval"
                else _normalize_tool_row(row, source_split=source_split, dataset=dataset)
            )
        if normalized is not None:
            normalized_rows.append(normalized)
    return normalized_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sciknoweval", "toolalpaca", "tooluse"), required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-split", choices=("train", "test"), required=True)
    parser.add_argument(
        "--domain",
        default=None,
        help="SciKnowEval domain for the reference flat format; defaults to the input parent directory name.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optionally limit output rows for a smoke run.")
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of normalized rows to hold out as an eval set. When >0, the held-out rows are "
            "written to <output.parent>/eval.jsonl and the rest to --output (train). "
            "Useful for a train/test split when only a single train source is available."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the eval/validation split.")
    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows < 0:
        parser.error("--max-rows must be non-negative")
    if not 0.0 <= args.eval_ratio < 1.0:
        parser.error("--eval-ratio must be in [0, 1)")

    rows = _read_rows(args.input)
    domain = args.domain or args.input.parent.name
    normalized = normalize_rows(args.dataset, rows, source_split=args.source_split, domain=domain)
    if args.max_rows is not None:
        normalized = normalized[: args.max_rows]
    if not normalized:
        raise ValueError(f"No rows matched dataset={args.dataset!r} from input {args.input}")

    if args.eval_ratio > 0.0:
        n_eval = int(round(len(normalized) * args.eval_ratio))
        if n_eval == 0:
            raise ValueError(
                f"--eval-ratio {args.eval_ratio} with {len(normalized)} rows yields 0 eval rows; "
                "raise the ratio or add more input rows."
            )
        rng = random.Random(args.seed)
        indices = list(range(len(normalized)))
        rng.shuffle(indices)
        eval_indices = set(indices[:n_eval])
        train_rows = [r for i, r in enumerate(normalized) if i not in eval_indices]
        eval_rows = [r for i, r in enumerate(normalized) if i in eval_indices]
        _write_jsonl(args.output, train_rows)
        eval_path = args.output.with_name("eval.jsonl")
        _write_jsonl(eval_path, eval_rows)
        logger.info(
            f"Split into train ({len(train_rows)} rows) and eval ({len(eval_rows)} rows); eval written to {eval_path}"
        )
    else:
        _write_jsonl(args.output, normalized)


if __name__ == "__main__":
    main()
