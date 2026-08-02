#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Download a small ModelScope GSM8K slice for Task 22."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_REPO_ID = "AI-ModelScope/gsm8k"
DEFAULT_SPLIT = "main"
DEFAULT_LIMIT = 16
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "task22_gsm8k_main16.jsonl"
DEFAULT_DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "task22-modelscope-gsm8k"
ANSWER_PATTERN = re.compile(r"####\s*(.+?)\s*$", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Task 22 GSM8K subset from ModelScope.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="ModelScope dataset repo id.")
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=("main", "socratic"),
        help="GSM8K config to export.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of rows to keep.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Local directory used by the ModelScope CLI download.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild the subset even if it already exists.")
    return parser.parse_args()


def extract_final_answer(answer: str) -> str:
    match = ANSWER_PATTERN.search(answer)
    if match is None:
        raise ValueError(f"Unable to extract a final answer from GSM8K response: {answer!r}")
    return " ".join(match.group(1).split())


def download_dataset(repo_id: str, download_dir: Path, force: bool) -> None:
    cmd = [
        "ms",
        "download",
        "--repo-type",
        "dataset",
        "--local-dir",
        str(download_dir),
    ]
    if force:
        cmd.append("--force")
    cmd.append(repo_id)
    subprocess.run(cmd, check=True)


def subset_records(frame: pd.DataFrame, limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if len(frame) < limit:
        raise ValueError(f"dataset only has {len(frame)} rows, cannot take {limit}")

    records: list[dict[str, str]] = []
    for _, row in frame.head(limit).iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        if not question:
            raise ValueError("Encountered an empty GSM8K question")
        records.append({"prompt": question, "label": extract_final_answer(answer)})
    return records


def write_jsonl(records: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        print(f"Task 22 dataset already exists: {output}")
        return 0

    download_dataset(args.repo_id, args.download_dir, args.force)
    source_path = args.download_dir / args.split / "train-00000-of-00001.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing GSM8K parquet shard: {source_path}")

    frame = pd.read_parquet(source_path)
    records = subset_records(frame, args.limit)
    write_jsonl(records, output)
    print(f"Wrote {len(records)} Task 22 rows from {args.repo_id}:{args.split}/train to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
