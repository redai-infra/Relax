# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build the pinned GSM8K train subset used by the Dr.GRPO comparison."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq


ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")
SOURCE_ROWS = 7473
OUTPUT_ROWS = 256
SHUFFLE_SEED = 42
SOURCE_SHA256 = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"
OUTPUT_SHA256 = "8f8580875e50e5da2828ad586f97ee20e55f3ac1dfd7a6f019103ddad1a0f9d1"


def sha256(path: Path) -> str:
    """Return a file's hexadecimal SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args()

    input_sha256 = sha256(args.input)
    if input_sha256 != SOURCE_SHA256:
        raise ValueError(f"Unexpected source SHA256 for {args.input}: {input_sha256}")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}; pass --force to replace it")

    rows = pq.read_table(args.input, columns=["question", "answer"]).to_pylist()
    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"Expected {SOURCE_ROWS} source rows, found {len(rows)}")
    random.Random(SHUFFLE_SEED).shuffle(rows)
    rows = rows[:OUTPUT_ROWS]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            match = ANSWER_PATTERN.search(row["answer"])
            if match is None:
                raise ValueError(f"GSM8K answer does not contain a final label: {row['answer']!r}")
            label = match.group(1).replace(",", "").strip()
            prompt = (
                "Solve the following math problem step by step. The last line of your response must be "
                f"exactly Answer: \\boxed{{answer}}.\n\n{row['question']}"
            )
            output_file.write(json.dumps({"prompt": [{"role": "user", "content": prompt}], "label": label}) + "\n")

    output_sha256 = sha256(args.output)
    if output_sha256 != OUTPUT_SHA256:
        raise ValueError(f"Unexpected output SHA256 for {args.output}: {output_sha256}")
    print(f"saved={args.output}")
    print(f"samples={len(rows)}")
    print(f"sha256={output_sha256}")


if __name__ == "__main__":
    main()
