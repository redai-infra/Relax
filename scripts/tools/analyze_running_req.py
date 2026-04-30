#!/usr/bin/env python3
"""Summarize SGLang ``#running-req`` values from Relax logs."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ENGINE_RE = re.compile(r"\(SGLangEngine pid=(?P<pid>\d+)\)")
RUNNING_REQ_RE = re.compile(r"#running-req:\s*(?P<value>\d+)")
TIMESTAMP_RE = re.compile(r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bucket_start(value: int, bucket_size: int) -> int:
    if bucket_size <= 1:
        return value
    return (value // bucket_size) * bucket_size


def bucket_label(start: int, bucket_size: int) -> str:
    if bucket_size <= 1:
        return str(start)
    end = start + bucket_size - 1
    return f"{start}-{end}"


def print_summary(title: str, values: list[int], bucket_size: int, top: int) -> None:
    if not values:
        print(f"\n{title}: no samples")
        return

    sorted_values = sorted(values)
    print(f"\n{title}")
    print(f"  samples: {len(values)}")
    print(f"  min/max: {sorted_values[0]} / {sorted_values[-1]}")
    print(f"  mean:    {sum(values) / len(values):.2f}")
    print(
        "  p50/p90/p95/p99: "
        f"{percentile(sorted_values, 50):.1f} / "
        f"{percentile(sorted_values, 90):.1f} / "
        f"{percentile(sorted_values, 95):.1f} / "
        f"{percentile(sorted_values, 99):.1f}"
    )

    counts = Counter(bucket_start(value, bucket_size) for value in values)
    cumulative = 0
    print("  distribution (high to low):")
    for start, count in sorted(counts.items(), reverse=True)[:top]:
        cumulative += count
        print(
            f"    {bucket_label(start, bucket_size):>12}: "
            f"{count:>8} ({count / len(values) * 100:6.2f}%, cum {cumulative / len(values) * 100:6.2f}%)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to Relax/SGLang log file, for example _e.log")
    parser.add_argument(
        "--phase",
        choices=("decode", "prefill", "all"),
        default="decode",
        help="Which SGLang batch lines to include. Defaults to decode.",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=16,
        help="Histogram bucket size. Use 1 for exact values. Defaults to 16.",
    )
    parser.add_argument("--top", type=int, default=30, help="Maximum histogram buckets to print.")
    parser.add_argument("--by-engine", action="store_true", help="Also print one summary per SGLangEngine pid.")
    parser.add_argument(
        "--show-timeline",
        action="store_true",
        help="Print first/last timestamp and the max #running-req line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values: list[int] = []
    by_engine: dict[str, list[int]] = defaultdict(list)
    first_ts: str | None = None
    last_ts: str | None = None
    max_record: tuple[int, str | None, str | None, int] | None = None

    with args.log.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = ANSI_RE.sub("", raw_line)
            if args.phase == "decode" and "Decode batch" not in line:
                continue
            if args.phase == "prefill" and "Prefill batch" not in line:
                continue
            if args.phase == "all" and "Decode batch" not in line and "Prefill batch" not in line:
                continue

            match = RUNNING_REQ_RE.search(line)
            if match is None:
                continue

            value = int(match.group("value"))
            engine = (ENGINE_RE.search(line) or {}).group("pid") if ENGINE_RE.search(line) else "unknown"
            timestamp_match = TIMESTAMP_RE.search(line)
            timestamp = timestamp_match.group("timestamp") if timestamp_match else None

            if timestamp is not None:
                first_ts = first_ts or timestamp
                last_ts = timestamp

            values.append(value)
            by_engine[engine].append(value)

            if max_record is None or value > max_record[0]:
                max_record = (value, engine, timestamp, line_no)

    print(f"log: {args.log}")
    print(f"phase: {args.phase}")
    if args.show_timeline:
        print(f"first timestamp: {first_ts or 'n/a'}")
        print(f"last timestamp:  {last_ts or 'n/a'}")
        if max_record is not None:
            value, engine, timestamp, line_no = max_record
            print(f"max record:      value={value}, engine={engine}, timestamp={timestamp or 'n/a'}, line={line_no}")

    print_summary("overall", values, args.bucket_size, args.top)

    if args.by_engine:
        for engine in sorted(by_engine):
            print_summary(f"engine {engine}", by_engine[engine], args.bucket_size, args.top)


if __name__ == "__main__":
    main()
