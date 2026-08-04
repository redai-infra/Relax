# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Turn ReLax rollout reward logs into an auditable training summary."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_METRIC = "rollout/mem_agent_raw_reward/mean"
SCHEMA_VERSION = "mem-agent-reward-summary-v1"
_PERF_LINE = re.compile(r"\bperf\s+(\d+):\s+(\{.*\})")


def extract_reward_points(lines: Iterable[str], metric: str = DEFAULT_METRIC) -> list[tuple[int, float]]:
    """Extract one trajectory-level reward mean for every logged rollout.

    ReLax logs the complete rollout metric dictionary as ``perf N: {...}``
    before sending the same values to TensorBoard. Parsing that durable text
    log keeps the acceptance artifact independent of a tracking backend.
    Identical duplicated lines are tolerated because Ray may replay driver
    output, while conflicting values for one rollout fail closed.
    """
    by_rollout: dict[int, float] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _PERF_LINE.search(line)
        if match is None:
            continue
        try:
            payload = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Malformed ReLax perf payload at line {line_number}.") from exc
        if not isinstance(payload, dict) or metric not in payload:
            continue

        rollout_id = int(match.group(1))
        raw_value = payload[metric]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"Reward metric {metric!r} at rollout {rollout_id} is not numeric: {raw_value!r}.")
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Reward metric {metric!r} at rollout {rollout_id} is outside [0, 1]: {value!r}.")
        if rollout_id in by_rollout and by_rollout[rollout_id] != value:
            raise ValueError(
                f"Conflicting reward values for rollout {rollout_id}: {by_rollout[rollout_id]} and {value}."
            )
        by_rollout[rollout_id] = value

    return sorted(by_rollout.items())


def summarize_reward_points(
    points: list[tuple[int, float]],
    *,
    expected_steps: int | None = None,
    window_size: int = 10,
    metric: str = DEFAULT_METRIC,
) -> dict[str, Any]:
    """Summarize first/last windows without inventing an effect threshold."""
    if not points:
        raise ValueError(f"No {metric!r} points were found in the training log.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    rollout_ids = [rollout_id for rollout_id, _ in points]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("Reward points contain duplicate rollout ids.")
    if expected_steps is not None:
        if expected_steps <= 0:
            raise ValueError("expected_steps must be positive.")
        expected_ids = list(range(expected_steps))
        if rollout_ids != expected_ids:
            raise ValueError(f"Reward rollout ids are incomplete: expected {expected_ids}, got {rollout_ids}.")

    effective_window = min(window_size, len(points))
    first_values = [value for _, value in points[:effective_window]]
    last_values = [value for _, value in points[-effective_window:]]
    first_mean = fmean(first_values)
    last_mean = fmean(last_values)
    delta = last_mean - first_mean
    peak_rollout_id, peak_reward = max(points, key=lambda item: (item[1], -item[0]))
    return {
        "schema_version": SCHEMA_VERSION,
        "metric": metric,
        "num_steps": len(points),
        "window_size_requested": window_size,
        "window_size_used": effective_window,
        "first_window_mean": first_mean,
        "last_window_mean": last_mean,
        "last_minus_first": delta,
        # "Clearly improved" has no frozen numeric margin in Task 36. Report
        # the strict direction here and leave any later threshold explicit in
        # the experiment record rather than silently choosing one in code.
        "strictly_improved": delta > 0.0,
        "peak_reward": peak_reward,
        "peak_rollout_id": peak_rollout_id,
        "points": [{"rollout_id": rollout_id, "reward": value} for rollout_id, value in points],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--window-size", type=int, default=10)
    args = parser.parse_args()

    with args.log_file.open(encoding="utf-8", errors="replace") as source:
        points = extract_reward_points(source, metric=args.metric)
    summary = summarize_reward_points(
        points,
        expected_steps=args.expected_steps,
        window_size=args.window_size,
        metric=args.metric,
    )
    summary["log_file"] = str(args.log_file.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
