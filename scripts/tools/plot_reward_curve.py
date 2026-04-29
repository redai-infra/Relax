#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
ROLLOUT_PATTERN = re.compile(r"rollout\s+(\d+):\s+(\{.*\})")
PASSRATE_PATTERN = re.compile(r"passrate\s+(\d+):\s+(\{.*\})")
DEFAULT_METRIC = "rollout/raw_reward"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward curves from one or more Relax training logs.")
    parser.add_argument("logs", nargs="+", type=Path, help="Training log file(s) to parse.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <log>_reward_curve.png for one log, or reward_curve.png for multiple logs.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Rollout metric to plot from 'rollout N: ...' lines. Default: {DEFAULT_METRIC}.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Rolling mean window size. Set to 0 to disable. Default: 10.",
    )
    parser.add_argument(
        "--no-passrate",
        action="store_true",
        help="Do not overlay passrate/pass@1 and passrate/pass@8 for a single log.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title. Defaults to '<metric> vs Rollout Step'.",
    )
    return parser.parse_args()


def parse_log(log_path: Path, metric: str) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    metric_by_step: dict[int, float] = {}
    pass1_by_step: dict[int, float] = {}
    pass8_by_step: dict[int, float] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = ANSI_PATTERN.sub("", raw_line)

            rollout_match = ROLLOUT_PATTERN.search(line)
            if rollout_match:
                step = int(rollout_match.group(1))
                metrics = ast.literal_eval(rollout_match.group(2))
                if metric in metrics:
                    metric_by_step[step] = float(metrics[metric])
                continue

            passrate_match = PASSRATE_PATTERN.search(line)
            if passrate_match:
                step = int(passrate_match.group(1))
                metrics = ast.literal_eval(passrate_match.group(2))
                if "passrate/pass@1" in metrics:
                    pass1_by_step[step] = float(metrics["passrate/pass@1"])
                if "passrate/pass@8" in metrics:
                    pass8_by_step[step] = float(metrics["passrate/pass@8"])

    return metric_by_step, pass1_by_step, pass8_by_step


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        current_window = values[start : index + 1]
        smoothed.append(sum(current_window) / len(current_window))
    return smoothed


def default_output_path(logs: list[Path]) -> Path:
    if len(logs) == 1:
        return logs[0].with_name(f"{logs[0].stem}_reward_curve.png")
    return Path("reward_curve.png")


def plot_reward_curves(args: argparse.Namespace) -> None:
    logs: list[Path] = args.logs
    output_path = args.output or default_output_path(logs)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)
    parsed_logs = []

    for log_path in logs:
        metric_by_step, pass1_by_step, pass8_by_step = parse_log(log_path, args.metric)
        if not metric_by_step:
            print(f"Warning: no {args.metric!r} metrics found in {log_path}")
            continue

        steps = sorted(metric_by_step)
        values = [metric_by_step[step] for step in steps]
        label = log_path.stem
        ax.plot(steps, values, linewidth=1.2, marker="o", markersize=2.2, alpha=0.65, label=label)

        if args.rolling_window > 1:
            smoothed = rolling_mean(values, args.rolling_window)
            ax.plot(steps, smoothed, linewidth=2.0, label=f"{label} rolling mean ({args.rolling_window})")

        parsed_logs.append((log_path, steps, values, pass1_by_step, pass8_by_step))

    if not parsed_logs:
        raise SystemExit(f"No {args.metric!r} metrics found in any input log.")

    ax.set_title(args.title or f"{args.metric} vs Rollout Step")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(args.metric)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if len(parsed_logs) == 1 and not args.no_passrate:
        _, _, _, pass1_by_step, pass8_by_step = parsed_logs[0]
        if pass1_by_step:
            ax2 = ax.twinx()
            pass1_steps = sorted(pass1_by_step)
            ax2.plot(
                pass1_steps,
                [pass1_by_step[step] for step in pass1_steps],
                color="#2ca02c",
                linewidth=1.3,
                linestyle="--",
                alpha=0.8,
                label="passrate/pass@1",
            )
            if pass8_by_step:
                pass8_steps = sorted(pass8_by_step)
                ax2.plot(
                    pass8_steps,
                    [pass8_by_step[step] for step in pass8_steps],
                    color="#9467bd",
                    linewidth=1.1,
                    linestyle=":",
                    alpha=0.8,
                    label="passrate/pass@8",
                )
            ax2.set_ylabel("Pass rate")
            ax2.set_ylim(bottom=0)
            ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)

    print(f"Saved: {output_path}")
    for log_path, steps, values, _, _ in parsed_logs:
        print(
            f"{log_path}: points={len(steps)}, step_range={steps[0]}..{steps[-1]}, "
            f"min/max/last={min(values):.6g}/{max(values):.6g}/{values[-1]:.6g}"
        )


if __name__ == "__main__":
    plot_reward_curves(parse_args())
