#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
PERF_PATTERN = re.compile(r"perf\s+(\d+):\s+(\{.*\})")
DEFAULT_STEP_METRIC = "perf/step_time"
TRAIN_MICRO_BATCH_PATTERN = re.compile(r"perf/train_micro_batch_\d+_time")
TRAIN_MICRO_BATCH_TOTAL_KEY = "perf/train_micro_batch_total_time"
AGGREGATE_TIME_KEYS = {
    "perf/init_actor_time",
    "perf/step_time",
    "perf/train_time",
}
NON_TIME_KEYS = {
    "perf/wait_time_ratio",
    "perf/actor_train_tflops",
    "perf/actor_train_tok_per_s",
    "rollout/step",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-step timing curves from one or more Relax training logs.")
    parser.add_argument("logs", nargs="+", type=Path, help="Training log file(s) to parse.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <log>_perf_curve.png for one log, or perf_curve.png for multiple logs.",
    )
    parser.add_argument(
        "--step-metric",
        default=DEFAULT_STEP_METRIC,
        help=f"Metric to use as the per-step total line. Default: {DEFAULT_STEP_METRIC}.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Rolling mean window for the step metric. Set to 0 to disable. Default: 10.",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=1,
        help="Skip the first N parsed steps for summary and plotting. Default: 1 to ignore warmup/init.",
    )
    parser.add_argument(
        "--components",
        nargs="*",
        default=None,
        help="Specific perf component keys to stack, e.g. perf/actor_train_time perf/train_wait_time.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Also export an approximate Chrome Trace JSON timeline from parsed perf durations.",
    )
    parser.add_argument("--title", default=None, help="Plot title.")
    return parser.parse_args()


def parse_log(log_path: Path) -> dict[int, dict[str, float]]:
    perf_by_step: dict[int, dict[str, float]] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = ANSI_PATTERN.sub("", raw_line)
            match = PERF_PATTERN.search(line)
            if not match:
                continue

            step = int(match.group(1))
            metrics = ast.literal_eval(match.group(2))
            numeric_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, int | float) and key not in NON_TIME_KEYS
            }
            train_micro_batch_total = sum(
                value for key, value in numeric_metrics.items() if TRAIN_MICRO_BATCH_PATTERN.fullmatch(key)
            )
            if train_micro_batch_total > 0:
                numeric_metrics[TRAIN_MICRO_BATCH_TOTAL_KEY] = train_micro_batch_total
            perf_by_step.setdefault(step, {}).update(numeric_metrics)

    return perf_by_step


def default_output_path(logs: list[Path]) -> Path:
    if len(logs) == 1:
        return logs[0].with_name(f"{logs[0].stem}_perf_curve.png")
    return Path("perf_curve.png")


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        current_window = values[start : index + 1]
        smoothed.append(sum(current_window) / len(current_window))
    return smoothed


def sorted_steps(perf_by_step: dict[int, dict[str, float]], metric: str, skip_first: int) -> list[int]:
    steps = sorted(step for step, metrics in perf_by_step.items() if metric in metrics)
    return steps[skip_first:]


def discover_component_keys(
    perf_by_step: dict[int, dict[str, float]], steps: list[int], requested_components: list[str] | None
) -> list[str]:
    if requested_components is not None:
        return requested_components

    totals: dict[str, float] = {}
    for step in steps:
        for key, value in perf_by_step[step].items():
            if key in AGGREGATE_TIME_KEYS:
                continue
            if key == "perf/actor_train_time" and TRAIN_MICRO_BATCH_TOTAL_KEY in perf_by_step[step]:
                continue
            if TRAIN_MICRO_BATCH_PATTERN.fullmatch(key):
                continue
            if not key.startswith("perf/"):
                continue
            if "_time" not in key:
                continue
            totals[key] = totals.get(key, 0.0) + value

    return [key for key, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)]


def metric_values(perf_by_step: dict[int, dict[str, float]], steps: list[int], metric: str) -> list[float]:
    return [perf_by_step[step].get(metric, 0.0) for step in steps]


def short_label(metric: str) -> str:
    return metric.removeprefix("perf/").replace("_time", "").replace("_", " ")


def print_summary(log_path: Path, perf_by_step: dict[int, dict[str, float]], steps: list[int], components: list[str]) -> None:
    print(f"{log_path}: points={len(steps)}, step_range={steps[0]}..{steps[-1]}")

    if DEFAULT_STEP_METRIC in perf_by_step[steps[0]]:
        step_values = metric_values(perf_by_step, steps, DEFAULT_STEP_METRIC)
        print(
            f"  step_time avg/min/max/last="
            f"{sum(step_values) / len(step_values):.3f}/{min(step_values):.3f}/"
            f"{max(step_values):.3f}/{step_values[-1]:.3f}s"
        )

    component_avgs = []
    for component in components:
        values = metric_values(perf_by_step, steps, component)
        if any(values):
            component_avgs.append((component, sum(values) / len(values)))

    for component, avg_value in sorted(component_avgs, key=lambda item: item[1], reverse=True)[:8]:
        print(f"  component avg: {component}={avg_value:.3f}s")

    if component_avgs:
        bottleneck, avg_value = max(component_avgs, key=lambda item: item[1])
        print(f"  bottleneck candidate: {bottleneck} ({avg_value:.3f}s avg)")


def export_synthetic_trace(
    trace_output: Path,
    parsed_logs: list[tuple[Path, dict[int, dict[str, float]]]],
    args: argparse.Namespace,
) -> None:
    """Export coarse perf metrics as Chrome Trace events.

    The log only has per-step durations, not precise component start times. Each
    step is placed sequentially, and component timers are aligned at the step
    start on separate lanes so the trace shows relative duration and bottlenecks
    without pretending to know exact ordering.
    """
    trace_events = []

    for pid, (log_path, perf_by_step) in enumerate(parsed_logs):
        steps = sorted_steps(perf_by_step, args.step_metric, args.skip_first)
        if not steps:
            continue

        components = discover_component_keys(perf_by_step, steps, args.components)
        tid_by_component = {component: index + 1 for index, component in enumerate(components)}

        trace_events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": log_path.stem}})
        trace_events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": args.step_metric}})
        for component, tid in tid_by_component.items():
            trace_events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": component}})

        step_start_us = 0
        for step in steps:
            step_duration_s = perf_by_step[step].get(args.step_metric, 0.0)
            step_duration_us = int(step_duration_s * 1e6)
            trace_events.append(
                {
                    "name": args.step_metric,
                    "ph": "X",
                    "ts": step_start_us,
                    "dur": step_duration_us,
                    "pid": pid,
                    "tid": 0,
                    "args": {"step": step, "log": str(log_path)},
                }
            )

            for component in components:
                duration_s = perf_by_step[step].get(component, 0.0)
                if duration_s <= 0:
                    continue
                trace_events.append(
                    {
                        "name": component,
                        "ph": "X",
                        "ts": step_start_us,
                        "dur": int(duration_s * 1e6),
                        "pid": pid,
                        "tid": tid_by_component[component],
                        "args": {"step": step, "log": str(log_path)},
                    }
                )

            step_start_us += step_duration_us

    if not trace_events:
        raise SystemExit(f"No {args.step_metric!r} metrics found for trace export.")

    trace_output.parent.mkdir(parents=True, exist_ok=True)
    trace_output.write_text(json.dumps({"traceEvents": trace_events}, indent=2), encoding="utf-8")
    print(f"Saved trace: {trace_output}")


def plot_single_log(
    ax_time: plt.Axes,
    ax_breakdown: plt.Axes,
    log_path: Path,
    perf_by_step: dict[int, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    steps = sorted_steps(perf_by_step, args.step_metric, args.skip_first)
    if not steps:
        raise SystemExit(f"No {args.step_metric!r} metrics found in {log_path}")

    step_values = metric_values(perf_by_step, steps, args.step_metric)
    ax_time.plot(steps, step_values, linewidth=1.3, marker="o", markersize=2.2, label=args.step_metric)
    if args.rolling_window > 1:
        ax_time.plot(
            steps,
            rolling_mean(step_values, args.rolling_window),
            linewidth=2.0,
            label=f"rolling mean ({args.rolling_window})",
        )
    ax_time.set_ylabel("Seconds")
    ax_time.grid(True, alpha=0.25)
    ax_time.legend(loc="best")

    components = discover_component_keys(perf_by_step, steps, args.components)
    if components:
        ax_breakdown.stackplot(
            steps,
            [metric_values(perf_by_step, steps, component) for component in components],
            labels=[short_label(component) for component in components],
            alpha=0.85,
        )
        ax_breakdown.legend(loc="upper left")
    ax_breakdown.set_xlabel("Rollout step")
    ax_breakdown.set_ylabel("Component seconds")
    ax_breakdown.grid(True, alpha=0.25)

    print_summary(log_path, perf_by_step, steps, components)


def plot_multi_log(
    ax: plt.Axes,
    parsed_logs: list[tuple[Path, dict[int, dict[str, float]]]],
    args: argparse.Namespace,
) -> None:
    any_plotted = False
    for log_path, perf_by_step in parsed_logs:
        steps = sorted_steps(perf_by_step, args.step_metric, args.skip_first)
        if not steps:
            print(f"Warning: no {args.step_metric!r} metrics found in {log_path}")
            continue

        values = metric_values(perf_by_step, steps, args.step_metric)
        label = log_path.stem
        ax.plot(steps, values, linewidth=1.2, marker="o", markersize=2.0, alpha=0.65, label=label)
        if args.rolling_window > 1:
            ax.plot(steps, rolling_mean(values, args.rolling_window), linewidth=2.0, label=f"{label} rolling mean")

        components = discover_component_keys(perf_by_step, steps, args.components)
        print_summary(log_path, perf_by_step, steps, components)
        any_plotted = True

    if not any_plotted:
        raise SystemExit(f"No {args.step_metric!r} metrics found in any input log.")

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def plot_perf_curves(args: argparse.Namespace) -> None:
    output_path = args.output or default_output_path(args.logs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_logs = [(log_path, parse_log(log_path)) for log_path in args.logs]

    if len(parsed_logs) == 1:
        fig, (ax_time, ax_breakdown) = plt.subplots(2, 1, figsize=(12, 8), dpi=160, sharex=True)
        fig.suptitle(args.title or f"{args.step_metric} and Time Breakdown")
        plot_single_log(ax_time, ax_breakdown, parsed_logs[0][0], parsed_logs[0][1], args)
    else:
        fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)
        ax.set_title(args.title or f"{args.step_metric} Comparison")
        plot_multi_log(ax, parsed_logs, args)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")

    if args.trace_output is not None:
        export_synthetic_trace(args.trace_output, parsed_logs, args)


if __name__ == "__main__":
    plot_perf_curves(parse_args())
