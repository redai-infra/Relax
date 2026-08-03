#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Analyze Task40 runs without dropping failed or non-finite attempts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EXPECTED_SEEDS = (42, 1234, 2026)
MEMORY_PATTERN = re.compile(r"'allocated_GB': ([0-9.]+), 'reserved_GB': ([0-9.]+)")
SCENARIOS = {
    "on_policy": ("p3o_on_policy", "grpo_on_policy", 1.0),
    "temperature_0p6": ("p3o_temperature_0p6", "grpo_temperature_0p6", 0.6),
    "temperature_1p2": ("p3o_temperature_1p2", "grpo_temperature_1p2", 1.2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_exit_code(path: Path) -> int | None:
    try:
        return int(read_text(path).strip())
    except ValueError:
        return None


def load_scalars(tensorboard_dir: Path) -> dict[str, list[tuple[int, float]]]:
    event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    result: dict[str, list[tuple[int, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        result[tag] = sorted((event.step, event.value) for event in accumulator.Scalars(tag))
    return result


def series(scalars: dict[str, list[tuple[int, float]]], *tags: str) -> list[float]:
    for tag in tags:
        if tag in scalars:
            return [value for _, value in scalars[tag]]
    return []


def final_scalar(scalars: dict[str, list[tuple[int, float]]], *tags: str) -> float | None:
    values = series(scalars, *tags)
    return values[-1] if values else None


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def loss_spikes(losses: list[float]) -> tuple[int, int, float | None]:
    spikes = 0
    eligible = 0
    for index, value in enumerate(losses):
        window = losses[max(0, index - 5) : index]
        if len(window) < 3 or not all(math.isfinite(item) for item in [*window, value]):
            continue
        median = statistics.median(window)
        mad = statistics.median(abs(item - median) for item in window)
        eligible += 1
        spikes += abs(value - median) > 5 * max(mad, 1e-8)
    return spikes, eligible, spikes / eligible if eligible else None


def reward_collapse(rewards: list[float], unexpected_abort: bool, nonfinite_count: int) -> bool:
    if unexpected_abort or nonfinite_count:
        return True
    prior_peak = -math.inf
    below_count = 0
    for reward in rewards:
        if not math.isfinite(reward):
            return True
        if prior_peak > 0 and reward <= 0.5 * prior_peak:
            below_count += 1
            if below_count >= 3:
                return True
        else:
            below_count = 0
        prior_peak = max(prior_peak, reward)
    return False


def memory_peaks(log_text: str) -> tuple[float | None, float | None]:
    matches = list(MEMORY_PATTERN.finditer(log_text))
    if not matches:
        return None, None
    return max(float(match.group(1)) for match in matches), max(float(match.group(2)) for match in matches)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    identity = json.loads(read_text(run_dir / "run_identity.json"))
    config = run_dir.parents[1].name
    seed = int(identity["seed"])
    exit_code = read_exit_code(run_dir / "exit_code.txt")
    job_status = read_text(run_dir / "job_status.txt")
    log_text = read_text(run_dir / "stdout_stderr.log")
    event_files = list((run_dir / "tensorboard").glob("events.out.tfevents.*"))
    succeeded = (
        exit_code == 0
        and "succeeded" in job_status.lower()
        and "All training steps finished" in log_text
        and "Main func successfully" in log_text
        and bool(event_files)
    )
    scalars = load_scalars(run_dir / "tensorboard") if event_files else {}
    loss_values = series(scalars, "train/p3o/total_loss", "train/loss")
    rewards = series(scalars, "rollout/raw_reward")
    all_values = [value for values in scalars.values() for _, value in values]
    nonfinite_count = sum(not math.isfinite(value) for value in all_values)
    spikes, eligible_steps, spike_rate = loss_spikes(loss_values)
    peak_allocated, peak_reserved = memory_peaks(log_text)
    return {
        "config": config,
        "method": identity["method"],
        "temperature": identity["rollout_temperature"],
        "seed": seed,
        "run_id": run_dir.name,
        "run_path": str(run_dir),
        "success": succeeded,
        "exit_code": exit_code,
        "job_succeeded": "succeeded" in job_status.lower(),
        "training_completed": "All training steps finished" in log_text,
        "tensorboard_present": bool(event_files),
        "train_steps_observed": len(loss_values),
        "final_eval_pass_at_1": final_scalar(scalars, "eval/gsm8k-pass@1"),
        "final_eval_pass_at_16": final_scalar(scalars, "eval/gsm8k-pass@16"),
        "reward_mean": finite_mean(rewards),
        "reward_final": rewards[-1] if rewards else None,
        "loss_spikes": spikes,
        "loss_spike_eligible_steps": eligible_steps,
        "loss_spike_rate": spike_rate,
        "nonfinite_scalar_count": nonfinite_count,
        "unexpected_abort": not succeeded,
        "reward_collapse": reward_collapse(rewards, not succeeded, nonfinite_count),
        "normalized_ess_mean": finite_mean(series(scalars, "train/p3o/normalized_ess")),
        "adaptive_cap_mean": finite_mean(series(scalars, "train/p3o/adaptive_cap")),
        "ratio_mean": finite_mean(series(scalars, "train/p3o/ratio_mean")),
        "ratio_std_mean": finite_mean(series(scalars, "train/p3o/ratio_std")),
        "cap_fraction_mean": finite_mean(series(scalars, "train/p3o/cap_fraction")),
        "behavior_kl_proxy_mean": finite_mean(series(scalars, "train/p3o/behavior_kl_proxy")),
        "adaptive_kl_loss_mean": finite_mean(series(scalars, "train/p3o/adaptive_kl_loss")),
        "reference_kl_mean": finite_mean(series(scalars, "train/p3o/reference_kl")),
        "total_loss_mean": finite_mean(loss_values),
        "valid_tokens_mean": finite_mean(series(scalars, "train/p3o/valid_tokens")),
        "step_time_steady_mean_s": finite_mean(series(scalars, "perf/step_time")[1:]),
        "actor_train_tok_per_s_steady_mean": finite_mean(series(scalars, "perf/actor_train_tok_per_s")[1:]),
        "peak_allocated_gb": peak_allocated,
        "peak_reserved_gb": peak_reserved,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_metrics = [
        "final_eval_pass_at_1",
        "reward_mean",
        "loss_spike_rate",
        "normalized_ess_mean",
        "adaptive_cap_mean",
        "ratio_mean",
        "ratio_std_mean",
        "cap_fraction_mean",
        "behavior_kl_proxy_mean",
        "adaptive_kl_loss_mean",
        "reference_kl_mean",
        "total_loss_mean",
        "valid_tokens_mean",
        "step_time_steady_mean_s",
        "actor_train_tok_per_s_steady_mean",
        "peak_allocated_gb",
        "peak_reserved_gb",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    output = []
    for config, config_rows in sorted(grouped.items()):
        valid_rows = [row for row in config_rows if row["success"]]
        item: dict[str, Any] = {
            "config": config,
            "method": config_rows[0]["method"],
            "temperature": config_rows[0]["temperature"],
            "planned_runs": len(config_rows),
            "valid_runs": len(valid_rows),
            "seeds": ";".join(str(row["seed"]) for row in sorted(valid_rows, key=lambda row: row["seed"])),
            "collapse_count": sum(row["reward_collapse"] for row in config_rows),
            "nonfinite_scalar_count": sum(row["nonfinite_scalar_count"] for row in config_rows),
        }
        for metric in numeric_metrics:
            values = [row[metric] for row in valid_rows if row[metric] is not None]
            item[f"{metric}_mean"] = statistics.fmean(values) if values else None
            item[f"{metric}_sample_variance"] = statistics.variance(values) if len(values) >= 2 else None
            item[f"{metric}_sample_std"] = statistics.stdev(values) if len(values) >= 2 else None
            item[f"{metric}_min"] = min(values) if values else None
            item[f"{metric}_max"] = max(values) if values else None
        output.append(item)
    return output


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["config"], row["seed"]): row for row in rows}
    output = []
    for scenario, (p3o_config, grpo_config, temperature) in SCENARIOS.items():
        for seed in EXPECTED_SEEDS:
            p3o = indexed.get((p3o_config, seed))
            grpo = indexed.get((grpo_config, seed))
            pair_valid = bool(p3o and grpo and p3o["success"] and grpo["success"])
            item: dict[str, Any] = {
                "scenario": scenario,
                "temperature": temperature,
                "seed": seed,
                "pair_complete": pair_valid,
                "p3o_config": p3o_config,
                "grpo_config": grpo_config,
            }
            if pair_valid:
                quality_delta = (
                    p3o["final_eval_pass_at_1"] - grpo["final_eval_pass_at_1"]
                    if p3o["final_eval_pass_at_1"] is not None and grpo["final_eval_pass_at_1"] is not None
                    else None
                )
                grpo_spike_rate = grpo["loss_spike_rate"]
                p3o_spike_rate = p3o["loss_spike_rate"]
                relative_reduction = (
                    (grpo_spike_rate - p3o_spike_rate) / grpo_spike_rate
                    if grpo_spike_rate not in {None, 0} and p3o_spike_rate is not None
                    else None
                )
                item.update(
                    {
                        "p3o_minus_grpo_pass_at_1": quality_delta,
                        "quality_guard_within_5pp": quality_delta is not None and quality_delta >= -0.05,
                        "p3o_loss_spike_rate": p3o_spike_rate,
                        "grpo_loss_spike_rate": grpo_spike_rate,
                        "relative_loss_spike_reduction": relative_reduction,
                        "stability_same_direction": (
                            p3o_spike_rate is not None
                            and grpo_spike_rate is not None
                            and p3o_spike_rate < grpo_spike_rate
                        ),
                    }
                )
            output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_verdicts(analysis_dir: Path, rows: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    lines = [
        "# Frozen gate verdict",
        "",
        "The original 10% spike-reduction, 2/3 direction, and -5pp quality gates are unchanged.",
        "",
        "| scenario | valid pairs | quality guard | spike reduction | direction | verdict |",
        "|---|---:|---|---:|---:|---|",
    ]
    for scenario in SCENARIOS:
        pairs = [item for item in paired if item["scenario"] == scenario and item["pair_complete"]]
        quality = len(pairs) == 3 and all(item.get("quality_guard_within_5pp") for item in pairs)
        p3o_rates = [item.get("p3o_loss_spike_rate") for item in pairs]
        grpo_rates = [item.get("grpo_loss_spike_rate") for item in pairs]
        reduction = None
        if len(pairs) == 3 and all(value is not None for value in [*p3o_rates, *grpo_rates]):
            grpo_mean = statistics.fmean(grpo_rates)
            reduction = (grpo_mean - statistics.fmean(p3o_rates)) / grpo_mean if grpo_mean else None
        direction = sum(bool(item.get("stability_same_direction")) for item in pairs)
        stability = reduction is not None and reduction >= 0.10 and direction >= 2
        verdict = "PASS" if len(pairs) == 3 and quality and (scenario == "on_policy" or stability) else "FAIL"
        lines.append(
            f"| {scenario} | {len(pairs)}/3 | {'PASS' if quality else 'FAIL'} | "
            f"{reduction if reduction is not None else 'NA'} | {direction}/3 | {verdict} |"
        )
    analysis_dir.joinpath("frozen_gate_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    successful = [row for row in rows if row["success"]]
    acceptance = [
        "# Official acceptance matrix",
        "",
        "| requirement | status | evidence | limitation |",
        "|---|---|---|---|",
        f"| three temperature configs, two methods, three seeds | "
        f"{'PASS' if len(successful) == 18 else 'PARTIAL'} | per_run_metrics.csv | {len(successful)}/18 valid runs |",
        "| mean, sample variance, and sample standard deviation | PASS | aggregate_metrics.csv | valid runs only; counts retained |",
        "| mismatch improvement | See frozen verdict | paired_seed_comparison.csv | no post-hoc threshold changes |",
        "| failures and non-finite values retained | PASS | failures.csv | failed runs are excluded only from numeric means |",
    ]
    analysis_dir.joinpath("official_acceptance_matrix.md").write_text("\n".join(acceptance) + "\n", encoding="utf-8")


def write_performance_table(analysis_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Throughput and memory",
        "",
        "| config | valid/planned | actor train tok/s mean | peak allocated GiB mean | peak reserved GiB mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['valid_runs']}/{row['planned_runs']} | "
            f"{row['actor_train_tok_per_s_steady_mean_mean']} | {row['peak_allocated_gb_mean']} | "
            f"{row['peak_reserved_gb_mean']} |"
        )
    analysis_dir.joinpath("throughput_memory_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(analysis_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use(Path(__file__).with_name("task40_academic.mplstyle"))

    curves_dir = analysis_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    valid = [row for row in rows if row["success"]]
    temperatures = (0.6, 1.0, 1.2)
    temperature_colors = {0.6: "#557A95", 1.0: "#8A8878", 1.2: "#A65F5F"}
    method_styles = {"p3o": ("-", "o"), "grpo": ("--", "s")}
    step_plot_specs = {
        "reward_vs_step_by_temperature": (
            {"p3o": ("rollout/raw_reward",), "grpo": ("rollout/raw_reward",)},
            "mean rollout reward",
            "Reward",
        ),
        "ess_vs_step_by_temperature": (
            {"p3o": ("train/p3o/normalized_ess",)},
            "normalized ESS",
            "Normalized ESS",
        ),
        "kl_vs_step_by_temperature": (
            {
                "p3o": ("train/p3o/behavior_kl_proxy",),
                "grpo": ("train/ppo_kl", "train/approx_kl"),
            },
            "sampled-token KL proxy",
            "KL proxy",
        ),
    }
    step_plot_claims = {
        "reward_vs_step_by_temperature": (
            "P3O versus GRPO rollout reward by behavior temperature; mean ± sample standard deviation across "
            "three seeds."
        ),
        "ess_vs_step_by_temperature": (
            "P3O optimizer-step normalized ESS by behavior temperature; mean ± sample standard deviation across "
            "three seeds."
        ),
        "kl_vs_step_by_temperature": (
            "P3O behavior-KL proxy versus GRPO PPO-KL proxy by behavior temperature; mean ± sample standard "
            "deviation across three seeds."
        ),
    }
    catalog = []
    for stem, (tags_by_method, ylabel, title) in step_plot_specs.items():
        figure, axis = plt.subplots(figsize=(7.2, 3.8))
        plotted_methods = []
        for method, tags in tags_by_method.items():
            linestyle, marker = method_styles[method]
            method_plotted = False
            for temperature in temperatures:
                by_step: dict[int, list[float]] = defaultdict(list)
                for row in valid:
                    if row["method"] != method or row["temperature"] != temperature:
                        continue
                    scalars = load_scalars(Path(row["run_path"]) / "tensorboard")
                    selected = next((scalars[tag] for tag in tags if tag in scalars), [])
                    for step, value in selected:
                        if math.isfinite(value):
                            by_step[step].append(value)
                if not by_step:
                    continue
                steps = sorted(by_step)
                means = [statistics.fmean(by_step[step]) for step in steps]
                stds = [statistics.stdev(by_step[step]) if len(by_step[step]) >= 2 else 0.0 for step in steps]
                color = temperature_colors[temperature]
                axis.plot(
                    steps,
                    means,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=2,
                )
                axis.fill_between(
                    steps,
                    [mean - std for mean, std in zip(means, stds, strict=True)],
                    [mean + std for mean, std in zip(means, stds, strict=True)],
                    color=color,
                    alpha=0.10,
                )
                method_plotted = True
            if method_plotted:
                plotted_methods.append(method)
        axis.set_xlabel("optimizer step")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if stem == "reward_vs_step_by_temperature":
            axis.axhline(0.0, color="#9CA3AF", linewidth=0.8, zorder=0)
        elif stem in {"ess_vs_step_by_temperature", "kl_vs_step_by_temperature"}:
            axis.set_ylim(bottom=0.0)
        legend_handles = [
            Line2D([0], [0], color=temperature_colors[temperature], label=f"T={temperature}")
            for temperature in temperatures
        ]
        if len(plotted_methods) > 1:
            legend_handles.extend(
                Line2D(
                    [0],
                    [0],
                    color="#4B5563",
                    linestyle=method_styles[method][0],
                    marker=method_styles[method][1],
                    label=method.upper(),
                )
                for method in plotted_methods
            )
        axis.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(legend_handles),
        )
        figure.tight_layout()
        for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
            figure.savefig(curves_dir / f"{stem}.{suffix}", **options)
        plt.close(figure)
        catalog.append(
            {
                "stem": stem,
                "surface_class": "paper_main",
                "source_data": "analysis/per_run_metrics.csv and per-run TensorBoard events",
                "generator": "Relax/scripts/experiments/task40/analyze_overnight.py",
                "exports": [f"analysis/curves/{stem}.png", f"analysis/curves/{stem}.pdf"],
                "main_comparison": step_plot_claims[stem],
                "review_revision": (
                    "Unified temperature colors and method line styles; moved the compact legend above the data; "
                    "corrected proxy-KL terminology."
                ),
            }
        )

    temperature_plot_specs = {
        "loss_spike_rate_by_temperature": (
            "loss_spike_rate",
            "loss-spike rate",
            "Pre-registered loss-spike rate",
        ),
        "final_quality_by_temperature": (
            "final_eval_pass_at_1",
            "final GSM8K Pass@1",
            "Final GSM8K quality",
        ),
    }
    for stem, (metric, ylabel, title) in temperature_plot_specs.items():
        figure, axis = plt.subplots(figsize=(5.2, 3.5))
        positions = list(range(len(temperatures)))
        for method, offset in (("p3o", -0.08), ("grpo", 0.08)):
            _, marker = method_styles[method]
            plotted_positions = []
            means = []
            errors = []
            colors = []
            for position, temperature in zip(positions, temperatures, strict=True):
                values = [
                    row[metric]
                    for row in valid
                    if row["method"] == method and row["temperature"] == temperature and row[metric] is not None
                ]
                if values:
                    plotted_positions.append(position + offset)
                    means.append(statistics.fmean(values))
                    errors.append(statistics.stdev(values) if len(values) >= 2 else 0.0)
                    colors.append(temperature_colors[temperature])
            if plotted_positions:
                for position, mean, error, color in zip(plotted_positions, means, errors, colors, strict=True):
                    axis.errorbar(
                        position,
                        mean,
                        yerr=error,
                        color=color,
                        marker=marker,
                        linestyle="none",
                        markeredgecolor="white",
                        markeredgewidth=0.7,
                    )
        axis.set_xlabel("rollout behavior temperature")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xticks(positions, [str(temperature) for temperature in temperatures])
        axis.set_ylim(0.0, 1.0 if metric == "final_eval_pass_at_1" else None)
        axis.legend(
            handles=[
                Line2D([0], [0], color="#4B5563", marker="o", linestyle="none", label="P3O"),
                Line2D([0], [0], color="#4B5563", marker="s", linestyle="none", label="GRPO"),
            ],
            loc="upper right",
        )
        figure.tight_layout()
        for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
            figure.savefig(curves_dir / f"{stem}.{suffix}", **options)
        plt.close(figure)
        catalog.append(
            {
                "stem": stem,
                "surface_class": "paper_main",
                "source_data": "analysis/per_run_metrics.csv",
                "generator": "Relax/scripts/experiments/task40/analyze_overnight.py",
                "exports": [f"analysis/curves/{stem}.png", f"analysis/curves/{stem}.pdf"],
                "main_comparison": f"{title}: mean ± sample standard deviation across three seeds",
                "review_revision": (
                    "Changed the crowded continuous line chart to a categorical point-range plot with method offsets."
                ),
            }
        )
    curves_dir.joinpath("figure_catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    analysis_dir = args.evidence_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path.parent for path in (args.evidence_root / "runs").rglob("run_identity.json"))
    if not run_dirs:
        raise SystemExit(f"no run_identity.json files under {args.evidence_root / 'runs'}")
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    for row in rows:
        Path(row["run_path"]).joinpath("metrics.json").write_text(
            json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    aggregate_rows = aggregate(rows)
    paired_rows = paired_comparisons(rows)
    failure_rows = [row for row in rows if not row["success"] or row["nonfinite_scalar_count"]]
    write_csv(analysis_dir / "per_run_metrics.csv", rows)
    write_csv(analysis_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(analysis_dir / "paired_seed_comparison.csv", paired_rows)
    write_csv(analysis_dir / "failures.csv", failure_rows)
    write_verdicts(analysis_dir, rows, paired_rows)
    write_performance_table(analysis_dir, aggregate_rows)
    make_plots(analysis_dir, rows)
    print(f"planned_attempts={len(rows)} valid_attempts={sum(row['success'] for row in rows)}")
    print(analysis_dir)


if __name__ == "__main__":
    main()
