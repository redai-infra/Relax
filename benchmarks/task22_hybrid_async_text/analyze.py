#!/usr/bin/env python3
"""Analyze Task 22 paired Hybrid-async benchmark artifacts."""

from __future__ import annotations

import ast
import csv
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ARTIFACT_ROOT = Path(
    os.environ.get("TASK22_ARTIFACT_ROOT", REPO_ROOT / "benchmark_artifacts" / "task22-hybrid-async-text-v2")
)
OUTPUT_ROOT = Path(
    os.environ.get("TASK22_OUTPUT_ROOT", REPO_ROOT / "benchmarks" / "results" / "task22-hybrid-async-text")
)
VARIANTS = ("baseline", "zero_kl", "optimized")
RUN_IDS = (1, 2, 3)
STABLE_FIRST_STEP = 5
STABLE_LAST_STEP = 15
GLOBAL_BATCH_SIZE = 32
UPDATE_WEIGHT_BUFFER_SIZE = 512 * 1024**2
TRAIN_TOKEN_BUDGETS = {"baseline": 8192, "zero_kl": 8192, "optimized": 12288}
LOG_PROB_TOKEN_BUDGETS = {"baseline": 8192, "zero_kl": 8192, "optimized": 24576}
WORKLOAD_KEYS = (
    "git_commit",
    "python",
    "torch",
    "torch_cuda",
    "ray",
    "sglang",
    "transformers",
    "gpu_name",
    "gpu_driver",
    "cpu_model",
    "cpu_logical_count",
    "memory_total_kib",
    "model_path",
    "dataset_sha256",
    "dataset_repo_id",
    "dataset_split",
    "dataset_subset_size",
    "model_sha256",
    "cuda_visible_devices",
    "actor_gpu",
    "rollout_gpu",
    "num_rollout",
    "rollout_batch_size",
    "n_samples_per_prompt",
    "global_batch_size",
    "max_response_len",
    "max_staleness",
    "update_weights_interval",
    "kl_loss_coef",
    "use_tis",
)


class _NonFiniteNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.Constant:
        if node.id == "inf":
            return ast.copy_location(ast.Constant(float("inf")), node)
        if node.id == "nan":
            return ast.copy_location(ast.Constant(float("nan")), node)
        raise ValueError(f"Unexpected name in metric dictionary: {node.id}")


def parse_metric_dict(value: str) -> dict[str, Any] | None:
    try:
        expression = ast.parse(value, mode="eval")
        parsed = ast.literal_eval(_NonFiniteNormalizer().visit(expression))
    except (SyntaxError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def metric_records(path: Path, kind: str, required_key: str | None = None) -> dict[int, dict[str, Any]]:
    pattern = re.compile(rf"{kind} (\d+): (\{{.*\}})")
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        record = parse_metric_dict(match.group(2))
        if record is not None and (required_key is None or required_key in record):
            records[int(match.group(1))] = record
    return records


def actor_completion_timestamps(path: Path) -> dict[int, datetime]:
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
        r".*Actor training completed step (\d+)/(\d+)"
    )
    records: dict[int, datetime] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            records[int(match.group(2))] = datetime.fromisoformat(match.group(1).replace(",", "."))
    return records


def manifest_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(math.ceil(len(ordered) * fraction)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def gpu_stats(
    path: Path,
    stable_start: datetime,
    stable_end: datetime,
    actor_gpu: int,
    rollout_gpu: int,
) -> dict[str, float]:
    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?),"
        r"\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*$"
    )
    util: dict[int, list[float]] = defaultdict(list)
    peak_memory: dict[int, float] = defaultdict(float)
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        timestamp = datetime.fromisoformat(match.group(1)).replace(tzinfo=None)
        gpu_id = int(match.group(2))
        gpu_util = float(match.group(3))
        memory_used = float(match.group(4))
        if gpu_id in (actor_gpu, rollout_gpu) and stable_start <= timestamp <= stable_end:
            util[gpu_id].append(gpu_util)
            peak_memory[gpu_id] = max(peak_memory[gpu_id], memory_used)
    combined = util[actor_gpu] + util[rollout_gpu]
    return {
        "gpu_util_pct": mean(combined),
        "actor_gpu_util_pct": mean(util[actor_gpu]),
        "rollout_gpu_util_pct": mean(util[rollout_gpu]),
        "actor_peak_memory_mib": peak_memory[actor_gpu],
        "rollout_peak_memory_mib": peak_memory[rollout_gpu],
        "peak_memory_mib": max(peak_memory[actor_gpu], peak_memory[rollout_gpu]),
    }


def nonfinite_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    known = 0
    unexpected = 0
    for record in records:
        for key, value in record.items():
            if not isinstance(value, (int, float)) or math.isfinite(float(value)):
                continue
            if key == "perf/device_peak_tflops" and math.isinf(float(value)):
                known += 1
            else:
                unexpected += 1
    return known, unexpected


def dynamic_microbatch_counts(path: Path) -> list[int]:
    pattern = re.compile(r"After dynamic batching, num_microbatches: \[(\d+)\]")
    return [int(match.group(1)) for match in pattern.finditer(path.read_text(errors="replace"))]


def summarize_run(variant: str, run_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = ARTIFACT_ROOT / variant / f"run-{run_id}"
    log_path = run_dir / "train.log"
    manifest = manifest_values(run_dir / "manifest.txt")
    if manifest.get("job_status") != "SUCCEEDED":
        raise RuntimeError(f"{variant} run {run_id} did not succeed")

    perf = metric_records(log_path, "perf", required_key="perf/step_time")
    rollout_perf = metric_records(log_path, "perf", required_key="perf/rollout_time")
    rollout = metric_records(log_path, "rollout", required_key="rollout/response_lengths")
    train = metric_records(log_path, "step", required_key="train/loss")
    completion_timestamps = actor_completion_timestamps(log_path)
    expected_steps = list(range(int(manifest["num_rollout"])))
    for kind, records in (
        ("actor perf", perf),
        ("rollout", rollout),
        ("train", train),
        ("actor completion", completion_timestamps),
    ):
        if sorted(records) != expected_steps:
            raise RuntimeError(f"{variant} run {run_id} has incomplete {kind} steps: {sorted(records)}")

    stable_steps = [step for step in expected_steps if STABLE_FIRST_STEP <= step <= STABLE_LAST_STEP]
    if stable_steps != list(range(STABLE_FIRST_STEP, STABLE_LAST_STEP + 1)):
        raise RuntimeError(
            f"{variant} run {run_id} does not cover required steps {STABLE_FIRST_STEP}-{STABLE_LAST_STEP}"
        )
    step_rows: list[dict[str, Any]] = []
    for step in stable_steps:
        perf_record = perf[step]
        rollout_record = rollout[step]
        step_time = (completion_timestamps[step] - completion_timestamps[step - 1]).total_seconds()
        if step_time <= 0:
            raise RuntimeError(f"{variant} run {run_id} has non-positive completion interval at step {step}")
        response_length = float(rollout_record["rollout/response_lengths"])
        framework_step_time = float(perf_record["perf/step_time"])
        step_rows.append(
            {
                "variant": variant,
                "run_id": run_id,
                "step": step,
                "e2e_step_time_s": step_time,
                "framework_step_time_s": framework_step_time,
                "response_tokens_per_s": response_length * GLOBAL_BATCH_SIZE / framework_step_time,
                "samples_per_s": GLOBAL_BATCH_SIZE / framework_step_time,
                "train_wait_s": float(perf_record.get("perf/train_wait_time", 0.0)),
                "actor_train_s": float(perf_record.get("perf/actor_train_time", 0.0)),
                "preceding_update_weights_s": float(perf_record.get("perf/update_weights_time", 0.0)),
                "rollout_s": (float(rollout_perf[step]["perf/rollout_time"]) if step in rollout_perf else None),
                "ref_log_probs_s": perf_record.get("perf/ref_log_probs_time"),
                "raw_reward": float(rollout_record.get("rollout/raw_reward", 0.0)),
            }
        )

    step_times = [float(row["e2e_step_time_s"]) for row in step_rows]
    framework_step_times = [float(row["framework_step_time_s"]) for row in step_rows]
    total_framework_time = sum(framework_step_times)
    total_response_tokens = sum(
        float(rollout[step]["rollout/response_lengths"]) * GLOBAL_BATCH_SIZE for step in stable_steps
    )
    gpu = gpu_stats(
        run_dir / "gpu.csv",
        completion_timestamps[STABLE_FIRST_STEP - 1],
        completion_timestamps[stable_steps[-1]],
        int(manifest["actor_gpu"]),
        int(manifest["rollout_gpu"]),
    )
    known_nonfinite, unexpected_nonfinite = nonfinite_counts(
        list(perf.values()) + list(rollout_perf.values()) + list(rollout.values()) + list(train.values())
    )
    error_pattern = re.compile(r"Actor training failed|CUDA out of memory|Traceback \(most recent call last\)")
    error_count = len(error_pattern.findall(log_path.read_text(errors="replace")))
    train_values = [train[step] for step in stable_steps]
    microbatch_counts = dynamic_microbatch_counts(log_path)
    summary = {
        "variant": variant,
        "run_id": run_id,
        "git_commit": manifest["git_commit"],
        "git_status": manifest["git_status"],
        "dataset_sha256": manifest["dataset_sha256"],
        "model_sha256": manifest["model_sha256"],
        "python": manifest["python"],
        "torch": manifest["torch"],
        "torch_cuda": manifest["torch_cuda"],
        "ray": manifest["ray"],
        "sglang": manifest["sglang"],
        "transformers": manifest["transformers"],
        "gpu_name": manifest["gpu_name"],
        "gpu_driver": manifest["gpu_driver"],
        "seed": int(manifest["seed"]),
        "update_weight_buffer_size": int(manifest["update_weight_buffer_size"]),
        "max_tokens_per_gpu": int(manifest["max_tokens_per_gpu"]),
        "log_probs_max_tokens_per_gpu": int(manifest["log_probs_max_tokens_per_gpu"]),
        "total_steps": len(perf),
        "stable_steps": len(stable_steps),
        "e2e_step_time_s": mean(step_times),
        "e2e_step_time_p50_s": median(step_times),
        "e2e_step_time_p95_s": percentile(step_times, 0.95),
        "framework_step_time_s": mean(framework_step_times),
        "response_tokens_per_s": total_response_tokens / total_framework_time,
        "samples_per_s": len(stable_steps) * GLOBAL_BATCH_SIZE / total_framework_time,
        "train_wait_s": mean([float(row["train_wait_s"]) for row in step_rows]),
        "actor_train_s": mean([float(row["actor_train_s"]) for row in step_rows]),
        "preceding_update_weights_s": mean([float(row["preceding_update_weights_s"]) for row in step_rows]),
        "rollout_s": mean([float(row["rollout_s"]) for row in step_rows if row["rollout_s"] is not None]),
        "ref_log_probs_s": mean(
            [float(row["ref_log_probs_s"]) for row in step_rows if row["ref_log_probs_s"] is not None]
        ),
        "raw_reward": mean([float(rollout[step].get("rollout/raw_reward", 0.0)) for step in stable_steps]),
        "loss": mean([float(record.get("train/loss", 0.0)) for record in train_values]),
        "tis": mean([float(record.get("train/tis", 1.0)) for record in train_values]),
        "tis_clipfrac": mean([float(record.get("train/tis_clipfrac", 0.0)) for record in train_values]),
        "observed_samples": len(stable_steps) * GLOBAL_BATCH_SIZE,
        "total_samples": len(rollout) * GLOBAL_BATCH_SIZE,
        "actual_response_tokens": sum(
            float(record["rollout/response_lengths"]) * GLOBAL_BATCH_SIZE for record in rollout.values()
        ),
        "known_peak_inf": known_nonfinite,
        "unexpected_nonfinite": unexpected_nonfinite,
        "runtime_errors": error_count,
        "has_reference_forward": all("perf/ref_log_probs_time" in record for record in perf.values()),
        "dynamic_microbatches_mean": mean(microbatch_counts),
        **gpu,
    }
    return summary, step_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_chart(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 960, 420
    margin = 55
    values = [float(row["response_tokens_per_s"]) for row in rows]
    y_min, y_max = min(values) * 0.97, max(values) * 1.03
    colors = {"baseline": "#4b5563", "zero_kl": "#2563eb", "optimized": "#0f766e"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Task 22 stable response throughput</text>',
    ]
    for variant in VARIANTS:
        points = []
        variant_rows = [row for row in rows if row["variant"] == variant]
        for index, row in enumerate(variant_rows):
            x = margin + index * (width - 2 * margin) / max(len(variant_rows) - 1, 1)
            value = float(row["response_tokens_per_s"])
            y = height - margin - (value - y_min) * (height - 2 * margin) / (y_max - y_min)
            points.append(f"{x:.1f},{y:.1f}")
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[variant]}" stroke-width="2"/>'
        )
    elements.extend(
        [
            f'<text x="{margin}" y="{height - 15}" font-family="sans-serif" font-size="12">steps 5-15 across 3 paired trials</text>',
            '<rect x="760" y="45" width="12" height="12" fill="#4b5563"/><text x="780" y="56" font-family="sans-serif" font-size="12">baseline</text>',
            '<rect x="760" y="65" width="12" height="12" fill="#2563eb"/><text x="780" y="76" font-family="sans-serif" font-size="12">zero_kl</text>',
            '<rect x="760" y="85" width="12" height="12" fill="#0f766e"/><text x="780" y="96" font-family="sans-serif" font-size="12">optimized</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements) + "\n")


def aggregate(rows: list[dict[str, Any]], variant: str, key: str) -> float:
    return mean([float(row[key]) for row in rows if row["variant"] == variant])


def write_report(summaries: list[dict[str, Any]], path: Path) -> None:
    baseline_tps = aggregate(summaries, "baseline", "response_tokens_per_s")
    zero_kl_tps = aggregate(summaries, "zero_kl", "response_tokens_per_s")
    optimized_tps = aggregate(summaries, "optimized", "response_tokens_per_s")
    baseline_step = aggregate(summaries, "baseline", "e2e_step_time_s")
    zero_kl_step = aggregate(summaries, "zero_kl", "e2e_step_time_s")
    optimized_step = aggregate(summaries, "optimized", "e2e_step_time_s")
    zero_kl_throughput_gain = (zero_kl_tps / baseline_tps - 1.0) * 100.0
    token_budget_throughput_gain = (optimized_tps / zero_kl_tps - 1.0) * 100.0
    total_throughput_gain = (optimized_tps / baseline_tps - 1.0) * 100.0
    zero_kl_latency_change = (zero_kl_step / baseline_step - 1.0) * 100.0
    token_budget_latency_change = (optimized_step / zero_kl_step - 1.0) * 100.0
    total_latency_change = (optimized_step / baseline_step - 1.0) * 100.0
    acceptance = "PASS" if token_budget_throughput_gain >= 5.0 else "NOT MET"
    commit = summaries[0]["git_commit"]
    dataset_sha = summaries[0]["dataset_sha256"]
    model_sha = summaries[0]["model_sha256"]
    environment = manifest_values(ARTIFACT_ROOT / "baseline" / "run-1" / "manifest.txt")
    lines = [
        "# Task 22 Hybrid-async text performance report",
        "",
        "## Result",
        "",
        f"Three paired trials on commit `{commit}` show that pruning the zero-coefficient KL reference path "
        f"changes response throughput by {zero_kl_throughput_gain:+.2f}% and E2E latency by "
        f"{zero_kl_latency_change:+.2f}%. Applying the 12288/24576 train/log-prob token budgets then "
        f"changes throughput by another {token_budget_throughput_gain:+.2f}% and latency by "
        f"{token_budget_latency_change:+.2f}%. The combined change is {total_throughput_gain:+.2f}% throughput "
        f"and {total_latency_change:+.2f}% latency versus baseline.",
        f"Acceptance: **{acceptance}** against the frozen target of at least +5% response throughput for "
        "`optimized` versus `zero_kl`.",
        "",
        "| variant | train/log-prob budget | microbatches | framework/E2E step (s) | response tok/s | samples/s | GPU util | actor peak MiB | rollout peak MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        variant_rows = [row for row in summaries if row["variant"] == variant]
        lines.append(
            f"| {variant} | {TRAIN_TOKEN_BUDGETS[variant]}/{LOG_PROB_TOKEN_BUDGETS[variant]} | "
            f"{aggregate(summaries, variant, 'dynamic_microbatches_mean'):.2f} | "
            f"{aggregate(summaries, variant, 'framework_step_time_s'):.3f}/"
            f"{aggregate(summaries, variant, 'e2e_step_time_s'):.3f} | "
            f"{aggregate(summaries, variant, 'response_tokens_per_s'):.1f} | "
            f"{aggregate(summaries, variant, 'samples_per_s'):.3f} | "
            f"{aggregate(summaries, variant, 'gpu_util_pct'):.1f}% | "
            f"{max(float(row['actor_peak_memory_mib']) for row in variant_rows):.0f} | "
            f"{max(float(row['rollout_peak_memory_mib']) for row in variant_rows):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Per-run evidence",
            "",
            "| variant | run | steps | E2E p50/p95 (s) | response tok/s | reward | loss | TIS | samples | errors |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['run_id']} | {row['total_steps']} | "
            f"{row['e2e_step_time_p50_s']:.3f}/{row['e2e_step_time_p95_s']:.3f} | "
            f"{row['response_tokens_per_s']:.1f} | {row['raw_reward']:.4f} | {row['loss']:.6g} | "
            f"{row['tis']:.6f} | {row['observed_samples']} | "
            f"{row['runtime_errors'] + row['unexpected_nonfinite']} |"
        )
    lines.extend(
        [
            "",
            "## Paired changes",
            "",
            "| run | zero-KL throughput | token-budget throughput | total throughput | total latency | seed |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run_id in RUN_IDS:
        baseline = next(row for row in summaries if row["variant"] == "baseline" and row["run_id"] == run_id)
        zero_kl = next(row for row in summaries if row["variant"] == "zero_kl" and row["run_id"] == run_id)
        optimized = next(row for row in summaries if row["variant"] == "optimized" and row["run_id"] == run_id)
        zero_kl_gain = zero_kl["response_tokens_per_s"] / baseline["response_tokens_per_s"] - 1.0
        token_budget_gain = optimized["response_tokens_per_s"] / zero_kl["response_tokens_per_s"] - 1.0
        total_gain = optimized["response_tokens_per_s"] / baseline["response_tokens_per_s"] - 1.0
        run_latency_change = optimized["e2e_step_time_s"] / baseline["e2e_step_time_s"] - 1.0
        lines.append(
            f"| {run_id} | {zero_kl_gain * 100:+.2f}% | {token_budget_gain * 100:+.2f}% | "
            f"{total_gain * 100:+.2f}% | {run_latency_change * 100:+.2f}% | {baseline['seed']} |"
        )
    lines.extend(
        [
            "",
            "## Fixed workload and method",
            "",
            "- Hardware: physical GPUs 2 and 3, both NVIDIA RTX PRO 6000 Blackwell; actor 1 GPU, rollout 1 GPU.",
            f"- Host: {environment['cpu_logical_count']} logical CPUs ({environment['cpu_model']}), "
            f"{int(environment['memory_total_kib']) / 1024**2:.1f} GiB RAM.",
            f"- Runtime: {environment['python']}; Torch {environment['torch']} (CUDA {environment['torch_cuda']}); "
            f"driver {environment['gpu_driver']}; Ray {environment['ray']}; SGLang {environment['sglang']}; "
            f"Transformers {environment['transformers']}.",
            "- Model: `/home/zhengbaowei/model/Qwen3-0.6B`, " + f"model SHA256 `{model_sha}`.",
            "- Data: "
            f"ModelScope `{environment['dataset_repo_id']}` {environment['dataset_split']} "
            f"subset of {environment['dataset_subset_size']} prompts, SHA256 `{dataset_sha}`; "
            "no hand-written large dataset.",
            "- Each component run: 20 steps, 8 prompts/step, 4 samples/prompt, effective batch 32, response cap 512.",
            "- Async policy: Hybrid, max staleness 2, TIS enabled, weight update interval 1.",
            "- Performance window: logged steps 5-15 inclusive (11 observations). Primary throughput uses high-resolution framework `perf/step_time`; the E2E interval from actor completion N-1 to N is secondary because those completion logs have only one-second timestamp resolution.",
            "- GPU window: actor completion 4 through completion 15. Utilization and peak memory use only this window; utilization includes idle-zero samples.",
            "- Trial order uses a three-way rotation to reduce warm-cache and order bias.",
            "",
            "## Optimization rationale",
            "",
            "`baseline` deliberately preserves the original misconfiguration: `--use-kl-loss --kl-loss-coef 0.00` plus `--ref-load`. `zero_kl` removes the unused reference path while retaining the 8192-token dynamic-batch budget. Because the KL term is multiplied by exactly zero, this removes compute but does not change the scalar objective.",
            "",
            "`optimized` builds on `zero_kl` and uses a 12288-token training budget plus a 24576-token forward-only log-prob budget. The role-specific budgets target two backward microbatches and one forward microbatch instead of three each, reducing scheduling and kernel-launch overhead without changing samples, generated-token caps, global batch, optimizer updates, staleness, or weight-publication frequency.",
            "",
            "All variants keep the weight-update buffer fixed at 512 MiB; the previously tested 1 GiB buffer is intentionally excluded from this experiment.",
            "",
            "## Correctness guardrails",
            "",
            f"All {len(summaries)} component jobs completed 20 steps. Unexpected non-finite metrics: "
            f"{sum(int(row['unexpected_nonfinite']) for row in summaries)}; runtime errors: "
            f"{sum(int(row['runtime_errors']) for row in summaries)}.",
            "The existing unknown-device `perf/device_peak_tflops=inf` sentinel is counted separately and is not treated as a training anomaly.",
            "This short experiment establishes runtime equivalence guards, not long-horizon convergence equivalence.",
            "",
            "## Reproduction and rollback",
            "",
            "```bash",
            "cd /home/zhengbaowei/relax_ft/Relax",
            "CUDA_VISIBLE_DEVICES=2,3 TOTAL_TRIALS=3 bash benchmarks/task22_hybrid_async_text/run_paired_trials.sh",
            "```",
            "",
            "Run `TASK22_VARIANT=zero_kl MAX_TOKENS_PER_GPU=8192` to disable only the token-budget optimization, or `TASK22_VARIANT=baseline` to roll back both changes. Raw logs, manifests, submitted commands, and one-second GPU samples are under `benchmark_artifacts/task22-hybrid-async-text-v2/`.",
            "",
            "Generated files: `summary.csv`, `step_metrics.csv`, and `throughput_curves.svg`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    summaries: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for run_id in RUN_IDS:
            summary, step_rows = summarize_run(variant, run_id)
            summaries.append(summary)
            steps.extend(step_rows)

    manifests = {
        (variant, run_id): manifest_values(ARTIFACT_ROOT / variant / f"run-{run_id}" / "manifest.txt")
        for variant in VARIANTS
        for run_id in RUN_IDS
    }
    fixed_workloads = {tuple(manifest[key] for key in WORKLOAD_KEYS) for manifest in manifests.values()}
    if len(fixed_workloads) != 1:
        raise RuntimeError("Runs do not share one fixed model, data, hardware, and workload configuration")
    for run_id in RUN_IDS:
        paired_seeds = {manifests[(variant, run_id)]["seed"] for variant in VARIANTS}
        if len(paired_seeds) != 1:
            raise RuntimeError(f"Paired run {run_id} does not use the same seed")
    if any(row["git_status"] != "clean" for row in summaries):
        raise RuntimeError("Formal runs must use a clean worktree")
    if any(row["runtime_errors"] or row["unexpected_nonfinite"] for row in summaries):
        raise RuntimeError("Runtime errors or unexpected non-finite metrics found")
    expected_reference = {"baseline": True, "zero_kl": False, "optimized": False}
    if any(row["has_reference_forward"] != expected_reference[row["variant"]] for row in summaries):
        raise RuntimeError("Reference-forward execution does not match the three-stage design")
    if any(row["update_weight_buffer_size"] != UPDATE_WEIGHT_BUFFER_SIZE for row in summaries):
        raise RuntimeError("Weight-update buffer size must stay fixed at 512 MiB")
    if any(row["max_tokens_per_gpu"] != TRAIN_TOKEN_BUDGETS[row["variant"]] for row in summaries):
        raise RuntimeError("Training token budgets do not match the three-stage design")
    if any(row["log_probs_max_tokens_per_gpu"] != LOG_PROB_TOKEN_BUDGETS[row["variant"]] for row in summaries):
        raise RuntimeError("Dynamic-batch token budgets do not match the three-stage design")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "summary.csv", summaries)
    write_csv(OUTPUT_ROOT / "step_metrics.csv", steps)
    svg_chart(steps, OUTPUT_ROOT / "throughput_curves.svg")
    write_report(summaries, OUTPUT_ROOT / "report.md")
    print(f"Wrote Task 22 report to {OUTPUT_ROOT / 'report.md'}")


if __name__ == "__main__":
    main()
