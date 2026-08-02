#!/usr/bin/env python3
"""Analyze Task 22 paired Hybrid-async benchmark artifacts."""

from __future__ import annotations

import ast
import csv
import hashlib
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
    os.environ.get("TASK22_ARTIFACT_ROOT", REPO_ROOT / "benchmark_artifacts" / "task22-hybrid-async-text-v3")
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "TASK22_OUTPUT_ROOT",
        REPO_ROOT / "benchmark_artifacts" / "task22-pr-attachments" / "task22-hybrid-async-text",
    )
)
VARIANTS = ("baseline", "zero_kl", "optimized")
RUN_IDS = (1, 2, 3)
STABLE_FIRST_STEP = 5
STABLE_LAST_STEP = 15
GLOBAL_BATCH_SIZE = 32
TOTAL_STEPS = 20
UPDATE_WEIGHT_BUFFER_SIZE = 512 * 1024**2
TRAIN_TOKEN_BUDGETS = {variant: 8192 for variant in VARIANTS}
LOG_PROB_TOKEN_BUDGETS = {variant: 8192 for variant in VARIANTS}
UPDATE_WEIGHTS_INTERVALS = {"baseline": 1, "zero_kl": 1, "optimized": 2}
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


def weight_publication_count(path: Path) -> int:
    return path.read_text(errors="replace").count("before update_weights")


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
        "update_weights_interval": int(manifest["update_weights_interval"]),
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
        "weight_publications": weight_publication_count(log_path),
        "dynamic_microbatches_mean": mean(microbatch_counts),
        **gpu,
    }
    return summary, step_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_window_svg_chart(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    metric_key: str,
    title: str,
    y_axis_label: str,
    y_interval: float,
    y_decimals: int,
) -> None:
    width, height = 960, 460
    plot_left, plot_right = 88, 930
    plot_top, plot_bottom = 70, 375
    values = [float(row[metric_key]) for row in rows]
    y_min = math.floor(min(values) / y_interval) * y_interval
    y_max = math.ceil(max(values) / y_interval) * y_interval
    if y_min == y_max:
        y_max += y_interval
    colors = {"baseline": "#4b5563", "zero_kl": "#2563eb", "optimized": "#0f766e"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title">',
        f'<title id="chart-title">{title}</title>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="480" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
    ]
    tick_count = int(round((y_max - y_min) / y_interval))
    for tick_index in range(tick_count + 1):
        tick_value = y_min + tick_index * y_interval
        y = plot_bottom - (tick_value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)
        elements.extend(
            [
                f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>',
                f'<text x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick_value:.{y_decimals}f}</text>',
            ]
        )

    reference_rows = [row for row in rows if row["variant"] == VARIANTS[0]]
    denominator = max(len(reference_rows) - 1, 1)
    for index, row in enumerate(reference_rows):
        if int(row["step"]) not in (STABLE_FIRST_STEP, 10, STABLE_LAST_STEP):
            continue
        x = plot_left + index * (plot_right - plot_left) / denominator
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 5}" stroke="#111827"/>',
                f'<text x="{x:.1f}" y="{plot_bottom + 20}" text-anchor="middle" font-family="sans-serif" font-size="10">R{row["run_id"]}/S{row["step"]}</text>',
            ]
        )
    stable_steps_per_run = STABLE_LAST_STEP - STABLE_FIRST_STEP + 1
    for boundary in range(1, len(RUN_IDS)):
        boundary_index = boundary * stable_steps_per_run - 0.5
        x = plot_left + boundary_index * (plot_right - plot_left) / denominator
        elements.append(
            f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 4"/>'
        )

    elements.extend(
        [
            f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#111827" stroke-width="1.5"/>',
            f'<text x="{(plot_left + plot_right) / 2:.1f}" y="438" text-anchor="middle" font-family="sans-serif" font-size="12">配对运行 / 训练 step</text>',
            f'<text x="20" y="{(plot_top + plot_bottom) / 2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" transform="rotate(-90 20 {(plot_top + plot_bottom) / 2:.1f})">{y_axis_label}</text>',
        ]
    )
    for variant in VARIANTS:
        points = []
        variant_rows = [row for row in rows if row["variant"] == variant]
        for index, row in enumerate(variant_rows):
            x = plot_left + index * (plot_right - plot_left) / max(len(variant_rows) - 1, 1)
            value = float(row[metric_key])
            y = plot_bottom - (value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)
            points.append(f"{x:.1f},{y:.1f}")
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[variant]}" stroke-width="2"/>'
        )
    elements.extend(
        [
            '<line x1="590" y1="50" x2="612" y2="50" stroke="#4b5563" stroke-width="2"/><text x="620" y="54" font-family="sans-serif" font-size="12">baseline</text>',
            '<line x1="700" y1="50" x2="722" y2="50" stroke="#2563eb" stroke-width="2"/><text x="730" y="54" font-family="sans-serif" font-size="12">zero_kl</text>',
            '<line x1="800" y1="50" x2="822" y2="50" stroke="#0f766e" stroke-width="2"/><text x="830" y="54" font-family="sans-serif" font-size="12">optimized</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements) + "\n")


def svg_chart(rows: list[dict[str, Any]], path: Path) -> None:
    stable_window_svg_chart(
        rows,
        path,
        metric_key="response_tokens_per_s",
        title="Task 22 稳定窗口响应吞吐",
        y_axis_label="响应吞吐（tokens/s）",
        y_interval=500,
        y_decimals=0,
    )


def aggregate(rows: list[dict[str, Any]], variant: str, key: str) -> float:
    return mean([float(row[key]) for row in rows if row["variant"] == variant])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(summaries: list[dict[str, Any]], path: Path) -> None:
    baseline_tps = aggregate(summaries, "baseline", "response_tokens_per_s")
    zero_kl_tps = aggregate(summaries, "zero_kl", "response_tokens_per_s")
    optimized_tps = aggregate(summaries, "optimized", "response_tokens_per_s")
    baseline_step = aggregate(summaries, "baseline", "e2e_step_time_s")
    zero_kl_step = aggregate(summaries, "zero_kl", "e2e_step_time_s")
    optimized_step = aggregate(summaries, "optimized", "e2e_step_time_s")
    zero_kl_throughput_gain = (zero_kl_tps / baseline_tps - 1.0) * 100.0
    interval_throughput_gain = (optimized_tps / zero_kl_tps - 1.0) * 100.0
    total_throughput_gain = (optimized_tps / baseline_tps - 1.0) * 100.0
    zero_kl_latency_change = (zero_kl_step / baseline_step - 1.0) * 100.0
    interval_latency_change = (optimized_step / zero_kl_step - 1.0) * 100.0
    total_latency_change = (optimized_step / baseline_step - 1.0) * 100.0
    acceptance = "通过" if interval_throughput_gain >= 5.0 else "未达到"
    commit = summaries[0]["git_commit"]
    dataset_sha = summaries[0]["dataset_sha256"]
    model_sha = summaries[0]["model_sha256"]
    environment = manifest_values(ARTIFACT_ROOT / "baseline" / "run-1" / "manifest.txt")
    tis_values = [float(row["tis"]) for row in summaries]
    max_tis_clipfrac = max(float(row["tis_clipfrac"]) for row in summaries)
    baseline_actor_peak = max(float(row["actor_peak_memory_mib"]) for row in summaries if row["variant"] == "baseline")
    optimized_actor_peak = max(
        float(row["actor_peak_memory_mib"]) for row in summaries if row["variant"] == "optimized"
    )
    actor_peak_delta = optimized_actor_peak - baseline_actor_peak
    actor_peak_delta_pct = (optimized_actor_peak / baseline_actor_peak - 1.0) * 100.0
    evidence_archive = path.parent / "raw-evidence.tar.gz"
    evidence_index = path.parent / "raw-evidence-index.csv"
    evidence_checksum = path.parent / "raw-evidence.sha256"
    if evidence_archive.is_file() and evidence_index.is_file() and evidence_checksum.is_file():
        evidence_lines = [
            "- 脱敏原始证据包：[raw-evidence.tar.gz](raw-evidence.tar.gz)",
            "- 文件级索引：[raw-evidence-index.csv](raw-evidence-index.csv)",
            "- 压缩包校验：[raw-evidence.sha256](raw-evidence.sha256)",
            f"- 当前压缩包 SHA-256：`{sha256_file(evidence_archive)}`。",
        ]
    else:
        evidence_lines = [
            "- **阻塞：当前 checkout 尚未包含脱敏原始证据包或长期可访问链接。**",
            "- `benchmark_artifacts/` 被 `.gitignore` 排除，不能把本地目录位置当成交付证据。",
            "- 在提交 PR 前，必须从原实验机生成并提交上述三个文件，或在此处填写长期可访问链接及 SHA-256。",
        ]
    lines = [
        "# Task 22 Hybrid-async 纯文本性能报告",
        "",
        "## 结果",
        "",
        f"在实验提交 `{commit}` 上完成三组配对实验。移除系数为零的 KL reference 路径后，"
        f"响应吞吐变化 {zero_kl_throughput_gain:+.2f}%，端到端延迟变化 "
        f"{zero_kl_latency_change:+.2f}%；随后每两次 Actor 更新发布一次 Rollout 权重，吞吐进一步变化 "
        f"{interval_throughput_gain:+.2f}%，延迟进一步变化 {interval_latency_change:+.2f}%。"
        f"相对 baseline，组合改动使吞吐变化 {total_throughput_gain:+.2f}%，延迟变化 "
        f"{total_latency_change:+.2f}%。",
        f"验收：**{acceptance}**。冻结目标为 `optimized` 相对 `zero_kl` 的响应吞吐至少提升 5%。",
        "",
        "| 变体 | 权重发布间隔 | framework/E2E step (s) | 响应 tok/s | samples/s | 发布耗时/step (s) | TIS | 整体 GPU 利用率 | Actor GPU 利用率 | Rollout GPU 利用率 | Actor 峰值 MiB | Rollout 峰值 MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        variant_rows = [row for row in summaries if row["variant"] == variant]
        lines.append(
            f"| {variant} | {UPDATE_WEIGHTS_INTERVALS[variant]} | "
            f"{aggregate(summaries, variant, 'framework_step_time_s'):.3f}/"
            f"{aggregate(summaries, variant, 'e2e_step_time_s'):.3f} | "
            f"{aggregate(summaries, variant, 'response_tokens_per_s'):.1f} | "
            f"{aggregate(summaries, variant, 'samples_per_s'):.3f} | "
            f"{aggregate(summaries, variant, 'preceding_update_weights_s'):.3f} | "
            f"{aggregate(summaries, variant, 'tis'):.6f} | "
            f"{aggregate(summaries, variant, 'gpu_util_pct'):.2f}% | "
            f"{aggregate(summaries, variant, 'actor_gpu_util_pct'):.2f}% | "
            f"{aggregate(summaries, variant, 'rollout_gpu_util_pct'):.2f}% | "
            f"{max(float(row['actor_peak_memory_mib']) for row in variant_rows):.0f} | "
            f"{max(float(row['rollout_peak_memory_mib']) for row in variant_rows):.0f} |"
        )
    lines.extend(
        [
            "",
            "GPU 利用率来自稳定窗口内的一秒采样，包含空闲的 0% 样本。`optimized` 的整体/Actor/Rollout "
            f"平均利用率分别为 {aggregate(summaries, 'optimized', 'gpu_util_pct'):.2f}% / "
            f"{aggregate(summaries, 'optimized', 'actor_gpu_util_pct'):.2f}% / "
            f"{aggregate(summaries, 'optimized', 'rollout_gpu_util_pct'):.2f}%；`zero_kl` 分别为 "
            f"{aggregate(summaries, 'zero_kl', 'gpu_util_pct'):.2f}% / "
            f"{aggregate(summaries, 'zero_kl', 'actor_gpu_util_pct'):.2f}% / "
            f"{aggregate(summaries, 'zero_kl', 'rollout_gpu_util_pct'):.2f}%。",
            "",
            f"Actor 峰值显存从 baseline 的 {baseline_actor_peak:.0f} MiB "
            f"（{baseline_actor_peak / 1024:.2f} GiB）增至 optimized 的 {optimized_actor_peak:.0f} MiB "
            f"（{optimized_actor_peak / 1024:.2f} GiB），增加 {actor_peak_delta:.0f} MiB "
            f"（{actor_peak_delta_pct:.2f}%）；Rollout 峰值基本不变。当前采样只记录设备已用显存，"
            "没有区分 PyTorch allocated/reserved 或张量生命周期，因此不能把增量归因于某个单独张量。"
            "它与权重发布间隔改变后的流水重叠及缓存高水位同时出现，应作为性能收益的显存代价记录；"
            "在取得 allocator/timeline 原始证据前不作更强因果结论。",
            "",
            "## 单次运行证据",
            "",
            "| 变体 | run | steps | E2E p50/p95 (s) | 响应 tok/s | reward | loss | TIS | samples | errors |",
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
            "## 配对变化",
            "",
            "| run | zero-KL 吞吐变化 | interval-two 吞吐变化 | 总吞吐变化 | 总延迟变化 | seed |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run_id in RUN_IDS:
        baseline = next(row for row in summaries if row["variant"] == "baseline" and row["run_id"] == run_id)
        zero_kl = next(row for row in summaries if row["variant"] == "zero_kl" and row["run_id"] == run_id)
        optimized = next(row for row in summaries if row["variant"] == "optimized" and row["run_id"] == run_id)
        zero_kl_gain = zero_kl["response_tokens_per_s"] / baseline["response_tokens_per_s"] - 1.0
        interval_gain = optimized["response_tokens_per_s"] / zero_kl["response_tokens_per_s"] - 1.0
        total_gain = optimized["response_tokens_per_s"] / baseline["response_tokens_per_s"] - 1.0
        run_latency_change = optimized["e2e_step_time_s"] / baseline["e2e_step_time_s"] - 1.0
        lines.append(
            f"| {run_id} | {zero_kl_gain * 100:+.2f}% | {interval_gain * 100:+.2f}% | "
            f"{total_gain * 100:+.2f}% | {run_latency_change * 100:+.2f}% | {baseline['seed']} |"
        )
    lines.extend(
        [
            "",
            "## 性能曲线",
            "",
            "![三组配对实验稳定窗口响应吞吐](throughput_curves.svg)",
            "",
            "![三组配对实验稳定窗口 framework step 耗时](step_time_curves.svg)",
            "",
            "![三组配对实验稳定窗口前序权重发布耗时](weight_update_curves.svg)",
            "",
            "## 固定工作量与方法",
            "",
            "- 硬件：物理 GPU 2 和 3，均为 NVIDIA RTX PRO 6000 Blackwell；Actor 1 卡、Rollout 1 卡。",
            f"- 主机：{environment['cpu_logical_count']} 个逻辑 CPU（{environment['cpu_model']}），"
            f"内存 {int(environment['memory_total_kib']) / 1024**2:.1f} GiB。",
            f"- 运行时：{environment['python']}；Torch {environment['torch']}（CUDA "
            f"{environment['torch_cuda']}）；driver {environment['gpu_driver']}；Ray {environment['ray']}；"
            f"SGLang {environment['sglang']}；Transformers {environment['transformers']}。",
            f"- 模型：Qwen3-0.6B；模型 SHA-256 `{model_sha}`。本地绝对路径不进入交付报告。",
            "- 数据："
            f"ModelScope `{environment['dataset_repo_id']}` {environment['dataset_split']} "
            f"中固定 {environment['dataset_subset_size']} 个 prompt，SHA-256 `{dataset_sha}`；没有手写大数据集。",
            "- 每个组件运行 20 steps；每 step 8 prompts × 4 samples，有效 batch 32，response cap 512。",
            "- 异步策略：Hybrid、max staleness 2、启用 TIS。baseline/zero_kl 每 step 发布；optimized 每两 step 发布。",
            "- 动态 batch：所有变体每卡训练/log-prob token budget 均为 8192/8192。",
            "- 性能窗口：记录的 step 5-15（共 11 个观测）。主要吞吐使用高精度 `perf/step_time`；"
            "Actor completion N-1 到 N 的 E2E 间隔只有一秒时间戳精度，因此作为次要指标。",
            "- GPU 窗口：Actor completion 4 至 15；利用率和峰值显存只使用该窗口，且包含空闲 0% 样本。",
            "- 三组实验采用轮换顺序，降低 warm-cache 和运行顺序偏差。",
            "",
            "## 优化原理",
            "",
            "`baseline` 保留原始配置：`--use-kl-loss --kl-loss-coef 0.00` 加 `--ref-load`。"
            "`zero_kl` 移除未使用的 reference 路径，同时保留 8192-token 动态 batch 预算。"
            "KL 项严格乘以零，因此该改动移除计算但不改变标量目标。",
            "",
            "`optimized` 基于 `zero_kl` 设置 `--update-weights-interval 2`。Hybrid 在奇数完成 step 跳过 "
            "Rollout pause、权重传输和 resume endpoint，同时仍在间隔边界、最后一步以及评测前强制发布。",
            "",
            "该方法有意用额外一次 Actor update 的 Rollout policy freshness 换取更低的发布开销。"
            "`--max-staleness 2` 约束现有异步流水；TIS、loss、reward 和 clipping 指标是正确性护栏，"
            "不能证明长期收敛等价。",
            "",
            f"日志记录每个 zero_kl 作业平均发布 {aggregate(summaries, 'zero_kl', 'weight_publications'):.0f} "
            f"次权重，每个 optimized 作业发布 {aggregate(summaries, 'optimized', 'weight_publications'):.0f} 次；"
            "两者均包含一次共同的初始化发布。",
            "",
            "所有变体的 weight-update buffer 固定为 512 MiB；先前测试的 1 GiB buffer 不进入本实验。",
            "",
            "## 未采用的方向",
            "",
            "- weight-update buffer 从 512 MiB 增至 1 GiB，三组实验吞吐只提升 +1.12%。",
            "- train/log-prob budget 增至 12288/24576 后，相对 `zero_kl` 吞吐下降 -1.52%；"
            "log-prob forward 加快，但 Actor training 的回退抵消了收益。",
            "",
            "## 正确性护栏",
            "",
            f"全部 {len(summaries)} 个组件作业均完成 20 steps；意外非有限指标为 "
            f"{sum(int(row['unexpected_nonfinite']) for row in summaries)}，运行时错误为 "
            f"{sum(int(row['runtime_errors']) for row in summaries)}。",
            f"每个作业生成 {TOTAL_STEPS * GLOBAL_BATCH_SIZE} 个样本。单次运行平均 TIS 范围为 "
            f"{min(tis_values):.6f}-{max(tis_values):.6f}，最大平均 TIS clip fraction 为 "
            f"{max_tis_clipfrac:.6f}。",
            "已有的未知设备 `perf/device_peak_tflops=inf` sentinel 单独计数，不视为训练异常。",
            "该短实验只建立运行时等价护栏，不证明长期收敛等价。",
            "",
            "## 复现与回退",
            "",
            "```bash",
            "cd /path/to/Relax",
            "MODEL_PATH=/path/to/Qwen3-0.6B CUDA_VISIBLE_DEVICES=2,3 TOTAL_TRIALS=3 \\",
            "  bash benchmarks/task22_hybrid_async_text/run_paired_trials.sh",
            "```",
            "",
            "设置 `TASK22_VARIANT=zero_kl UPDATE_WEIGHTS_INTERVAL=1` 可只关闭 interval-two 发布；"
            "设置 `TASK22_VARIANT=baseline` 可回退两个改动。",
            "",
            "生成文件：`summary.csv`、`step_metrics.csv`、`throughput_curves.svg`、"
            "`step_time_curves.svg` 和 `weight_update_curves.svg`。",
            "这些结果文件位于被忽略的附件目录，不应提交到仓库。上传 Draft PR 后，需在 PR 和 Issue "
            "中记录附件 URL 与 SHA-256。",
            "",
            "## 原始证据交付",
            "",
        ]
    )
    lines.extend(evidence_lines)
    lines.extend(
        [
            "",
            "拿到实验机上的 `benchmark_artifacts/task22-hybrid-async-text-v3/` 后，执行：",
            "",
            "```bash",
            "python benchmarks/task22_hybrid_async_text/package_evidence.py \\",
            "  --artifact-root benchmark_artifacts/task22-hybrid-async-text-v3 \\",
            "  --output-dir benchmark_artifacts/task22-pr-attachments/task22-hybrid-async-text",
            "python benchmarks/task22_hybrid_async_text/analyze.py",
            "```",
            "",
            "打包器要求 3 个变体 × 3 次运行全部具备 `manifest.txt`、`submit.log`、`train.log` 和 "
            "`gpu.csv`；它会脱敏 home path、私网 IP、邮箱、URL credentials 和 secret-like assignments，"
            "并同时记录原文件与交付文件的 SHA-256。",
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
    if any(row["total_steps"] != TOTAL_STEPS for row in summaries):
        raise RuntimeError(f"Every component run must contain exactly {TOTAL_STEPS} steps")
    if any(row["total_samples"] != TOTAL_STEPS * GLOBAL_BATCH_SIZE for row in summaries):
        raise RuntimeError("Component runs contain missing or extra samples")
    expected_reference = {"baseline": True, "zero_kl": False, "optimized": False}
    if any(row["has_reference_forward"] != expected_reference[row["variant"]] for row in summaries):
        raise RuntimeError("Reference-forward execution does not match the three-stage design")
    if any(row["update_weight_buffer_size"] != UPDATE_WEIGHT_BUFFER_SIZE for row in summaries):
        raise RuntimeError("Weight-update buffer size must stay fixed at 512 MiB")
    if any(row["update_weights_interval"] != UPDATE_WEIGHTS_INTERVALS[row["variant"]] for row in summaries):
        raise RuntimeError("Weight-update intervals do not match the three-stage design")
    if any(
        row["weight_publications"] != 1 + math.ceil(TOTAL_STEPS / UPDATE_WEIGHTS_INTERVALS[row["variant"]])
        for row in summaries
    ):
        raise RuntimeError("Observed weight-publication counts do not match configured intervals")
    if any(row["max_tokens_per_gpu"] != TRAIN_TOKEN_BUDGETS[row["variant"]] for row in summaries):
        raise RuntimeError("Training token budgets do not match the three-stage design")
    if any(row["log_probs_max_tokens_per_gpu"] != LOG_PROB_TOKEN_BUDGETS[row["variant"]] for row in summaries):
        raise RuntimeError("Dynamic-batch token budgets do not match the three-stage design")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "summary.csv", summaries)
    write_csv(OUTPUT_ROOT / "step_metrics.csv", steps)
    svg_chart(steps, OUTPUT_ROOT / "throughput_curves.svg")
    stable_window_svg_chart(
        steps,
        OUTPUT_ROOT / "step_time_curves.svg",
        metric_key="framework_step_time_s",
        title="Task 22 稳定窗口 framework step 耗时",
        y_axis_label="framework step 耗时（s）",
        y_interval=0.5,
        y_decimals=1,
    )
    stable_window_svg_chart(
        steps,
        OUTPUT_ROOT / "weight_update_curves.svg",
        metric_key="preceding_update_weights_s",
        title="Task 22 稳定窗口前序权重发布耗时",
        y_axis_label="前序权重发布耗时（s）",
        y_interval=0.1,
        y_decimals=1,
    )
    write_report(summaries, OUTPUT_ROOT / "report.md")
    print(f"Wrote Task 22 report to {OUTPUT_ROOT / 'report.md'}")


if __name__ == "__main__":
    main()
