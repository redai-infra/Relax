# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build auditable Qwen3.5-4B GSM8K GRPO/Dr.GRPO artifacts."""

import argparse
import ast
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


RUNS = {"grpo": "GRPO", "dr_grpo": "Dr.GRPO"}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRICS_PATTERN = re.compile(r"(?:step|rollout) (\d+): (\{.*\})")
LOG_FIELDS = (
    "rollout/raw_reward",
    "train/loss",
    "train/pg_loss",
    "train/entropy_loss",
    "train/pg_clipfrac",
    "train/ppo_kl",
    "train/kl_loss",
    "train/train_rollout_logprob_abs_diff",
    "train/train_rollout_prob_abs_diff",
    "train/grad_norm",
    "train/lr-pg_0",
)
OUTCOME_FIELDS = (
    "rollout/response_lengths/correct",
    "rollout/response_lengths/incorrect",
    "rollout/response_counts/correct",
    "rollout/response_counts/incorrect",
    "rollout/truncated_counts/correct",
    "rollout/truncated_counts/incorrect",
    "train/loss_mask_tokens",
    "train/fully_masked_responses",
    "train/reference_kl",
)
FIELDS = (*LOG_FIELDS, *OUTCOME_FIELDS)


def find_training_log(run_dir: Path) -> Path:
    """Return the complete Ray job log, with a recipe-log fallback."""
    ray_job_log = run_dir / "ray_job_final.log"
    if ray_job_log.is_file():
        return ray_job_log

    logs = sorted((run_dir / "log").glob("*.log"))
    if len(logs) != 1:
        raise ValueError(f"Expected ray_job_final.log or exactly one recipe log in {run_dir}, found {len(logs)}")
    return logs[0]


def parse_log(log_path: Path) -> dict[int, dict[str, float]]:
    """Read selected training and rollout metrics keyed by optimizer step."""
    records: dict[int, dict[str, float]] = {}
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            match = METRICS_PATTERN.search(ANSI_ESCAPE.sub("", raw_line))
            if match is None:
                continue
            step = int(match.group(1))
            metrics = ast.literal_eval(match.group(2))
            records.setdefault(step, {}).update(
                {field: float(metrics[field]) for field in LOG_FIELDS if field in metrics}
            )
    return records


def is_correct(reward: Any) -> bool:
    """Return the task correctness encoded by Relax's reward record."""
    if isinstance(reward, dict):
        if "acc" in reward:
            return bool(reward["acc"])
        reward = reward.get("score", 0)
    return float(reward) == 1.0


def add_exact_loss_mask_tokens(
    run_dir: Path,
    records: dict[int, dict[str, float]],
    num_steps: int,
    world_size: int,
    model_parallel_size: int,
    global_batch_size: int,
) -> None:
    """Recover exact global ``T`` from unmodified per-rank train dumps."""
    import torch

    if world_size <= 0 or model_parallel_size <= 0 or world_size % model_parallel_size:
        raise ValueError(f"Invalid topology: world_size={world_size}, model_parallel_size={model_parallel_size}")

    train_data_dir = run_dir / "train_data"
    all_paths = sorted(train_data_dir.glob("*_*.pt"))
    if len(all_paths) != num_steps * world_size:
        raise ValueError(
            f"Expected {num_steps * world_size} total train dumps in {train_data_dir}, found {len(all_paths)}"
        )
    for step in range(num_steps):
        paths = sorted(train_data_dir.glob(f"{step}_*.pt"))
        if len(paths) != world_size:
            raise ValueError(f"Expected {world_size} train dumps for step {step}, found {len(paths)}")

        ranks = set()
        replicated_samples = 0
        replicated_tokens = 0
        replicated_fully_masked_responses = 0
        replica_signatures = Counter()
        for path in paths:
            dump = torch.load(path, map_location="cpu", weights_only=False)
            if dump.get("rollout_id") != step:
                raise ValueError(f"Unexpected rollout_id in {path}: {dump.get('rollout_id')}")
            rank = int(dump["rank"])
            if rank in ranks:
                raise ValueError(f"Duplicate rank {rank} for step {step}")
            ranks.add(rank)

            samples = dump.get("samples")
            if not isinstance(samples, list):
                raise ValueError(f"Missing samples list in {path}")
            replicated_samples += len(samples)
            rank_signature = []
            for sample in samples:
                loss_mask = sample.get("loss_masks")
                if loss_mask is None:
                    raise ValueError(f"Missing loss_masks in {path}, sample {sample.get('sample_index')}")
                sample_mask = []
                for value in loss_mask:
                    if value not in (0, 1, False, True):
                        raise ValueError(f"Non-binary loss mask value in {path}: {value!r}")
                    mask_value = int(value)
                    sample_mask.append(mask_value)
                    replicated_tokens += mask_value
                replicated_fully_masked_responses += not any(sample_mask)
                rank_signature.append(tuple(sample_mask))
            replica_signatures[tuple(rank_signature)] += 1

        if ranks != set(range(world_size)):
            raise ValueError(f"Unexpected ranks for step {step}: {sorted(ranks)}")
        if any(replica_count % model_parallel_size for replica_count in replica_signatures.values()):
            raise ValueError(f"Model-parallel rank dumps disagree for step {step}")
        if (
            replicated_samples % model_parallel_size
            or replicated_tokens % model_parallel_size
            or replicated_fully_masked_responses % model_parallel_size
        ):
            raise ValueError(f"Model-parallel replicas disagree for step {step}")

        sample_count = replicated_samples // model_parallel_size
        if sample_count != global_batch_size:
            raise ValueError(f"Expected {global_batch_size} samples for step {step}, found {sample_count}")
        records.setdefault(step, {}).update(
            {
                "train/loss_mask_tokens": replicated_tokens // model_parallel_size,
                "train/fully_masked_responses": replicated_fully_masked_responses // model_parallel_size,
            }
        )


def add_outcome_lengths(
    run: str,
    run_dir: Path,
    records: dict[int, dict[str, float]],
    num_steps: int,
    response_budget: int,
    global_batch_size: int,
) -> None:
    """Add per-step correct/incorrect response lengths from rollout JSONL."""
    result_dir = run_dir / "rollout_result" / "train"
    actual_filenames = {path.name for path in result_dir.glob("*.jsonl")}
    expected_filenames = {f"{step}.jsonl" for step in range(num_steps)}
    if actual_filenames != expected_filenames:
        missing = sorted(expected_filenames - actual_filenames)
        extra = sorted(actual_filenames - expected_filenames)
        raise ValueError(f"Unexpected rollout result files: missing={missing}, extra={extra}")
    for step in range(num_steps):
        path = result_dir / f"{step}.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing rollout result for step {step}: {path}")

        lengths = {True: [], False: []}
        truncated_counts = {True: 0, False: 0}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                sample = json.loads(line)
                correct = is_correct(sample["reward"])
                lengths[correct].append(float(sample["response_length"]))
                truncated_counts[correct] += sample["status"] == "truncated"

        if len(lengths[True]) + len(lengths[False]) != global_batch_size:
            raise ValueError(f"Expected {global_batch_size} rollout samples in {path}")
        # Dr.GRPO logs sum(KL) / (N * B). CP1 GRPO preserves the historical
        # per-response floor and logs sum(KL) / (T + Z), where Z is the number
        # of fully masked responses. Undo the applicable denominator here so
        # both runs report strict sum(KL) / T.
        loss_mask_tokens = records[step]["train/loss_mask_tokens"]
        logged_kl = records[step]["train/kl_loss"]
        if loss_mask_tokens == 0:
            reference_kl = math.nan
        elif run == "dr_grpo":
            reference_kl = logged_kl * (global_batch_size * response_budget) / loss_mask_tokens
        else:
            legacy_denominator = loss_mask_tokens + records[step]["train/fully_masked_responses"]
            reference_kl = logged_kl * legacy_denominator / loss_mask_tokens

        records.setdefault(step, {}).update(
            {
                "train/reference_kl": reference_kl,
                "rollout/response_lengths/correct": (statistics.fmean(lengths[True]) if lengths[True] else math.nan),
                "rollout/response_lengths/incorrect": (
                    statistics.fmean(lengths[False]) if lengths[False] else math.nan
                ),
                "rollout/response_counts/correct": len(lengths[True]),
                "rollout/response_counts/incorrect": len(lengths[False]),
                "rollout/truncated_counts/correct": truncated_counts[True],
                "rollout/truncated_counts/incorrect": truncated_counts[False],
            }
        )


def validate_records(run: str, records: dict[int, dict[str, float]], num_steps: int) -> None:
    """Validate that every requested metric is present and finite when
    defined."""
    expected_steps = set(range(num_steps))
    if set(records) != expected_steps:
        missing = sorted(expected_steps - set(records))
        extra = sorted(set(records) - expected_steps)
        raise ValueError(f"Unexpected steps for {run}: missing={missing}, extra={extra}")
    for step, metrics in records.items():
        missing_fields = [field for field in FIELDS if field not in metrics]
        if missing_fields:
            raise ValueError(f"Missing fields for {run} step {step}: {missing_fields}")
        for field, value in metrics.items():
            if field in OUTCOME_FIELDS and field.startswith("rollout/response_lengths/") and math.isnan(value):
                continue
            if field == "train/reference_kl" and metrics["train/loss_mask_tokens"] == 0 and math.isnan(value):
                continue
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {field} for {run} step {step}: {value}")
        loss_mask_tokens = metrics["train/loss_mask_tokens"]
        if not isinstance(loss_mask_tokens, int) or not 0 <= loss_mask_tokens:
            raise ValueError(f"Invalid train/loss_mask_tokens for {run} step {step}: {loss_mask_tokens}")
        fully_masked_responses = metrics["train/fully_masked_responses"]
        if (
            not isinstance(fully_masked_responses, int)
            or not 0
            <= fully_masked_responses
            <= metrics["rollout/response_counts/correct"] + metrics["rollout/response_counts/incorrect"]
        ):
            raise ValueError(f"Invalid train/fully_masked_responses for {run} step {step}: {fully_masked_responses}")


def write_csv(path: Path, all_records: dict[str, dict[int, dict[str, float]]]) -> None:
    """Write one row per algorithm and optimizer step."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run", "algorithm", "step", *FIELDS))
        writer.writeheader()
        for run, records in all_records.items():
            for step in sorted(records):
                writer.writerow({"run": run, "algorithm": RUNS[run], "step": step, **records[step]})


def write_summary(path: Path, all_records: dict[str, dict[int, dict[str, float]]]) -> None:
    """Write plot metadata and full-run pooled outcome statistics."""
    summary: dict[str, Any] = {
        "kl_note": (
            "train/kl_loss is Relax's logged algorithm-normalized reference-KL component; "
            "train/reference_kl reconstructs sum(KL)/T using exact loss masks and the applicable training "
            "denominator; it is unavailable when T=0."
        ),
        "loss_mask_tokens_source": (
            "sum(loss_masks) from every actor-rank --save-debug-train-data dump, divided by TP*PP*CP replicas"
        ),
        "runs": {},
    }
    for run, records in all_records.items():
        correct_count = sum(records[step]["rollout/response_counts/correct"] for step in records)
        incorrect_count = sum(records[step]["rollout/response_counts/incorrect"] for step in records)
        summary["runs"][run] = {
            "algorithm": RUNS[run],
            "steps": len(records),
            "correct_count": int(correct_count),
            "incorrect_count": int(incorrect_count),
            "accuracy": correct_count / (correct_count + incorrect_count),
            "mean_raw_reward": statistics.fmean(record["rollout/raw_reward"] for record in records.values()),
            "loss_mask_tokens": int(sum(record["train/loss_mask_tokens"] for record in records.values())),
            "fully_masked_responses": int(sum(record["train/fully_masked_responses"] for record in records.values())),
            "zero_token_windows": sum(record["train/loss_mask_tokens"] == 0 for record in records.values()),
            "mean_grad_norm": statistics.fmean(record["train/grad_norm"] for record in records.values()),
            "max_grad_norm": max(record["train/grad_norm"] for record in records.values()),
        }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_tables(path: Path, all_records: dict[str, dict[int, dict[str, float]]]) -> None:
    """Write the report table from the same validated per-step records."""
    lines = [
        "| Algorithm | Reward (all) | Reward (last 20) | Accuracy | Exact loss-mask tokens | Grad norm (mean / max) |",  # noqa: E501
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run, records in all_records.items():
        ordered = [records[step] for step in sorted(records)]
        correct = sum(record["rollout/response_counts/correct"] for record in ordered)
        total = correct + sum(record["rollout/response_counts/incorrect"] for record in ordered)
        lines.append(
            "| {algorithm} | {reward:.6f} | {last20:.6f} | {accuracy:.6f} | {tokens} | "
            "{grad_mean:.6f} / {grad_max:.6f} |".format(
                algorithm=RUNS[run],
                reward=statistics.fmean(record["rollout/raw_reward"] for record in ordered),
                last20=statistics.fmean(record["rollout/raw_reward"] for record in ordered[-20:]),
                accuracy=correct / total,
                tokens=sum(record["train/loss_mask_tokens"] for record in ordered),
                grad_mean=statistics.fmean(record["train/grad_norm"] for record in ordered),
                grad_max=max(record["train/grad_norm"] for record in ordered),
            )
        )
    lines.extend(
        [
            "",
            "`Exact loss-mask tokens` is reconstructed from the unmodified per-rank train dumps, not from response length.",  # noqa: E501
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def pooled_outcome_lengths(records: dict[int, dict[str, float]], outcome: str, window_size: int) -> list[float]:
    """Return trailing-window lengths pooled across samples of one outcome."""
    means = f"rollout/response_lengths/{outcome}"
    counts = f"rollout/response_counts/{outcome}"
    values = []
    for step in sorted(records):
        if step < window_size - 1:
            values.append(math.nan)
            continue
        window = range(max(0, step - window_size + 1), step + 1)
        count = sum(records[index][counts] for index in window)
        if count == 0:
            values.append(math.nan)
            continue
        length_sum = sum(records[index][means] * records[index][counts] for index in window if records[index][counts])
        values.append(length_sum / count)
    return values


def pooled_reference_kl(records: dict[int, dict[str, float]], window_size: int) -> list[float]:
    """Trailing-window reference KL pooled by token count.

    A plain moving average of a per-token ratio would weight sparse steps as
    heavily as dense ones; pooling sum(KL) over sum(T) keeps the token-mean
    definition intact inside the window.
    """
    values = []
    for step in sorted(records):
        if step < window_size - 1:
            values.append(math.nan)
            continue
        window = range(max(0, step - window_size + 1), step + 1)
        tokens = sum(records[index]["train/loss_mask_tokens"] for index in window)
        if tokens == 0:
            values.append(math.nan)
            continue
        kl_sum = sum(
            records[index]["train/reference_kl"] * records[index]["train/loss_mask_tokens"]
            for index in window
            if records[index]["train/loss_mask_tokens"]
        )
        values.append(kl_sum / tokens)
    return values


def write_plot(path: Path, all_records: dict[str, dict[int, dict[str, float]]], num_steps: int) -> None:
    """Write the five-panel plot using the established GSM8K comparison
    style."""
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "relax-dr-grpo-qwen35-gsm8k"
    panels = (
        ("rollout/raw_reward", "Mean raw reward"),
        ("train/reference_kl", "Policy-reference KL, sum(KL) / T (10-step pooled)"),
        ("train/grad_norm", "Gradient norm"),
        ("rollout/response_lengths/correct", "Correct response length"),
        ("rollout/response_lengths/incorrect", "Incorrect length (10-step pooled)"),
    )
    colors = {"GRPO": "tab:blue", "Dr.GRPO": "tab:orange"}
    linestyles = {"GRPO": "--", "Dr.GRPO": "-"}
    figure = plt.figure(figsize=(10.2, 5.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 6)
    axes = (
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 0:3]),
        figure.add_subplot(grid[1, 3:6]),
    )
    for axis, (field, title) in zip(axes, panels, strict=True):
        for run, records in all_records.items():
            algorithm = RUNS[run]
            steps = sorted(records)
            values = [records[step][field] for step in steps]
            if field == "rollout/response_lengths/incorrect":
                axis.plot(steps, values, color=colors[algorithm], linewidth=0.7, alpha=0.18)
                values = pooled_outcome_lengths(records, "incorrect", 10)
            elif field == "train/reference_kl":
                axis.plot(steps, values, color=colors[algorithm], linewidth=0.7, alpha=0.18)
                values = pooled_reference_kl(records, 10)
            axis.plot(
                steps,
                values,
                color=colors[algorithm],
                linestyle=linestyles[algorithm],
                linewidth=1.3,
                label=(
                    f"{algorithm} (10-step pooled)"
                    if field in ("rollout/response_lengths/incorrect", "train/reference_kl")
                    else algorithm
                ),
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Optimizer step", fontsize=9)
        axis.set_xlim(0, num_steps - 1)
        ticks = list(range(0, num_steps, 50))
        if ticks[-1] != num_steps - 1:
            ticks.append(num_steps - 1)
        axis.set_xticks(ticks)
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    axes[0].set_ylim(-1.05, 1.05)
    axes[4].axhline(4096, color="black", linestyle=":", linewidth=1.0, alpha=0.7, label="Response cap (4096)")
    axes[4].legend(loc="best", fontsize=8)
    figure.suptitle("GSM8K — Qwen3.5-4B — CP1/DP2", fontsize=12)
    figure.savefig(path, dpi=160, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)


def write_pooled_incorrect_length_plot(
    path: Path, all_records: dict[str, dict[int, dict[str, float]]], num_steps: int, window_size: int = 10
) -> None:
    """Plot raw per-step and trailing pooled incorrect response lengths."""
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "relax-dr-grpo-qwen35-gsm8k"
    colors = {"GRPO": "tab:blue", "Dr.GRPO": "tab:orange"}
    linestyles = {"GRPO": "--", "Dr.GRPO": "-"}
    figure, axis = plt.subplots(figsize=(10.2, 4.2), constrained_layout=True)
    for run, records in all_records.items():
        algorithm = RUNS[run]
        steps = sorted(records)
        raw_values = [records[step]["rollout/response_lengths/incorrect"] for step in steps]
        pooled_values = pooled_outcome_lengths(records, "incorrect", window_size)
        axis.plot(steps, raw_values, color=colors[algorithm], linewidth=0.7, alpha=0.18)
        axis.plot(
            steps,
            pooled_values,
            color=colors[algorithm],
            linestyle=linestyles[algorithm],
            linewidth=2.0,
            label=f"{algorithm} ({window_size}-step pooled)",
        )
    axis.axhline(4096, color="black", linestyle=":", linewidth=1.0, alpha=0.7, label="Response cap (4096)")
    axis.set_title("Incorrect response length — trailing 10-step pooled conditional mean", fontsize=11)
    axis.set_xlabel("Optimizer step", fontsize=9)
    axis.set_ylabel("Response length (tokens)", fontsize=9)
    axis.set_xlim(0, num_steps - 1)
    ticks = list(range(0, num_steps, 50))
    if ticks[-1] != num_steps - 1:
        ticks.append(num_steps - 1)
    axis.set_xticks(ticks)
    axis.tick_params(labelsize=8)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.suptitle("GSM8K — Qwen3.5-4B — CP1/DP2", fontsize=12)
    figure.savefig(path, dpi=160, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--response-budget", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--model-parallel-size", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records: dict[str, dict[int, dict[str, float]]] = {}
    for run in RUNS:
        run_dir = args.experiment_root / run
        records = parse_log(find_training_log(run_dir))
        add_exact_loss_mask_tokens(
            run_dir,
            records,
            args.num_steps,
            args.world_size,
            args.model_parallel_size,
            args.global_batch_size,
        )
        add_outcome_lengths(run, run_dir, records, args.num_steps, args.response_budget, args.global_batch_size)
        validate_records(run, records, args.num_steps)
        all_records[run] = records

    write_csv(args.output_dir / "training_metrics_long.csv", all_records)
    write_summary(args.output_dir / "training_summary.json", all_records)
    write_tables(args.output_dir / "tables.md", all_records)
    write_plot(args.output_dir / "training_curves.svg", all_records, args.num_steps)
    write_pooled_incorrect_length_plot(
        args.output_dir / "incorrect_length_10step_pooled.svg", all_records, args.num_steps
    )


if __name__ == "__main__":
    main()
