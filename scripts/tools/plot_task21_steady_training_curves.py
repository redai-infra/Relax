# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Plot Task 21 training curves from validated TensorBoard scalar exports."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence


METRICS = {
    "train/grad_norm": ("Gradient norm", "task21_steady_grad_norm.png"),
    "train/loss": ("Training loss", "task21_steady_loss.png"),
    "rollout/raw_reward": ("Raw reward", "task21_steady_reward.png"),
}
CONDITION_ORDER = ("B", "P", "P+R", "P+S")
COLORS = {
    "B": "#343A40",
    "P": "#168C83",
    "P+R": "#C43D3D",
    "P+S": "#2867B2",
}
SEED_STYLES = ("--", ":")


def _parse_run(value: str) -> tuple[str, int, Path]:
    try:
        condition, seed_text, path_text = value.split("=", 2)
        seed = int(seed_text)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("--run must be CONDITION=SEED=RUN_DIR") from None
    if condition not in CONDITION_ORDER:
        raise argparse.ArgumentTypeError(f"unsupported condition {condition!r}")
    run_dir = Path(path_text).resolve()
    scalar_path = run_dir / "analysis" / "tensorboard_scalars.csv"
    if not scalar_path.is_file():
        raise argparse.ArgumentTypeError(f"missing scalar export: {scalar_path}")
    return condition, seed, run_dir


def _load_series(run_dir: Path, *, expected_steps: int) -> dict[str, dict[int, float]]:
    scalar_path = run_dir / "analysis" / "tensorboard_scalars.csv"
    series = {tag: {} for tag in METRICS}
    with scalar_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tag = row["tag"]
            if tag not in series:
                continue
            step = int(row["step"])
            if step in series[tag]:
                raise ValueError(f"{scalar_path}: duplicate {tag} step {step}")
            series[tag][step] = float(row["value"])

    expected = set(range(expected_steps))
    for tag, values in series.items():
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"{scalar_path}: {tag} step mismatch: missing={missing}, extra={extra}")
    return series


def _write_csv(
    output_path: Path,
    rows: Sequence[tuple[str, int, Path, dict[str, dict[int, float]]]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("condition", "seed", "step", "grad_norm", "loss", "raw_reward", "run"))
        for condition, seed, run_dir, series in rows:
            for step in sorted(series["train/grad_norm"]):
                writer.writerow(
                    (
                        condition,
                        seed,
                        step,
                        series["train/grad_norm"][step],
                        series["train/loss"][step],
                        series["rollout/raw_reward"][step],
                        run_dir.name,
                    )
                )


def _plot_metric(
    output_path: Path,
    tag: str,
    grouped: dict[str, list[tuple[int, dict[int, float]]]],
    *,
    expected_steps: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib") from exc

    title, _ = METRICS[tag]
    steps = list(range(expected_steps))
    figure, axis = plt.subplots(figsize=(12, 6.75), dpi=160)
    axis.axvspan(-0.5, 9.5, color="#E9ECEF", alpha=0.7, label="Warmup (0-9)", zorder=0)
    axis.axvline(9.5, color="#868E96", linewidth=1, zorder=1)

    for condition in CONDITION_ORDER:
        seed_series = sorted(grouped[condition], key=lambda item: item[0])
        values_by_seed = [[values[step] for step in steps] for _, values in seed_series]
        lower = [min(values[index] for values in values_by_seed) for index in steps]
        upper = [max(values[index] for values in values_by_seed) for index in steps]
        mean = [sum(values[index] for values in values_by_seed) / len(values_by_seed) for index in steps]
        color = COLORS[condition]
        axis.fill_between(steps, lower, upper, color=color, alpha=0.1, linewidth=0)
        for index, (seed, values) in enumerate(seed_series):
            axis.plot(
                steps,
                [values[step] for step in steps],
                color=color,
                linestyle=SEED_STYLES[index],
                linewidth=1,
                alpha=0.45,
            )
        axis.plot(steps, mean, color=color, linewidth=2.4, label=f"{condition} mean", zorder=3)

    if tag == "train/grad_norm":
        axis.set_yscale("log")
        spike_seed, spike_values = max(
            grouped["P+R"],
            key=lambda item: max(item[1].values()),
        )
        spike_step, spike_value = max(spike_values.items(), key=lambda item: item[1])
        axis.annotate(
            f"P+R seed {spike_seed}: {spike_value:.2f}",
            xy=(spike_step, spike_value),
            xytext=(spike_step + 2, spike_value * 0.72),
            arrowprops={"arrowstyle": "->", "color": COLORS["P+R"]},
            color=COLORS["P+R"],
            fontsize=9,
        )
    elif tag == "train/loss":
        axis.axhline(0, color="#ADB5BD", linewidth=1)
    elif tag == "rollout/raw_reward":
        axis.set_ylim(0, 0.7)

    axis.set_title(f"Task 21 steady training: {title}", fontsize=16, pad=14)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel(title)
    axis.set_xlim(-0.5, expected_steps - 0.5)
    axis.set_xticks(range(0, expected_steps, 5))
    axis.grid(axis="y", color="#DEE2E6", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="#6C757D", linestyle="--", linewidth=1, alpha=0.6),
            Patch(facecolor="#6C757D", alpha=0.12, edgecolor="none"),
        ]
    )
    labels.extend(("Individual seed", "Two-seed range"))
    axis.legend(handles, labels, ncol=4, frameon=False, fontsize=9, loc="best")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run,
        help="Validated formal run as CONDITION=SEED=RUN_DIR; repeat twice per condition.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=40)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_steps <= 0:
        raise SystemExit("--expected-steps must be positive")

    runs_by_condition: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for condition, seed, run_dir in args.run:
        runs_by_condition[condition].append((seed, run_dir))
    invalid = {
        condition: len(runs_by_condition.get(condition, []))
        for condition in CONDITION_ORDER
        if len(runs_by_condition.get(condition, [])) != 2
    }
    if invalid:
        raise SystemExit(f"expected exactly two runs per condition, got {invalid}")

    loaded = []
    grouped: dict[str, dict[str, list[tuple[int, dict[int, float]]]]] = {tag: defaultdict(list) for tag in METRICS}
    for condition in CONDITION_ORDER:
        for seed, run_dir in sorted(runs_by_condition[condition]):
            series = _load_series(run_dir, expected_steps=args.expected_steps)
            loaded.append((condition, seed, run_dir, series))
            for tag in METRICS:
                grouped[tag][condition].append((seed, series[tag]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "task21_steady_training_curves.csv", loaded)
    for tag, (_, filename) in METRICS.items():
        _plot_metric(
            args.output_dir / filename,
            tag,
            grouped[tag],
            expected_steps=args.expected_steps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
