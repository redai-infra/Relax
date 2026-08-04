# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compare paired VIME and ReLax evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMPATIBILITY_FIELDS = (
    "data_file",
    "mode",
    "tokenizer",
    "temperature",
    "top_p",
    "sampling_count",
    "chunk_tokens",
    "max_memory_tokens",
    "max_final_tokens",
    "max_chunks",
    "max_input_tokens",
    "server_max_model_len",
    "total",
)


def validate_compatible_summaries(*summaries: dict[str, Any]) -> None:
    """Reject a comparison when any controlled evaluation field differs."""
    if len(summaries) < 2:
        raise ValueError("At least two summaries are required for compatibility validation.")
    for field in COMPATIBILITY_FIELDS:
        if any(field not in summary for summary in summaries):
            raise KeyError(f"Compatibility field {field!r} must exist in every summary.")
        values = [summary[field] for summary in summaries]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Evaluation summaries differ on controlled field {field!r}: {values}")


def compare_pair(
    label: str,
    vime_summary: dict[str, Any],
    relax_summary: dict[str, Any],
    metric: str = "sub_em_pct",
    tolerance_pp: float = 3.0,
) -> dict[str, Any]:
    """Build one auditable percentage-point comparison.

    Both summaries must come from the same evaluator/data recipe. The function
    intentionally compares the reported percentage field directly: ``3.0``
    therefore means three percentage points, not a three-percent relative gap.
    """
    if metric not in vime_summary or metric not in relax_summary:
        raise KeyError(f"Metric {metric!r} must exist in both summaries.")
    vime_value = float(vime_summary[metric])
    relax_value = float(relax_summary[metric])
    gap_pp = abs(relax_value - vime_value)
    return {
        "label": label,
        "metric": metric,
        "vime": vime_value,
        "relax": relax_value,
        "absolute_gap_pp": gap_pp,
        "tolerance_pp": tolerance_pp,
        "passed": gap_pp <= tolerance_pp,
    }


def compare_baseline(
    label: str,
    base_summary: dict[str, Any],
    relax_summary: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    """Require the trained ReLax checkpoint to strictly beat frozen base."""
    if metric not in base_summary or metric not in relax_summary:
        raise KeyError(f"Metric {metric!r} must exist in both summaries.")
    base_value = float(base_summary[metric])
    relax_value = float(relax_summary[metric])
    improvement_pp = relax_value - base_value
    return {
        "label": label,
        "metric": metric,
        "base": base_value,
        "relax": relax_value,
        "improvement_pp": improvement_pp,
        "passed": improvement_pp > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "VIME_SUMMARY", "RELAX_SUMMARY"),
        required=True,
        help="Repeat for every RULER-HQA length selected for acceptance.",
    )
    parser.add_argument("--metric", default="sub_em_pct")
    parser.add_argument("--tolerance-pp", type=float, default=3.0)
    parser.add_argument(
        "--baseline-pair",
        nargs=4,
        action="append",
        default=[],
        metavar=("LABEL", "METRIC", "BASE_SUMMARY", "RELAX_SUMMARY"),
        help="Require ReLax to strictly exceed frozen base; repeat for every required metric/dataset.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparisons = []
    for label, vime_path, relax_path in args.pair:
        with Path(vime_path).open(encoding="utf-8") as source:
            vime_summary = json.load(source)
        with Path(relax_path).open(encoding="utf-8") as source:
            relax_summary = json.load(source)
        validate_compatible_summaries(vime_summary, relax_summary)
        comparison = compare_pair(label, vime_summary, relax_summary, args.metric, args.tolerance_pp)
        comparison["vime_summary"] = str(vime_path)
        comparison["relax_summary"] = str(relax_path)
        comparisons.append(comparison)

    baseline_comparisons = []
    for label, metric, base_path, relax_path in args.baseline_pair:
        with Path(base_path).open(encoding="utf-8") as source:
            base_summary = json.load(source)
        with Path(relax_path).open(encoding="utf-8") as source:
            relax_summary = json.load(source)
        validate_compatible_summaries(base_summary, relax_summary)
        comparison = compare_baseline(label, base_summary, relax_summary, metric)
        comparison["base_summary"] = str(base_path)
        comparison["relax_summary"] = str(relax_path)
        baseline_comparisons.append(comparison)

    report = {
        "metric": args.metric,
        "tolerance_pp": args.tolerance_pp,
        "passed": all(item["passed"] for item in comparisons + baseline_comparisons),
        "comparisons": comparisons,
        "baseline_comparisons": baseline_comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
