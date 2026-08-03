#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail closed when Task40 evidence is incomplete or internally
inconsistent."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


ALL_CONFIGS = (
    "p3o_on_policy",
    "grpo_on_policy",
    "p3o_temperature_0p6",
    "grpo_temperature_0p6",
    "p3o_temperature_1p2",
    "grpo_temperature_1p2",
)
EXPECTED_SEEDS = (42, 1234, 2026)
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Z]:(?:\\[^\\\r\n]{2,}\\|/[^/\r\n]{2,}/)")
MODEL_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-config", action="append", choices=ALL_CONFIGS)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_complete(run_dir: Path) -> bool:
    return (
        read_text(run_dir / "exit_code.txt").strip() == "0"
        and "succeeded" in read_text(run_dir / "job_status.txt").lower()
        and "All training steps finished" in read_text(run_dir / "stdout_stderr.log")
        and bool(list((run_dir / "tensorboard").glob("events.out.tfevents.*")))
    )


def compare_controls(identities: list[dict[str, Any]]) -> list[str]:
    errors = []
    ignored = {"method", "rollout_temperature", "seed"}
    reference = {key: value for key, value in identities[0].items() if key not in ignored}
    for identity in identities[1:]:
        controls = {key: value for key, value in identity.items() if key not in ignored}
        if controls != reference:
            errors.append(f"control mismatch: expected={reference} actual={controls}")
    return errors


def scan_text_artifacts(root: Path) -> list[str]:
    errors = []
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024 or "tensorboard" in path.parts:
            continue
        if b"\0" in path.read_bytes()[:8192]:
            continue
        text = read_text(path)
        relative_parts = path.relative_to(root).parts
        packaged_test_source = relative_parts[:4] == ("delivery", "source", "Relax", "tests")
        if WINDOWS_PATH.search(text) and not packaged_test_source:
            errors.append(f"Windows absolute path found: {path}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"secret-like content found: {path} pattern={pattern.pattern}")
    return errors


def scan_tracked_models(repo_root: Path) -> list[str]:
    output = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    errors = []
    for item in output.split(b"\0"):
        if not item:
            continue
        relative = Path(item.decode())
        path = repo_root / relative
        if path.exists() and (
            (path.suffix.lower() in MODEL_SUFFIXES and path.stat().st_size > 1024 * 1024)
            or path.stat().st_size > 50 * 1024 * 1024
        ):
            errors.append(f"tracked model/checkpoint-like file: {relative}")
    return errors


def main() -> None:
    args = parse_args()
    expected_configs = tuple(args.expected_config or ALL_CONFIGS)
    errors: list[str] = []
    run_identity_paths = sorted((args.evidence_root / "runs").rglob("run_identity.json"))
    attempts: dict[tuple[str, int], list[Path]] = {}
    successful_identities = []
    git_shas = set()
    for identity_path in run_identity_paths:
        run_dir = identity_path.parent
        identity = json.loads(read_text(identity_path))
        config = run_dir.parents[1].name
        seed = int(identity["seed"])
        attempts.setdefault((config, seed), []).append(run_dir)
        required = ("command.sh", "exit_code.txt", "job_status.txt", "run_identity.json")
        for filename in required:
            if not (run_dir / filename).is_file():
                errors.append(f"missing {filename}: {run_dir}")
        if run_complete(run_dir):
            if not (run_dir / "metrics.json").is_file():
                errors.append(f"successful run missing metrics.json: {run_dir}")
            successful_identities.append(identity)
            git_shas.add(identity["git_sha"])

    for config in expected_configs:
        for seed in EXPECTED_SEEDS:
            candidates = attempts.get((config, seed), [])
            if not candidates:
                errors.append(f"missing planned run: config={config} seed={seed}")
                continue
            successful = [run_dir for run_dir in candidates if run_complete(run_dir)]
            if len(successful) != 1:
                errors.append(
                    f"expected exactly one successful identity: config={config} seed={seed} "
                    f"successes={len(successful)} attempts={len(candidates)}"
                )

    if len(git_shas) > 1:
        errors.append(f"successful run git SHA mismatch: {sorted(git_shas)}")
    if successful_identities:
        errors.extend(compare_controls(successful_identities))

    per_run_rows = load_csv(args.evidence_root / "analysis" / "per_run_metrics.csv")
    analyzed_paths = {row["run_path"] for row in per_run_rows}
    identity_run_paths = {str(path.parent) for path in run_identity_paths}
    if analyzed_paths != identity_run_paths:
        errors.append(
            f"analysis coverage mismatch: analyzed={len(analyzed_paths)} identities={len(identity_run_paths)}"
        )
    for row in per_run_rows:
        try:
            nonfinite = int(row["nonfinite_scalar_count"])
        except (KeyError, ValueError):
            errors.append(f"invalid nonfinite count in per_run_metrics.csv: {row}")
            continue
        if nonfinite:
            errors.append(f"non-finite scalars: run={row['run_path']} count={nonfinite}")
        for key, value in row.items():
            if value and value.lower() in {"nan", "inf", "-inf"}:
                errors.append(f"non-finite analysis value: run={row['run_path']} field={key}")
            try:
                if value and not math.isfinite(float(value)):
                    errors.append(f"non-finite numeric value: run={row['run_path']} field={key}")
            except ValueError:
                pass

    for required_analysis in (
        "aggregate_metrics.csv",
        "paired_seed_comparison.csv",
        "failures.csv",
        "official_acceptance_matrix.md",
        "frozen_gate_verdict.md",
        "throughput_memory_table.md",
    ):
        if not (args.evidence_root / "analysis" / required_analysis).is_file():
            errors.append(f"missing analysis artifact: {required_analysis}")

    errors.extend(scan_text_artifacts(args.evidence_root))
    errors.extend(scan_tracked_models(args.repo_root))
    print(
        f"planned_runs={len(expected_configs) * len(EXPECTED_SEEDS)} "
        f"attempts={len(run_identity_paths)} successful={len(successful_identities)} errors={len(errors)}"
    )
    for error in errors:
        print(f"ERROR {error}")
    if errors:
        raise SystemExit(1)
    print("TASK40_ARTIFACT_VERIFICATION_PASS")


if __name__ == "__main__":
    main()
