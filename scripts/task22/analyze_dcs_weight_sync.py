#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Analyze joint Hybrid DCS and targeted-retirement markers."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from pathlib import Path
from statistics import mean, median
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MARKER = "TASK22_DCS_WEIGHT_SYNC"
SNAPSHOT_MARKER = "TASK22_WEIGHT_SNAPSHOT"
FLOAT_FIELDS = (
    "topology_seconds",
    "group_setup_seconds",
    "source_materialize_seconds",
    "tp_gather_seconds",
    "hf_conversion_seconds",
    "lock_wait_seconds",
    "broadcast_seconds",
    "receiver_finalize_seconds",
    "pause_flush_seconds",
    "continue_seconds",
    "targeted_prepare_seconds",
    "backend_total_seconds",
    "client_total_seconds",
)
INT_FIELDS = (
    "weight_version",
    "group_world_size",
    "rollout_receivers",
    "source_h2d_bytes",
    "source_local_bytes",
    "broadcast_buckets",
    "broadcast_tensors",
    "broadcast_bytes",
    "fanout_bytes",
    "targeted_active_requests",
    "targeted_expired_requests",
    "targeted_safe_requests",
)
RETIRE_MARKER = "TASK22_TARGETED_RETIRE"


class AnalysisError(ValueError):
    pass


def _rows(driver_log: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(driver_log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = ANSI_RE.sub("", raw)
        marker = line.find(MARKER)
        if marker < 0:
            continue
        fields: dict[str, str] = {}
        for item in shlex.split(line[marker + len(MARKER) :]):
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value
        try:
            row: dict[str, Any] = {
                "logical_step": int(fields["logical_step"]),
                "group_reused": fields["group_reused"].lower() == "true",
                "_line": line_no,
            }
            for name in FLOAT_FIELDS:
                value = float(fields[name])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name}={value!r}")
                row[name] = value
            for name in INT_FIELDS:
                value = int(fields[name])
                if value < 0:
                    raise ValueError(f"{name}={value!r}")
                row[name] = value
        except (KeyError, ValueError) as exc:
            raise AnalysisError(f"malformed DCS marker at line {line_no}: {exc}") from exc
        rows.append(row)
    return rows


def _snapshot_rows(driver_log: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(driver_log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = ANSI_RE.sub("", raw)
        marker = line.find(SNAPSHOT_MARKER)
        if marker < 0:
            continue
        fields = {}
        for item in shlex.split(line[marker + len(SNAPSHOT_MARKER) :]):
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value
        try:
            elapsed_seconds = float(fields["elapsed_seconds"])
            if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
                raise ValueError(f"elapsed_seconds={elapsed_seconds!r}")
            rows.append(
                {
                    "logical_step": int(fields["logical_step"]),
                    "on_device": fields["on_device"].lower() == "true",
                    "local_tensors": int(fields["local_tensors"]),
                    "local_bytes": int(fields["local_bytes"]),
                    "elapsed_seconds": elapsed_seconds,
                    "_line": line_no,
                }
            )
        except (KeyError, ValueError) as exc:
            raise AnalysisError(f"malformed snapshot marker at line {line_no}: {exc}") from exc
    return rows


def _retirement_events(driver_log: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(driver_log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = ANSI_RE.sub("", raw)
        marker = line.find(RETIRE_MARKER)
        if marker < 0:
            continue
        fields = {}
        for item in shlex.split(line[marker + len(RETIRE_MARKER) :]):
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value
        try:
            rows.append(
                {
                    "event": fields["event"],
                    "publication_id": fields["publication_id"],
                    "target_version": int(fields["target_version"]),
                    "_line": line_no,
                }
            )
        except (KeyError, ValueError) as exc:
            raise AnalysisError(f"malformed targeted retirement marker at line {line_no}: {exc}") from exc
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in FLOAT_FIELDS:
        values = [row[field] for row in rows]
        result[field] = {
            "count": len(values),
            "sum": sum(values),
            "mean": mean(values),
            "p50": median(values),
            "max": max(values),
        }
    for field in INT_FIELDS:
        values = [row[field] for row in rows]
        result[field] = {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def analyze(driver_log: Path, *, num_rollout: int = 11, headline_lo: int = 2, headline_hi: int = 9):
    rows = _rows(driver_log)
    snapshots = _snapshot_rows(driver_log)
    retirement_events = _retirement_events(driver_log)
    init_rows = [row for row in rows if row["logical_step"] == -1]
    train_rows = [row for row in rows if row["logical_step"] >= 0]
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in train_rows:
        by_step.setdefault(row["logical_step"], []).append(row)

    errors = []
    if len(init_rows) != 1:
        errors.append(f"init_marker_count={len(init_rows)}")
    expected_steps = set(range(num_rollout))
    observed_steps = set(by_step)
    if observed_steps != expected_steps:
        errors.append(
            f"training_step_coverage:missing={sorted(expected_steps - observed_steps)}:"
            f"unexpected={sorted(observed_steps - expected_steps)}"
        )
    duplicates = sorted(step for step, step_rows in by_step.items() if len(step_rows) != 1)
    if duplicates:
        errors.append(f"nonunique_training_steps={duplicates}")

    snapshot_by_step: dict[int, list[dict[str, Any]]] = {}
    for row in snapshots:
        snapshot_by_step.setdefault(row["logical_step"], []).append(row)
    expected_snapshot_steps = {-1, *expected_steps}
    if set(snapshot_by_step) != expected_snapshot_steps:
        errors.append(
            f"snapshot_step_coverage:missing={sorted(expected_snapshot_steps - set(snapshot_by_step))}:"
            f"unexpected={sorted(set(snapshot_by_step) - expected_snapshot_steps)}"
        )
    snapshot_duplicates = sorted(step for step, step_rows in snapshot_by_step.items() if len(step_rows) != 1)
    if snapshot_duplicates:
        errors.append(f"nonunique_snapshot_steps={snapshot_duplicates}")

    for row in train_rows:
        step = row["logical_step"]
        if not row["group_reused"]:
            errors.append(f"logical_step={step}:group_not_reused")
        if row["group_world_size"] != 3 or row["rollout_receivers"] != 2:
            errors.append(
                f"logical_step={step}:topology={row['group_world_size']}/{row['rollout_receivers']}:expected=3/2"
            )
        if row["source_h2d_bytes"] != 0:
            errors.append(f"logical_step={step}:gpu_snapshot_missed:h2d_bytes={row['source_h2d_bytes']}")
        if row["broadcast_bytes"] <= 0 or row["fanout_bytes"] != 2 * row["broadcast_bytes"]:
            errors.append(f"logical_step={step}:invalid_fanout_bytes:{row['broadcast_bytes']}/{row['fanout_bytes']}")
        if row["targeted_active_requests"] != (row["targeted_expired_requests"] + row["targeted_safe_requests"]):
            errors.append(f"logical_step={step}:invalid_targeted_request_partition")
        snapshot_rows = snapshot_by_step.get(step, [])
        if len(snapshot_rows) == 1:
            snapshot = snapshot_rows[0]
            if not snapshot["on_device"]:
                errors.append(f"logical_step={step}:snapshot_not_on_device")
            if snapshot["local_tensors"] <= 0 or snapshot["local_bytes"] <= 0:
                errors.append(f"logical_step={step}:invalid_snapshot_size")
            row["snapshot_elapsed_seconds"] = snapshot["elapsed_seconds"]
            row["snapshot_plus_client_seconds"] = snapshot["elapsed_seconds"] + row["client_total_seconds"]

    headline_rows = [row for row in train_rows if headline_lo <= row["logical_step"] <= headline_hi]
    if len(headline_rows) != headline_hi - headline_lo + 1:
        errors.append(f"headline_count={len(headline_rows)}")
    if sum(row["targeted_safe_requests"] for row in train_rows) == 0:
        errors.append("targeted_safe_requests_not_exercised")
    if sum(row["targeted_expired_requests"] for row in train_rows) == 0:
        errors.append("targeted_expired_requests_not_exercised")

    events_by_publication: dict[str, list[str]] = {}
    versions_by_publication: dict[str, set[int]] = {}
    for event in retirement_events:
        events_by_publication.setdefault(event["publication_id"], []).append(event["event"])
        versions_by_publication.setdefault(event["publication_id"], set()).add(event["target_version"])
    if len(events_by_publication) != num_rollout + 1:
        errors.append(f"targeted_publication_count={len(events_by_publication)}")
    for publication_id, events in events_by_publication.items():
        if events != ["prepare", "commit"]:
            errors.append(f"publication={publication_id}:events={events}")
        if len(versions_by_publication[publication_id]) != 1:
            errors.append(
                f"publication={publication_id}:target_versions={sorted(versions_by_publication[publication_id])}"
            )
    event_versions = sorted(
        next(iter(versions)) for versions in versions_by_publication.values() if len(versions) == 1
    )
    marker_versions = sorted(row["weight_version"] for row in rows)
    if event_versions != marker_versions:
        errors.append(f"publication_version_mismatch:events={event_versions}:markers={marker_versions}")

    return {
        "schema_version": 1,
        "verdict": "PASS" if not errors else "INVALID",
        "errors": errors,
        "contract": {
            "num_rollout": num_rollout,
            "headline": {"logical_step_lo": headline_lo, "logical_step_hi": headline_hi},
            "expected_group_world_size": 3,
            "expected_rollout_receivers": 2,
            "gpu_snapshot_requires_zero_h2d": True,
        },
        "init": init_rows,
        "snapshots": {str(row["logical_step"]): row for row in snapshots},
        "targeted_retirement_events": retirement_events,
        "per_step": {str(row["logical_step"]): row for row in train_rows},
        "headline_summary": (
            {
                **_summary(headline_rows),
                "snapshot_elapsed_seconds": {
                    "count": len(headline_rows),
                    "sum": sum(row.get("snapshot_elapsed_seconds", 0.0) for row in headline_rows),
                    "mean": mean(row.get("snapshot_elapsed_seconds", 0.0) for row in headline_rows),
                    "p50": median(row.get("snapshot_elapsed_seconds", 0.0) for row in headline_rows),
                    "max": max(row.get("snapshot_elapsed_seconds", 0.0) for row in headline_rows),
                },
                "snapshot_plus_client_seconds": {
                    "count": len(headline_rows),
                    "sum": sum(row.get("snapshot_plus_client_seconds", 0.0) for row in headline_rows),
                    "mean": mean(row.get("snapshot_plus_client_seconds", 0.0) for row in headline_rows),
                    "p50": median(row.get("snapshot_plus_client_seconds", 0.0) for row in headline_rows),
                    "max": max(row.get("snapshot_plus_client_seconds", 0.0) for row in headline_rows),
                },
            }
            if headline_rows
            else {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.driver_log)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    raise SystemExit(0 if result["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
