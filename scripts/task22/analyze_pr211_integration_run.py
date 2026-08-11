#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate publication topology for the Task 22 PR211 integration experiment.

This is an experiment-harness check, not a product runtime policy.  Its only
purpose is to reject A/B runs whose effective Rollout topology drifted from the
predeclared launch contract.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MARKER = "DCS_WEIGHT_SYNC"


class AnalysisError(ValueError):
    """The run artifacts do not satisfy the experiment contract."""


def _option(argv: list[str], flag: str) -> str:
    positions = [index for index, item in enumerate(argv) if item == flag or item.startswith(f"{flag}=")]
    if len(positions) != 1:
        raise AnalysisError(f"{flag} must appear exactly once in train_argv, found {len(positions)}")
    index = positions[0]
    item = argv[index]
    if item.startswith(f"{flag}="):
        return item.split("=", maxsplit=1)[1]
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise AnalysisError(f"{flag} is missing a value in train_argv")
    return argv[index + 1]


def _has_option(argv: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in argv)


def _derive_expected_topology(contract: dict[str, Any]) -> dict[str, int]:
    argv = contract.get("train_argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise AnalysisError("run_contract.train_argv must be a string array")

    try:
        resource = json.loads(_option(argv, "--resource"))
        rollout_resource = resource["rollout"]
        rollout_total_gpus = int(rollout_resource[1])
        rollout_gpus_per_engine = int(_option(argv, "--rollout-num-gpus-per-engine"))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, IndexError) as exc:
        raise AnalysisError(f"cannot derive Rollout topology from train_argv: {exc}") from exc

    if rollout_total_gpus <= 0 or rollout_gpus_per_engine <= 0:
        raise AnalysisError(
            f"Rollout GPU counts must be positive: total={rollout_total_gpus}, per_engine={rollout_gpus_per_engine}"
        )
    if rollout_total_gpus % rollout_gpus_per_engine != 0:
        raise AnalysisError(
            "Rollout total GPUs must be divisible by GPUs per engine: "
            f"{rollout_total_gpus} % {rollout_gpus_per_engine} != 0"
        )

    return {
        "rollout_total_gpus": rollout_total_gpus,
        "rollout_gpus_per_engine": rollout_gpus_per_engine,
        "rollout_receiver_count": rollout_total_gpus // rollout_gpus_per_engine,
        # DCS group rank 0 is the Actor sender; all Rollout GPUs are receivers.
        "group_world_size": 1 + rollout_total_gpus,
    }


def _publication_rows(driver_log: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = driver_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise AnalysisError(f"cannot read driver log: {exc}") from exc
    for line_no, raw in enumerate(lines, 1):
        line = ANSI_RE.sub("", raw)
        marker = line.find(MARKER)
        if marker < 0:
            continue
        fields: dict[str, str] = {}
        for item in shlex.split(line[marker + len(MARKER) :]):
            if "=" in item:
                key, value = item.split("=", maxsplit=1)
                fields[key] = value
        try:
            reused_raw = fields["group_reused"].lower()
            if reused_raw not in {"true", "false"}:
                raise ValueError(f"group_reused={reused_raw!r}")
            row = {
                "logical_step": int(fields["logical_step"]),
                "weight_version": int(fields["weight_version"]),
                "group_reused": reused_raw == "true",
                "group_world_size": int(fields["group_world_size"]),
                "rollout_receiver_count": int(fields["rollout_receivers"]),
                "line": line_no,
            }
        except (KeyError, ValueError) as exc:
            raise AnalysisError(f"malformed {MARKER} at driver.log line {line_no}: {exc}") from exc
        rows.append(row)
    return rows


def _expected_publication_steps(num_rollout: int, interval: int) -> list[int]:
    if num_rollout <= 0 or interval <= 0:
        raise AnalysisError(f"invalid publication cadence: num_rollout={num_rollout}, interval={interval}")
    steps = [-1]
    for rollout_id in range(num_rollout):
        if rollout_id == num_rollout - 1 or (rollout_id + 1) % interval == 0:
            steps.append(rollout_id)
    return steps


def analyze(run_contract: Path, driver_log: Path) -> dict[str, Any]:
    try:
        contract = json.loads(run_contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read run contract: {exc}") from exc
    if not isinstance(contract, dict):
        raise AnalysisError("run contract must be a JSON object")

    argv = contract.get("train_argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise AnalysisError("run_contract.train_argv must be a string array")

    expected_topology = _derive_expected_topology(contract)
    declared_topology = contract.get("expected_topology")
    errors: list[str] = []
    if declared_topology is not None and declared_topology != expected_topology:
        errors.append(f"declared_topology_mismatch:{declared_topology!r}!={expected_topology!r}")

    rows = _publication_rows(driver_log)
    argv_cross_version_kv = _has_option(argv, "--enable-cross-version-kv-continuation")
    cross_version_kv = contract.get("cross_version_kv", argv_cross_version_kv)
    if not isinstance(cross_version_kv, bool):
        errors.append("cross_version_kv_not_boolean")
        cross_version_kv = argv_cross_version_kv
    if cross_version_kv != argv_cross_version_kv:
        errors.append(f"cross_version_kv_contract_drift:{cross_version_kv}!={argv_cross_version_kv}")

    if not cross_version_kv:
        if rows:
            errors.append(f"kv_off_arm_has_dcs_publications:count={len(rows)}")
        return {
            "schema_version": 1,
            "verdict": "INVALID_INPUT" if errors else "VALID",
            "applicable": False,
            "errors": errors,
            "contract": {"expected_topology": expected_topology},
            "observed": {"publication_count": len(rows), "publications": rows},
        }

    try:
        argv_num_rollout = int(_option(argv, "--num-rollout"))
        argv_interval = int(_option(argv, "--update-weights-interval"))
        num_rollout = int(contract.get("num_rollout", argv_num_rollout))
        interval = int(contract.get("update_weights_interval", argv_interval))
        if num_rollout != argv_num_rollout:
            errors.append(f"num_rollout_contract_drift:{num_rollout}!={argv_num_rollout}")
        if interval != argv_interval:
            errors.append(f"update_weights_interval_contract_drift:{interval}!={argv_interval}")
        expected_steps = _expected_publication_steps(num_rollout, interval)
    except (KeyError, TypeError, ValueError, AnalysisError) as exc:
        errors.append(f"invalid_publication_cadence:{exc}")
        expected_steps = []

    observed_steps = [row["logical_step"] for row in rows]
    if observed_steps != expected_steps:
        errors.append(f"publication_step_coverage:{observed_steps!r}!={expected_steps!r}")

    expected_receivers = expected_topology["rollout_receiver_count"]
    expected_world_size = expected_topology["group_world_size"]
    for row in rows:
        step = row["logical_step"]
        if row["rollout_receiver_count"] != expected_receivers or row["group_world_size"] != expected_world_size:
            errors.append(
                f"logical_step={step}:topology="
                f"{row['group_world_size']}/{row['rollout_receiver_count']}:"
                f"expected={expected_world_size}/{expected_receivers}"
            )
        if step == -1 and row["group_reused"]:
            errors.append("logical_step=-1:initial_group_unexpectedly_reused")
        if step >= 0 and not row["group_reused"]:
            errors.append(f"logical_step={step}:group_not_reused")

    return {
        "schema_version": 1,
        "verdict": "INVALID_INPUT" if errors else "VALID",
        "applicable": True,
        "errors": errors,
        "contract": {
            "num_rollout": num_rollout,
            "update_weights_interval": interval,
            "expected_publication_steps": expected_steps,
            "expected_topology": expected_topology,
        },
        "observed": {"publication_count": len(rows), "publications": rows},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = analyze(args.run_contract, args.driver_log)
    except AnalysisError as exc:
        result = {
            "schema_version": 1,
            "verdict": "INVALID_INPUT",
            "applicable": None,
            "errors": [str(exc)],
        }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "TASK22_PR211_INTEGRATION_ANALYSIS "
        f"verdict={result['verdict']} applicable={result.get('applicable')} errors={len(result['errors'])}"
    )
    raise SystemExit(0 if result["verdict"] == "VALID" else 2)


if __name__ == "__main__":
    main()
