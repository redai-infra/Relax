#!/usr/bin/env python3
"""Replay Joint Carry-Admit decisions against historical Task 22 traces."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_joint_planner():
    from relax.utils.cross_version_kv import plan_joint_carry_admission

    return plan_joint_carry_admission


plan_joint_carry_admission = load_joint_planner()


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RUNS = (
    ("instrumented_on_1", "phase_feedback_fix_on_20260802_212004"),
    ("instrumented_on_2", "phase_feedback_on20_20260802_230341"),
)
LOG_TIMEZONE = timezone(timedelta(hours=8))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_epoch(line: str) -> float | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOG_TIMEZONE).timestamp()


def parse_driver_log(path: Path) -> dict:
    text = ANSI_RE.sub("", path.read_text(errors="replace")).replace("\r", "\n")
    rollout_start: dict[int, float] = {}
    carry_over: dict[int, dict[str, int]] = {}
    debt_close: dict[int, float] = {}
    gate_begin: dict[int, float] = {}
    for line in text.splitlines():
        match = re.search(r"Starting rollout step (\d+)", line)
        if match is not None and int(match.group(1)) not in rollout_start:
            rollout_start[int(match.group(1))] = parse_epoch(line)
        match = re.search(
            r"Rollout step (\d+) carry-over: committed_current=(\d+) "
            r"next_step_deficit=(\d+) oversample_surplus=(\d+) aborted=(\d+)",
            line,
        )
        if match is not None:
            rollout_id, committed, deficit, surplus, aborted = map(int, match.groups())
            carry_over[rollout_id] = {
                "committed": committed,
                "deficit": deficit,
                "surplus": surplus,
                "aborted": aborted,
            }
        match = re.search(r"phase=debt_partition_closed .*rollout_id=(\d+) groups=(\d+)", line)
        if match is not None:
            debt_close[int(match.group(1))] = parse_epoch(line)
        match = re.search(r"phase=actor_phase .*actor_phase=quiesce .*rollout_id=(\d+)", line)
        if match is not None:
            gate_begin.setdefault(int(match.group(1)), parse_epoch(line))
    return {
        "rollout_start": rollout_start,
        "carry_over": carry_over,
        "debt_close": debt_close,
        "gate_begin": gate_begin,
    }


def load_permit_rows(run_path: Path) -> list[dict]:
    rows: list[dict] = []
    files = sorted(
        run_path.glob("permit_wait_rollout_*.jsonl"),
        key=lambda item: int(re.search(r"(\d+)$", item.stem).group(1)),
    )
    for path in files:
        rows.extend(load_jsonl(path))
    return rows


def group_rows(rows: list[dict], field: str) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row[field])].append(row)
    return grouped


def group_is_useful(rows: list[dict]) -> bool:
    return bool(rows) and all("ABORTED" not in row["sample_status"] for row in rows)


def group_end_epoch(rows: list[dict]) -> float:
    return max(float(row["terminal_epoch"]) for row in rows)


def replay_run(name: str, path: Path, analysis_wall_seconds: float) -> dict:
    contract = json.loads((path / "run_contract.json").read_text())
    analysis_lo = int(contract["headline"]["logical_step_lo"])
    analysis_hi = int(contract["headline"]["logical_step_hi"])
    contract_text = json.dumps(contract)
    a3_enabled = (
        "--enable-cross-version-kv-continuation" in contract_text
        or '"enable_cross_version_kv_continuation": true' in contract_text
    )
    joint_planner_rows = sum(len(load_jsonl(ledger_path)) for ledger_path in path.glob("joint_planner_ledger_*.jsonl"))
    log_data = parse_driver_log(path / "driver.log")
    rows = load_permit_rows(path)
    by_rollout = group_rows(rows, "physical_rollout_id")
    events = []
    quorum_savings = 0.0
    soft_floor_savings = 0.0
    soft_floor_hits = 0
    debt_timing_regressions = 0
    excluded_events = []

    for source_rollout, carry in sorted(log_data["carry_over"].items()):
        debt_target = carry["deficit"]
        physical_rollout = source_rollout + 1
        if debt_target > 0 and not (analysis_lo <= physical_rollout <= analysis_hi):
            excluded_events.append(
                {
                    "source_rollout": source_rollout,
                    "physical_rollout": physical_rollout,
                    "debt_target": debt_target,
                    "reason": "outside_headline_analysis_window",
                    "outside_analysis_window": True,
                }
            )
            continue
        if (
            debt_target <= 0
            or source_rollout not in by_rollout
            or physical_rollout not in by_rollout
            or physical_rollout not in log_data["debt_close"]
            or physical_rollout not in log_data["gate_begin"]
            or physical_rollout not in log_data["rollout_start"]
        ):
            if debt_target > 0:
                missing = [
                    name
                    for name, present in (
                        ("source_trace", source_rollout in by_rollout),
                        ("next_trace", physical_rollout in by_rollout),
                        ("debt_close", physical_rollout in log_data["debt_close"]),
                        ("gate_begin", physical_rollout in log_data["gate_begin"]),
                        ("rollout_start", physical_rollout in log_data["rollout_start"]),
                    )
                    if not present
                ]
                excluded_events.append(
                    {
                        "source_rollout": source_rollout,
                        "physical_rollout": physical_rollout,
                        "debt_target": debt_target,
                        "reason": f"missing:{','.join(missing)}",
                        "outside_analysis_window": physical_rollout > analysis_hi,
                    }
                )
            continue

        source_groups = group_rows(by_rollout[source_rollout], "group_index")
        debt_eligible = {
            group_id
            for group_id, group in source_groups.items()
            if any("ABORTED" in row["sample_status"] for row in group)
        }
        high_groups = group_rows(by_rollout[physical_rollout], "group_index")
        completions = sorted(
            (
                {
                    "group_id": group_id,
                    "debt_eligible": group_id in debt_eligible,
                    "end_epoch": group_end_epoch(group),
                }
                for group_id, group in high_groups.items()
                if group_is_useful(group)
            ),
            key=lambda item: (item["end_epoch"], item["group_id"]),
        )

        debt_done = 0
        current_done = 0
        debt_publish = None
        floor_publish = None
        previous_ids: list[int] = []
        current_ids: list[int] = []
        for completion in completions:
            if completion["debt_eligible"] and debt_done < debt_target:
                debt_done += 1
                previous_ids.append(completion["group_id"])
            else:
                current_done += 1
                current_ids.append(completion["group_id"])
            if debt_done >= debt_target and debt_publish is None:
                debt_publish = completion["end_epoch"]
            if debt_done >= debt_target and current_done >= 2 and floor_publish is None:
                floor_publish = completion["end_epoch"]

        rollout_start = log_data["rollout_start"][physical_rollout]
        actual_close_epoch = log_data["debt_close"][physical_rollout]
        gate_begin_epoch = log_data["gate_begin"][physical_rollout]
        actual_wait = max(actual_close_epoch - gate_begin_epoch, 0.0)
        debt_savings = 0.0
        floor_savings = 0.0
        floor_hit = False
        if debt_publish is not None:
            raw_debt_savings = actual_wait - max(debt_publish - gate_begin_epoch, 0.0)
            debt_timing_regressions += int(raw_debt_savings < -1.0)
            debt_savings = max(raw_debt_savings, 0.0)
            deadline = debt_publish + 2.0
            selected_publish = deadline
            if floor_publish is not None and floor_publish <= deadline:
                selected_publish = floor_publish
                floor_hit = True
            floor_savings = actual_wait - max(selected_publish - gate_begin_epoch, 0.0)
        quorum_savings += debt_savings
        soft_floor_savings += floor_savings
        soft_floor_hits += int(floor_hit)

        plan = plan_joint_carry_admission(
            phase=None,
            debt_target=debt_target,
            debt_committed=0,
            current_target=8,
            current_committed=0,
            resident_group_ids=[],
            carry_groups=[],
            debt_eligible_group_ids=[],
            carry_current_group_ids=[],
            fresh_current_group_ids=[],
            rollout_batch_size=8,
            max_response_length=8192,
        )
        planned_envelope = plan.debt_admit_groups + plan.fresh_admit_groups
        historical_unique_groups = len(high_groups)
        assigned = previous_ids + current_ids
        high_sample_ids = [int(row["sample_index"]) for row in by_rollout[physical_rollout]]
        identity_unique = len(high_sample_ids) == len(set(high_sample_ids))
        assignments_cover_completions = len(assigned) == len(completions)
        events.append(
            {
                "source_rollout": source_rollout,
                "physical_rollout": physical_rollout,
                "debt_target": debt_target,
                "debt_eligible_groups": len(debt_eligible),
                "historical_cumulative_unique_groups": historical_unique_groups,
                "joint_debt_admit_groups": plan.debt_admit_groups,
                "joint_fresh_admit_groups": plan.fresh_admit_groups,
                "buffer_refill_initial_admit_groups": planned_envelope,
                "joint_resident_cap": plan.resident_cap,
                "joint_work_overcommit_equivalents": plan.work_overcommit_equivalents,
                "actual_current_commit": log_data["carry_over"].get(physical_rollout, {}).get("committed"),
                "useful_completion_groups": len(completions),
                "dynamic_quorum_previous_group_ids": previous_ids,
                "dynamic_quorum_current_group_ids": current_ids,
                "assignment_unique": len(assigned) == len(set(assigned)),
                "assignments_cover_useful_completions": assignments_cover_completions,
                "physical_rollout_sample_identity_unique": identity_unique,
                "debt_quorum_hit": debt_publish is not None,
                "actual_debt_close_seconds": actual_close_epoch - rollout_start,
                "gate_begin_seconds": gate_begin_epoch - rollout_start,
                "dynamic_debt_close_seconds": debt_publish - rollout_start if debt_publish is not None else None,
                "debt_quorum_savings_upper_bound_seconds": debt_savings,
                "soft_floor_2_cap_2_hit": floor_hit,
                "soft_floor_2_cap_2_savings_upper_bound_seconds": floor_savings,
            }
        )

    return {
        "name": name,
        "path": str(path),
        "analysis_wall_seconds": analysis_wall_seconds,
        "analysis_logical_step_lo": analysis_lo,
        "analysis_logical_step_hi": analysis_hi,
        "a3_enabled": a3_enabled,
        "joint_planner_ledger_rows": joint_planner_rows,
        "permit_rows": len(rows),
        "events": events,
        "summary": {
            "events": len(events),
            "excluded_nonreplayable_events": excluded_events,
            "excluded_within_analysis_window": sum(not event["outside_analysis_window"] for event in excluded_events),
            "historical_max_cumulative_unique_groups": max(
                (event["historical_cumulative_unique_groups"] for event in events), default=0
            ),
            "buffer_refill_max_initial_admit_groups": max(
                (event["buffer_refill_initial_admit_groups"] for event in events), default=0
            ),
            "buffer_refill_resident_cap_violations": sum(
                event["buffer_refill_initial_admit_groups"] > event["joint_resident_cap"] for event in events
            ),
            "assignment_uniqueness_violations": sum(not event["assignment_unique"] for event in events),
            "assignment_coverage_violations": sum(
                not event["assignments_cover_useful_completions"] for event in events
            ),
            "sample_identity_violations": sum(
                not event["physical_rollout_sample_identity_unique"] for event in events
            ),
            "debt_timing_regressions_over_1s": debt_timing_regressions,
            "debt_quorum_hits": sum(event["debt_quorum_hit"] for event in events),
            "debt_quorum_savings_upper_bound_seconds": quorum_savings,
            "soft_floor_2_cap_2_hits": soft_floor_hits,
            "soft_floor_2_cap_2_savings_upper_bound_seconds": soft_floor_savings,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_path = args.reference_results or (
        args.artifact_root / "current_floor_lane_simulation_20260803" / "simulation_results.json"
    )
    reference = json.loads(reference_path.read_text())
    runs = [
        replay_run(
            name,
            args.artifact_root / relative_path,
            float(reference["runs"][name]["analysis_wall_seconds"]),
        )
        for name, relative_path in RUNS
    ]
    pooled_wall = sum(run["analysis_wall_seconds"] for run in runs)
    pooled_quorum = sum(run["summary"]["debt_quorum_savings_upper_bound_seconds"] for run in runs)
    pooled_soft = sum(run["summary"]["soft_floor_2_cap_2_savings_upper_bound_seconds"] for run in runs)
    coverage_ok = sum(run["summary"]["events"] for run in runs) == 19 and all(
        run["summary"]["excluded_within_analysis_window"] == 0 for run in runs
    )
    identity_ok = all(
        run["summary"]["assignment_uniqueness_violations"] == 0
        and run["summary"]["assignment_coverage_violations"] == 0
        and run["summary"]["sample_identity_violations"] == 0
        for run in runs
    )
    timing_ok = all(
        run["summary"]["debt_timing_regressions_over_1s"] == 0
        and run["summary"]["debt_quorum_hits"] == run["summary"]["events"]
        for run in runs
    )
    buffer_budget_ok = all(run["summary"]["buffer_refill_resident_cap_violations"] == 0 for run in runs)
    closed_loop_a3_available = all(run["a3_enabled"] and run["joint_planner_ledger_rows"] > 0 for run in runs)
    prerequisite_checks_ok = coverage_ok and identity_ok and timing_ok and buffer_budget_ok
    if not prerequisite_checks_ok:
        decision = "fail"
    elif not closed_loop_a3_available:
        decision = "hold"
    else:
        decision = "go"
    output = {
        "schema_version": 1,
        "reference_results": str(reference_path),
        "driver_log_timezone": "UTC+08:00",
        "evidence_scope": {
            "admission_budget": "Historical pre-A3 buffer-refill initial-plan diagnostic",
            "dynamic_quorum": "Observed completion-order upper bound",
            "limitation": (
                "The budget diagnostic starts with zero resident carry and is not comparable "
                "to cumulative historical physical-rollout attempts. It does not predict "
                "closed-loop A3 carry/KV throughput."
            ),
        },
        "runs": runs,
        "pooled": {
            "events": sum(run["summary"]["events"] for run in runs),
            "analysis_wall_seconds": pooled_wall,
            "historical_max_cumulative_unique_groups": max(
                run["summary"]["historical_max_cumulative_unique_groups"] for run in runs
            ),
            "buffer_refill_max_initial_admit_groups": max(
                run["summary"]["buffer_refill_max_initial_admit_groups"] for run in runs
            ),
            "buffer_refill_resident_cap_violations": sum(
                run["summary"]["buffer_refill_resident_cap_violations"] for run in runs
            ),
            "assignment_uniqueness_violations": sum(
                run["summary"]["assignment_uniqueness_violations"] for run in runs
            ),
            "assignment_coverage_violations": sum(run["summary"]["assignment_coverage_violations"] for run in runs),
            "sample_identity_violations": sum(run["summary"]["sample_identity_violations"] for run in runs),
            "debt_timing_regressions_over_1s": sum(run["summary"]["debt_timing_regressions_over_1s"] for run in runs),
            "debt_quorum_savings_upper_bound_seconds": pooled_quorum,
            "debt_quorum_gain_upper_bound_fraction": pooled_wall / (pooled_wall - pooled_quorum) - 1,
            "soft_floor_2_cap_2_savings_upper_bound_seconds": pooled_soft,
            "soft_floor_2_cap_2_gain_upper_bound_fraction": pooled_wall / (pooled_wall - pooled_soft) - 1,
        },
        "j1_gate": {
            "analysis_window_coverage": "pass" if coverage_ok else "fail",
            "identity_checks": "pass" if identity_ok else "fail",
            "zero_resident_buffer_refill_cap_diagnostic": "pass" if buffer_budget_ok else "fail",
            "observed_dynamic_quorum_upper_bound": "pass" if coverage_ok and identity_ok and timing_ok else "fail",
            "joint_carry_budget_replay": "pass" if closed_loop_a3_available else "blocked",
            "closed_loop_a3_trace": "available" if closed_loop_a3_available else "missing",
            "decision": decision,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
