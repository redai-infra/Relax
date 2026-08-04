# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "task22" / "analyze_phase_elastic_calibration.py"
SPEC = importlib.util.spec_from_file_location("task22_phase_elastic_calibration_analyzer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)
analyze = ANALYZER.analyze


def _gate_lines(
    *,
    gate_walls: dict[int, float] | None = None,
    step_lo: int = 5,
    step_hi: int = 14,
) -> list[str]:
    walls = gate_walls or {}
    return [
        (
            "prefix TASK22_CALIBRATION_GATE "
            f"logical_step={step} sync_id={step + 1} "
            f"t_begin={100.0 + (step - step_lo) * 10:.6f} "
            f"t_end={100.0 + (step - step_lo) * 10 + walls.get(step, 4.0):.6f} "
            f"dur={walls.get(step, 4.0):.6f}"
        )
        for step in range(step_lo, step_hi + 1)
    ]


def _actor_lines(
    *,
    gate_walls: dict[int, float] | None = None,
    step_lo: int = 5,
    step_hi: int = 14,
) -> list[str]:
    walls = gate_walls or {}
    lines = []
    for step in range(step_lo, step_hi + 1):
        gate_begin = 100.0 + (step - step_lo) * 10
        gate_end = gate_begin + walls.get(step, 4.0)
        lines.extend(
            [
                (f"prefix TASK22_CALIBRATION_ACTOR logical_step={step} phase=data_wait_begin t={gate_begin - 10:.6f}"),
                (
                    "prefix TASK22_CALIBRATION_ACTOR "
                    f"logical_step={step} phase=first_subbatch_ready t={gate_begin - 8:.6f}"
                ),
                (
                    "prefix TASK22_CALIBRATION_ACTOR "
                    f"logical_step={step} phase=last_subbatch_ready t={gate_begin - 6:.6f}"
                ),
                (
                    "prefix TASK22_CALIBRATION_ACTOR "
                    f"logical_step={step} phase=actor_train "
                    f"t_begin={gate_begin - 5:.6f} t_end={gate_begin - 2:.6f} dur=3.000000"
                ),
                (
                    "prefix TASK22_CALIBRATION_ACTOR "
                    f"logical_step={step} phase=weight_sync "
                    f"t_begin={gate_end + 0.5:.6f} t_end={gate_end + 1.5:.6f} dur=1.000000"
                ),
            ]
        )
    return lines


def _standard_metric_lines(*, step_lo: int = 5, step_hi: int = 14) -> list[str]:
    lines = []
    for step in range(step_lo, step_hi + 1):
        actor_perf = {
            "perf/step_time": 100.0 + step,
            "perf/train_wait_time": 50.0 + step,
            "perf/log_probs_time": 12.0,
            "perf/actor_train_time": 38.0,
            "perf/train_time": 39.0,
            "perf/train_get_data_time": 0.1,
            "perf/update_weights_time": 3.5,
            "perf/step_token_per_s": 4000.0,
        }
        train_quality = {
            "train/tis": 1.0,
            "train/tis_clipfrac": 0.0,
            "train/tis_abs": 0.01,
            "train/train_rollout_logprob_abs_diff": 0.01,
            "train/mismatch_kl": 0.0005,
            "train/mismatch_k3_kl": 0.0006,
        }
        rollout_data = {
            "rollout/response_lengths": 6000.0 + step,
            "rollout/raw_reward": -0.75,
            "rollout/rewards": 0.0,
            "rollout/total_lengths/max": 8400.0,
            "rollout/total_lengths/min": 1800.0,
        }
        lines.extend(
            [
                f"actor perf {step}: {actor_perf!r}",
                f"actor step {step}: {train_quality!r}",
                f"actor rollout {step}: {rollout_data!r}",
            ]
        )
    return lines


def _scheduler_event(
    pid: int,
    timestamp: float,
    *,
    running_rids: list[str],
    queued_rids: list[str],
    forward_mode: str = "ForwardMode.DECODE",
    idle: bool | None = None,
    decode_tokens_cumulative: int | None = None,
    prefill_tokens_cumulative: int | None = None,
    cached_tokens_cumulative: int | None = None,
    include_decode_tokens_cumulative: bool = True,
    include_prefill_tokens_cumulative: bool = True,
    include_cached_tokens_cumulative: bool = True,
    timestamp_iso: str | None = None,
    include_timestamp_epoch: bool = True,
) -> str:
    rendered_timestamp = timestamp_iso or datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    event = {
        "timestamp": rendered_timestamp,
        "event": "scheduler.status",
        "forward_mode": forward_mode,
        "idle": (not running_rids and not queued_rids) if idle is None else idle,
        "running_rids": running_rids,
        "running_seq_lens": [100 + index for index in range(len(running_rids))],
        "running_origin_input_lens": [40 + index for index in range(len(running_rids))],
        "running_output_lens": [60 for _ in running_rids],
        "queued_rids": queued_rids,
        "queued_origin_input_lens": [30 + index for index in range(len(queued_rids))],
        "queued_output_lens": [5 for _ in queued_rids],
    }
    if include_timestamp_epoch:
        event["timestamp_epoch"] = timestamp
    if include_decode_tokens_cumulative:
        event["decode_tokens_cumulative"] = (
            int(round(timestamp * 10)) if decode_tokens_cumulative is None else decode_tokens_cumulative
        )
    if include_prefill_tokens_cumulative:
        event["prefill_tokens_cumulative"] = (
            int(round(timestamp * 5)) if prefill_tokens_cumulative is None else prefill_tokens_cumulative
        )
    if include_cached_tokens_cumulative:
        event["cached_tokens_cumulative"] = (
            int(round(timestamp * 2)) if cached_tokens_cumulative is None else cached_tokens_cumulative
        )
    return f"(SGLangEngine pid={pid}) {json.dumps(event, sort_keys=True)}"


def _scheduler_lines(
    *,
    engine_pids: tuple[int, ...] = (100, 200),
    pre_age: float = 0.5,
    first_pre_rids: dict[int, tuple[list[str], list[str]]] | None = None,
    first_interval_rids: dict[int, tuple[list[str], list[str]]] | None = None,
    step_lo: int = 5,
    step_hi: int = 14,
) -> list[str]:
    lines = []
    for step in range(step_lo, step_hi + 1):
        gate_begin = 100.0 + (step - step_lo) * 10
        for pid in engine_pids:
            if step == 5 and first_pre_rids and pid in first_pre_rids:
                pre_running_rids, pre_queued_rids = first_pre_rids[pid]
            else:
                pre_running_rids = [f"HEALTH_CHECK_pre-running-{pid}-{step}"]
                pre_queued_rids = [f"HEALTH_CHECK_pre-queued-{pid}-{step}"]
            lines.append(
                _scheduler_event(
                    pid,
                    gate_begin - pre_age,
                    running_rids=pre_running_rids,
                    queued_rids=pre_queued_rids,
                )
            )
            if step == 5 and first_interval_rids and pid in first_interval_rids:
                running_rids, queued_rids = first_interval_rids[pid]
            else:
                running_rids = [f"HEALTH_CHECK_gate-running-{pid}-{step}"]
                queued_rids = [f"HEALTH_CHECK_gate-queued-{pid}-{step}"]
            lines.append(
                _scheduler_event(
                    pid,
                    gate_begin + 1.0,
                    running_rids=running_rids,
                    queued_rids=queued_rids,
                )
            )
    return lines


def _wait(
    wait_id: str,
    *,
    begin: float,
    wait_end: float,
    status: str = "terminal",
    kind: str = "fresh",
    group: int = 1,
    context: int | None = 100,
    generated: int | None = 20,
    request_wall: float | None = 3.0,
    sglang_timing: bool = True,
    exception_type: str | None = None,
    physical_rollout_id: int = 1,
) -> dict:
    row = {
        "schema_version": 1,
        "record_type": "permit_wait",
        "permit_wait_id": wait_id,
        "physical_rollout_id": physical_rollout_id,
        "group_index": group,
        "sample_index": group * 10,
        "attempt_kind": kind,
        "response_tokens_before": 10 if kind == "resume" else 0,
        "context_tokens_before": context,
        "permit_acquire_begin_epoch": begin,
        "permit_wait_status": status,
        "generated_tokens_this_attempt": generated,
        "request_wall_seconds": request_wall,
        "exception_type": exception_type,
        "engine_request_started": status == "terminal",
    }
    if status == "terminal":
        row["permit_acquire_granted_epoch"] = wait_end
        row["terminal_epoch"] = wait_end + (request_wall or 0.0)
    else:
        row["terminal_epoch"] = wait_end
    if status == "terminal" and sglang_timing and exception_type is None:
        _with_sglang_timing(row, forward_entry=1000.0 + group)
    return row


def _with_sglang_timing(
    row: dict,
    *,
    queue: float = 0.2,
    forward_entry: float = 1000.0,
    prefill: float = 0.3,
) -> dict:
    row.update(
        {
            "sglang_queue_seconds": queue,
            "sglang_forward_entry_epoch": forward_entry,
            "sglang_prefill_finished_epoch": forward_entry + prefill,
            "sglang_prefill_seconds": prefill,
            "sglang_queue_plus_prefill_seconds": queue + prefill,
        }
    )
    return row


def _write_manifest(calibration: Path) -> None:
    files = []
    for path in sorted(calibration.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "PERSISTED"}:
            payload = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(calibration).as_posix(),
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    (calibration / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (calibration / "PERSISTED").write_text("verified\n", encoding="utf-8")


def _write_inputs(
    tmp_path,
    *,
    gate_lines=None,
    actor_lines=None,
    scheduler_lines=None,
    waits=None,
    permit_rollout_ids=None,
    run_contract=None,
    run_lifecycle=None,
    status="SUCCEEDED",
    fill_permit_rollouts=True,
    standard_metric_lines=None,
):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    driver = calibration / "driver.log"
    contract = run_contract or {
        "schema_version": 1,
        "workload": {"num_rollout": 16},
        "headline": {"logical_step_lo": 5, "logical_step_hi": 14},
    }
    contract_headline = contract.get("headline", {})
    metric_step_lo = contract_headline.get("logical_step_lo", 5)
    metric_step_hi = contract_headline.get("logical_step_hi", 14)
    driver_rows = [
        *(_gate_lines() if gate_lines is None else gate_lines),
        *(_actor_lines() if actor_lines is None else actor_lines),
        *(_scheduler_lines() if scheduler_lines is None else scheduler_lines),
        *(
            _standard_metric_lines(step_lo=metric_step_lo, step_hi=metric_step_hi)
            if standard_metric_lines is None
            else standard_metric_lines
        ),
    ]
    driver.write_text("\n".join(driver_rows) + "\n", encoding="utf-8")
    rows = [] if waits is None else waits
    rollout_ids = list(range(16)) if permit_rollout_ids is None else permit_rollout_ids
    for rollout_id in rollout_ids:
        rollout_rows = [row for row in rows if row.get("physical_rollout_id") == rollout_id]
        if fill_permit_rollouts and not rollout_rows:
            rollout_rows = [
                _wait(
                    f"fixture-rollout-{rollout_id}",
                    begin=10.0 + rollout_id,
                    wait_end=10.5 + rollout_id,
                    group=1000 + rollout_id,
                    physical_rollout_id=rollout_id,
                )
            ]
        (calibration / f"permit_wait_rollout_{rollout_id}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rollout_rows),
            encoding="utf-8",
        )
    lifecycle = run_lifecycle or {"schema_version": 1, "status": "SUCCEEDED", "exit_code": 0}
    (calibration / "run_contract.json").write_text(json.dumps(contract) + "\n", encoding="utf-8")
    (calibration / "run_lifecycle.json").write_text(json.dumps(lifecycle) + "\n", encoding="utf-8")
    (calibration / "STATUS").write_text(status + "\n", encoding="utf-8")
    _write_manifest(calibration)
    return driver, calibration


def test_analyzer_measures_gate_waiters_overlap_cancel_and_inline_metrics(tmp_path) -> None:
    resume = _wait("resume", begin=99.0, wait_end=102.0, kind="resume", group=7, context=140, generated=60)
    cancelled = _wait(
        "cancelled",
        begin=101.0,
        wait_end=103.0,
        status="cancelled_before_grant",
        kind="fresh",
        group=8,
        context=80,
        generated=0,
        request_wall=None,
    )
    before_gate = _wait("before", begin=97.0, wait_end=98.0, group=9)
    driver, calibration = _write_inputs(tmp_path, waits=[resume, cancelled, before_gate])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    assert result["headline"]["mapping"] == "sync_id = logical_step + 1"
    gate = result["gates"][0]
    assert (gate["logical_step"], gate["sync_id"]) == (5, 6)
    assert gate["waiting_at_gate_begin"]["request_count"] == 1
    assert gate["waiting_at_gate_begin"]["attempt_kind_counts"]["resume"] == 1
    assert gate["waiting_at_gate_begin"]["unique_group_count"] == 1
    assert gate["permit_wait_overlap"]["request_count"] == 2
    assert gate["permit_wait_overlap"]["waiter_slot_seconds"] == pytest.approx(4.0)
    assert gate["permit_wait_overlap"]["gate_seconds_with_any_waiter"] == pytest.approx(3.0)
    assert gate["permit_wait_overlap"]["gate_time_coverage_ratio"] == pytest.approx(0.75)
    assert gate["cancelled_before_grant_inside_gate"]["request_count"] == 1
    metrics = gate["permit_wait_overlap"]["inline_metrics"]
    assert metrics["context_tokens_before"]["sum"] == 220
    assert metrics["generated_tokens_this_attempt"]["sum"] == 60
    assert metrics["request_wall_seconds"]["eligible_count"] == 1
    assert metrics["request_wall_seconds"]["available_count"] == 1


def test_analyzer_marks_missing_inline_fields_unavailable_without_inference(tmp_path) -> None:
    row = _wait(
        "missing",
        begin=99.0,
        wait_end=102.0,
        context=None,
        generated=None,
        request_wall=None,
    )
    driver, calibration = _write_inputs(tmp_path, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    metrics = result["gates"][0]["permit_wait_overlap"]["inline_metrics"]
    assert metrics["context_tokens_before"]["availability"] == "unavailable"
    assert metrics["generated_tokens_this_attempt"]["availability"] == "unavailable"
    assert metrics["request_wall_seconds"]["availability"] == "unavailable"
    assert "sum" not in metrics["request_wall_seconds"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong_mapping"])
def test_analyzer_rejects_incomplete_nonunique_or_mismapped_headline_gate(tmp_path, defect) -> None:
    lines = _gate_lines()
    target = next(line for line in lines if "logical_step=10 " in line)
    if defect == "missing":
        lines.remove(target)
    elif defect == "duplicate":
        lines.append(target)
    else:
        lines[lines.index(target)] = target.replace("sync_id=11", "sync_id=12")
    driver, calibration = _write_inputs(tmp_path, gate_lines=lines, waits=[])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "incomplete_or_nonunique_headline_gates" in result["errors"][0]


def test_analyzer_rejects_nonterminal_permit_row(tmp_path) -> None:
    row = _wait("waiting", begin=99.0, wait_end=102.0)
    row["permit_wait_status"] = "waiting"
    row.pop("permit_acquire_granted_epoch")
    driver, calibration = _write_inputs(tmp_path, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "nonterminal_or_unknown_wait" in result["errors"][0]


def test_analyzer_rejects_empty_permit_artifacts(tmp_path) -> None:
    driver, calibration = _write_inputs(tmp_path, waits=[], fill_permit_rollouts=False)

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "invalid_permit_physical_rollout_coverage" in result["errors"][0]
    assert "empty=" in result["errors"][0]


def test_analyzer_summarizes_two_engines_at_gate_begin_and_inside_gate(tmp_path) -> None:
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    assert result["scheduler"]["engine_pids"] == ["100", "200"]
    assert result["scheduler"]["health_threshold_seconds"] == 5.0
    scheduler = result["gates"][0]["scheduler"]
    assert scheduler["engine_count"] == 2
    for engine in scheduler["engines"].values():
        nearest = engine["nearest_at_or_before_gate_begin"]
        assert nearest["age_at_gate_begin_seconds"] == pytest.approx(0.5)
        assert nearest["running_request_count"] == 1
        assert nearest["queued_request_count"] == 1
        assert nearest["running_seq_lens"]["sum"] == 100
        interval = engine["gate_interval"]
        assert interval["snapshot_count"] == 1
        assert interval["forward_mode_counts"] == {"ForwardMode.DECODE": 1}
        assert interval["running_seq_len_sum_per_snapshot"]["sum"] == 100
        assert interval["running_origin_input_len_sum_per_snapshot"]["sum"] == 40
        assert interval["running_output_len_sum_per_snapshot"]["sum"] == 60


def test_analyzer_rejects_missing_scheduler_engine(tmp_path) -> None:
    scheduler_lines = _scheduler_lines(engine_pids=(100,))
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, scheduler_lines=scheduler_lines, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "scheduler_engine_count" in result["errors"][0]


def test_analyzer_reports_stale_short_gate_without_invalidating_run(tmp_path) -> None:
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(
        tmp_path,
        scheduler_lines=_scheduler_lines(pre_age=6.0),
        waits=[row],
    )

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    scheduler = result["gates"][0]["scheduler"]
    assert not scheduler["is_long_gate"]
    assert scheduler["engines"]["100"]["nearest_at_or_before_gate_begin"]["health_status"] == "stale_active_state"


def test_analyzer_reports_missing_short_gate_snapshot_without_invalidating_run(tmp_path) -> None:
    scheduler_lines = _scheduler_lines()
    scheduler_lines.pop(0)  # engine 100 has no snapshot before the first short gate
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, scheduler_lines=scheduler_lines, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    nearest = result["gates"][0]["scheduler"]["engines"]["100"]["nearest_at_or_before_gate_begin"]
    assert nearest == {
        "availability": "unavailable",
        "health_status": "missing",
        "required_for_validity": False,
        "freshness_required": False,
    }


@pytest.mark.parametrize("defect", ["stale", "missing"])
def test_analyzer_rejects_stale_or_missing_snapshot_for_long_gate(tmp_path, defect) -> None:
    gate_walls = {5: 6.0}
    scheduler_lines = _scheduler_lines(pre_age=6.0) if defect == "stale" else _scheduler_lines()
    if defect == "missing":
        scheduler_lines.pop(0)
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(
        tmp_path,
        gate_lines=_gate_lines(gate_walls=gate_walls),
        actor_lines=_actor_lines(gate_walls=gate_walls),
        scheduler_lines=scheduler_lines,
        waits=[row],
    )

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    expected = (
        "incomplete_scheduler_heartbeat_during_long_gate"
        if defect == "stale"
        else "missing_scheduler_snapshot_before_long_gate"
    )
    assert expected in result["errors"][0]


def test_analyzer_maps_scheduler_rid_only_by_exact_permit_wait_id(tmp_path) -> None:
    fresh = _wait("permit-fresh", begin=99.0, wait_end=102.0, kind="fresh", group=1)
    resume = _wait("permit-resume", begin=99.0, wait_end=102.0, kind="resume", group=2)
    scheduler_lines = _scheduler_lines(
        first_interval_rids={
            100: ([fresh["permit_wait_id"], "HEALTH_CHECK_unmatched-rid"], [resume["permit_wait_id"]]),
        }
    )
    driver, calibration = _write_inputs(
        tmp_path,
        scheduler_lines=scheduler_lines,
        waits=[fresh, resume],
    )

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    classification = result["gates"][0]["scheduler"]["engines"]["100"]["gate_interval"][
        "all_rid_occurrence_classification"
    ]
    assert classification["rid_count"] == 3
    assert classification["external_rid_count"] == 2
    assert classification["health_check_rid_count"] == 1
    assert classification["exact_permit_id_match_count"] == 2
    assert classification["exact_permit_id_match_coverage_ratio"] == 1.0
    assert classification["kind_counts"] == {
        "fresh": 1,
        "resume": 1,
        "unknown_external": 0,
        "health_check": 1,
    }


def test_analyzer_accepts_complete_long_gate_rid_mapping_and_excludes_health_checks(tmp_path) -> None:
    gate_walls = {5: 6.0}
    fresh = _wait("permit-fresh", begin=99.0, wait_end=102.0, kind="fresh", group=1)
    resume = _wait("permit-resume", begin=99.0, wait_end=102.0, kind="resume", group=2)
    scheduler_lines = _scheduler_lines(
        first_pre_rids={
            100: ([fresh["permit_wait_id"], "HEALTH_CHECK_probe"], []),
            200: ([resume["permit_wait_id"]], []),
        },
        first_interval_rids={
            100: ([fresh["permit_wait_id"]], []),
            200: ([resume["permit_wait_id"]], []),
        },
    )
    driver, calibration = _write_inputs(
        tmp_path,
        gate_lines=_gate_lines(gate_walls=gate_walls),
        actor_lines=_actor_lines(gate_walls=gate_walls),
        scheduler_lines=scheduler_lines,
        waits=[fresh, resume],
    )

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    health = result["scheduler"]["headline_rid_health"]
    assert health["long_gate_count"] == 1
    assert health["external_unique_rid_count"] == 2
    assert health["health_check_unique_rid_count"] > 0
    assert health["exact_permit_id_match_coverage_ratio"] == 1.0
    assert health["cross_engine_external_rid_count"] == 0
    assert health["status"] == "pass"


@pytest.mark.parametrize("lost_mode", ["zero", "partial"])
def test_analyzer_rejects_lost_headline_scheduler_rid(tmp_path, lost_mode) -> None:
    matched = _wait("permit-matched", begin=99.0, wait_end=102.0)
    running = ["lost-rid"] if lost_mode == "zero" else [matched["permit_wait_id"], "lost-rid"]
    scheduler_lines = _scheduler_lines(
        first_pre_rids={100: (running, []), 200: (["HEALTH_CHECK_probe-200"], [])},
        first_interval_rids={100: (running, []), 200: (["HEALTH_CHECK_probe-200"], [])},
    )
    driver, calibration = _write_inputs(tmp_path, scheduler_lines=scheduler_lines, waits=[matched])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    health = result["scheduler"]["headline_rid_health"]
    expected_status = "invalid_zero_exact_matches" if lost_mode == "zero" else "invalid_below_minimum_coverage"
    assert health["status"] == expected_status
    assert health["minimum_required_coverage_ratio"] == 1.0


def test_analyzer_rejects_one_scheduler_rid_observed_on_both_engines(tmp_path) -> None:
    row = _wait("permit-shared", begin=99.0, wait_end=102.0)
    scheduler_lines = _scheduler_lines(
        first_pre_rids={100: ([row["permit_wait_id"]], []), 200: ([row["permit_wait_id"]], [])},
        first_interval_rids={100: ([row["permit_wait_id"]], []), 200: ([row["permit_wait_id"]], [])},
    )
    driver, calibration = _write_inputs(tmp_path, scheduler_lines=scheduler_lines, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    health = result["scheduler"]["headline_rid_health"]
    assert health["status"] == "invalid_cross_engine_rid_ownership"
    assert health["cross_engine_external_rid_count"] == 1


def test_analyzer_summarizes_actor_timeline_and_idle_window(tmp_path) -> None:
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, waits=[row])

    result = analyze(driver, calibration)

    timeline = result["gates"][0]["actor_timeline"]
    assert timeline["first_subbatch_wait_seconds"] == 2.0
    assert timeline["all_subbatches_ready_wait_seconds"] == 4.0
    assert timeline["remaining_subbatches_after_first_seconds"] == 2.0
    assert timeline["post_last_subbatch_forward_prepare_seconds"] == 1.0
    assert timeline["actor_train_seconds"] == 3.0
    assert timeline["gate_seconds"] == 4.0
    assert timeline["weight_sync_seconds"] == 1.0
    assert timeline["actor_idle_before_weight_sync_seconds"] == 6.5
    assert result["summary"]["actor_timeline"]["actor_idle_before_weight_sync_seconds"]["mean"] == 6.5


@pytest.mark.parametrize("defect", ["missing", "duplicate", "reversed"])
def test_analyzer_rejects_missing_duplicate_or_reversed_actor_phase(tmp_path, defect) -> None:
    actor_lines = _actor_lines()
    target = next(line for line in actor_lines if "logical_step=10 " in line and "phase=first_subbatch_ready " in line)
    if defect == "missing":
        actor_lines.remove(target)
    elif defect == "duplicate":
        actor_lines.append(target)
    else:
        actor_lines[actor_lines.index(target)] = target.replace("t=142.000000", "t=139.000000")
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, actor_lines=actor_lines, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "invalid_actor_timeline" in result["errors"][0]


@pytest.mark.parametrize("defect", ["schema", "rollout_id"])
def test_analyzer_rejects_permit_schema_or_filename_rollout_mismatch(tmp_path, defect) -> None:
    row = _wait("wait", begin=99.0, wait_end=102.0)
    if defect == "schema":
        row["schema_version"] = 2
    driver, calibration = _write_inputs(tmp_path, waits=[row])
    if defect == "rollout_id":
        row["physical_rollout_id"] = 2
        (calibration / "permit_wait_rollout_1.jsonl").write_text(
            json.dumps(row, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_manifest(calibration)

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    expected = "invalid_permit_schema_version" if defect == "schema" else "permit_artifact_rollout_mismatch"
    assert expected in result["errors"][0]


def test_analyzer_rejects_stale_idle_heartbeat_during_long_gate(tmp_path) -> None:
    gate_walls = {5: 6.0}
    scheduler_lines = _scheduler_lines()
    scheduler_lines = scheduler_lines[2:]
    scheduler_lines.append(
        _scheduler_event(
            100,
            90.0,
            running_rids=[],
            queued_rids=[],
            idle=True,
            decode_tokens_cumulative=900,
        )
    )
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(
        tmp_path,
        gate_lines=_gate_lines(gate_walls=gate_walls),
        actor_lines=_actor_lines(gate_walls=gate_walls),
        scheduler_lines=scheduler_lines,
        waits=[row],
    )

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "incomplete_scheduler_heartbeat_during_long_gate" in result["errors"][0]
    assert "heartbeat_start_gap" in result["errors"][0]


@pytest.mark.parametrize("transition", ["idle_to_active", "active_to_idle"])
def test_analyzer_tracks_idle_active_transitions_through_long_gate(tmp_path, transition) -> None:
    gate_walls = {5: 6.0}
    scheduler_lines = _scheduler_lines()
    if transition == "idle_to_active":
        scheduler_lines[0] = _scheduler_event(
            100,
            99.5,
            running_rids=[],
            queued_rids=[],
            idle=True,
            decode_tokens_cumulative=995,
        )
    else:
        scheduler_lines[1] = _scheduler_event(
            100,
            101.0,
            running_rids=[],
            queued_rids=[],
            idle=True,
            decode_tokens_cumulative=1010,
        )
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(
        tmp_path,
        gate_lines=_gate_lines(gate_walls=gate_walls),
        actor_lines=_actor_lines(gate_walls=gate_walls),
        scheduler_lines=scheduler_lines,
        waits=[row],
    )

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    continuity = result["gates"][0]["scheduler"]["engines"]["100"]["state_continuity"]
    if transition == "idle_to_active":
        assert continuity["initial_state"] == "idle"
        assert continuity["final_state"] == "active"
        assert continuity["active_transition_count"] == 1
    else:
        assert continuity["initial_state"] == "active"
        assert continuity["final_state"] == "idle"
        assert continuity["idle_transition_count"] == 1


def test_analyzer_rejects_active_scheduler_logger_death_during_long_gate(tmp_path) -> None:
    gate_walls = {5: 6.0}
    scheduler_lines = _scheduler_lines()
    scheduler_lines.pop(1)  # engine 100 remains active but has no status during the long gate
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(
        tmp_path,
        gate_lines=_gate_lines(gate_walls=gate_walls),
        actor_lines=_actor_lines(gate_walls=gate_walls),
        scheduler_lines=scheduler_lines,
        waits=[row],
    )

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "incomplete_scheduler_heartbeat_during_long_gate" in result["errors"][0]
    assert "heartbeat_end_gap" in result["errors"][0]


def test_analyzer_reports_exact_token_progress_per_gate_and_headline(tmp_path) -> None:
    scheduler_lines = _scheduler_lines()
    scheduler_lines.extend(
        [
            _scheduler_event(
                pid,
                102.5,
                running_rids=[f"HEALTH_CHECK_gate-running-{pid}-5"],
                queued_rids=[],
                decode_tokens_cumulative=1025,
            )
            for pid in (100, 200)
        ]
    )
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, scheduler_lines=scheduler_lines, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    gate_progress = result["gates"][0]["scheduler"]["engines"]["100"]["gate_interval"]["token_progress_exact"]
    assert gate_progress["availability"] == "available"
    assert gate_progress["interval_count"] == 1
    assert gate_progress["decode_tokens"] == 15
    assert gate_progress["decode_tok_per_s"] == 10.0
    assert gate_progress["prefill_tokens"] == 7
    assert gate_progress["cached_tokens"] == 3
    assert gate_progress["cache_hit_ratio"] == 0.3
    assert gate_progress["boundary_crossing_interval_count_excluded"] == 2
    headline_progress = result["scheduler"]["headline_interval_token_progress_exact"]["100"]
    assert headline_progress["availability"] == "available"
    assert headline_progress["decode_tok_per_s"] == 10.0


@pytest.mark.parametrize(
    "defect",
    [
        "missing_counter",
        "missing_prefill_counter",
        "nonmonotonic_counter",
        "nonmonotonic_cached",
        "missing_epoch",
        "string_epoch",
    ],
)
def test_analyzer_rejects_invalid_scheduler_decode_or_epoch_contract(tmp_path, defect) -> None:
    scheduler_lines = _scheduler_lines()
    if defect == "missing_counter":
        scheduler_lines[0] = _scheduler_event(
            100,
            99.5,
            running_rids=["HEALTH_CHECK_pre-running-100-5"],
            queued_rids=["HEALTH_CHECK_pre-queued-100-5"],
            include_decode_tokens_cumulative=False,
        )
    elif defect == "missing_prefill_counter":
        scheduler_lines[0] = _scheduler_event(
            100,
            99.5,
            running_rids=["HEALTH_CHECK_pre-running-100-5"],
            queued_rids=["HEALTH_CHECK_pre-queued-100-5"],
            include_prefill_tokens_cumulative=False,
        )
    elif defect == "nonmonotonic_counter":
        scheduler_lines[1] = _scheduler_event(
            100,
            101.0,
            running_rids=["HEALTH_CHECK_gate-running-100-5"],
            queued_rids=["HEALTH_CHECK_gate-queued-100-5"],
            decode_tokens_cumulative=900,
        )
    elif defect == "nonmonotonic_cached":
        scheduler_lines[1] = _scheduler_event(
            100,
            101.0,
            running_rids=["HEALTH_CHECK_gate-running-100-5"],
            queued_rids=["HEALTH_CHECK_gate-queued-100-5"],
            cached_tokens_cumulative=100,
        )
    elif defect == "missing_epoch":
        scheduler_lines[0] = _scheduler_event(
            100,
            99.5,
            running_rids=["HEALTH_CHECK_pre-running-100-5"],
            queued_rids=["HEALTH_CHECK_pre-queued-100-5"],
            include_timestamp_epoch=False,
        )
    else:
        event_text = scheduler_lines[0]
        payload = json.loads(event_text[event_text.index("{") :])
        payload["timestamp_epoch"] = "99.5"
        scheduler_lines[0] = f"(SGLangEngine pid=100) {json.dumps(payload, sort_keys=True)}"
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, scheduler_lines=scheduler_lines, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    expected = {
        "missing_counter": "invalid_scheduler_cumulative_tokens",
        "missing_prefill_counter": "invalid_scheduler_cumulative_tokens",
        "nonmonotonic_counter": "nonmonotonic_scheduler_cumulative_tokens",
        "nonmonotonic_cached": "nonmonotonic_scheduler_cumulative_tokens",
        "missing_epoch": "invalid_scheduler_timestamp_epoch",
        "string_epoch": "invalid_scheduler_timestamp_epoch",
    }[defect]
    assert expected in result["errors"][0]


def test_analyzer_accepts_real_naive_scheduler_timestamp_with_authoritative_epoch(tmp_path) -> None:
    scheduler_lines = _scheduler_lines()
    scheduler_lines[0] = _scheduler_event(
        100,
        99.5,
        running_rids=["HEALTH_CHECK_pre-running-100-5"],
        queued_rids=["HEALTH_CHECK_pre-queued-100-5"],
        timestamp_iso="1970-01-01T00:01:39.500000",
    )
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, scheduler_lines=scheduler_lines, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    nearest = result["gates"][0]["scheduler"]["engines"]["100"]["nearest_at_or_before_gate_begin"]
    assert nearest["timestamp_epoch"] == 99.5
    assert nearest["timestamp_iso"] == "1970-01-01T00:01:39.500000"


def test_analyzer_rejects_idle_true_with_nonempty_scheduler_work(tmp_path) -> None:
    scheduler_lines = _scheduler_lines()
    scheduler_lines[0] = _scheduler_event(
        100,
        99.5,
        running_rids=["HEALTH_CHECK_impossible-running"],
        queued_rids=[],
        idle=True,
    )
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, scheduler_lines=scheduler_lines, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "inconsistent_scheduler_idle_state" in result["errors"][0]


def test_analyzer_rejects_non_bool_scheduler_idle(tmp_path) -> None:
    scheduler_lines = _scheduler_lines()
    event_text = scheduler_lines[0]
    payload = json.loads(event_text[event_text.index("{") :])
    payload["idle"] = 0
    scheduler_lines[0] = f"(SGLangEngine pid=100) {json.dumps(payload, sort_keys=True)}"
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, scheduler_lines=scheduler_lines, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "invalid_scheduler_idle" in result["errors"][0]


@pytest.mark.parametrize("defect", ["lifecycle", "status", "contract_rollout", "contract_headline"])
def test_analyzer_rejects_unsuccessful_or_mismatched_run_contract(tmp_path, defect) -> None:
    lifecycle = {"schema_version": 1, "status": "SUCCEEDED", "exit_code": 0}
    contract = {
        "schema_version": 1,
        "workload": {"num_rollout": 16},
        "headline": {"logical_step_lo": 5, "logical_step_hi": 14},
    }
    status = "SUCCEEDED"
    if defect == "lifecycle":
        lifecycle.update({"status": "FAILED", "exit_code": 1})
    elif defect == "status":
        status = "FAILED exit_code=1"
    elif defect == "contract_rollout":
        contract["workload"]["num_rollout"] = 14
    else:
        contract["headline"]["logical_step_hi"] = 13
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(
        tmp_path,
        waits=[row],
        run_contract=contract,
        run_lifecycle=lifecycle,
        status=status,
    )

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"


def test_analyzer_accepts_short_zero_kl_diagnostic_contract(tmp_path) -> None:
    contract = {
        "schema_version": 1,
        "workload": {"num_rollout": 11},
        "headline": {"logical_step_lo": 2, "logical_step_hi": 9},
    }
    driver, calibration = _write_inputs(
        tmp_path,
        gate_lines=_gate_lines(step_lo=2, step_hi=9),
        actor_lines=_actor_lines(step_lo=2, step_hi=9),
        scheduler_lines=_scheduler_lines(step_lo=2, step_hi=9),
        permit_rollout_ids=list(range(11)),
        run_contract=contract,
    )

    result = analyze(driver, calibration, headline_lo=2, headline_hi=9)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    assert result["headline"]["logical_step_lo"] == 2
    assert result["headline"]["logical_step_hi"] == 9
    assert result["inputs"]["permit_physical_rollout_coverage"]["required_physical_rollout_ids"] == list(range(11))
    headline = result["summary"]["pipeline_headline"]
    assert headline["actor_perf"]["perf/step_time"]["count"] == 8
    assert headline["actor_perf"]["perf/train_wait_time"]["count"] == 8
    assert headline["train_quality"]["train/tis"]["mean"] == 1.0
    assert headline["rollout_data"]["rollout/response_lengths"]["count"] == 8
    assert headline["zero_kl_reference_metric_absence"] == "pass"


@pytest.mark.parametrize("defect", ["missing_perf", "duplicate_perf", "ref_perf", "ref_rollout"])
def test_analyzer_rejects_incomplete_or_non_zero_kl_standard_metrics(tmp_path, defect) -> None:
    lines = _standard_metric_lines()
    target_index = next(index for index, line in enumerate(lines) if line.startswith("actor perf 7:"))
    if defect == "missing_perf":
        lines.pop(target_index)
    elif defect == "duplicate_perf":
        lines.append(lines[target_index])
    elif defect == "ref_perf":
        payload = ast.literal_eval(lines[target_index].split(": ", maxsplit=1)[1])
        payload["perf/ref_log_probs_time"] = 12.0
        lines[target_index] = f"actor perf 7: {payload!r}"
    else:
        target_index = next(index for index, line in enumerate(lines) if line.startswith("actor rollout 7:"))
        payload = ast.literal_eval(lines[target_index].split(": ", maxsplit=1)[1])
        payload["rollout/ref_log_probs"] = -0.3
        lines[target_index] = f"actor rollout 7: {payload!r}"
    driver, calibration = _write_inputs(tmp_path, standard_metric_lines=lines)

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    expected = (
        "incomplete_or_nonunique_actor_perf"
        if defect in {"missing_perf", "duplicate_perf"}
        else "zero_kl_reference_metrics_present"
    )
    assert expected in result["errors"][0]


@pytest.mark.parametrize(
    ("rollout_ids", "expected_verdict"),
    [
        (list(range(16)), "RECALIBRATE_REQUIRED"),
        (list(range(17)), "RECALIBRATE_REQUIRED"),
        (list(range(1, 16)), "INVALID_INPUT"),
        ([*range(16), 17], "INVALID_INPUT"),
    ],
)
def test_analyzer_enforces_permit_physical_rollout_coverage(tmp_path, rollout_ids, expected_verdict) -> None:
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, waits=[row], permit_rollout_ids=rollout_ids)

    result = analyze(driver, calibration)

    assert result["verdict"] == expected_verdict
    if expected_verdict == "RECALIBRATE_REQUIRED":
        coverage = result["inputs"]["permit_physical_rollout_coverage"]
        assert coverage["observed_physical_rollout_ids"] == rollout_ids
        assert coverage["final_backfill_present"] == (16 in rollout_ids)
        assert all(count > 0 for count in coverage["per_physical_rollout_row_counts"].values())
    else:
        assert "invalid_permit_physical_rollout_coverage" in result["errors"][0]


def test_analyzer_rejects_empty_optional_final_backfill_artifact(tmp_path) -> None:
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, waits=[row], permit_rollout_ids=list(range(17)))
    (calibration / "permit_wait_rollout_16.jsonl").write_text("", encoding="utf-8")
    _write_manifest(calibration)

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "invalid_permit_physical_rollout_coverage" in result["errors"][0]
    assert "empty=[16]" in result["errors"][0]


@pytest.mark.parametrize("defect", ["marker", "tamper", "missing_driver", "unlisted_permit"])
def test_analyzer_rejects_unverified_or_incomplete_persistence(tmp_path, defect) -> None:
    row = _wait("wait", begin=99.0, wait_end=102.0)
    driver, calibration = _write_inputs(tmp_path, waits=[row])
    if defect == "marker":
        (calibration / "PERSISTED").write_text("copying\n", encoding="utf-8")
    elif defect == "tamper":
        (calibration / "STATUS").write_text("SUCCEEDED\ntampered\n", encoding="utf-8")
    else:
        manifest_path = calibration / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        omitted = "driver.log" if defect == "missing_driver" else "permit_wait_rollout_5.jsonl"
        manifest["files"] = [item for item in manifest["files"] if item["path"] != omitted]
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    expected = {
        "marker": "invalid_persisted_marker",
        "tamper": "artifact_manifest_verification_failed",
        "missing_driver": "artifact_manifest_missing_required_files",
        "unlisted_permit": "artifact_manifest_missing_permit_files",
    }[defect]
    assert expected in result["errors"][0]


def test_analyzer_summarizes_sglang_timing_coverage_by_attempt_kind(tmp_path) -> None:
    fresh = _with_sglang_timing(_wait("fresh-timing", begin=99.0, wait_end=102.0, kind="fresh", group=1))
    resume = _wait("resume-timing", begin=99.0, wait_end=102.0, kind="resume", group=2)
    cancelled = _wait(
        "cancelled-no-timing",
        begin=101.0,
        wait_end=103.0,
        status="cancelled_before_grant",
        kind="fresh",
        group=3,
    )
    driver, calibration = _write_inputs(tmp_path, waits=[fresh, resume, cancelled])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    timing = result["summary"]["sglang_queue_prefill_timing"]["headline_overlap_terminal_granted"]
    assert timing["all_attempt_kinds"]["eligible_terminal_granted_count"] == 2
    assert timing["all_attempt_kinds"]["timing_available_count"] == 2
    assert timing["all_attempt_kinds"]["timing_coverage_ratio"] == 1.0
    assert timing["all_attempt_kinds"]["availability"] == "available"
    assert timing["fresh"]["availability"] == "available"
    assert timing["fresh"]["distributions"]["sglang_queue_plus_prefill_seconds"]["mean"] == 0.5
    assert timing["resume"]["availability"] == "available"


def test_analyzer_excludes_terminal_before_engine_request_from_timing_coverage(tmp_path) -> None:
    pre_engine_abort = _wait(
        "pre-engine-abort",
        begin=99.0,
        wait_end=102.0,
        status="terminal",
        kind="fresh",
        group=1,
        sglang_timing=False,
    )
    pre_engine_abort["engine_request_started"] = False
    timed = _wait("timed", begin=99.0, wait_end=102.0, kind="resume", group=2)
    driver, calibration = _write_inputs(tmp_path, waits=[pre_engine_abort, timed])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    timing = result["summary"]["sglang_queue_prefill_timing"]["headline_overlap_terminal_granted"]["all_attempt_kinds"]
    assert timing["eligible_terminal_granted_count"] == 1
    assert timing["terminal_before_engine_request_excluded_count"] == 1
    assert timing["timing_coverage_ratio"] == 1.0


@pytest.mark.parametrize("missing_kind", ["fresh", "resume"])
def test_analyzer_rejects_incomplete_headline_sglang_timing(tmp_path, missing_kind) -> None:
    rows = [
        _wait(
            f"{kind}-timing",
            begin=99.0,
            wait_end=102.0,
            kind=kind,
            group=index,
            sglang_timing=kind != missing_kind,
            physical_rollout_id=4 + index,
        )
        for index, kind in enumerate(("fresh", "resume"), 1)
    ]
    driver, calibration = _write_inputs(tmp_path, waits=rows)

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "incomplete_headline_sglang_timing" in result["errors"][0]


def test_analyzer_does_not_require_complete_timing_for_cancelled_before_grant(tmp_path) -> None:
    cancelled = _wait(
        "cancelled-partial-timing",
        begin=99.0,
        wait_end=102.0,
        status="cancelled_before_grant",
    )
    cancelled["sglang_queue_seconds"] = 0.2
    terminal = _wait("terminal-no-timing", begin=99.0, wait_end=102.0, group=2)
    driver, calibration = _write_inputs(tmp_path, waits=[cancelled, terminal])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    timing = result["summary"]["sglang_queue_prefill_timing"]["headline_overlap_terminal_granted"]["all_attempt_kinds"]
    assert timing["eligible_terminal_granted_count"] == 1
    assert timing["timing_available_count"] == 1
    assert timing["availability"] == "available"


def test_analyzer_excludes_terminal_local_exception_from_timing_requirement(tmp_path) -> None:
    failed = _wait(
        "terminal-local-exception",
        begin=99.0,
        wait_end=102.0,
        sglang_timing=False,
        exception_type="CancelledError",
        physical_rollout_id=5,
    )
    driver, calibration = _write_inputs(tmp_path, waits=[failed])

    result = analyze(driver, calibration)

    assert result["verdict"] == "RECALIBRATE_REQUIRED"
    timing = result["summary"]["sglang_queue_prefill_timing"]["headline_overlap_terminal_granted"]["all_attempt_kinds"]
    assert timing["eligible_terminal_granted_count"] == 0
    assert timing["started_terminal_with_exception_excluded_count"] == 1
    assert timing["availability"] == "unavailable"


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_analyzer_rejects_invalid_engine_request_started(value, tmp_path) -> None:
    row = _wait("invalid-started", begin=99.0, wait_end=102.0)
    row["engine_request_started"] = value
    driver, calibration = _write_inputs(tmp_path, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "invalid_engine_request_started" in result["errors"][0]


@pytest.mark.parametrize("defect", ["partial", "negative", "inconsistent", "string"])
def test_analyzer_rejects_malformed_sglang_timing(tmp_path, defect) -> None:
    row = _wait("bad-timing", begin=99.0, wait_end=102.0, sglang_timing=defect != "partial")
    if defect == "partial":
        row["sglang_queue_seconds"] = 0.2
    else:
        _with_sglang_timing(row)
        if defect == "negative":
            row["sglang_queue_seconds"] = -0.1
        elif defect == "inconsistent":
            row["sglang_queue_plus_prefill_seconds"] = 0.6
        else:
            row["sglang_queue_seconds"] = "0.2"
    driver, calibration = _write_inputs(tmp_path, waits=[row])

    result = analyze(driver, calibration)

    assert result["verdict"] == "INVALID_INPUT"
    assert "sglang" in result["errors"][0]
