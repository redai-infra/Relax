#!/usr/bin/env python3
"""Analyze Task 22 gate/permit overlap from a latest-main calibration run.

The output is evidence for recalibrating the phase-elastic design.  It never
converts queue overlap into a predicted end-to-end gain or an automatic GO.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shlex
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ENGINE_PID_RE = re.compile(r"\(SGLangEngine pid=(\d+)\)")
PERMIT_FILENAME_RE = re.compile(r"^permit_wait_rollout_(\d+)\.jsonl$")
GATE_MARKER = "TASK22_CALIBRATION_GATE"
ACTOR_MARKER = "TASK22_CALIBRATION_ACTOR"
FINAL_WAIT_STATUSES = {"terminal", "cancelled_before_grant", "ended_before_grant"}
SCHEDULER_HEALTH_FLOOR_SECONDS = 5.0
HEADLINE_RID_MATCH_MIN_COVERAGE = 1.0
HEALTH_CHECK_RID_PREFIX = "HEALTH_CHECK_"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCHEDULER_CUMULATIVE_TOKEN_FIELDS = (
    "decode_tokens_cumulative",
    "prefill_tokens_cumulative",
    "cached_tokens_cumulative",
)
SGLANG_TIMING_FIELDS = (
    "sglang_queue_seconds",
    "sglang_forward_entry_epoch",
    "sglang_prefill_finished_epoch",
    "sglang_prefill_seconds",
    "sglang_queue_plus_prefill_seconds",
)
SGLANG_TIMING_CONSISTENCY_ABS_TOLERANCE_SECONDS = 1e-4
SGLANG_TIMING_CONSISTENCY_REL_TOLERANCE = 1e-5
ACTOR_PERF_REQUIRED_FIELDS = (
    "perf/step_time",
    "perf/train_wait_time",
    "perf/log_probs_time",
    "perf/actor_train_time",
    "perf/train_time",
    "perf/train_get_data_time",
    "perf/update_weights_time",
    "perf/step_token_per_s",
)
TRAIN_QUALITY_REQUIRED_FIELDS = (
    "train/tis",
    "train/tis_clipfrac",
    "train/tis_abs",
    "train/train_rollout_logprob_abs_diff",
    "train/mismatch_kl",
    "train/mismatch_k3_kl",
)
ROLLOUT_DATA_REQUIRED_FIELDS = (
    "rollout/response_lengths",
    "rollout/raw_reward",
    "rollout/rewards",
    "rollout/total_lengths/max",
    "rollout/total_lengths/min",
)


class CalibrationInputError(ValueError):
    """Raised when calibration evidence is incomplete or ambiguous."""


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _finite_nonnegative(value: Any) -> float | None:
    parsed = _finite_float(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _complete_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CalibrationInputError(f"unreadable_driver_log:{path}:{exc}") from exc
    lines = text.splitlines()
    if text and not text.endswith(("\n", "\r")):
        lines.pop()
    return lines


def _structured_rows(path: Path, marker_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_no, raw_line in enumerate(_complete_lines(path), 1):
        line = ANSI_RE.sub("", raw_line)
        marker = line.find(marker_text)
        if marker < 0:
            continue
        row = {"_line": str(line_no)}
        try:
            fields = shlex.split(line[marker + len(marker_text) :].strip())
        except ValueError as exc:
            raise CalibrationInputError(f"malformed_gate_line:{line_no}:{exc}") from exc
        for field in fields:
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            row[key] = value
        rows.append(row)
    return rows


def _python_dict_log_rows(path: Path) -> list[dict[str, Any]]:
    patterns = (
        ("perf", re.compile(r"\bperf (\d+): (\{.*\})$")),
        ("train", re.compile(r"\bstep (\d+): (\{.*\})$")),
        ("rollout", re.compile(r"\brollout (\d+): (\{.*\})$")),
    )
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(_complete_lines(path), 1):
        line = ANSI_RE.sub("", raw_line)
        for kind, pattern in patterns:
            match = pattern.search(line)
            if match is None:
                continue
            try:
                payload = ast.literal_eval(match.group(2))
            except (SyntaxError, ValueError) as exc:
                raise CalibrationInputError(f"malformed_standard_metric_log:{kind}:line_{line_no}:{exc}") from exc
            if not isinstance(payload, dict):
                raise CalibrationInputError(f"non_object_standard_metric_log:{kind}:line_{line_no}")
            rows.append(
                {
                    "_line": line_no,
                    "kind": kind,
                    "logical_step": int(match.group(1)),
                    "metrics": payload,
                }
            )
            break
    return rows


def _validate_metric_fields(
    row: dict[str, Any],
    required_fields: tuple[str, ...],
    *,
    category: str,
) -> dict[str, float]:
    metrics = row["metrics"]
    missing = [field for field in required_fields if field not in metrics]
    if missing:
        raise CalibrationInputError(
            f"missing_{category}_fields:logical_{row['logical_step']}:line_{row['_line']}:{missing}"
        )
    parsed: dict[str, float] = {}
    for field in required_fields:
        value = _json_finite_number(metrics[field])
        if value is None:
            raise CalibrationInputError(
                f"invalid_{category}_field:logical_{row['logical_step']}:line_{row['_line']}:"
                f"{field}={metrics[field]!r}"
            )
        parsed[field] = value
    return parsed


def _headline_standard_metrics(
    driver_log: Path,
    headline_lo: int,
    headline_hi: int,
) -> dict[str, Any]:
    rows = _python_dict_log_rows(driver_log)
    expected_steps = set(range(headline_lo, headline_hi + 1))

    actor_perf_candidates = [
        row
        for row in rows
        if row["kind"] == "perf" and row["logical_step"] in expected_steps and "perf/step_time" in row["metrics"]
    ]
    train_quality_candidates = [
        row
        for row in rows
        if row["kind"] == "train" and row["logical_step"] in expected_steps and "train/tis" in row["metrics"]
    ]
    rollout_data_candidates = [
        row
        for row in rows
        if row["kind"] == "rollout"
        and row["logical_step"] in expected_steps
        and "rollout/response_lengths" in row["metrics"]
    ]

    def unique_by_step(candidates: list[dict[str, Any]], category: str) -> dict[int, dict[str, Any]]:
        selected: dict[int, dict[str, Any]] = {}
        errors = []
        for step in sorted(expected_steps):
            matching = [row for row in candidates if row["logical_step"] == step]
            if len(matching) != 1:
                errors.append(f"logical_{step}:count={len(matching)}")
            else:
                selected[step] = matching[0]
        if errors:
            raise CalibrationInputError(f"incomplete_or_nonunique_{category}:" + ";".join(errors))
        return selected

    actor_perf_rows = unique_by_step(actor_perf_candidates, "actor_perf")
    train_quality_rows = unique_by_step(train_quality_candidates, "train_quality")
    rollout_data_rows = unique_by_step(rollout_data_candidates, "rollout_data")

    per_step: dict[str, Any] = {}
    for step in sorted(expected_steps):
        actor_perf = actor_perf_rows[step]
        train_quality = train_quality_rows[step]
        rollout_data = rollout_data_rows[step]
        forbidden_actor = sorted(field for field in actor_perf["metrics"] if field.startswith("perf/ref_log_probs"))
        forbidden_rollout = sorted(field for field in rollout_data["metrics"] if field == "rollout/ref_log_probs")
        if forbidden_actor or forbidden_rollout:
            raise CalibrationInputError(
                f"zero_kl_reference_metrics_present:logical_{step}:actor={forbidden_actor}:rollout={forbidden_rollout}"
            )
        per_step[str(step)] = {
            "actor_perf": _validate_metric_fields(
                actor_perf,
                ACTOR_PERF_REQUIRED_FIELDS,
                category="actor_perf",
            ),
            "train_quality": _validate_metric_fields(
                train_quality,
                TRAIN_QUALITY_REQUIRED_FIELDS,
                category="train_quality",
            ),
            "rollout_data": _validate_metric_fields(
                rollout_data,
                ROLLOUT_DATA_REQUIRED_FIELDS,
                category="rollout_data",
            ),
        }

    return {
        "per_step": per_step,
        "actor_perf": {
            field: _number_summary([per_step[str(step)]["actor_perf"][field] for step in sorted(expected_steps)])
            for field in ACTOR_PERF_REQUIRED_FIELDS
        },
        "train_quality": {
            field: _number_summary([per_step[str(step)]["train_quality"][field] for step in sorted(expected_steps)])
            for field in TRAIN_QUALITY_REQUIRED_FIELDS
        },
        "rollout_data": {
            field: _number_summary([per_step[str(step)]["rollout_data"][field] for step in sorted(expected_steps)])
            for field in ROLLOUT_DATA_REQUIRED_FIELDS
        },
        "zero_kl_reference_metric_absence": "pass",
    }


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        return None
    return value


def _rid_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return None
    return value


def _scheduler_snapshot(engine_pid: str, event: dict[str, Any], source: str) -> dict[str, Any]:
    timestamp_epoch = _json_finite_number(event.get("timestamp_epoch"))
    if timestamp_epoch is None or timestamp_epoch <= 0:
        raise CalibrationInputError(f"invalid_scheduler_timestamp_epoch:{source}:{event.get('timestamp_epoch')!r}")
    timestamp_iso = event.get("timestamp")
    if not isinstance(timestamp_iso, str) or not timestamp_iso:
        raise CalibrationInputError(f"invalid_scheduler_timestamp:{source}:{event.get('timestamp')!r}")
    forward_mode = event.get("forward_mode")
    if isinstance(forward_mode, bool) or not isinstance(forward_mode, (str, int)):
        raise CalibrationInputError(f"invalid_scheduler_forward_mode:{source}:{forward_mode!r}")
    idle = event.get("idle")
    if not isinstance(idle, bool):
        raise CalibrationInputError(f"invalid_scheduler_idle:{source}:{idle!r}")
    cumulative_tokens = {field: _nonnegative_int(event.get(field)) for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS}
    invalid_cumulative_fields = [field for field, value in cumulative_tokens.items() if value is None]
    if invalid_cumulative_fields:
        raise CalibrationInputError(
            f"invalid_scheduler_cumulative_tokens:{source}:fields={','.join(invalid_cumulative_fields)}"
        )
    list_fields = {
        "running_rids": _rid_list(event.get("running_rids")),
        "running_seq_lens": _nonnegative_int_list(event.get("running_seq_lens")),
        "running_origin_input_lens": _nonnegative_int_list(event.get("running_origin_input_lens")),
        "running_output_lens": _nonnegative_int_list(event.get("running_output_lens")),
        "queued_rids": _rid_list(event.get("queued_rids")),
        "queued_origin_input_lens": _nonnegative_int_list(event.get("queued_origin_input_lens")),
        "queued_output_lens": _nonnegative_int_list(event.get("queued_output_lens")),
    }
    invalid_fields = [field for field, value in list_fields.items() if value is None]
    if invalid_fields:
        raise CalibrationInputError(f"invalid_scheduler_list_fields:{source}:{','.join(invalid_fields)}")
    running_count = len(list_fields["running_rids"] or [])
    queued_count = len(list_fields["queued_rids"] or [])
    running_lengths = {
        len(list_fields[field] or [])
        for field in ("running_rids", "running_seq_lens", "running_origin_input_lens", "running_output_lens")
    }
    queued_lengths = {
        len(list_fields[field] or []) for field in ("queued_rids", "queued_origin_input_lens", "queued_output_lens")
    }
    if len(running_lengths) != 1 or len(queued_lengths) != 1:
        raise CalibrationInputError(
            f"scheduler_parallel_list_length_mismatch:{source}:running={sorted(running_lengths)}:queued={sorted(queued_lengths)}"
        )
    if idle and (running_count or queued_count):
        raise CalibrationInputError(
            f"inconsistent_scheduler_idle_state:{source}:idle=true:running={running_count}:queued={queued_count}"
        )
    return {
        "engine_pid": engine_pid,
        "timestamp_epoch": timestamp_epoch,
        "timestamp_iso": timestamp_iso,
        "idle": idle,
        **cumulative_tokens,
        "forward_mode": str(forward_mode),
        "running_rids": list_fields["running_rids"],
        "running_seq_lens": list_fields["running_seq_lens"],
        "running_origin_input_lens": list_fields["running_origin_input_lens"],
        "running_output_lens": list_fields["running_output_lens"],
        "queued_rids": list_fields["queued_rids"],
        "queued_origin_input_lens": list_fields["queued_origin_input_lens"],
        "queued_output_lens": list_fields["queued_output_lens"],
        "running_request_count": running_count,
        "queued_request_count": queued_count,
        "_source": source,
    }


def _scheduler_snapshots(driver_log: Path) -> dict[str, list[dict[str, Any]]]:
    by_engine: dict[str, list[dict[str, Any]]] = {}
    for line_no, raw_line in enumerate(_complete_lines(driver_log), 1):
        line = ANSI_RE.sub("", raw_line)
        pid_match = ENGINE_PID_RE.search(line)
        if pid_match is None:
            continue
        marker = line.find("{")
        if marker < 0:
            continue
        try:
            event = json.loads(line[marker:])
        except json.JSONDecodeError as exc:
            if "scheduler.status" in line:
                raise CalibrationInputError(f"malformed_scheduler_json:line_{line_no}:{exc.msg}") from exc
            continue
        if not isinstance(event, dict) or event.get("event") != "scheduler.status":
            continue
        engine_pid = pid_match.group(1)
        by_engine.setdefault(engine_pid, []).append(_scheduler_snapshot(engine_pid, event, f"driver.log:{line_no}"))
    if len(by_engine) != 2:
        raise CalibrationInputError(f"scheduler_engine_count:{len(by_engine)}:expected=2:pids={sorted(by_engine)}")
    for snapshots in by_engine.values():
        snapshots.sort(key=lambda row: row["timestamp_epoch"])
        for previous, current in zip(snapshots, snapshots[1:], strict=False):
            if current["timestamp_epoch"] <= previous["timestamp_epoch"]:
                raise CalibrationInputError(
                    f"nonincreasing_scheduler_timestamp:engine_pid={current['engine_pid']}:"
                    f"previous={previous['timestamp_epoch']:.6f}:current={current['timestamp_epoch']:.6f}"
                )
            for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS:
                if current[field] < previous[field]:
                    raise CalibrationInputError(
                        f"nonmonotonic_scheduler_cumulative_tokens:engine_pid={current['engine_pid']}:"
                        f"field={field}:previous={previous[field]}:current={current[field]}"
                    )
    return by_engine


def _gate_interval(row: dict[str, str]) -> tuple[float, float, float] | None:
    begin = _finite_float(row.get("t_begin"))
    end = _finite_float(row.get("t_end"))
    duration = _finite_float(row.get("dur"))
    if begin is None or end is None or duration is None:
        return None
    if begin > end or duration < 0:
        return None
    if not math.isclose(duration, end - begin, rel_tol=1e-4, abs_tol=1e-3):
        return None
    return begin, end, duration


def _headline_gates(driver_log: Path, headline_lo: int, headline_hi: int) -> list[dict[str, Any]]:
    rows = _structured_rows(driver_log, GATE_MARKER)
    expected_pairs = {(step, step + 1) for step in range(headline_lo, headline_hi + 1)}
    keyed: dict[tuple[int, int], list[dict[str, str]]] = {pair: [] for pair in expected_pairs}
    errors: list[str] = []
    for row in rows:
        try:
            logical_step = int(row["logical_step"])
            sync_id = int(row["sync_id"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"line_{row['_line']}:missing_or_invalid_step_mapping")
            continue
        pair = (logical_step, sync_id)
        touches_headline = headline_lo <= logical_step <= headline_hi or headline_lo + 1 <= sync_id <= headline_hi + 1
        if touches_headline and pair not in expected_pairs:
            errors.append(f"line_{row['_line']}:invalid_mapping:{logical_step}->{sync_id}")
            continue
        if pair in expected_pairs:
            keyed[pair].append(row)

    gates = []
    for logical_step in range(headline_lo, headline_hi + 1):
        pair = (logical_step, logical_step + 1)
        matching = keyed[pair]
        if len(matching) != 1:
            errors.append(f"logical_{logical_step}:sync_{logical_step + 1}:count={len(matching)}")
            continue
        interval = _gate_interval(matching[0])
        if interval is None:
            errors.append(f"logical_{logical_step}:sync_{logical_step + 1}:invalid_interval")
            continue
        begin, end, duration = interval
        gates.append(
            {
                "logical_step": logical_step,
                "sync_id": logical_step + 1,
                "gate_begin_epoch": begin,
                "gate_end_epoch": end,
                "gate_wall_seconds": duration,
            }
        )
    if errors:
        raise CalibrationInputError("incomplete_or_nonunique_headline_gates:" + ";".join(errors))
    return gates


def _actor_point(row: dict[str, str]) -> float | None:
    return _finite_float(row.get("t"))


def _actor_interval(row: dict[str, str]) -> tuple[float, float, float] | None:
    return _gate_interval(row)


def _headline_actor_timelines(
    driver_log: Path,
    gates: list[dict[str, Any]],
    headline_lo: int,
    headline_hi: int,
) -> dict[int, dict[str, Any]]:
    rows = _structured_rows(driver_log, ACTOR_MARKER)
    required_phases = (
        "data_wait_begin",
        "first_subbatch_ready",
        "last_subbatch_ready",
        "actor_train",
        "weight_sync",
    )
    by_step_phase: dict[tuple[int, str], list[dict[str, str]]] = {
        (step, phase): [] for step in range(headline_lo, headline_hi + 1) for phase in required_phases
    }
    errors: list[str] = []
    for row in rows:
        try:
            logical_step = int(row["logical_step"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"actor_line_{row['_line']}:missing_or_invalid_logical_step")
            continue
        phase = row.get("phase")
        if headline_lo <= logical_step <= headline_hi and phase in required_phases:
            by_step_phase[(logical_step, phase)].append(row)

    gate_by_step = {gate["logical_step"]: gate for gate in gates}
    timelines: dict[int, dict[str, Any]] = {}
    for logical_step in range(headline_lo, headline_hi + 1):
        selected: dict[str, dict[str, str]] = {}
        for phase in required_phases:
            matching = by_step_phase[(logical_step, phase)]
            if len(matching) != 1:
                errors.append(f"actor_logical_{logical_step}:{phase}:count={len(matching)}")
                continue
            selected[phase] = matching[0]
        if len(selected) != len(required_phases):
            continue
        data_wait_begin = _actor_point(selected["data_wait_begin"])
        first_ready = _actor_point(selected["first_subbatch_ready"])
        last_ready = _actor_point(selected["last_subbatch_ready"])
        actor_train = _actor_interval(selected["actor_train"])
        weight_sync = _actor_interval(selected["weight_sync"])
        if data_wait_begin is None or first_ready is None or last_ready is None:
            errors.append(f"actor_logical_{logical_step}:invalid_point_timestamp")
            continue
        if actor_train is None or weight_sync is None:
            errors.append(f"actor_logical_{logical_step}:invalid_interval")
            continue
        actor_train_begin, actor_train_end, actor_train_wall = actor_train
        weight_sync_begin, weight_sync_end, weight_sync_wall = weight_sync
        gate = gate_by_step[logical_step]
        ordered = (
            data_wait_begin,
            first_ready,
            last_ready,
            actor_train_begin,
            actor_train_end,
            gate["gate_begin_epoch"],
            gate["gate_end_epoch"],
            weight_sync_begin,
            weight_sync_end,
        )
        if any(previous > current for previous, current in zip(ordered, ordered[1:])):
            errors.append(f"actor_logical_{logical_step}:phase_order_invalid:{ordered}")
            continue
        timelines[logical_step] = {
            "data_wait_begin_epoch": data_wait_begin,
            "first_subbatch_ready_epoch": first_ready,
            "last_subbatch_ready_epoch": last_ready,
            "actor_train_begin_epoch": actor_train_begin,
            "actor_train_end_epoch": actor_train_end,
            "gate_begin_epoch": gate["gate_begin_epoch"],
            "gate_end_epoch": gate["gate_end_epoch"],
            "weight_sync_begin_epoch": weight_sync_begin,
            "weight_sync_end_epoch": weight_sync_end,
            "first_subbatch_wait_seconds": first_ready - data_wait_begin,
            "all_subbatches_ready_wait_seconds": last_ready - data_wait_begin,
            "remaining_subbatches_after_first_seconds": last_ready - first_ready,
            "post_last_subbatch_forward_prepare_seconds": actor_train_begin - last_ready,
            "actor_train_seconds": actor_train_wall,
            "gate_seconds": gate["gate_wall_seconds"],
            "weight_sync_seconds": weight_sync_wall,
            "actor_idle_before_weight_sync_seconds": weight_sync_begin - actor_train_end,
            "post_train_before_gate_seconds": gate["gate_begin_epoch"] - actor_train_end,
            "post_gate_before_weight_sync_seconds": weight_sync_begin - gate["gate_end_epoch"],
            "observed_pipeline_seconds": weight_sync_end - data_wait_begin,
        }
    if errors:
        raise CalibrationInputError("invalid_actor_timeline:" + ";".join(errors))
    return timelines


def _read_json_object(path: Path, artifact_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationInputError(f"unreadable_{artifact_name}:{path}:{exc}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationInputError(f"malformed_{artifact_name}:{path.name}:{exc.msg}") from exc
    if not isinstance(payload, dict):
        raise CalibrationInputError(f"non_object_{artifact_name}:{path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CalibrationInputError(f"unreadable_manifest_artifact:{path}:{exc}") from exc
    return digest.hexdigest()


def _validate_persisted_artifacts(
    directory: Path,
    driver_log: Path,
) -> tuple[dict[str, Any], set[str]]:
    persisted_path = directory / "PERSISTED"
    try:
        persisted_status = persisted_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CalibrationInputError(f"unreadable_persisted_marker:{persisted_path}:{exc}") from exc
    if persisted_status != "verified":
        raise CalibrationInputError(f"invalid_persisted_marker:{persisted_status!r}:expected='verified'")

    manifest = _read_json_object(directory / "artifact_manifest.json", "artifact_manifest")
    if _nonnegative_int(manifest.get("schema_version")) != 1:
        raise CalibrationInputError(f"invalid_artifact_manifest_schema_version:{manifest.get('schema_version')!r}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise CalibrationInputError("invalid_artifact_manifest_files")

    directory_resolved = directory.resolve()
    listed_paths: set[str] = set()
    verified_bytes = 0
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise CalibrationInputError(f"invalid_artifact_manifest_entry:index={index}")
        raw_path = item.get("path")
        size = _nonnegative_int(item.get("size"))
        expected_sha256 = item.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise CalibrationInputError(f"invalid_artifact_manifest_path:index={index}:{raw_path!r}")
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise CalibrationInputError(f"unsafe_artifact_manifest_path:index={index}:{raw_path!r}")
        normalized = relative.as_posix()
        if normalized in {"artifact_manifest.json", "PERSISTED"}:
            raise CalibrationInputError(f"forbidden_artifact_manifest_entry:{normalized}")
        if normalized in listed_paths:
            raise CalibrationInputError(f"duplicate_artifact_manifest_path:{normalized}")
        if size is None or not isinstance(expected_sha256, str) or SHA256_RE.fullmatch(expected_sha256) is None:
            raise CalibrationInputError(
                f"invalid_artifact_manifest_metadata:index={index}:path={normalized}:"
                f"size={item.get('size')!r}:sha256={expected_sha256!r}"
            )
        candidate = directory.joinpath(*relative.parts)
        try:
            candidate_resolved = candidate.resolve(strict=True)
            candidate_resolved.relative_to(directory_resolved)
        except (OSError, ValueError) as exc:
            raise CalibrationInputError(f"missing_or_escaped_manifest_artifact:{normalized}:{exc}") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise CalibrationInputError(f"non_regular_manifest_artifact:{normalized}")
        actual_size = candidate.stat().st_size
        actual_sha256 = _sha256(candidate)
        if actual_size != size or actual_sha256 != expected_sha256:
            raise CalibrationInputError(
                f"artifact_manifest_verification_failed:{normalized}:"
                f"size={actual_size}/{size}:sha256={actual_sha256}/{expected_sha256}"
            )
        listed_paths.add(normalized)
        verified_bytes += actual_size

    try:
        driver_relative = driver_log.resolve(strict=True).relative_to(directory_resolved).as_posix()
    except (OSError, ValueError) as exc:
        raise CalibrationInputError(f"driver_log_not_inside_persisted_calibration_dir:{driver_log}:{exc}") from exc
    required = {"STATUS", "run_lifecycle.json", "run_contract.json", driver_relative}
    missing_required = sorted(required - listed_paths)
    if missing_required:
        raise CalibrationInputError(f"artifact_manifest_missing_required_files:{missing_required}")
    return (
        {
            "schema_version": 1,
            "persisted_marker": "verified",
            "manifest_file_count": len(listed_paths),
            "verified_bytes": verified_bytes,
            "driver_log_manifest_path": driver_relative,
            "status": "pass",
        },
        listed_paths,
    )


def _validate_run_artifacts(
    directory: Path,
    *,
    headline_lo: int,
    headline_hi: int,
) -> dict[str, Any]:
    if headline_lo < 0 or headline_hi < headline_lo:
        raise CalibrationInputError(f"invalid_calibration_headline_contract:{headline_lo}..{headline_hi}")

    contract = _read_json_object(directory / "run_contract.json", "run_contract")
    if _nonnegative_int(contract.get("schema_version")) != 1:
        raise CalibrationInputError(f"invalid_run_contract_schema_version:{contract.get('schema_version')!r}")
    workload = contract.get("workload")
    if not isinstance(workload, dict):
        raise CalibrationInputError("invalid_run_contract_workload")
    num_rollout = _nonnegative_int(workload.get("num_rollout"))
    if num_rollout is None or num_rollout <= headline_hi:
        raise CalibrationInputError(f"invalid_run_contract_num_rollout:{num_rollout!r}:headline_hi={headline_hi}")
    contract_headline = contract.get("headline")
    expected_headline = {"logical_step_lo": headline_lo, "logical_step_hi": headline_hi}
    if not isinstance(contract_headline, dict) or any(
        _nonnegative_int(contract_headline.get(key)) != value for key, value in expected_headline.items()
    ):
        raise CalibrationInputError(
            f"invalid_run_contract_headline:{contract_headline!r}:expected={expected_headline!r}"
        )

    lifecycle = _read_json_object(directory / "run_lifecycle.json", "run_lifecycle")
    if _nonnegative_int(lifecycle.get("schema_version")) != 1:
        raise CalibrationInputError(f"invalid_run_lifecycle_schema_version:{lifecycle.get('schema_version')!r}")
    if lifecycle.get("status") != "SUCCEEDED" or _nonnegative_int(lifecycle.get("exit_code")) != 0:
        raise CalibrationInputError(
            f"unsuccessful_run_lifecycle:status={lifecycle.get('status')!r}:exit_code={lifecycle.get('exit_code')!r}"
        )

    status_path = directory / "STATUS"
    try:
        status = status_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CalibrationInputError(f"unreadable_run_status:{status_path}:{exc}") from exc
    if status != "SUCCEEDED":
        raise CalibrationInputError(f"unsuccessful_run_status:{status!r}:expected='SUCCEEDED'")

    return {
        "run_contract": {
            "schema_version": 1,
            "num_rollout": num_rollout,
            "headline": expected_headline,
        },
        "run_lifecycle": {
            "schema_version": 1,
            "status": "SUCCEEDED",
            "exit_code": 0,
        },
        "status": status,
    }


def _read_permit_rows(directory: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(directory.glob("permit_wait_rollout_*.jsonl"))
    if not paths:
        raise CalibrationInputError("missing_permit_wait_artifacts:permit_wait_rollout_*.jsonl")
    rows: list[dict[str, Any]] = []
    for path in paths:
        filename_match = PERMIT_FILENAME_RE.fullmatch(path.name)
        if filename_match is None:
            raise CalibrationInputError(f"invalid_permit_wait_filename:{path.name}")
        artifact_rollout_id = int(filename_match.group(1))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CalibrationInputError(f"unreadable_permit_wait:{path}:{exc}") from exc
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationInputError(f"malformed_permit_wait:{path.name}:{line_no}:{exc.msg}") from exc
            if not isinstance(row, dict):
                raise CalibrationInputError(f"non_object_permit_wait:{path.name}:{line_no}")
            if (
                not isinstance(row.get("schema_version"), int)
                or isinstance(row.get("schema_version"), bool)
                or row["schema_version"] != 1
            ):
                raise CalibrationInputError(
                    f"invalid_permit_schema_version:{path.name}:{line_no}:{row.get('schema_version')!r}"
                )
            if (
                not isinstance(row.get("physical_rollout_id"), int)
                or isinstance(row.get("physical_rollout_id"), bool)
                or row["physical_rollout_id"] != artifact_rollout_id
            ):
                raise CalibrationInputError(
                    f"permit_artifact_rollout_mismatch:{path.name}:{line_no}:"
                    f"row={row.get('physical_rollout_id')!r}:filename={artifact_rollout_id}"
                )
            row["_source"] = f"{path.name}:{line_no}"
            rows.append(row)
    return paths, rows


def _validate_permit_rollout_coverage(
    paths: list[Path],
    rows: list[dict[str, Any]],
    *,
    num_rollout: int,
) -> dict[str, Any]:
    rollout_ids = []
    for path in paths:
        match = PERMIT_FILENAME_RE.fullmatch(path.name)
        if match is None:
            raise CalibrationInputError(f"invalid_permit_wait_filename:{path.name}")
        rollout_ids.append(int(match.group(1)))
    observed = set(rollout_ids)
    required = set(range(num_rollout))
    allowed = required | {num_rollout}
    missing = sorted(required - observed)
    unexpected = sorted(observed - allowed)
    row_counts = Counter(row["physical_rollout_id"] for row in rows)
    empty = sorted(rollout_id for rollout_id in observed if row_counts[rollout_id] == 0)
    if missing or unexpected or empty:
        raise CalibrationInputError(
            f"invalid_permit_physical_rollout_coverage:missing={missing}:unexpected={unexpected}:empty={empty}:"
            f"required=0..{num_rollout - 1}:optional_final_backfill={num_rollout}"
        )
    return {
        "required_physical_rollout_ids": list(range(num_rollout)),
        "optional_final_backfill_physical_rollout_id": num_rollout,
        "observed_physical_rollout_ids": sorted(observed),
        "per_physical_rollout_row_counts": {
            str(rollout_id): row_counts[rollout_id] for rollout_id in sorted(observed)
        },
        "final_backfill_present": num_rollout in observed,
        "status": "pass",
    }


def _wait_end(row: dict[str, Any]) -> float:
    status = row["permit_wait_status"]
    field = "permit_acquire_granted_epoch" if status == "terminal" else "terminal_epoch"
    value = _finite_float(row.get(field))
    if value is None:
        raise CalibrationInputError(f"missing_wait_end:{row['_source']}:{status}:{field}")
    return value


def _validate_sglang_timing(row: dict[str, Any]) -> bool:
    if (
        row.get("permit_wait_status") != "terminal"
        or row.get("exception_type") is not None
        or row.get("engine_request_started") is not True
    ):
        return False
    present = [row.get(field) is not None for field in SGLANG_TIMING_FIELDS]
    if not any(present):
        return False
    if not all(present):
        missing = [field for field, is_present in zip(SGLANG_TIMING_FIELDS, present, strict=True) if not is_present]
        raise CalibrationInputError(f"partial_sglang_timing:{row['_source']}:missing={','.join(missing)}")

    queue = _json_finite_number(row["sglang_queue_seconds"])
    forward_entry = _json_finite_number(row["sglang_forward_entry_epoch"])
    prefill_finished = _json_finite_number(row["sglang_prefill_finished_epoch"])
    prefill = _json_finite_number(row["sglang_prefill_seconds"])
    queue_plus_prefill = _json_finite_number(row["sglang_queue_plus_prefill_seconds"])
    if (
        queue is None
        or queue < 0
        or forward_entry is None
        or forward_entry <= 0
        or prefill_finished is None
        or prefill_finished < forward_entry
        or prefill is None
        or prefill < 0
        or queue_plus_prefill is None
        or queue_plus_prefill < 0
    ):
        raise CalibrationInputError(f"invalid_sglang_timing_values:{row['_source']}")
    derived_prefill = prefill_finished - forward_entry
    if not math.isclose(
        prefill,
        derived_prefill,
        rel_tol=SGLANG_TIMING_CONSISTENCY_REL_TOLERANCE,
        abs_tol=SGLANG_TIMING_CONSISTENCY_ABS_TOLERANCE_SECONDS,
    ):
        raise CalibrationInputError(
            f"inconsistent_sglang_prefill_seconds:{row['_source']}:"
            f"reported={prefill:.9f}:derived={derived_prefill:.9f}"
        )
    derived_queue_plus_prefill = queue + prefill
    if not math.isclose(
        queue_plus_prefill,
        derived_queue_plus_prefill,
        rel_tol=SGLANG_TIMING_CONSISTENCY_REL_TOLERANCE,
        abs_tol=SGLANG_TIMING_CONSISTENCY_ABS_TOLERANCE_SECONDS,
    ):
        raise CalibrationInputError(
            f"inconsistent_sglang_queue_plus_prefill_seconds:{row['_source']}:"
            f"reported={queue_plus_prefill:.9f}:derived={derived_queue_plus_prefill:.9f}"
        )
    return True


def _validate_permit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise CalibrationInputError("no_permit_wait_rows")
    seen_ids: dict[str, str] = {}
    validated = []
    for row in rows:
        if row.get("record_type") != "permit_wait":
            raise CalibrationInputError(f"invalid_permit_record_type:{row['_source']}:{row.get('record_type')!r}")
        wait_id = row.get("permit_wait_id")
        if not isinstance(wait_id, str) or not wait_id:
            raise CalibrationInputError(f"missing_permit_wait_id:{row['_source']}")
        if wait_id in seen_ids:
            raise CalibrationInputError(f"duplicate_permit_wait_id:{wait_id}:{seen_ids[wait_id]}:{row['_source']}")
        seen_ids[wait_id] = row["_source"]
        status = row.get("permit_wait_status")
        if status not in FINAL_WAIT_STATUSES:
            raise CalibrationInputError(f"nonterminal_or_unknown_wait:{row['_source']}:{status!r}")
        begin = _finite_float(row.get("permit_acquire_begin_epoch"))
        if begin is None:
            raise CalibrationInputError(f"missing_wait_begin:{row['_source']}")
        end = _wait_end(row)
        if end < begin:
            raise CalibrationInputError(f"reversed_wait_interval:{row['_source']}:{begin}>{end}")
        if status == "terminal":
            terminal = _finite_float(row.get("terminal_epoch"))
            if terminal is None or terminal < end:
                raise CalibrationInputError(f"invalid_request_terminal:{row['_source']}:{terminal!r}<{end}")
            if not isinstance(row.get("engine_request_started"), bool):
                raise CalibrationInputError(
                    f"invalid_engine_request_started:{row['_source']}:{row.get('engine_request_started')!r}"
                )
        row["_sglang_timing_available"] = _validate_sglang_timing(row)
        row["_wait_begin_epoch"] = begin
        row["_wait_end_epoch"] = end
        validated.append(row)
    return validated


def _kind_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = Counter(str(row.get("attempt_kind", "unknown")) for row in rows)
    return {
        "fresh": exact.get("fresh", 0),
        "resume": exact.get("resume", 0),
        "other": sum(value for key, value in exact.items() if key not in {"fresh", "resume"}),
        "exact": dict(sorted(exact.items())),
    }


def _unique_groups(rows: list[dict[str, Any]]) -> tuple[int, int]:
    groups = {
        json.dumps(row["group_index"], sort_keys=True, default=str)
        for row in rows
        if row.get("group_index") is not None
    }
    return len(groups), sum(row.get("group_index") is None for row in rows)


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    current_begin, current_end = ordered[0]
    total = 0.0
    for begin, end in ordered[1:]:
        if begin <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_begin
        current_begin, current_end = begin, end
    return total + current_end - current_begin


def _metric_summary(
    rows: list[dict[str, Any]],
    field: str,
    *,
    eligible_statuses: set[str] | None = None,
) -> dict[str, Any]:
    eligible = (
        rows if eligible_statuses is None else [row for row in rows if row["permit_wait_status"] in eligible_statuses]
    )
    values = [value for row in eligible if (value := _finite_nonnegative(row.get(field))) is not None]
    result: dict[str, Any] = {
        "source_field": field,
        "row_count": len(rows),
        "eligible_count": len(eligible),
        "available_count": len(values),
        "eligible_coverage_ratio": _safe_ratio(len(values), len(eligible)),
        "all_row_coverage_ratio": _safe_ratio(len(values), len(rows)),
    }
    if not values:
        result["availability"] = "unavailable"
        return result
    result.update(
        {
            "availability": "available" if len(values) == len(eligible) else "partial",
            "sum": sum(values),
            "mean": mean(values),
            "p50": median(values),
            "max": max(values),
        }
    )
    return result


def _inline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "response_tokens_before": _metric_summary(rows, "response_tokens_before"),
        "context_tokens_before": _metric_summary(rows, "context_tokens_before"),
        "generated_tokens_this_attempt": _metric_summary(rows, "generated_tokens_this_attempt"),
        "request_wall_seconds": _metric_summary(
            rows,
            "request_wall_seconds",
            eligible_statuses={"terminal"},
        ),
    }


def _sglang_timing_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    terminal = [row for row in rows if row["permit_wait_status"] == "terminal"]
    started = [row for row in terminal if row.get("engine_request_started") is True]
    eligible = [row for row in started if row.get("exception_type") is None]
    available = [row for row in eligible if row["_sglang_timing_available"]]
    if not eligible or not available:
        availability = "unavailable"
    elif len(available) < len(eligible):
        availability = "insufficient_evidence"
    else:
        availability = "available"
    return {
        "availability": availability,
        "eligible_terminal_granted_count": len(eligible),
        "terminal_before_engine_request_excluded_count": len(terminal) - len(started),
        "started_terminal_with_exception_excluded_count": len(started) - len(eligible),
        "timing_available_count": len(available),
        "timing_coverage_ratio": _safe_ratio(len(available), len(eligible)),
        "distributions": {field: _number_summary([row[field] for row in available]) for field in SGLANG_TIMING_FIELDS},
    }


def _sglang_timing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_attempt_kinds": _sglang_timing_group_summary(rows),
        "fresh": _sglang_timing_group_summary([row for row in rows if row.get("attempt_kind") == "fresh"]),
        "resume": _sglang_timing_group_summary([row for row in rows if row.get("attempt_kind") == "resume"]),
    }


def _validate_headline_sglang_timing_coverage(rows: list[dict[str, Any]]) -> None:
    groups = {
        "all_attempt_kinds": rows,
        "fresh": [row for row in rows if row.get("attempt_kind") == "fresh"],
        "resume": [row for row in rows if row.get("attempt_kind") == "resume"],
    }
    for group_name, group_rows in groups.items():
        summary = _sglang_timing_group_summary(group_rows)
        eligible = summary["eligible_terminal_granted_count"]
        available = summary["timing_available_count"]
        if eligible > 0 and available != eligible:
            raise CalibrationInputError(
                f"incomplete_headline_sglang_timing:group={group_name}:"
                f"eligible={eligible}:available={available}:"
                f"coverage={summary['timing_coverage_ratio']}"
            )


def _number_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"availability": "unavailable", "count": 0}
    return {
        "availability": "available",
        "count": len(values),
        "sum": sum(values),
        "mean": mean(values),
        "p50": median(values),
        "max": max(values),
    }


def _scheduler_token_intervals(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals = []
    for previous, current in zip(snapshots, snapshots[1:], strict=False):
        duration = current["timestamp_epoch"] - previous["timestamp_epoch"]
        token_deltas = {
            field.removesuffix("_cumulative"): current[field] - previous[field]
            for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS
        }
        intervals.append(
            {
                "start_timestamp_epoch": previous["timestamp_epoch"],
                "end_timestamp_epoch": current["timestamp_epoch"],
                "start_timestamp_iso": previous["timestamp_iso"],
                "end_timestamp_iso": current["timestamp_iso"],
                "duration_seconds": duration,
                **{f"start_{field}": previous[field] for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS},
                **{f"end_{field}": current[field] for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS},
                **token_deltas,
                **{
                    f"{field.removesuffix('_tokens_cumulative')}_tok_per_s": (
                        token_deltas[field.removesuffix("_cumulative")] / duration
                    )
                    for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS
                },
                "cache_hit_ratio": _safe_ratio(
                    token_deltas["cached_tokens"],
                    token_deltas["cached_tokens"] + token_deltas["prefill_tokens"],
                ),
                "start_idle": previous["idle"],
                "end_idle": current["idle"],
            }
        )
    return intervals


def _token_progress_window_summary(
    intervals: list[dict[str, Any]],
    *,
    window_begin: float,
    window_end: float,
) -> dict[str, Any]:
    contained = [
        interval
        for interval in intervals
        if window_begin <= interval["start_timestamp_epoch"] and interval["end_timestamp_epoch"] <= window_end
    ]
    boundary_crossing = [
        interval
        for interval in intervals
        if interval["start_timestamp_epoch"] < window_end
        and window_begin < interval["end_timestamp_epoch"]
        and interval not in contained
    ]
    covered_seconds = sum(interval["duration_seconds"] for interval in contained)
    token_totals = {
        field.removesuffix("_cumulative"): sum(interval[field.removesuffix("_cumulative")] for interval in contained)
        for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS
    }
    result: dict[str, Any] = {
        "availability": "available" if contained else "unavailable",
        "window_begin_epoch": window_begin,
        "window_end_epoch": window_end,
        "window_seconds": window_end - window_begin,
        "interval_count": len(contained),
        "covered_interval_seconds": covered_seconds,
        "window_coverage_ratio": _safe_ratio(covered_seconds, window_end - window_begin),
        "boundary_crossing_interval_count_excluded": len(boundary_crossing),
        "intervals": contained,
        "boundary_crossing_intervals_excluded": boundary_crossing,
    }
    if contained:
        result.update(
            {
                **token_totals,
                **{
                    f"{field.removesuffix('_tokens_cumulative')}_tok_per_s": _safe_ratio(
                        token_totals[field.removesuffix("_cumulative")], covered_seconds
                    )
                    for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS
                },
                "cache_hit_ratio": _safe_ratio(
                    token_totals["cached_tokens"],
                    token_totals["cached_tokens"] + token_totals["prefill_tokens"],
                ),
                "per_interval_tok_per_s": {
                    field.removesuffix("_tokens_cumulative"): _number_summary(
                        [interval[f"{field.removesuffix('_tokens_cumulative')}_tok_per_s"] for interval in contained]
                    )
                    for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS
                },
            }
        )
    return result


def _rid_classification(rids: list[str], permit_kinds: dict[str, str]) -> dict[str, Any]:
    external_rids = [rid for rid in rids if not rid.startswith(HEALTH_CHECK_RID_PREFIX)]
    health_check_count = len(rids) - len(external_rids)
    kinds = Counter()
    matched = 0
    classified = 0
    for rid in external_rids:
        kind = permit_kinds.get(rid)
        if kind is not None:
            matched += 1
        if kind in {"fresh", "resume"}:
            kinds[kind] += 1
            classified += 1
        else:
            kinds["unknown"] += 1
    return {
        "rid_count": len(rids),
        "external_rid_count": len(external_rids),
        "health_check_rid_count": health_check_count,
        "exact_permit_id_match_count": matched,
        "exact_permit_id_match_coverage_ratio": _safe_ratio(matched, len(external_rids)),
        "fresh_resume_classified_count": classified,
        "fresh_resume_classification_coverage_ratio": _safe_ratio(classified, len(external_rids)),
        "kind_counts": {
            "fresh": kinds.get("fresh", 0),
            "resume": kinds.get("resume", 0),
            "unknown_external": kinds.get("unknown", 0),
            "health_check": health_check_count,
        },
    }


def _scheduler_snapshot_summary(
    snapshot: dict[str, Any],
    permit_kinds: dict[str, str],
    *,
    gate_begin: float,
) -> dict[str, Any]:
    running_rids = snapshot["running_rids"]
    queued_rids = snapshot["queued_rids"]
    return {
        "timestamp_iso": snapshot["timestamp_iso"],
        "timestamp_epoch": snapshot["timestamp_epoch"],
        "age_at_gate_begin_seconds": gate_begin - snapshot["timestamp_epoch"],
        "idle": snapshot["idle"],
        **{field: snapshot[field] for field in SCHEDULER_CUMULATIVE_TOKEN_FIELDS},
        "forward_mode": snapshot["forward_mode"],
        "running_request_count": snapshot["running_request_count"],
        "queued_request_count": snapshot["queued_request_count"],
        "running_seq_lens": _number_summary(snapshot["running_seq_lens"]),
        "running_origin_input_lens": _number_summary(snapshot["running_origin_input_lens"]),
        "running_output_lens": _number_summary(snapshot["running_output_lens"]),
        "queued_origin_input_lens": _number_summary(snapshot["queued_origin_input_lens"]),
        "queued_output_lens": _number_summary(snapshot["queued_output_lens"]),
        "running_rid_classification": _rid_classification(running_rids, permit_kinds),
        "queued_rid_classification": _rid_classification(queued_rids, permit_kinds),
        "all_rid_classification": _rid_classification([*running_rids, *queued_rids], permit_kinds),
    }


def _scheduler_gate_state_continuity(
    gate: dict[str, Any],
    engine_pid: str,
    snapshots: list[dict[str, Any]],
    *,
    health_threshold_seconds: float,
    is_long_gate: bool,
) -> dict[str, Any]:
    begin = gate["gate_begin_epoch"]
    end = gate["gate_end_epoch"]
    before = [snapshot for snapshot in snapshots if snapshot["timestamp_epoch"] <= begin]
    if not before:
        if is_long_gate:
            raise CalibrationInputError(
                f"missing_scheduler_snapshot_before_long_gate:logical={gate['logical_step']}:engine_pid={engine_pid}"
            )
        return {
            "status": "missing_state_at_gate_begin",
            "required_for_validity": False,
            "state_semantics": "interval heartbeat with explicit idle/active state",
        }

    current = before[-1]
    initial_idle = current["idle"]
    current_idle = initial_idle
    last_observation_epoch = current["timestamp_epoch"]
    heartbeat_gaps: list[float] = []
    violations: list[str] = []
    idle_transition_count = 0
    active_transition_count = 0

    start_gap = begin - last_observation_epoch
    heartbeat_gaps.append(start_gap)
    if start_gap > health_threshold_seconds:
        violations.append(f"heartbeat_start_gap={start_gap:.6f}")

    inside_after_begin = [snapshot for snapshot in snapshots if begin < snapshot["timestamp_epoch"] <= end]
    for snapshot in inside_after_begin:
        gap = snapshot["timestamp_epoch"] - last_observation_epoch
        heartbeat_gaps.append(gap)
        if gap > health_threshold_seconds:
            violations.append(f"heartbeat_observation_gap={gap:.6f}")
        if snapshot["idle"] != current_idle:
            if snapshot["idle"]:
                idle_transition_count += 1
            else:
                active_transition_count += 1
        current_idle = snapshot["idle"]
        last_observation_epoch = snapshot["timestamp_epoch"]

    tail_gap = end - last_observation_epoch
    heartbeat_gaps.append(tail_gap)
    if tail_gap > health_threshold_seconds:
        violations.append(f"heartbeat_end_gap={tail_gap:.6f}")

    if is_long_gate and violations:
        raise CalibrationInputError(
            f"incomplete_scheduler_heartbeat_during_long_gate:logical={gate['logical_step']}:"
            f"engine_pid={engine_pid}:violations={','.join(violations)}:"
            f"threshold={health_threshold_seconds:.6f}"
        )
    return {
        "status": "pass" if not violations else "insufficient_short_gate_state_evidence",
        "required_for_validity": is_long_gate,
        "state_semantics": "interval heartbeat with explicit idle/active state",
        "initial_state": "idle" if initial_idle else "active",
        "final_state": "idle" if current_idle else "active",
        "snapshot_count_after_gate_begin": len(inside_after_begin),
        "idle_transition_count": idle_transition_count,
        "active_transition_count": active_transition_count,
        "heartbeat_gap_seconds": _number_summary(heartbeat_gaps),
        "violations": violations,
    }


def _scheduler_interval_summary(
    snapshots: list[dict[str, Any]],
    permit_kinds: dict[str, str],
) -> dict[str, Any]:
    running_rids = [rid for snapshot in snapshots for rid in snapshot["running_rids"]]
    queued_rids = [rid for snapshot in snapshots for rid in snapshot["queued_rids"]]
    return {
        "snapshot_count": len(snapshots),
        "forward_mode_counts": dict(sorted(Counter(snapshot["forward_mode"] for snapshot in snapshots).items())),
        "running_request_count_per_snapshot": _number_summary(
            [snapshot["running_request_count"] for snapshot in snapshots]
        ),
        "queued_request_count_per_snapshot": _number_summary(
            [snapshot["queued_request_count"] for snapshot in snapshots]
        ),
        "running_seq_len_sum_per_snapshot": _number_summary(
            [sum(snapshot["running_seq_lens"]) for snapshot in snapshots]
        ),
        "running_origin_input_len_sum_per_snapshot": _number_summary(
            [sum(snapshot["running_origin_input_lens"]) for snapshot in snapshots]
        ),
        "running_output_len_sum_per_snapshot": _number_summary(
            [sum(snapshot["running_output_lens"]) for snapshot in snapshots]
        ),
        "queued_origin_input_len_sum_per_snapshot": _number_summary(
            [sum(snapshot["queued_origin_input_lens"]) for snapshot in snapshots]
        ),
        "queued_output_len_sum_per_snapshot": _number_summary(
            [sum(snapshot["queued_output_lens"]) for snapshot in snapshots]
        ),
        "running_seq_lens": _number_summary(
            [value for snapshot in snapshots for value in snapshot["running_seq_lens"]]
        ),
        "running_origin_input_lens": _number_summary(
            [value for snapshot in snapshots for value in snapshot["running_origin_input_lens"]]
        ),
        "running_output_lens": _number_summary(
            [value for snapshot in snapshots for value in snapshot["running_output_lens"]]
        ),
        "queued_origin_input_lens": _number_summary(
            [value for snapshot in snapshots for value in snapshot["queued_origin_input_lens"]]
        ),
        "queued_output_lens": _number_summary(
            [value for snapshot in snapshots for value in snapshot["queued_output_lens"]]
        ),
        "running_rid_occurrence_classification": _rid_classification(running_rids, permit_kinds),
        "queued_rid_occurrence_classification": _rid_classification(queued_rids, permit_kinds),
        "all_rid_occurrence_classification": _rid_classification([*running_rids, *queued_rids], permit_kinds),
        "unique_rid_classification": _rid_classification(sorted(set(running_rids + queued_rids)), permit_kinds),
    }


def _scheduler_gate_evidence(
    gate: dict[str, Any],
    snapshots_by_engine: dict[str, list[dict[str, Any]]],
    permit_kinds: dict[str, str],
    *,
    health_threshold_seconds: float,
    long_gate_floor_seconds: float,
) -> dict[str, Any]:
    begin = gate["gate_begin_epoch"]
    end = gate["gate_end_epoch"]
    is_long_gate = gate["gate_wall_seconds"] >= long_gate_floor_seconds
    engines: dict[str, Any] = {}
    for engine_pid, snapshots in sorted(snapshots_by_engine.items()):
        state_continuity = _scheduler_gate_state_continuity(
            gate,
            engine_pid,
            snapshots,
            health_threshold_seconds=health_threshold_seconds,
            is_long_gate=is_long_gate,
        )
        before = [snapshot for snapshot in snapshots if snapshot["timestamp_epoch"] <= begin]
        if not before:
            nearest_summary = {
                "availability": "unavailable",
                "health_status": "missing",
                "required_for_validity": False,
                "freshness_required": False,
            }
        else:
            nearest = before[-1]
            age = begin - nearest["timestamp_epoch"]
            is_fresh = age <= health_threshold_seconds
            nearest_summary = {
                **_scheduler_snapshot_summary(
                    nearest,
                    permit_kinds,
                    gate_begin=begin,
                ),
                "availability": "available",
                "health_status": (
                    f"{'fresh' if is_fresh else 'stale'}_{'idle' if nearest['idle'] else 'active'}_state"
                ),
                "required_for_validity": is_long_gate,
                "freshness_required": True,
            }
        inside = [snapshot for snapshot in snapshots if begin <= snapshot["timestamp_epoch"] <= end]
        gate_interval = _scheduler_interval_summary(inside, permit_kinds)
        gate_interval["token_progress_exact"] = _token_progress_window_summary(
            _scheduler_token_intervals(snapshots),
            window_begin=begin,
            window_end=end,
        )
        engines[engine_pid] = {
            "nearest_at_or_before_gate_begin": nearest_summary,
            "state_continuity": state_continuity,
            "gate_interval": gate_interval,
        }
    return {
        "health_threshold_seconds": health_threshold_seconds,
        "long_gate_floor_seconds": long_gate_floor_seconds,
        "is_long_gate": is_long_gate,
        "engine_count": len(engines),
        "engines": engines,
    }


def _headline_scheduler_rid_health(
    gates: list[dict[str, Any]],
    snapshots_by_engine: dict[str, list[dict[str, Any]]],
    permit_kinds: dict[str, str],
    *,
    long_gate_floor_seconds: float,
) -> dict[str, Any]:
    all_external_rids: set[str] = set()
    all_health_check_rids: set[str] = set()
    external_rid_engines: dict[str, set[str]] = {}
    per_gate: dict[str, Any] = {}
    long_gates = [gate for gate in gates if gate["gate_wall_seconds"] >= long_gate_floor_seconds]
    for gate in gates:
        gate_rids: set[str] = set()
        gate_rid_engines: dict[str, set[str]] = {}
        for engine_pid, snapshots in snapshots_by_engine.items():
            before = [snapshot for snapshot in snapshots if snapshot["timestamp_epoch"] <= gate["gate_begin_epoch"]]
            selected = ([before[-1]] if before else []) + [
                snapshot
                for snapshot in snapshots
                if gate["gate_begin_epoch"] <= snapshot["timestamp_epoch"] <= gate["gate_end_epoch"]
            ]
            for snapshot in selected:
                for rid in [*snapshot["running_rids"], *snapshot["queued_rids"]]:
                    gate_rids.add(rid)
                    gate_rid_engines.setdefault(rid, set()).add(engine_pid)
        health_check_rids = {rid for rid in gate_rids if rid.startswith(HEALTH_CHECK_RID_PREFIX)}
        external_rids = gate_rids - health_check_rids
        exact_matches = external_rids & permit_kinds.keys()
        all_external_rids.update(external_rids)
        all_health_check_rids.update(health_check_rids)
        for rid in external_rids:
            external_rid_engines.setdefault(rid, set()).update(gate_rid_engines[rid])
        per_gate[str(gate["logical_step"])] = {
            "is_long_gate": gate in long_gates,
            "external_unique_rid_count": len(external_rids),
            "health_check_unique_rid_count": len(health_check_rids),
            "exact_permit_id_match_count": len(exact_matches),
            "exact_permit_id_match_coverage_ratio": _safe_ratio(len(exact_matches), len(external_rids)),
            "cross_engine_external_rid_count": sum(len(gate_rid_engines[rid]) != 1 for rid in external_rids),
        }
    all_exact_matches = all_external_rids & permit_kinds.keys()
    cross_engine_rids = {rid for rid, engine_pids in external_rid_engines.items() if len(engine_pids) != 1}
    coverage = _safe_ratio(len(all_exact_matches), len(all_external_rids))
    if not all_external_rids:
        status = "not_applicable_no_external_work"
    elif cross_engine_rids:
        status = "invalid_cross_engine_rid_ownership"
    elif not all_exact_matches:
        status = "invalid_zero_exact_matches"
    elif coverage is not None and coverage < HEADLINE_RID_MATCH_MIN_COVERAGE:
        status = "invalid_below_minimum_coverage"
    else:
        status = "pass"
    return {
        "long_gate_count": len(long_gates),
        "long_gate_floor_seconds": long_gate_floor_seconds,
        "health_check_rid_prefix": HEALTH_CHECK_RID_PREFIX,
        "external_unique_rid_count": len(all_external_rids),
        "health_check_unique_rid_count": len(all_health_check_rids),
        "exact_permit_id_match_count": len(all_exact_matches),
        "exact_permit_id_match_coverage_ratio": coverage,
        "minimum_required_coverage_ratio": HEADLINE_RID_MATCH_MIN_COVERAGE,
        "cross_engine_external_rid_count": len(cross_engine_rids),
        "single_engine_ownership_coverage_ratio": _safe_ratio(
            len(all_external_rids) - len(cross_engine_rids),
            len(all_external_rids),
        ),
        "status": status,
        "per_gate": per_gate,
    }


def _gate_evidence(
    gate: dict[str, Any],
    waits: list[dict[str, Any]],
    snapshots_by_engine: dict[str, list[dict[str, Any]]],
    permit_kinds: dict[str, str],
    *,
    health_threshold_seconds: float,
    long_gate_floor_seconds: float,
) -> dict[str, Any]:
    begin = gate["gate_begin_epoch"]
    end = gate["gate_end_epoch"]
    wall = gate["gate_wall_seconds"]
    at_begin = [row for row in waits if row["_wait_begin_epoch"] <= begin < row["_wait_end_epoch"]]
    overlaps: list[tuple[dict[str, Any], float, float]] = []
    for row in waits:
        clipped_begin = max(begin, row["_wait_begin_epoch"])
        clipped_end = min(end, row["_wait_end_epoch"])
        if clipped_begin < clipped_end:
            overlaps.append((row, clipped_begin, clipped_end))
    overlap_rows = [row for row, _, _ in overlaps]
    slot_seconds = sum(clipped_end - clipped_begin for _, clipped_begin, clipped_end in overlaps)
    any_waiter_seconds = _union_seconds([(clipped_begin, clipped_end) for _, clipped_begin, clipped_end in overlaps])
    cancelled_inside = [
        row
        for row in waits
        if row["permit_wait_status"] == "cancelled_before_grant" and begin <= row["terminal_epoch"] < end
    ]
    begin_groups, begin_missing_groups = _unique_groups(at_begin)
    overlap_groups, overlap_missing_groups = _unique_groups(overlap_rows)
    return {
        **gate,
        "waiting_at_gate_begin": {
            "request_count": len(at_begin),
            "attempt_kind_counts": _kind_counts(at_begin),
            "unique_group_count": begin_groups,
            "missing_group_index_count": begin_missing_groups,
        },
        "permit_wait_overlap": {
            "request_count": len(overlap_rows),
            "attempt_kind_counts": _kind_counts(overlap_rows),
            "unique_group_count": overlap_groups,
            "missing_group_index_count": overlap_missing_groups,
            "waiter_slot_seconds": slot_seconds,
            "gate_seconds_with_any_waiter": any_waiter_seconds,
            "gate_time_coverage_ratio": _safe_ratio(any_waiter_seconds, wall),
            "mean_waiters_per_gate_second": _safe_ratio(slot_seconds, wall),
            "inline_metrics": _inline_metrics(overlap_rows),
            "sglang_queue_prefill_timing": _sglang_timing_summary(overlap_rows),
        },
        "cancelled_before_grant_inside_gate": {
            "request_count": len(cancelled_inside),
            "attempt_kind_counts": _kind_counts(cancelled_inside),
            "unique_group_count": _unique_groups(cancelled_inside)[0],
        },
        "scheduler": _scheduler_gate_evidence(
            gate,
            snapshots_by_engine,
            permit_kinds,
            health_threshold_seconds=health_threshold_seconds,
            long_gate_floor_seconds=long_gate_floor_seconds,
        ),
    }


def _invalid_result(
    driver_log: Path,
    calibration_dir: Path,
    headline_lo: int,
    headline_hi: int,
    scheduler_interval_seconds: float,
    error: str,
    *,
    scheduler_rid_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed_scheduler_interval = _finite_float(scheduler_interval_seconds)
    health_threshold = (
        max(2.5 * parsed_scheduler_interval, SCHEDULER_HEALTH_FLOOR_SECONDS)
        if parsed_scheduler_interval is not None and parsed_scheduler_interval > 0
        else None
    )
    scheduler = {
        "configured_interval_seconds": parsed_scheduler_interval,
        "health_threshold_seconds": health_threshold,
        "health_threshold_formula": "max(2.5 * scheduler_interval_seconds, 5.0)",
        "long_gate_floor_seconds": SCHEDULER_HEALTH_FLOOR_SECONDS,
    }
    if scheduler_rid_health is not None:
        scheduler["headline_rid_health"] = scheduler_rid_health
    return {
        "schema_version": 1,
        "verdict": "INVALID_INPUT",
        "decision": "NO_OPTIMIZATION_DECISION",
        "headline": {
            "logical_step_lo": headline_lo,
            "logical_step_hi": headline_hi,
            "expected_sync_id_lo": headline_lo + 1,
            "expected_sync_id_hi": headline_hi + 1,
        },
        "scheduler": scheduler,
        "inputs": {"driver_log": str(driver_log), "calibration_dir": str(calibration_dir)},
        "errors": [error],
    }


def analyze(
    driver_log: Path,
    calibration_dir: Path,
    *,
    headline_lo: int = 5,
    headline_hi: int = 14,
    scheduler_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Measure gate/permit overlap without making an optimization decision."""

    if headline_lo < 0 or headline_hi < headline_lo:
        return _invalid_result(
            driver_log,
            calibration_dir,
            headline_lo,
            headline_hi,
            scheduler_interval_seconds,
            f"invalid_headline_range:{headline_lo}..{headline_hi}",
        )
    parsed_scheduler_interval = _finite_float(scheduler_interval_seconds)
    if parsed_scheduler_interval is None or parsed_scheduler_interval <= 0:
        return _invalid_result(
            driver_log,
            calibration_dir,
            headline_lo,
            headline_hi,
            scheduler_interval_seconds,
            f"invalid_scheduler_interval_seconds:{scheduler_interval_seconds!r}",
        )
    health_threshold_seconds = max(2.5 * parsed_scheduler_interval, SCHEDULER_HEALTH_FLOOR_SECONDS)
    try:
        persistence_summary, manifest_paths = _validate_persisted_artifacts(
            calibration_dir,
            driver_log,
        )
        run_artifacts = _validate_run_artifacts(
            calibration_dir,
            headline_lo=headline_lo,
            headline_hi=headline_hi,
        )
        standard_metrics = _headline_standard_metrics(driver_log, headline_lo, headline_hi)
        gates = _headline_gates(driver_log, headline_lo, headline_hi)
        actor_timelines = _headline_actor_timelines(driver_log, gates, headline_lo, headline_hi)
        permit_paths, permit_rows = _read_permit_rows(calibration_dir)
        unlisted_permit_paths = sorted(path.name for path in permit_paths if path.name not in manifest_paths)
        if unlisted_permit_paths:
            raise CalibrationInputError(f"artifact_manifest_missing_permit_files:{unlisted_permit_paths}")
        permit_rollout_coverage = _validate_permit_rollout_coverage(
            permit_paths,
            permit_rows,
            num_rollout=run_artifacts["run_contract"]["num_rollout"],
        )
        waits = _validate_permit_rows(permit_rows)
        snapshots_by_engine = _scheduler_snapshots(driver_log)
        permit_kinds = {
            row["permit_wait_id"]: (
                row["attempt_kind"] if row.get("attempt_kind") in {"fresh", "resume"} else "unknown"
            )
            for row in waits
        }
        gate_results = [
            _gate_evidence(
                gate,
                waits,
                snapshots_by_engine,
                permit_kinds,
                health_threshold_seconds=health_threshold_seconds,
                long_gate_floor_seconds=SCHEDULER_HEALTH_FLOOR_SECONDS,
            )
            for gate in gates
        ]
        for gate_result in gate_results:
            gate_result["actor_timeline"] = actor_timelines[gate_result["logical_step"]]
        overlap_wait_ids = {
            row["permit_wait_id"]
            for row in waits
            if any(
                max(gate["gate_begin_epoch"], row["_wait_begin_epoch"])
                < min(gate["gate_end_epoch"], row["_wait_end_epoch"])
                for gate in gates
            )
        }
        overlap_rows = [row for row in waits if row["permit_wait_id"] in overlap_wait_ids]
        headline_physical_rows = [row for row in waits if headline_lo <= row["physical_rollout_id"] <= headline_hi]
        _validate_headline_sglang_timing_coverage(headline_physical_rows)
    except CalibrationInputError as exc:
        return _invalid_result(
            driver_log,
            calibration_dir,
            headline_lo,
            headline_hi,
            parsed_scheduler_interval,
            str(exc),
        )

    scheduler_rid_health = _headline_scheduler_rid_health(
        gates,
        snapshots_by_engine,
        permit_kinds,
        long_gate_floor_seconds=SCHEDULER_HEALTH_FLOOR_SECONDS,
    )
    if scheduler_rid_health["status"] != "pass" and not scheduler_rid_health["status"].startswith("not_applicable"):
        return _invalid_result(
            driver_log,
            calibration_dir,
            headline_lo,
            headline_hi,
            parsed_scheduler_interval,
            "invalid_headline_scheduler_rid_health:"
            f"status={scheduler_rid_health['status']}:"
            f"external={scheduler_rid_health['external_unique_rid_count']}:"
            f"exact={scheduler_rid_health['exact_permit_id_match_count']}:"
            f"coverage={scheduler_rid_health['exact_permit_id_match_coverage_ratio']}",
            scheduler_rid_health=scheduler_rid_health,
        )

    total_wall = sum(gate["gate_wall_seconds"] for gate in gate_results)
    total_any_waiter = sum(gate["permit_wait_overlap"]["gate_seconds_with_any_waiter"] for gate in gate_results)
    total_slot_seconds = sum(gate["permit_wait_overlap"]["waiter_slot_seconds"] for gate in gate_results)
    cycles_with_waiters = sum(gate["permit_wait_overlap"]["request_count"] > 0 for gate in gate_results)
    status_counts = Counter(row["permit_wait_status"] for row in waits)
    headline_snapshots_by_engine = {
        engine_pid: [
            snapshot
            for snapshot in snapshots
            if any(gate["gate_begin_epoch"] <= snapshot["timestamp_epoch"] <= gate["gate_end_epoch"] for gate in gates)
        ]
        for engine_pid, snapshots in snapshots_by_engine.items()
    }
    headline_begin = gates[0]["gate_begin_epoch"]
    headline_end = gates[-1]["gate_end_epoch"]
    headline_token_progress = {
        engine_pid: _token_progress_window_summary(
            _scheduler_token_intervals(snapshots),
            window_begin=headline_begin,
            window_end=headline_end,
        )
        for engine_pid, snapshots in sorted(snapshots_by_engine.items())
    }
    return {
        "schema_version": 1,
        "verdict": "RECALIBRATE_REQUIRED",
        "decision": "EVIDENCE_ONLY_NO_AUTOMATIC_GO",
        "headline": {
            "logical_step_lo": headline_lo,
            "logical_step_hi": headline_hi,
            "expected_sync_id_lo": headline_lo + 1,
            "expected_sync_id_hi": headline_hi + 1,
            "gate_count": len(gate_results),
            "mapping": "sync_id = logical_step + 1",
        },
        "inputs": {
            "driver_log": str(driver_log),
            "calibration_dir": str(calibration_dir),
            "permit_wait_files": [path.name for path in permit_paths],
            "validated_run_artifacts": run_artifacts,
            "validated_persistence": persistence_summary,
            "permit_physical_rollout_coverage": permit_rollout_coverage,
        },
        "scheduler": {
            "configured_interval_seconds": parsed_scheduler_interval,
            "health_threshold_seconds": health_threshold_seconds,
            "health_threshold_formula": "max(2.5 * scheduler_interval_seconds, 5.0)",
            "long_gate_floor_seconds": SCHEDULER_HEALTH_FLOOR_SECONDS,
            "long_gate_count": scheduler_rid_health["long_gate_count"],
            "headline_rid_health": scheduler_rid_health,
            "engine_pids": sorted(snapshots_by_engine),
            "snapshot_counts": {
                engine_pid: len(snapshots) for engine_pid, snapshots in sorted(snapshots_by_engine.items())
            },
            "headline_interval_token_progress_exact": headline_token_progress,
        },
        "semantics": {
            "wait_interval": "permit_acquire_begin_epoch to grant; cancelled/ended-before-grant uses terminal_epoch.",
            "waiter_slot_seconds": (
                "Sum of clipped permit-wait/gate overlap; one not-yet-granted request contributes one "
                "request-slot-second."
            ),
            "gate_time_coverage_ratio": (
                "Fraction of gate wall containing at least one observable waiter; queue opportunity, not speedup."
            ),
            "inline_metrics": "Only finite fields present in permit rows are summarized; no value is inferred.",
            "sglang_queue_prefill_timing": (
                "SGLang converts internal perf-counter forward/prefill timestamps to realtime before exposing HTTP "
                "meta, so the exported *_epoch fields are epoch values but only same-source duration differences are "
                f"used. Complete tuples are validated and summarized by fresh/resume. Every physical rollout "
                f"{headline_lo}..{headline_hi} "
                "terminal request with engine_request_started=true and exception_type=None requires complete timing "
                "(100% coverage); terminal requests stopped before HTTP dispatch, local exceptions, and "
                "cancelled-before-grant rows are excluded. Partial or inconsistent eligible tuples are invalid."
            ),
            "scheduler_rid_mapping": (
                "A scheduler RID is mapped only when it exactly equals a permit_wait_id; every other RID is unknown."
            ),
            "scheduler_timestamp": (
                "Every status requires finite positive timestamp_epoch, which is the sole authoritative clock. The "
                "SGLang-injected ISO timestamp may be naive or timezone-aware and is retained only for display."
            ),
            "scheduler_health": (
                "SGLang emits interval scheduler status while both idle and active. Every long gate requires both "
                "boundary gaps and every adjacent heartbeat gap within the health threshold for each engine; idle is "
                "a state classification, not permission to carry stale evidence. Short gates report incomplete "
                "evidence without invalidating the run."
            ),
            "scheduler_token_progress_exact": (
                "Decode, prefill-compute, and cache-hit cumulative token counters are monotonic per engine. Exact "
                "adjacent-snapshot delta/dt is aggregated only for intervals fully contained in a gate/headline "
                "window; boundary-crossing intervals are disclosed and excluded rather than proportionally inferred. "
                "cache_hit_ratio is cached/(cached+prefill); seq_len is never treated as batch-token work."
            ),
            "scheduler_request_shapes": (
                "running/queued seq, origin-input, and output lengths are sampled request-shape evidence only. Their "
                "per-snapshot sums are named *_len_sum_per_snapshot and are never used as processed-token counters."
            ),
            "actor_idle_before_weight_sync_seconds": (
                "Measured from actor_train end to weight_sync begin; includes the gate and any observed pre/post-gate gaps."
            ),
            "hybrid_subbatch_timeline": (
                "first_subbatch_wait_seconds is pure wait until the first mini is fetched. "
                "all_subbatches_ready_wait_seconds ends when the final mini is fetched but may include forward work "
                "for earlier minis, so it is not pure Transfer Queue wait. post_last_subbatch_forward_prepare_seconds "
                "covers the final mini forward plus merge/advantage/data-iterator preparation before actor training."
            ),
            "standard_pipeline_metrics": (
                "Existing Actor perf logs provide the end-to-end step headline and train-wait/forward/train/sync "
                "decomposition. Existing train logs provide TIS and mismatch health. Existing Actor rollout logs "
                "provide response-length mean plus global total-length min/max and reward summaries. Response median "
                "and truncated ratio are not emitted reliably on this Hybrid path and are not inferred. The standard "
                "perf/step_time timer is reset before the same-numbered gate and therefore includes prior-cycle "
                "publication time inside train_wait; do not subtract or add it to the same-numbered gate timeline. "
                "The TASK22 actor timeline is the same-logical-step data_wait_begin-to-weight_sync-end view."
            ),
        },
        "gates": gate_results,
        "summary": {
            "pipeline_headline": standard_metrics,
            "movable_queue_coverage": {
                "headline_gate_wall_seconds": total_wall,
                "gate_seconds_with_any_waiter": total_any_waiter,
                "gate_time_coverage_ratio": _safe_ratio(total_any_waiter, total_wall),
                "waiter_slot_seconds": total_slot_seconds,
                "mean_waiters_per_gate_second": _safe_ratio(total_slot_seconds, total_wall),
                "gate_cycles_with_waiters": cycles_with_waiters,
                "gate_cycle_coverage_ratio": _safe_ratio(cycles_with_waiters, len(gate_results)),
                "unique_overlap_wait_count": len(overlap_wait_ids),
            },
            "permit_wait": {
                "row_count": len(waits),
                "status_counts": dict(sorted(status_counts.items())),
                "attempt_kind_counts": _kind_counts(waits),
                "unique_group_count": _unique_groups(waits)[0],
            },
            "headline_overlap_data_completeness": _inline_metrics(overlap_rows),
            "sglang_queue_prefill_timing": {
                "all_terminal_granted": _sglang_timing_summary(waits),
                "headline_physical_rollouts_terminal_granted": _sglang_timing_summary(headline_physical_rows),
                "headline_overlap_terminal_granted": _sglang_timing_summary(overlap_rows),
            },
            "headline_gate_interval_scheduler": {
                engine_pid: _scheduler_interval_summary(snapshots, permit_kinds)
                for engine_pid, snapshots in sorted(headline_snapshots_by_engine.items())
            },
            "actor_timeline": {
                field: _number_summary([timeline[field] for timeline in actor_timelines.values()])
                for field in (
                    "first_subbatch_wait_seconds",
                    "all_subbatches_ready_wait_seconds",
                    "remaining_subbatches_after_first_seconds",
                    "post_last_subbatch_forward_prepare_seconds",
                    "actor_train_seconds",
                    "gate_seconds",
                    "weight_sync_seconds",
                    "actor_idle_before_weight_sync_seconds",
                    "post_train_before_gate_seconds",
                    "post_gate_before_weight_sync_seconds",
                    "observed_pipeline_seconds",
                )
            },
        },
    }


def _atomic_write(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(rendered)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--headline-lo", type=int, required=True)
    parser.add_argument("--headline-hi", type=int, required=True)
    parser.add_argument("--scheduler-interval-seconds", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = analyze(
        args.driver_log,
        args.calibration_dir,
        headline_lo=args.headline_lo,
        headline_hi=args.headline_hi,
        scheduler_interval_seconds=args.scheduler_interval_seconds,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        _atomic_write(args.output_json, rendered)
    raise SystemExit(2 if result["verdict"] == "INVALID_INPUT" else 0)


if __name__ == "__main__":
    main()
