# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in, prompt-free observability for Task 22 calibration runs."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from time import monotonic
from typing import Any


CALIBRATION_DIR_ENV = "TASK22_CALIBRATION_DIR"
CALIBRATION_RID_METADATA_KEY = "_task22_calibration_rid"
CALIBRATION_TIMING_METADATA_KEY = "_task22_sglang_timing"
CALIBRATION_ENGINE_REQUEST_STARTED_METADATA_KEY = "_task22_engine_request_started"


def output_dir() -> str | None:
    value = os.environ.get(CALIBRATION_DIR_ENV)
    return value if value else None


def begin_wait(
    sample: Any,
    *,
    physical_rollout_id: int | None,
    permit_snapshot: dict[str, int] | None,
) -> dict[str, Any]:
    begin_mono = monotonic()
    response_before = int(getattr(sample, "response_length", 0) or 0)
    tokens = getattr(sample, "tokens", None)
    wait_id = f"permit-wait:{uuid.uuid4().hex}"
    return {
        "schema_version": 1,
        "record_type": "permit_wait",
        "permit_wait_id": wait_id,
        "rid": wait_id,
        "physical_rollout_id": physical_rollout_id,
        "group_index": getattr(sample, "group_index", None),
        "sample_index": getattr(sample, "index", None),
        "abort_count": int(getattr(sample, "abort_count", 0) or 0),
        "attempt_kind": "resume" if response_before > 0 else "fresh",
        "response_tokens_before": response_before,
        "context_tokens_before": len(tokens) if tokens else None,
        "permit_acquire_begin_monotonic": begin_mono,
        "permit_acquire_begin_epoch": time.time(),
        "permit_snapshot_before": permit_snapshot,
        "permit_wait_status": "waiting",
    }


def mark_granted(row: dict[str, Any], permit_snapshot: dict[str, int] | None) -> None:
    granted_mono = monotonic()
    row.update(
        {
            "permit_acquire_granted_monotonic": granted_mono,
            "permit_acquire_granted_epoch": time.time(),
            "permit_acquire_wait_seconds": granted_mono - row["permit_acquire_begin_monotonic"],
            "permit_snapshot_after": permit_snapshot,
            "permit_wait_status": "granted",
        }
    )


def capture_sglang_timing(sample: Any, meta_info: dict[str, Any]) -> None:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return
    timing: dict[str, float] = {}
    for field in ("queue_time", "forward_entry_time", "prefill_finished_time"):
        value = meta_info.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        finite_value = float(value)
        if math.isfinite(finite_value):
            timing[field] = finite_value
    metadata[CALIBRATION_TIMING_METADATA_KEY] = timing


def mark_engine_request_started(sample: Any) -> None:
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        metadata[CALIBRATION_ENGINE_REQUEST_STARTED_METADATA_KEY] = True


def mark_terminal(row: dict[str, Any], sample: Any, error: BaseException | None = None) -> None:
    terminal_mono = monotonic()
    if row.get("permit_wait_status") == "waiting":
        row["permit_wait_status"] = "cancelled_before_grant" if error is not None else "ended_before_grant"
        row["permit_acquire_wait_seconds"] = terminal_mono - row["permit_acquire_begin_monotonic"]
    elif row.get("permit_wait_status") == "granted":
        row["permit_wait_status"] = "terminal"

    response_after = int(getattr(sample, "response_length", 0) or 0)
    generated_tokens = max(response_after - row["response_tokens_before"], 0)
    tokens_after = getattr(sample, "tokens", None)
    if tokens_after:
        context_tokens = len(tokens_after) - generated_tokens
        if context_tokens >= 0:
            row["context_tokens_before"] = context_tokens

    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        timing = metadata.pop(CALIBRATION_TIMING_METADATA_KEY, {})
        engine_request_started = metadata.pop(CALIBRATION_ENGINE_REQUEST_STARTED_METADATA_KEY, False)
    else:
        timing = {}
        engine_request_started = False
    queue_seconds = timing.get("queue_time")
    forward_epoch = timing.get("forward_entry_time")
    prefill_finished_epoch = timing.get("prefill_finished_time")
    prefill_seconds = None
    if (
        isinstance(forward_epoch, float)
        and isinstance(prefill_finished_epoch, float)
        and forward_epoch > 0
        and prefill_finished_epoch >= forward_epoch
    ):
        prefill_seconds = prefill_finished_epoch - forward_epoch
    queue_plus_prefill_seconds = None
    if isinstance(queue_seconds, float) and queue_seconds >= 0 and prefill_seconds is not None:
        queue_plus_prefill_seconds = queue_seconds + prefill_seconds

    row.update(
        {
            "terminal_monotonic": terminal_mono,
            "terminal_epoch": time.time(),
            "request_wall_seconds": (
                terminal_mono - row["permit_acquire_granted_monotonic"]
                if "permit_acquire_granted_monotonic" in row
                else None
            ),
            "response_tokens_after": response_after,
            "generated_tokens_this_attempt": generated_tokens,
            "sample_status": str(getattr(sample, "status", "unknown")),
            "exception_type": type(error).__name__ if error is not None else None,
            "engine_request_started": engine_request_started is True,
            "sglang_queue_seconds": queue_seconds if isinstance(queue_seconds, float) and queue_seconds >= 0 else None,
            "sglang_forward_entry_epoch": (
                forward_epoch if isinstance(forward_epoch, float) and forward_epoch > 0 else None
            ),
            "sglang_prefill_finished_epoch": (
                prefill_finished_epoch
                if isinstance(prefill_finished_epoch, float) and prefill_finished_epoch > 0
                else None
            ),
            "sglang_prefill_seconds": prefill_seconds,
            "sglang_queue_plus_prefill_seconds": queue_plus_prefill_seconds,
        }
    )


def export_rows(rows: list[dict[str, Any]], *, directory: str, physical_rollout_id: int) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"permit_wait_rollout_{physical_rollout_id}.jsonl"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output_path
