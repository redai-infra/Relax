#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${TASK22_PYTHON:?Set TASK22_PYTHON to an absolute executable launcher}"
BASE_PATCH="$REPO/scripts/task22/sglang_rollout_observability.patch"
IDLE_PATCH="$REPO/scripts/task22/sglang_idle_heartbeat.patch"
LEGACY_IDLE_PATCH="$REPO/scripts/task22/sglang_idle_heartbeat_legacy_upgrade.patch"
RID_ONLY_PATCH="$REPO/scripts/task22/sglang_rid_only_request_logging.patch"

SGLANG_IMPORT_ROOT="$(
    "$PYTHON_BIN" -c \
        'from pathlib import Path; from sglang.srt.utils import scheduler_status_logger; print(Path(scheduler_status_logger.__file__).resolve().parents[3])'
)"
TIMING_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/observability/req_time_stats.py"
METRICS_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/observability/scheduler_metrics_mixin.py"
STATUS_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/utils/scheduler_status_logger.py"
IDLE_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/managers/scheduler_runtime_checker_mixin.py"
REQUEST_LOGGER_FILE="$SGLANG_IMPORT_ROOT/sglang/srt/utils/request_logger.py"

if grep -Eq -- "_relax_forward_timing_payload|_task22_forward_timing_payload" "$TIMING_FILE" \
    && grep -Eq -- "RELAX_REQUEST_SHAPE_OBSERVABILITY|TASK22_REQUEST_SHAPE_PREFILL_STATUS" "$METRICS_FILE" \
    && grep -q -- '"running_seq_lens"' "$STATUS_FILE"; then
    printf '%s\n' "SGLang timing/shape calibration patch already present"
elif patch --dry-run --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$BASE_PATCH" >/dev/null 2>&1; then
    patch --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$BASE_PATCH"
    printf '%s\n' "Applied SGLang timing/shape calibration patch"
else
    printf '%s\n' "SGLang timing/shape patch state is incompatible" >&2
    exit 4
fi

if grep -q -- "RELAX_SCHEDULER_IDLE_HEARTBEAT" "$IDLE_FILE" \
    && grep -q -- '"idle": idle' "$STATUS_FILE" \
    && grep -q -- '"timestamp_epoch": now' "$STATUS_FILE" \
    && grep -q -- '"decode_tokens_cumulative"' "$STATUS_FILE" \
    && grep -q -- '"prefill_tokens_cumulative"' "$STATUS_FILE" \
    && grep -q -- '"cached_tokens_cumulative"' "$STATUS_FILE" \
    && grep -q -- 'decode_tokens=decode_tokens' "$METRICS_FILE" \
    && grep -q -- 'prefill_tokens=prefill_stats.log_input_tokens' "$METRICS_FILE" \
    && grep -q -- 'cached_tokens=prefill_stats.log_hit_tokens' "$METRICS_FILE"; then
    printf '%s\n' "SGLang idle-transition calibration patch already present"
elif grep -q -- "TASK22_REQUEST_SHAPE_PREFILL_STATUS" "$METRICS_FILE" \
    && patch --dry-run --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$LEGACY_IDLE_PATCH" >/dev/null 2>&1; then
    patch --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$LEGACY_IDLE_PATCH"
    printf '%s\n' "Upgraded legacy SGLang shape observability to the calibration heartbeat contract"
elif patch --dry-run --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$IDLE_PATCH" >/dev/null 2>&1; then
    patch --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$IDLE_PATCH"
    printf '%s\n' "Applied SGLang idle-transition calibration patch"
else
    printf '%s\n' "SGLang idle-transition patch state is incompatible" >&2
    exit 4
fi

if grep -q -- "RELAX_RID_ONLY_REQUEST_LOGGING" "$REQUEST_LOGGER_FILE"; then
    printf '%s\n' "SGLang RID-only request logging patch already present"
elif patch --dry-run --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$RID_ONLY_PATCH" >/dev/null 2>&1; then
    patch --batch --forward -d "$SGLANG_IMPORT_ROOT" -p2 < "$RID_ONLY_PATCH"
    printf '%s\n' "Applied SGLang RID-only request logging patch"
else
    printf '%s\n' "SGLang RID-only request logging patch state is incompatible" >&2
    exit 4
fi

"$PYTHON_BIN" "$REPO/scripts/task22/smoke_sglang_calibration.py"
