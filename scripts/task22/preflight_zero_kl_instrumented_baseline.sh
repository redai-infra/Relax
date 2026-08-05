#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${TASK22_PYTHON:?Set TASK22_PYTHON to the Relax environment python}"
EXPECTED_SHA="${TASK22_EXPECTED_GIT_SHA:?Set TASK22_EXPECTED_GIT_SHA}"
RUNTIME_ENV_JSON="${TASK22_RUNTIME_ENV_JSON:?Set TASK22_RUNTIME_ENV_JSON}"
TRAIN_ARGS_JSON="${TASK22_TRAIN_ARGS_JSON:?Set TASK22_TRAIN_ARGS_JSON}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR}"

cd "$REPO"

[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "Zero-KL diagnostic preflight requires a clean worktree" >&2
    exit 4
}
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]] || {
    echo "Zero-KL diagnostic HEAD does not match TASK22_EXPECTED_GIT_SHA" >&2
    exit 4
}

"$PYTHON_BIN" scripts/task22/validate_zero_kl_baseline_contract.py \
    --runtime-env-json "$RUNTIME_ENV_JSON" \
    --train-args-json "$TRAIN_ARGS_JSON"

"$PYTHON_BIN" -m py_compile \
    relax/backends/megatron/actor.py \
    relax/distributed/checkpoint_service/backends/device_direct.py \
    relax/distributed/checkpoint_service/client/engine.py \
    relax/engine/rollout/permit_observability.py \
    relax/engine/rollout/request_permit.py \
    relax/engine/rollout/sglang_rollout.py \
    relax/engine/rollout/sync_intent_rollout.py \
    relax/engine/router/request_version_ledger.py \
    relax/engine/router/router.py \
    relax/engine/router/work_accounting.py \
    relax/utils/cross_version_kv.py \
    scripts/task22/analyze_dcs_weight_sync.py \
    scripts/task22/analyze_phase_elastic_calibration.py \
    scripts/task22/smoke_sglang_calibration.py \
    scripts/task22/validate_zero_kl_baseline_contract.py

if [[ "${TASK22_FAST_PREFLIGHT:-0}" != "1" ]]; then
    "$PYTHON_BIN" -m pytest -q \
        tests/engine/rollout/test_permit_observability.py \
        tests/engine/rollout/test_request_permit.py \
        tests/engine/rollout/test_request_permit_snapshot.py \
        tests/engine/rollout/test_cross_version_kv.py \
        tests/engine/rollout/test_sync_intent.py \
        tests/engine/router/test_request_version_ledger.py \
        tests/engine/router/test_work_accounting.py \
        tests/engine/router/test_slime_router_work_lifecycle.py \
        tests/distributed/checkpoint_service/test_dcs_weight_sync_metrics.py \
        tests/tools/test_task22_dcs_weight_sync_analyzer.py \
        tests/tools/test_task22_phase_elastic_calibration_analyzer.py \
        tests/tools/test_task22_zero_kl_baseline_contract.py \
        tests/utils/test_hybrid_dcs_router_contract_source.py \
        tests/utils/test_tensor_backuper_gpu_snapshot.py
else
    echo "TASK22_ZERO_KL_PREFLIGHT tests=SKIPPED_LOCAL_VERIFIED"
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "$gpu_count" == "4" ]] || {
    echo "Zero-KL diagnostic requires exactly four GPUs, found $gpu_count" >&2
    exit 4
}
"$PYTHON_BIN" - <<'PY'
import csv
import io
import subprocess

expected_name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
raw = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
rows = [row for row in csv.reader(io.StringIO(raw)) if row]
if len(rows) != 4:
    raise SystemExit(f"expected four GPU inventory rows, found {len(rows)}")
for index, name, memory_mib in ([field.strip() for field in row] for row in rows):
    if name != expected_name or int(memory_mib) < 95_000:
        raise SystemExit(f"GPU {index} contract mismatch: {name}, {memory_mib} MiB")
print("TASK22_ZERO_KL_PREFLIGHT gpu_contract=PASS")
PY

TASK22_PYTHON="$PYTHON_BIN" scripts/task22/prepare_sglang_calibration.sh

if [[ "${TASK22_EXPERIMENT_ARM:-baseline}" == "dcs_joint_on" || "${TASK22_EXPERIMENT_ARM:-baseline}" == "dcs_carry_aware_kv_continuation_on" ]]; then
    "$PYTHON_BIN" - <<'PY'
from sglang.srt.managers.io_struct import GenerateReqInput

fields = getattr(GenerateReqInput, "__dataclass_fields__", {})
if "extra_key" not in fields:
    raise SystemExit("joint DCS requires SGLang GenerateReqInput.extra_key")
print("TASK22_ZERO_KL_PREFLIGHT sglang_extra_key=PASS")
PY
fi

[[ -r "$MODEL_DIR/Qwen3-4B/config.json" ]] || {
    echo "Qwen3-4B checkpoint is missing" >&2
    exit 4
}
[[ -r "$DATA_DIR/dapo-math-17k/dapo-math-17k.jsonl" ]] || {
    echo "Dapo dataset is missing" >&2
    exit 4
}

echo "TASK22_ZERO_KL_PREFLIGHT verdict=PASS"
