#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -uo pipefail

TASK40_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TASK40_REPO_ROOT="$(cd -- "${TASK40_SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
TASK40_HOST_BASE="${TASK40_HOST_BASE:-$(dirname -- "${TASK40_REPO_ROOT}")}"
TASK40_CONTAINER_BASE="${TASK40_CONTAINER_BASE:-/workspace}"
TASK40_EVIDENCE_ROOT="${TASK40_EVIDENCE_ROOT:-}"
TASK40_SIF="${TASK40_SIF:-}"
TASK40_RAY_DASHBOARD="${TASK40_RAY_DASHBOARD:-http://127.0.0.1:8265}"
TASK40_EVAL_DATA="${TASK40_EVAL_DATA:-${TASK40_HOST_BASE}/Output/task40/task40_a100_20260730_d11a9a6/data/gsm8k_test_64_random_state_0.parquet}"
TASK40_RUN_TIMEOUT="${TASK40_RUN_TIMEOUT:-7200}"
TASK40_TRAIN_STEPS="${TASK40_TRAIN_STEPS:-11}"
TASK40_EVAL_SIZE="${TASK40_EVAL_SIZE:-64}"
TASK40_MAX_RESPONSE_LENGTH="${TASK40_MAX_RESPONSE_LENGTH:-4096}"
TASK40_MODEL="${TASK40_MODEL:-${TASK40_HOST_BASE}/Qwen3-0.6B}"
TASK40_TRAIN_DATA="${TASK40_TRAIN_DATA:-${TASK40_HOST_BASE}/gsm8k/main/train-00000-of-00001.parquet}"

readonly TASK40_EXPECTED_SEEDS="42 1234 2026"
readonly TASK40_HARDWARE="4xA100-PCIE-40GB"

declare -a TASK40_DEFAULT_MATRIX=(
    "p3o_on_policy|p3o|1.0|run_p3o_on_policy_a100x4.sh"
    "grpo_on_policy|grpo|1.0|run_grpo_on_policy_a100x4.sh"
    "p3o_temperature_0p6|p3o|0.6|run_p3o_temperature_0p6_a100x4.sh"
    "grpo_temperature_0p6|grpo|0.6|run_grpo_temperature_0p6_a100x4.sh"
    "p3o_temperature_1p2|p3o|1.2|run_p3o_temperature_1p2_a100x4.sh"
    "grpo_temperature_1p2|grpo|1.2|run_grpo_temperature_1p2_a100x4.sh"
)

usage() {
    cat <<'EOF'
Usage: run_overnight_matrix.sh [config ...]

With no arguments, runs the frozen 18-run matrix. Positional config names
select whole three-seed blocks, for example:
  run_overnight_matrix.sh p3o_temperature_0p6 grpo_temperature_0p6
EOF
}

if [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -z "${TASK40_EVIDENCE_ROOT}" || -z "${TASK40_SIF}" ]]; then
    echo "TASK40_RUNNER_ERROR set TASK40_EVIDENCE_ROOT and TASK40_SIF" >&2
    exit 2
fi

for required in "${TASK40_SIF}" "${TASK40_MODEL}" "${TASK40_TRAIN_DATA}" "${TASK40_EVAL_DATA}"; do
    if [[ ! -e "${required}" ]]; then
        echo "TASK40_RUNNER_ERROR missing=${required}" >&2
        exit 2
    fi
done
if [[ ! "${TASK40_RUN_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TASK40_RUNNER_ERROR TASK40_RUN_TIMEOUT must be a positive integer" >&2
    exit 2
fi

mkdir -p "${TASK40_EVIDENCE_ROOT}/runs" "${TASK40_EVIDENCE_ROOT}/logs"
TASK40_GIT_SHA="$(git -C "${TASK40_REPO_ROOT}" rev-parse HEAD)"
TASK40_GIT_SHORT="$(git -C "${TASK40_REPO_ROOT}" rev-parse --short HEAD)"
TASK40_CONTAINER_OUTPUT="${TASK40_CONTAINER_BASE}/${TASK40_EVIDENCE_ROOT#"${TASK40_HOST_BASE}/"}/runs"
TASK40_CONTAINER_EVAL_DATA="${TASK40_CONTAINER_BASE}/${TASK40_EVAL_DATA#"${TASK40_HOST_BASE}/"}"
TASK40_CONTAINER_MODEL="${TASK40_CONTAINER_BASE}/${TASK40_MODEL#"${TASK40_HOST_BASE}/"}"
TASK40_CONTAINER_TRAIN_DATA="${TASK40_CONTAINER_BASE}/${TASK40_TRAIN_DATA#"${TASK40_HOST_BASE}/"}"

matrix_selected() {
    local config="$1"
    shift
    if [[ "$#" -eq 0 ]]; then
        return 0
    fi
    local requested
    for requested in "$@"; do
        if [[ "${requested}" == "${config}" ]]; then
            return 0
        fi
    done
    return 1
}

write_identity() {
    local path="$1" method="$2" temperature="$3" seed="$4"
    TASK40_IDENTITY_PATH="${path}" \
    TASK40_IDENTITY_METHOD="${method}" \
    TASK40_IDENTITY_TEMPERATURE="${temperature}" \
    TASK40_IDENTITY_SEED="${seed}" \
    TASK40_IDENTITY_GIT_SHA="${TASK40_GIT_SHA}" \
    TASK40_IDENTITY_MODEL="${TASK40_CONTAINER_MODEL}" \
    TASK40_IDENTITY_DATASET="${TASK40_CONTAINER_TRAIN_DATA}" \
    TASK40_IDENTITY_EVAL_SET="${TASK40_CONTAINER_EVAL_DATA}" \
    TASK40_IDENTITY_EVAL_SIZE="${TASK40_EVAL_SIZE}" \
    TASK40_IDENTITY_TRAIN_STEPS="${TASK40_TRAIN_STEPS}" \
    TASK40_IDENTITY_MAX_RESPONSE="${TASK40_MAX_RESPONSE_LENGTH}" \
    TASK40_IDENTITY_HARDWARE="${TASK40_HARDWARE}" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "git_sha": os.environ["TASK40_IDENTITY_GIT_SHA"],
    "method": os.environ["TASK40_IDENTITY_METHOD"],
    "model": os.environ["TASK40_IDENTITY_MODEL"],
    "dataset": os.environ["TASK40_IDENTITY_DATASET"],
    "seed": int(os.environ["TASK40_IDENTITY_SEED"]),
    "rollout_temperature": float(os.environ["TASK40_IDENTITY_TEMPERATURE"]),
    "update_weights_interval": 1,
    "train_steps": int(os.environ["TASK40_IDENTITY_TRAIN_STEPS"]),
    "global_batch_size": 48,
    "micro_batch_size": 1,
    "max_prompt_length": 512,
    "max_response_length": int(os.environ["TASK40_IDENTITY_MAX_RESPONSE"]),
    "eval_set": os.environ["TASK40_IDENTITY_EVAL_SET"],
    "eval_size": int(os.environ["TASK40_IDENTITY_EVAL_SIZE"]),
    "hardware": os.environ["TASK40_IDENTITY_HARDWARE"],
}
Path(os.environ["TASK40_IDENTITY_PATH"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

same_identity() {
    local left="$1" right="$2"
    python3 - "${left}" "${right}" <<'PY'
import json
import sys
from pathlib import Path

left, right = (json.loads(Path(item).read_text(encoding="utf-8")) for item in sys.argv[1:])
raise SystemExit(0 if left == right else 1)
PY
}

run_is_complete() {
    local run_dir="$1" desired_identity="$2"
    [[ -f "${run_dir}/run_identity.json" ]] || return 1
    [[ -f "${run_dir}/exit_code.txt" ]] || return 1
    [[ -f "${run_dir}/job_status.txt" ]] || return 1
    [[ -f "${run_dir}/stdout_stderr.log" ]] || return 1
    same_identity "${run_dir}/run_identity.json" "${desired_identity}" || return 1
    [[ "$(<"${run_dir}/exit_code.txt")" == "0" ]] || return 1
    rg -qi "succeeded" "${run_dir}/job_status.txt" || return 1
    rg -q "All training steps finished" "${run_dir}/stdout_stderr.log" || return 1
    find "${run_dir}/tensorboard" -type f -name 'events.out.tfevents.*' -print -quit | rg -q .
}

wait_for_gpu_cleanup() {
    local attempts=24
    local index
    for ((index = 1; index <= attempts; index++)); do
        if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
            return 0
        fi
        sleep 5
    done
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    return 1
}

overall_rc=0
selected_count=0
for entry in "${TASK40_DEFAULT_MATRIX[@]}"; do
    IFS='|' read -r config method temperature script_name <<<"${entry}"
    if ! matrix_selected "${config}" "$@"; then
        continue
    fi
    selected_count=$((selected_count + 1))
    for seed in ${TASK40_EXPECTED_SEEDS}; do
        seed_root="${TASK40_EVIDENCE_ROOT}/runs/${config}/seed_${seed}"
        mkdir -p "${seed_root}"
        desired_identity="${seed_root}/desired_run_identity.json"
        write_identity "${desired_identity}" "${method}" "${temperature}" "${seed}"

        resume_run=""
        for candidate in "${seed_root}"/*; do
            [[ -d "${candidate}" ]] || continue
            if run_is_complete "${candidate}" "${desired_identity}"; then
                resume_run="${candidate}"
                break
            fi
        done
        if [[ -n "${resume_run}" ]]; then
            echo "TASK40_MATRIX_RESUME_SKIP config=${config} seed=${seed} run=${resume_run}"
            continue
        fi

        attempt=1
        while [[ -e "${seed_root}/${TASK40_GIT_SHORT}_${config}_seed${seed}_attempt${attempt}" ]]; do
            attempt=$((attempt + 1))
        done
        run_id="${TASK40_GIT_SHORT}_${config}_seed${seed}_attempt${attempt}"
        run_dir="${seed_root}/${run_id}"
        driver_log="${TASK40_EVIDENCE_ROOT}/logs/driver_${config}_seed${seed}_attempt${attempt}.log"
        echo "TASK40_MATRIX_START config=${config} method=${method} temperature=${temperature} seed=${seed} run_id=${run_id}"

        if ! wait_for_gpu_cleanup; then
            echo "TASK40_MATRIX_BLOCKED gpu_cleanup_failed config=${config} seed=${seed}" | tee -a "${driver_log}" >&2
            overall_rc=1
            break 2
        fi

        set +e
        timeout --signal=TERM --kill-after=60 "${TASK40_RUN_TIMEOUT}" \
            apptainer exec \
            --cleanenv \
            --nv \
            --bind "${TASK40_HOST_BASE}:${TASK40_CONTAINER_BASE}" \
            "${TASK40_SIF}" \
            bash -lc "
                cd ${TASK40_CONTAINER_BASE}/Relax
                export TASK40_MODE=formal
                export TASK40_SEED=${seed}
                export TASK40_RUN_ID=${run_id}
                export TASK40_OUTPUT_ROOT=${TASK40_CONTAINER_OUTPUT}
                export TASK40_MODEL_DIR=${TASK40_CONTAINER_MODEL}
                export TASK40_TRAIN_DATA=${TASK40_CONTAINER_TRAIN_DATA}
                export TASK40_EVAL_DATA=${TASK40_CONTAINER_EVAL_DATA}
                export TASK40_RAY_DASHBOARD=${TASK40_RAY_DASHBOARD}
                export TASK40_NUM_ROLLOUT=${TASK40_TRAIN_STEPS}
                export TASK40_MAX_RESPONSE_LEN=${TASK40_MAX_RESPONSE_LENGTH}
                export TASK40_TORCH_DISTRIBUTED_DEBUG=OFF
                bash examples/algorithms/p3o/${script_name}
            " >"${driver_log}" 2>&1
        command_rc=$?
        set +e

        if [[ -d "${run_dir}" ]]; then
            cp "${desired_identity}" "${run_dir}/run_identity.json"
            printf '%s\n' "${command_rc}" >"${run_dir}/runner_exit_code.txt"
        fi
        if [[ "${command_rc}" -eq 0 ]] && run_is_complete "${run_dir}" "${desired_identity}"; then
            echo "TASK40_MATRIX_PASS config=${config} seed=${seed} run=${run_dir}"
        else
            echo "TASK40_MATRIX_FAIL config=${config} seed=${seed} command_rc=${command_rc} log=${driver_log}" >&2
            overall_rc=1
        fi
    done
done

if [[ "${selected_count}" -eq 0 ]]; then
    echo "TASK40_RUNNER_ERROR no requested config matched" >&2
    usage >&2
    exit 2
fi
echo "TASK40_MATRIX_COMPLETE selected_configs=${selected_count} status=${overall_rc}"
exit "${overall_rc}"
