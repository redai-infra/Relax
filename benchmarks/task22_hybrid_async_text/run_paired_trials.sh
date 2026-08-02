#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Run three paired Task 22 trials. Each trial contains baseline, zero-KL, and
# interval-two weight-publication runs with the same seed and workload.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." >/dev/null 2>&1 && pwd)"
RECIPE="${REPO_ROOT}/scripts/training/text/run-qwen3-0.6B-2xgpu-hybrid-async.sh"
DATASET_PREP="${SCRIPT_DIR}/prepare_task22_dataset.py"

export RELAX_ROOT="${REPO_ROOT}"
export RELAX="${REPO_ROOT}"
export MEGATRON="${WORKSPACE_ROOT}/.relax-deps/Megatron-LM"
source "${WORKSPACE_ROOT}/relax.env"
source "${WORKSPACE_ROOT}/.venv/bin/activate"
unset RUNTIME_ENV_JSON

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/model/Qwen3-0.6B}"
TOTAL_TRIALS="${TOTAL_TRIALS:-3}"
ARTIFACT_ROOT="${TASK22_ARTIFACT_ROOT:-${REPO_ROOT}/benchmark_artifacts/task22-hybrid-async-text-v3}"
export PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/benchmarks/data/task22_gsm8k_main16.jsonl}"
export TASK22_DATASET_REPO_ID="${TASK22_DATASET_REPO_ID:-AI-ModelScope/gsm8k}"
export TASK22_DATASET_SPLIT="${TASK22_DATASET_SPLIT:-main}"
export TASK22_DATASET_SUBSET_SIZE="${TASK22_DATASET_SUBSET_SIZE:-16}"
export TASK22_DATASET_DOWNLOAD_DIR="${TASK22_DATASET_DOWNLOAD_DIR:-${ARTIFACT_ROOT}/modelscope-cache}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
export RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=4
export MAX_RESPONSE_LEN=512
UPDATE_WEIGHT_BUFFER_SIZE=536870912

if [[ "${TOTAL_TRIALS}" != "3" ]]; then
    echo "Task 22 is fixed to exactly three paired trials, got TOTAL_TRIALS=${TOTAL_TRIALS}" >&2
    exit 2
fi
if [[ "${NUM_ROLLOUT}" != "20" ]]; then
    echo "Formal Task 22 trials require exactly 20 steps, got NUM_ROLLOUT=${NUM_ROLLOUT}" >&2
    exit 2
fi
IFS="," read -r ACTOR_GPU ROLLOUT_GPU EXTRA_GPU <<<"${CUDA_VISIBLE_DEVICES}"
if [[ -z "${ACTOR_GPU}" || -z "${ROLLOUT_GPU}" || -n "${EXTRA_GPU}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly two GPU IDs" >&2
    exit 2
fi
if ! ray status >/dev/null 2>&1; then
    echo "No reachable Ray cluster. Start a two-GPU cluster from the workspace .venv first." >&2
    exit 1
fi
python "${DATASET_PREP}" \
    --repo-id "${TASK22_DATASET_REPO_ID}" \
    --split "${TASK22_DATASET_SPLIT}" \
    --limit "${TASK22_DATASET_SUBSET_SIZE}" \
    --download-dir "${TASK22_DATASET_DOWNLOAD_DIR}" \
    --output "${PROMPT_DATA}"

job_status() {
    python - "$RAY_DASHBOARD_ADDRESS" "$1" <<'PY'
import sys
from ray.job_submission import JobSubmissionClient

print(JobSubmissionClient(sys.argv[1]).get_job_status(sys.argv[2]).value)
PY
}

cleanup_serve() {
    if ray serve shutdown -y >/dev/null 2>&1; then
        return
    fi
    local cleanup_id="task22-cleanup-$(date +%s%N)"
    ray job submit --address="${RAY_DASHBOARD_ADDRESS}" --submission-id="${cleanup_id}" \
        --log-style=record --log-color=false -- \
        python3 -c 'from ray import serve; serve.shutdown()' >/dev/null
}

run_component() {
    local variant="$1"
    local run_id="$2"
    local run_dir="${ARTIFACT_ROOT}/${variant}/run-${run_id}"
    local job_id="task22-${variant}-run-${run_id}-$(date +%s)"
    local monitor_pid=""
    local status=""
    local max_tokens_per_gpu="8192"
    local log_probs_max_tokens_per_gpu="8192"
    local reference_forward="disabled"
    local use_kl_loss="disabled"
    local update_weights_interval="1"
    if [[ "${variant}" == "baseline" ]]; then
        reference_forward="enabled"
        use_kl_loss="enabled"
    elif [[ "${variant}" == "optimized" ]]; then
        update_weights_interval="2"
    fi
    mkdir -p "${run_dir}"

    cleanup_serve
    (
        while true; do
            local sample_time
            sample_time="$(date --iso-8601=seconds)"
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits | while IFS= read -r row; do
                printf '%s,%s\n' "${sample_time}" "${row}"
            done
            sleep 1
        done
    ) >"${run_dir}/gpu.csv" 2>&1 &
    monitor_pid=$!

    {
        echo "variant=${variant}"
        echo "run_id=${run_id}"
        echo "job_id=${job_id}"
        echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        echo "git_branch=$(git -C "${REPO_ROOT}" branch --show-current)"
        echo "git_status=$(if [[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then echo clean; else echo dirty; fi)"
        echo "python=$(python --version 2>&1)"
        echo "torch=$(python -c 'import torch; print(torch.__version__)')"
        echo "torch_cuda=$(python -c 'import torch; print(torch.version.cuda)')"
        echo "ray=$(python -c 'import ray; print(ray.__version__)')"
        echo "sglang=$(python -c 'import sglang; print(sglang.__version__)')"
        echo "transformers=$(python -c 'import transformers; print(transformers.__version__)')"
        echo "gpu_name=$(nvidia-smi --id="${ACTOR_GPU}" --query-gpu=name --format=csv,noheader)"
        echo "gpu_driver=$(nvidia-smi --id="${ACTOR_GPU}" --query-gpu=driver_version --format=csv,noheader)"
        echo "cpu_model=$(lscpu | awk -F: '/Model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
        echo "cpu_logical_count=$(nproc --all)"
        echo "memory_total_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
        echo "model_path=${MODEL_PATH}"
        echo "dataset_sha256=$(sha256sum "${PROMPT_DATA}" | awk '{print $1}')"
        echo "dataset_repo_id=${TASK22_DATASET_REPO_ID}"
        echo "dataset_split=${TASK22_DATASET_SPLIT}"
        echo "dataset_subset_size=${TASK22_DATASET_SUBSET_SIZE}"
        echo "model_sha256=$(sha256sum "${MODEL_PATH}/model.safetensors" | awk '{print $1}')"
        echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
        echo "actor_gpu=${ACTOR_GPU}"
        echo "rollout_gpu=${ROLLOUT_GPU}"
        echo "num_rollout=${NUM_ROLLOUT}"
        echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
        echo "n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}"
        echo "global_batch_size=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
        echo "max_response_len=${MAX_RESPONSE_LEN}"
        echo "max_tokens_per_gpu=${max_tokens_per_gpu}"
        echo "log_probs_max_tokens_per_gpu=${log_probs_max_tokens_per_gpu}"
        echo "max_staleness=2"
        echo "update_weights_interval=${update_weights_interval}"
        echo "update_weight_buffer_size=${UPDATE_WEIGHT_BUFFER_SIZE}"
        echo "reference_forward=${reference_forward}"
        echo "use_kl_loss=${use_kl_loss}"
        echo "kl_loss_coef=0.00"
        echo "use_tis=enabled"
        echo "seed=$((20260801 + run_id))"
    } >"${run_dir}/manifest.txt"

    if ! TASK22_VARIANT="${variant}" UPDATE_WEIGHT_BUFFER_SIZE="${UPDATE_WEIGHT_BUFFER_SIZE}" \
        MAX_TOKENS_PER_GPU="${max_tokens_per_gpu}" \
        LOG_PROBS_MAX_TOKENS_PER_GPU="${log_probs_max_tokens_per_gpu}" \
        UPDATE_WEIGHTS_INTERVAL="${update_weights_interval}" \
        RUN_ID="${run_id}" TASK22_JOB_ID="${job_id}" \
        bash "${RECIPE}" >"${run_dir}/submit.log" 2>&1; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
        echo "Task 22 ${variant} run ${run_id} submission failed; see ${run_dir}/submit.log" >&2
        return 1
    fi

    while true; do
        if ! status="$(job_status "${job_id}")"; then
            printf '[%s] %s run %s: status query retry\n' "$(date '+%H:%M:%S')" "${variant}" "${run_id}"
            sleep 10
            continue
        fi
        printf '[%s] %s run %s: %s\n' "$(date '+%H:%M:%S')" "${variant}" "${run_id}" "${status}"
        case "${status}" in
            SUCCEEDED|FAILED|STOPPED) break ;;
        esac
        sleep 10
    done

    ray job logs --address="${RAY_DASHBOARD_ADDRESS}" "${job_id}" >"${run_dir}/train.log" 2>&1 || true
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
    echo "job_status=${status}" >>"${run_dir}/manifest.txt"
    if [[ "${status}" != "SUCCEEDED" ]]; then
        echo "Task 22 ${variant} run ${run_id} failed; see ${run_dir}/train.log" >&2
        return 1
    fi
}

mkdir -p "${ARTIFACT_ROOT}"
for run_id in 1 2 3; do
    case "${run_id}" in
        1) variants=(baseline zero_kl optimized) ;;
        2) variants=(zero_kl optimized baseline) ;;
        3) variants=(optimized baseline zero_kl) ;;
    esac
    for variant in "${variants[@]}"; do
        run_component "${variant}" "${run_id}"
    done
done

TASK22_ARTIFACT_ROOT="${ARTIFACT_ROOT}" python "${SCRIPT_DIR}/analyze.py"
