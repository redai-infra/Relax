#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

P3O_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
P3O_REPO_ROOT="$(cd -- "${P3O_SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
source "${P3O_REPO_ROOT}/scripts/models/qwen3-0.6B.sh"

P3O_ALGORITHM="${P3O_ALGORITHM:?set P3O_ALGORITHM to p3o or grpo}"
P3O_ENABLE_TEMPERATURE_OVERRIDE="${P3O_ENABLE_TEMPERATURE_OVERRIDE:-0}"
P3O_BEHAVIOR_TEMPERATURE="${P3O_BEHAVIOR_TEMPERATURE:-}"
P3O_MAX_STALENESS="${P3O_MAX_STALENESS:-0}"
P3O_UPDATE_WEIGHTS_INTERVAL="${P3O_UPDATE_WEIGHTS_INTERVAL:-1}"
P3O_PIPELINE_MODEL_PARALLEL_SIZE="${P3O_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
P3O_MODE="${P3O_MODE:-formal}"
P3O_SEED="${P3O_SEED:-42}"
P3O_DRY_RUN="${P3O_DRY_RUN:-0}"
P3O_NCCL_DEBUG="${P3O_NCCL_DEBUG:-WARN}"
P3O_TORCH_DISTRIBUTED_DEBUG="${P3O_TORCH_DISTRIBUTED_DEBUG:-OFF}"

if [[ "${P3O_DRY_RUN}" == "1" ]]; then
    P3O_MODEL_DIR="${P3O_MODEL_DIR:-/dummy/model}"
    P3O_TRAIN_DATA="${P3O_TRAIN_DATA:-/dummy/train.jsonl}"
    P3O_EVAL_DATA="${P3O_EVAL_DATA:-/dummy/eval.jsonl}"
    P3O_OUTPUT_ROOT="${P3O_OUTPUT_ROOT:-/dummy/output}"
    P3O_MEGATRON_DIR="${P3O_MEGATRON_DIR:-/dummy/megatron}"
else
    : "${P3O_MODEL_DIR:?P3O_MODEL_DIR must be set}"
    : "${P3O_TRAIN_DATA:?P3O_TRAIN_DATA must be set}"
    : "${P3O_OUTPUT_ROOT:?P3O_OUTPUT_ROOT must be set}"
    : "${P3O_MEGATRON_DIR:?P3O_MEGATRON_DIR must be set}"
    if [[ "${P3O_MODE}" == "formal" ]]; then
        : "${P3O_EVAL_DATA:?P3O_EVAL_DATA must be set in formal mode}"
    else
        P3O_EVAL_DATA="${P3O_EVAL_DATA:-}"
    fi
fi

: "${P3O_RAY_DASHBOARD:?P3O_RAY_DASHBOARD must be set}"

if [[ "${P3O_ALGORITHM}" != "p3o" && "${P3O_ALGORITHM}" != "grpo" ]]; then
    echo "Unsupported P3O_ALGORITHM=${P3O_ALGORITHM}" >&2
    exit 2
fi
if [[ "${P3O_ENABLE_TEMPERATURE_OVERRIDE}" != "0" && "${P3O_ENABLE_TEMPERATURE_OVERRIDE}" != "1" ]]; then
    echo "P3O_ENABLE_TEMPERATURE_OVERRIDE must be 0 or 1" >&2
    exit 2
fi
if [[ "${P3O_ENABLE_TEMPERATURE_OVERRIDE}" == "1" ]]; then
    python - "${P3O_BEHAVIOR_TEMPERATURE}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("P3O_BEHAVIOR_TEMPERATURE must be numeric") from exc

if not math.isfinite(value) or value <= 0.0:
    raise SystemExit("P3O_BEHAVIOR_TEMPERATURE must be finite and greater than zero")
PY
fi
if [[ "${P3O_MODE}" != "formal" && "${P3O_MODE}" != "smoke" ]]; then
    echo "P3O_MODE must be formal or smoke" >&2
    exit 2
fi
if [[ ! "${P3O_UPDATE_WEIGHTS_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "P3O_UPDATE_WEIGHTS_INTERVAL must be a positive integer" >&2
    exit 2
fi
if [[ "${P3O_UPDATE_WEIGHTS_INTERVAL}" != "1" && "${P3O_ENABLE_TEMPERATURE_OVERRIDE}" == "1" ]]; then
    echo "periodic policy synchronization and temperature override must be tested in separate runs" >&2
    exit 2
fi
if [[ ! "${P3O_PIPELINE_MODEL_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "P3O_PIPELINE_MODEL_PARALLEL_SIZE must be a positive integer" >&2
    exit 2
fi

if [[ "${P3O_MODE}" == "formal" ]]; then
    P3O_NUM_ROLLOUT="${P3O_NUM_ROLLOUT:-11}"
    P3O_ROLLOUT_BATCH_SIZE="${P3O_ROLLOUT_BATCH_SIZE:-12}"
    P3O_N_SAMPLES="${P3O_N_SAMPLES:-4}"
    P3O_GLOBAL_BATCH_SIZE="${P3O_GLOBAL_BATCH_SIZE:-48}"
    # Full-length responses make the FP32 logits conversion exceed A100-40GB at micro-batch 4.
    P3O_MICRO_BATCH_SIZE="${P3O_MICRO_BATCH_SIZE:-1}"
    P3O_MAX_RESPONSE_LEN="${P3O_MAX_RESPONSE_LEN:-4096}"
else
    P3O_NUM_ROLLOUT="${P3O_NUM_ROLLOUT:-1}"
    P3O_ROLLOUT_BATCH_SIZE="${P3O_ROLLOUT_BATCH_SIZE:-4}"
    P3O_N_SAMPLES="${P3O_N_SAMPLES:-4}"
    P3O_GLOBAL_BATCH_SIZE="${P3O_GLOBAL_BATCH_SIZE:-16}"
    P3O_MICRO_BATCH_SIZE="${P3O_MICRO_BATCH_SIZE:-1}"
    P3O_MAX_RESPONSE_LEN="${P3O_MAX_RESPONSE_LEN:-128}"
fi

P3O_CONFIG_NAME="${P3O_ALGORITHM}_$(
    if [[ "${P3O_ENABLE_TEMPERATURE_OVERRIDE}" == "1" ]]; then
        echo "temperature_${P3O_BEHAVIOR_TEMPERATURE//./p}"
    elif [[ "${P3O_UPDATE_WEIGHTS_INTERVAL}" != "1" ]]; then
        echo "periodic_sync_interval_${P3O_UPDATE_WEIGHTS_INTERVAL}"
    else
        echo "on_policy"
    fi
)"
if [[ "${P3O_PIPELINE_MODEL_PARALLEL_SIZE}" != "1" ]]; then
    P3O_CONFIG_NAME="${P3O_CONFIG_NAME}_pp${P3O_PIPELINE_MODEL_PARALLEL_SIZE}"
fi

P3O_build_args() {
    P3O_CKPT_ARGS=(
        --hf-checkpoint "${P3O_MODEL_DIR}"
        --megatron-to-hf-mode bridge
        --warm-hf-checkpoint-page-cache
    )

    P3O_ROLLOUT_ARGS=(
        --prompt-data "${P3O_TRAIN_DATA}"
        --input-key question
        --label-key answer
        --apply-chat-template
        --rollout-shuffle
        --rm-type mopd
        --num-rollout "${P3O_NUM_ROLLOUT}"
        --rollout-batch-size "${P3O_ROLLOUT_BATCH_SIZE}"
        --n-samples-per-prompt "${P3O_N_SAMPLES}"
        --rollout-max-prompt-len 512
        --rollout-max-response-len "${P3O_MAX_RESPONSE_LEN}"
        --rollout-temperature 1.0
        --rollout-top-p 1.0
        --rollout-top-k -1
        --global-batch-size "${P3O_GLOBAL_BATCH_SIZE}"
        --use-rollout-logprobs
        --balance-data
        --log-passrate
    )

    P3O_PERF_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size "${P3O_PIPELINE_MODEL_PARALLEL_SIZE}"
        --context-parallel-size 1
        --expert-model-parallel-size 1
        --expert-tensor-parallel-size 1
        --micro-batch-size "${P3O_MICRO_BATCH_SIZE}"
        --calculate-per-token-loss
    )

    P3O_OPTIMIZER_ARGS=(
        --optimizer adam
        --lr 1e-5
        --min-lr 0
        --lr-decay-style cosine
        --lr-warmup-fraction 0.1
        --weight-decay 0.01
        --adam-beta1 0.9
        --adam-beta2 0.95
        --clip-grad 1.0
    )

    P3O_ALGO_ARGS=(
        --advantage-estimator "${P3O_ALGORITHM}"
        --kl-coef 0.0
        --entropy-coef 0.0
    )
    if [[ "${P3O_ALGORITHM}" == "grpo" ]]; then
        P3O_ALGO_ARGS+=(--eps-clip 0.4 --eps-clip-high 0.4)
    fi
    if [[ "${P3O_ENABLE_TEMPERATURE_OVERRIDE}" == "1" ]]; then
        P3O_ALGO_ARGS+=(--custom-generate-function-path examples.algorithms.p3o.rollout.generate)
    fi

    P3O_SGLANG_ARGS=(
        --rollout-num-gpus 4
        --rollout-num-gpus-per-engine 1
        --sglang-mem-fraction-static 0.70
    )

    P3O_MISC_ARGS=(
        --seed "${P3O_SEED}"
        --rollout-seed "${P3O_SEED}"
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        --use-health-check
        --use-tensorboard
        --tb-project-name P3O-p3o-a100x4
        --tb-experiment-name "${P3O_CONFIG_NAME}-seed-${P3O_SEED}"
    )

    P3O_EVAL_ARGS=(--skip-eval-before-train)
    if [[ "${P3O_MODE}" == "formal" ]]; then
        P3O_EVAL_ARGS+=(
            --eval-interval "${P3O_NUM_ROLLOUT}"
            --eval-prompt-data gsm8k "${P3O_EVAL_DATA}"
            --n-samples-per-eval-prompt 16
            --eval-max-response-len 4096
            --eval-temperature 1.0
            --eval-top-p 0.95
        )
    fi

    P3O_TRAIN_ARGS=(
        --resource '{"actor":[1,4],"rollout":[1,4]}'
        --max-staleness "${P3O_MAX_STALENESS}"
        --update-weights-interval "${P3O_UPDATE_WEIGHTS_INTERVAL}"
        --num-iters-per-train-update 1
        --num-data-storage-units 1
        --colocate
        "${MODEL_ARGS[@]}"
        "${P3O_CKPT_ARGS[@]}"
        "${P3O_ROLLOUT_ARGS[@]}"
        "${P3O_PERF_ARGS[@]}"
        "${P3O_OPTIMIZER_ARGS[@]}"
        "${P3O_ALGO_ARGS[@]}"
        "${P3O_SGLANG_ARGS[@]}"
        "${P3O_EVAL_ARGS[@]}"
        "${P3O_MISC_ARGS[@]}"
    )
}

P3O_run() {
    P3O_build_args
    if [[ "${P3O_DRY_RUN:-0}" == "1" ]]; then
        printf '%s\n' "${P3O_TRAIN_ARGS[@]}"
        return 0
    fi

    for required_path in "${P3O_MODEL_DIR}" "${P3O_TRAIN_DATA}" "${P3O_MEGATRON_DIR}"; do
        if [[ ! -e "${required_path}" ]]; then
            echo "Required P3O asset is missing: ${required_path}" >&2
            exit 2
        fi
    done
    if [[ "${P3O_MODE}" == "formal" && ! -e "${P3O_EVAL_DATA}" ]]; then
        echo "Required P3O asset is missing: ${P3O_EVAL_DATA}" >&2
        exit 2
    fi

    P3O_RUN_ID="${P3O_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
    P3O_RUN_DIR="${P3O_OUTPUT_ROOT}/${P3O_CONFIG_NAME}/seed_${P3O_SEED}/${P3O_RUN_ID}"
    mkdir -p "$(dirname -- "${P3O_RUN_DIR}")"
    if ! mkdir "${P3O_RUN_DIR}"; then
        echo "Refusing to overwrite P3O run directory: ${P3O_RUN_DIR}" >&2
        exit 2
    fi
    mkdir "${P3O_RUN_DIR}/tensorboard"
    P3O_JOB_ID="${P3O_CONFIG_NAME}-seed-${P3O_SEED}-${P3O_RUN_ID}"
    P3O_GIT_COMMIT="$(git -C "${P3O_REPO_ROOT}" rev-parse HEAD)"
    P3O_GIT_BRANCH="$(git -C "${P3O_REPO_ROOT}" symbolic-ref --short -q HEAD || true)"
    P3O_GIT_DIRTY=0
    if [[ -n "$(git -C "${P3O_REPO_ROOT}" status --short)" ]]; then
        P3O_GIT_DIRTY=1
    fi

    printf '%s\n' "${P3O_TRAIN_ARGS[@]}" >"${P3O_RUN_DIR}/resolved_args.txt"
    {
        echo "GIT_COMMIT=${P3O_GIT_COMMIT}"
        echo "GIT_BRANCH=${P3O_GIT_BRANCH:-DETACHED}"
        echo "GIT_DIRTY=${P3O_GIT_DIRTY}"
        echo "config=${P3O_CONFIG_NAME}"
        echo "mode=${P3O_MODE}"
        echo "seed=${P3O_SEED}"
        echo "max_staleness=${P3O_MAX_STALENESS}"
        echo "update_weights_interval=${P3O_UPDATE_WEIGHTS_INTERVAL}"
        echo "pipeline_model_parallel_size=${P3O_PIPELINE_MODEL_PARALLEL_SIZE}"
        echo "nccl_debug=${P3O_NCCL_DEBUG}"
        echo "torch_distributed_debug=${P3O_TORCH_DISTRIBUTED_DEBUG}"
        echo "behavior_temperature=${P3O_BEHAVIOR_TEMPERATURE}"
        echo "ray_job_id=${P3O_JOB_ID}"
        echo "repo=${P3O_REPO_ROOT}"
        echo "model=${P3O_MODEL_DIR}"
        echo "train_data=${P3O_TRAIN_DATA}"
        echo "eval_data=${P3O_EVAL_DATA}"
        echo "ray_dashboard=${P3O_RAY_DASHBOARD}"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"${P3O_RUN_DIR}/run_identity.env"

    P3O_RUNTIME_ENV_JSON="$(
        P3O_RUNTIME_PYTHONPATH="${P3O_REPO_ROOT}:${P3O_MEGATRON_DIR}" \
        P3O_TENSORBOARD_DIR="${P3O_RUN_DIR}/tensorboard" \
        P3O_RUNTIME_ENABLE_TEMPERATURE_OVERRIDE="${P3O_ENABLE_TEMPERATURE_OVERRIDE}" \
        P3O_RUNTIME_BEHAVIOR_TEMPERATURE="${P3O_BEHAVIOR_TEMPERATURE}" \
        P3O_RUNTIME_NCCL_DEBUG="${P3O_NCCL_DEBUG}" \
        P3O_RUNTIME_TORCH_DISTRIBUTED_DEBUG="${P3O_TORCH_DISTRIBUTED_DEBUG}" \
        P3O_RUNTIME_CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}" \
        P3O_RUNTIME_OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
        P3O_RUNTIME_MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
        P3O_RUNTIME_OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" \
        P3O_RUNTIME_NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" \
        P3O_RUNTIME_NVSHMEM_DISABLE_NCCL="${NVSHMEM_DISABLE_NCCL:-1}" \
        python3 - <<'PY'
import json
import os

env_vars = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": os.environ["P3O_RUNTIME_PYTHONPATH"],
    "TENSORBOARD_DIR": os.environ["P3O_TENSORBOARD_DIR"],
    "NCCL_DEBUG": os.environ["P3O_RUNTIME_NCCL_DEBUG"],
    "TORCH_DISTRIBUTED_DEBUG": os.environ["P3O_RUNTIME_TORCH_DISTRIBUTED_DEBUG"],
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": os.environ["P3O_RUNTIME_CUDA_DEVICE_MAX_CONNECTIONS"],
    "OMP_NUM_THREADS": os.environ["P3O_RUNTIME_OMP_NUM_THREADS"],
    "MKL_NUM_THREADS": os.environ["P3O_RUNTIME_MKL_NUM_THREADS"],
    "OPENBLAS_NUM_THREADS": os.environ["P3O_RUNTIME_OPENBLAS_NUM_THREADS"],
    "NCCL_NVLS_ENABLE": os.environ["P3O_RUNTIME_NCCL_NVLS_ENABLE"],
    "NVSHMEM_DISABLE_NCCL": os.environ["P3O_RUNTIME_NVSHMEM_DISABLE_NCCL"],
}

if os.environ["P3O_RUNTIME_ENABLE_TEMPERATURE_OVERRIDE"] == "1":
    env_vars["P3O_BEHAVIOR_TEMPERATURE"] = os.environ["P3O_RUNTIME_BEHAVIOR_TEMPERATURE"]

print(json.dumps({"env_vars": env_vars}))
PY
    )"

    P3O_COMMAND=(
        ray job submit
        --address "${P3O_RAY_DASHBOARD}"
        --submission-id "${P3O_JOB_ID}"
        --runtime-env-json "${P3O_RUNTIME_ENV_JSON}"
        --
        python3 -m relax.entrypoints.train
        "${P3O_TRAIN_ARGS[@]}"
    )
    printf '%q ' "${P3O_COMMAND[@]}" >"${P3O_RUN_DIR}/command.sh"
    printf '\n' >>"${P3O_RUN_DIR}/command.sh"

    set -o pipefail
    set +e
    "${P3O_COMMAND[@]}" 2>&1 | tee "${P3O_RUN_DIR}/stdout_stderr.log"
    P3O_EXIT_CODE=${PIPESTATUS[0]}
    ray job status "${P3O_JOB_ID}" --address "${P3O_RAY_DASHBOARD}" >"${P3O_RUN_DIR}/job_status.txt" 2>&1
    P3O_STATUS_QUERY_EXIT_CODE=$?
    set -e
    echo "${P3O_EXIT_CODE}" >"${P3O_RUN_DIR}/exit_code.txt"
    echo "${P3O_STATUS_QUERY_EXIT_CODE}" >"${P3O_RUN_DIR}/job_status_query_exit_code.txt"
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${P3O_RUN_DIR}/run_identity.env"
    return "${P3O_EXIT_CODE}"
}
