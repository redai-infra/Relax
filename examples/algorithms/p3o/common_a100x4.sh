#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

TASK40_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TASK40_REPO_ROOT="$(cd -- "${TASK40_SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
source "${TASK40_REPO_ROOT}/scripts/models/qwen3-0.6B.sh"

TASK40_ALGORITHM="${TASK40_ALGORITHM:?set TASK40_ALGORITHM to p3o or grpo}"
TASK40_BEHAVIOR_MISMATCH="${TASK40_BEHAVIOR_MISMATCH:-0}"
TASK40_MODE="${TASK40_MODE:-formal}"
TASK40_SEED="${TASK40_SEED:-42}"
TASK40_MODEL_DIR="${TASK40_MODEL_DIR:-/workspace/Qwen3-0.6B}"
TASK40_TRAIN_DATA="${TASK40_TRAIN_DATA:-/workspace/gsm8k/main/train-00000-of-00001.parquet}"
TASK40_EVAL_DATA="${TASK40_EVAL_DATA:-/workspace/gsm8k/main/test-00000-of-00001.parquet}"
TASK40_OUTPUT_ROOT="${TASK40_OUTPUT_ROOT:-/workspace/Output/task40/formal}"
TASK40_RAY_DASHBOARD="${TASK40_RAY_DASHBOARD:-http://127.0.0.1:8265}"
TASK40_MEGATRON_DIR="${TASK40_MEGATRON_DIR:-/root/Megatron-LM}"

if [[ "${TASK40_ALGORITHM}" != "p3o" && "${TASK40_ALGORITHM}" != "grpo" ]]; then
    echo "Unsupported TASK40_ALGORITHM=${TASK40_ALGORITHM}" >&2
    exit 2
fi
if [[ "${TASK40_BEHAVIOR_MISMATCH}" != "0" && "${TASK40_BEHAVIOR_MISMATCH}" != "1" ]]; then
    echo "TASK40_BEHAVIOR_MISMATCH must be 0 or 1" >&2
    exit 2
fi
if [[ "${TASK40_MODE}" != "formal" && "${TASK40_MODE}" != "smoke" ]]; then
    echo "TASK40_MODE must be formal or smoke" >&2
    exit 2
fi

if [[ "${TASK40_MODE}" == "formal" ]]; then
    TASK40_NUM_ROLLOUT="${TASK40_NUM_ROLLOUT:-11}"
    TASK40_ROLLOUT_BATCH_SIZE="${TASK40_ROLLOUT_BATCH_SIZE:-12}"
    TASK40_N_SAMPLES="${TASK40_N_SAMPLES:-4}"
    TASK40_GLOBAL_BATCH_SIZE="${TASK40_GLOBAL_BATCH_SIZE:-48}"
    # Full-length responses make the FP32 logits conversion exceed A100-40GB at micro-batch 4.
    TASK40_MICRO_BATCH_SIZE="${TASK40_MICRO_BATCH_SIZE:-1}"
    TASK40_MAX_RESPONSE_LEN="${TASK40_MAX_RESPONSE_LEN:-4096}"
else
    TASK40_NUM_ROLLOUT="${TASK40_NUM_ROLLOUT:-1}"
    TASK40_ROLLOUT_BATCH_SIZE="${TASK40_ROLLOUT_BATCH_SIZE:-4}"
    TASK40_N_SAMPLES="${TASK40_N_SAMPLES:-4}"
    TASK40_GLOBAL_BATCH_SIZE="${TASK40_GLOBAL_BATCH_SIZE:-16}"
    TASK40_MICRO_BATCH_SIZE="${TASK40_MICRO_BATCH_SIZE:-1}"
    TASK40_MAX_RESPONSE_LEN="${TASK40_MAX_RESPONSE_LEN:-128}"
fi

TASK40_CONFIG_NAME="${TASK40_ALGORITHM}_$(
    if [[ "${TASK40_BEHAVIOR_MISMATCH}" == "1" ]]; then
        echo "temperature_1p2"
    else
        echo "on_policy"
    fi
)"

task40_build_args() {
    TASK40_CKPT_ARGS=(
        --hf-checkpoint "${TASK40_MODEL_DIR}"
        --megatron-to-hf-mode bridge
        --warm-hf-checkpoint-page-cache
    )

    TASK40_ROLLOUT_ARGS=(
        --prompt-data "${TASK40_TRAIN_DATA}"
        --input-key question
        --label-key answer
        --apply-chat-template
        --rollout-shuffle
        --rm-type mopd
        --num-rollout "${TASK40_NUM_ROLLOUT}"
        --rollout-batch-size "${TASK40_ROLLOUT_BATCH_SIZE}"
        --n-samples-per-prompt "${TASK40_N_SAMPLES}"
        --rollout-max-prompt-len 512
        --rollout-max-response-len "${TASK40_MAX_RESPONSE_LEN}"
        --rollout-temperature 1.0
        --rollout-top-p 1.0
        --rollout-top-k -1
        --global-batch-size "${TASK40_GLOBAL_BATCH_SIZE}"
        --use-rollout-logprobs
        --balance-data
        --log-passrate
    )

    TASK40_PERF_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 1
        --context-parallel-size 1
        --expert-model-parallel-size 1
        --expert-tensor-parallel-size 1
        --micro-batch-size "${TASK40_MICRO_BATCH_SIZE}"
        --calculate-per-token-loss
    )

    TASK40_OPTIMIZER_ARGS=(
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

    TASK40_ALGO_ARGS=(
        --advantage-estimator "${TASK40_ALGORITHM}"
        --kl-coef 0.0
        --entropy-coef 0.0
    )
    if [[ "${TASK40_ALGORITHM}" == "grpo" ]]; then
        TASK40_ALGO_ARGS+=(--eps-clip 0.4 --eps-clip-high 0.4)
    fi
    if [[ "${TASK40_BEHAVIOR_MISMATCH}" == "1" ]]; then
        TASK40_ALGO_ARGS+=(--custom-generate-function-path examples.algorithms.p3o.rollout.generate)
    fi

    TASK40_SGLANG_ARGS=(
        --rollout-num-gpus 4
        --rollout-num-gpus-per-engine 1
        --sglang-mem-fraction-static 0.70
    )

    TASK40_MISC_ARGS=(
        --seed "${TASK40_SEED}"
        --rollout-seed "${TASK40_SEED}"
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        --use-health-check
        --use-tensorboard
        --tb-project-name task40-p3o-a100x4
        --tb-experiment-name "${TASK40_CONFIG_NAME}-seed-${TASK40_SEED}"
    )

    TASK40_EVAL_ARGS=(--skip-eval-before-train)
    if [[ "${TASK40_MODE}" == "formal" ]]; then
        TASK40_EVAL_ARGS+=(
            --eval-interval "${TASK40_NUM_ROLLOUT}"
            --eval-prompt-data gsm8k "${TASK40_EVAL_DATA}"
            --n-samples-per-eval-prompt 16
            --eval-max-response-len 4096
            --eval-temperature 1.0
            --eval-top-p 0.95
        )
    fi

    TASK40_TRAIN_ARGS=(
        --resource '{"actor":[1,4],"rollout":[1,4]}'
        --max-staleness 0
        --num-iters-per-train-update 1
        --num-data-storage-units 1
        --colocate
        "${MODEL_ARGS[@]}"
        "${TASK40_CKPT_ARGS[@]}"
        "${TASK40_ROLLOUT_ARGS[@]}"
        "${TASK40_PERF_ARGS[@]}"
        "${TASK40_OPTIMIZER_ARGS[@]}"
        "${TASK40_ALGO_ARGS[@]}"
        "${TASK40_SGLANG_ARGS[@]}"
        "${TASK40_EVAL_ARGS[@]}"
        "${TASK40_MISC_ARGS[@]}"
    )
}

task40_run() {
    task40_build_args
    if [[ "${TASK40_DRY_RUN:-0}" == "1" ]]; then
        printf '%s\n' "${TASK40_TRAIN_ARGS[@]}"
        return 0
    fi

    for required_path in "${TASK40_MODEL_DIR}" "${TASK40_TRAIN_DATA}" "${TASK40_EVAL_DATA}"; do
        if [[ ! -e "${required_path}" ]]; then
            echo "Required Task40 asset is missing: ${required_path}" >&2
            exit 2
        fi
    done

    TASK40_RUN_ID="${TASK40_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
    TASK40_RUN_DIR="${TASK40_OUTPUT_ROOT}/${TASK40_CONFIG_NAME}/seed_${TASK40_SEED}/${TASK40_RUN_ID}"
    mkdir -p "$(dirname -- "${TASK40_RUN_DIR}")"
    if ! mkdir "${TASK40_RUN_DIR}"; then
        echo "Refusing to overwrite Task40 run directory: ${TASK40_RUN_DIR}" >&2
        exit 2
    fi
    mkdir "${TASK40_RUN_DIR}/tensorboard"
    TASK40_JOB_ID="${TASK40_CONFIG_NAME}-seed-${TASK40_SEED}-${TASK40_RUN_ID}"

    printf '%s\n' "${TASK40_TRAIN_ARGS[@]}" >"${TASK40_RUN_DIR}/resolved_args.txt"
    {
        echo "config=${TASK40_CONFIG_NAME}"
        echo "mode=${TASK40_MODE}"
        echo "seed=${TASK40_SEED}"
        echo "ray_job_id=${TASK40_JOB_ID}"
        echo "repo=${TASK40_REPO_ROOT}"
        echo "model=${TASK40_MODEL_DIR}"
        echo "train_data=${TASK40_TRAIN_DATA}"
        echo "eval_data=${TASK40_EVAL_DATA}"
        echo "ray_dashboard=${TASK40_RAY_DASHBOARD}"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"${TASK40_RUN_DIR}/run_identity.env"

    TASK40_RUNTIME_ENV_JSON="$(
        TASK40_RUNTIME_PYTHONPATH="${TASK40_REPO_ROOT}:${TASK40_MEGATRON_DIR}" \
        TASK40_TENSORBOARD_DIR="${TASK40_RUN_DIR}/tensorboard" \
        python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "env_vars": {
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": os.environ["TASK40_RUNTIME_PYTHONPATH"],
                "TENSORBOARD_DIR": os.environ["TASK40_TENSORBOARD_DIR"],
                "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "ALL_PROXY": "",
                "http_proxy": "",
                "https_proxy": "",
                "all_proxy": "",
                "NO_PROXY": "*",
                "no_proxy": "*",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8",
                "NCCL_NVLS_ENABLE": "0",
                "NVSHMEM_DISABLE_NCCL": "1",
            }
        }
    )
)
PY
    )"

    TASK40_COMMAND=(
        ray job submit
        --address "${TASK40_RAY_DASHBOARD}"
        --submission-id "${TASK40_JOB_ID}"
        --runtime-env-json "${TASK40_RUNTIME_ENV_JSON}"
        --
        python3 -m relax.entrypoints.train
        "${TASK40_TRAIN_ARGS[@]}"
    )
    printf '%q ' "${TASK40_COMMAND[@]}" >"${TASK40_RUN_DIR}/command.sh"
    printf '\n' >>"${TASK40_RUN_DIR}/command.sh"

    set -o pipefail
    set +e
    "${TASK40_COMMAND[@]}" 2>&1 | tee "${TASK40_RUN_DIR}/stdout_stderr.log"
    TASK40_EXIT_CODE=${PIPESTATUS[0]}
    ray job status "${TASK40_JOB_ID}" --address "${TASK40_RAY_DASHBOARD}" >"${TASK40_RUN_DIR}/job_status.txt" 2>&1
    TASK40_STATUS_QUERY_EXIT_CODE=$?
    set -e
    echo "${TASK40_EXIT_CODE}" >"${TASK40_RUN_DIR}/exit_code.txt"
    echo "${TASK40_STATUS_QUERY_EXIT_CODE}" >"${TASK40_RUN_DIR}/job_status_query_exit_code.txt"
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${TASK40_RUN_DIR}/run_identity.env"
    return "${TASK40_EXIT_CODE}"
}
