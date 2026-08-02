#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Two-GPU Hybrid-async Task 22 recipe. Baseline and optimized variants run
# identical training work and differ only in weight-update chunk size.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." >/dev/null 2>&1 && pwd)"

export RELAX_ROOT="${REPO_ROOT}"
export RELAX="${REPO_ROOT}"
source "${WORKSPACE_ROOT}/relax.env"
source "${WORKSPACE_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/scripts/models/qwen3-0.6B.sh"

MODEL_PATH="${MODEL_PATH:-${HOME}/model/Qwen3-0.6B}"
PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/benchmarks/data/task22_tiny_math.jsonl}"
TASK22_VARIANT="${TASK22_VARIANT:-baseline}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
RUN_ID="${RUN_ID:-1}"
SEED="${SEED:-$((20260801 + RUN_ID))}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
TASK22_JOB_ID="${TASK22_JOB_ID:-task22-${TASK22_VARIANT}-run-${RUN_ID}}"
RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-{}}"

case "${TASK22_VARIANT}" in
    baseline)
        UPDATE_WEIGHT_BUFFER_SIZE="${UPDATE_WEIGHT_BUFFER_SIZE:-536870912}"
        ;;
    optimized)
        UPDATE_WEIGHT_BUFFER_SIZE="${UPDATE_WEIGHT_BUFFER_SIZE:-1073741824}"
        ;;
    *)
        echo "TASK22_VARIANT must be baseline or optimized, got ${TASK22_VARIANT}" >&2
        exit 2
        ;;
esac

if ! [[ "${UPDATE_WEIGHT_BUFFER_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "UPDATE_WEIGHT_BUFFER_SIZE must be a positive integer, got ${UPDATE_WEIGHT_BUFFER_SIZE}" >&2
    exit 2
fi
if ! [[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || ((NUM_ROLLOUT > 20)); then
    echo "NUM_ROLLOUT must be an integer in [1, 20], got ${NUM_ROLLOUT}" >&2
    exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != *,* ]] || [[ "${CUDA_VISIBLE_DEVICES}" == *,*,* ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly two GPU IDs" >&2
    exit 2
fi
if [[ ! -s "${MODEL_PATH}/model.safetensors" ]]; then
    echo "Missing model checkpoint: ${MODEL_PATH}/model.safetensors" >&2
    exit 1
fi
if [[ ! -s "${PROMPT_DATA}" ]]; then
    echo "Missing prompt data: ${PROMPT_DATA}" >&2
    exit 1
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)
GRPO_ARGS=(
    --advantage-estimator grpo
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
)

TRAIN_CMD=(
    python3 -m relax.entrypoints.train
    --resource '{"actor": [1, 1], "rollout": [1, 1]}'
    --max-staleness 2
    --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE}"
    --num-data-storage-units 1
    --num-iters-per-train-update 1
    --hybrid
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --seed "${SEED}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --use-fault-tolerance
    --balance-data
    --skip-eval-before-train
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    "${GRPO_ARGS[@]}"
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --log-passrate
    --no-use-metrics-service
    --tb-project-name Relax/task22-qwen3-0.6b
    --tb-experiment-name "task22-${TASK22_VARIANT}-run-${RUN_ID}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

printf 'Task 22 command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "${TASK22_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

ray job submit \
    --no-wait \
    --log-style=record \
    --log-color=false \
    --address="${RAY_DASHBOARD_ADDRESS}" \
    --submission-id="${TASK22_JOB_ID}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    --working-dir="${REPO_ROOT}" \
    -- "${TRAIN_CMD[@]}"
