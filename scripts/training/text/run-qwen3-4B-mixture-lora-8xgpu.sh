#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B Mixture-of-LoRA GRPO on DAPO math with 8 colocated GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

now="$(date '+%Y-%m-%d-%H-%M-%S')"
PROJECT_NAME="${PROJECT_NAME:-Relax/dev/qwen3-4b-mixture-lora}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_DIR}/Qwen3-4B_mixture_lora_8xgpu}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
    --load "${OUTPUT_DIR}"
    --save "${OUTPUT_DIR}"
    --save-interval "${SAVE_INTERVAL:-50}"
)

LORA_ARGS=(
    --lora-rank "${LORA_RANK:-16}"
    --lora-alpha "${LORA_ALPHA:-32}"
    --lora-target-modules linear_qkv linear_proj
    --lora-dropout 0.0
    --lora-num-experts "${LORA_NUM_EXPERTS:-4}"
    --lora-router-top-k "${LORA_ROUTER_TOP_K:-2}"
    --lora-router-temperature "${LORA_ROUTER_TEMPERATURE:-1.0}"
    --lora-router-aux-loss-coef "${LORA_ROUTER_AUX_LOSS_COEF:-0.01}"
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
    --balance-data
    --use-fault-tolerance
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --initial-loss-scale 32768
    --min-loss-scale 1
    --use-precision-aware-optimizer
    --no-store-param-remainders
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
    --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU:-30720}"
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
)

LOGGING_ARGS=(
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "qwen3-4b-mixture-lora-${now}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --skip-eval-before-train
)

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"
if [[ -z "${RUNTIME_ENV_JSON:-}" ]]; then
    RUNTIME_ENV_JSON='{}'
fi

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --fp16 \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/logs/qwen3-4b-mixture-lora-${now}.log"
