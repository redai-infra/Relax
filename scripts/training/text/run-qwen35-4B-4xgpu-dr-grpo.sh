#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-4B 4xGPU colocate (sync) Dr.GRPO training script.
#
# Usage:
#   hf download openai/gsm8k main/train-00000-of-00001.parquet \
#     --repo-type dataset \
#     --revision 740312add88f781978c0658806c59bc2815b9866 \
#     --local-dir /path/to/gsm8k-pinned
#   python scripts/testing/convert_gsm8k_for_dr_grpo_e2e.py \
#     --input /path/to/gsm8k-pinned/main/train-00000-of-00001.parquet \
#     --output /path/to/gsm8k-train-shuffle42-first256.jsonl
#   MODEL_PATH=/path/to/Qwen3.5-4B \
#   PROMPT_SET=/path/to/gsm8k-train-shuffle42-first256.jsonl \
#   OUTPUT_DIR=/path/to/output \
#   TRAIN_DATA_DIR=/path/to/output/train_data \
#   NUM_ROLLOUT=200 \
#     bash scripts/training/text/run-qwen35-4B-4xgpu-dr-grpo.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
   source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-4B.sh"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to a complete Qwen3.5-4B HF checkpoint}"
PROMPT_SET="${PROMPT_SET:?PROMPT_SET must point to a math JSONL dataset}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-dr_grpo}"
RM_TYPE="${RM_TYPE:-dapo}"
REWARD_KEY="${REWARD_KEY:-score}"
APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS:-}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
RUN_ID="${RUN_ID:-qwen35-4b-${ADVANTAGE_ESTIMATOR}-cp${CONTEXT_PARALLEL_SIZE}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/log}"
ROLLOUT_RESULT_DIR="${ROLLOUT_RESULT_DIR:-${OUTPUT_DIR}/rollout_result}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
export TENSORBOARD_DIR

RUNTIME_ENV_JSON=$(python3 -c '
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime_env.setdefault("env_vars", {})["TENSORBOARD_DIR"] = os.environ["TENSORBOARD_DIR"]
print(json.dumps(runtime_env))
')
export RUNTIME_ENV_JSON

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --ref-load "${MODEL_PATH}"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)
if [ -n "${ACTOR_LOAD:-}" ]; then
   CKPT_ARGS+=(--load "${ACTOR_LOAD}")
fi
if [ "${SAVE_CHECKPOINT:-0}" = "1" ]; then
   CKPT_ARGS+=(--save "${OUTPUT_DIR}/checkpoint" --save-interval "${SAVE_INTERVAL:-50}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rollout-seed 42
   --rm-type "${RM_TYPE}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   --rollout-temperature 0.7
   --rollout-top-p 0.8
   --rollout-top-k 20
   --global-batch-size 16
   --log-passrate
   --skip-eval-before-train
   --rollout-result-dir "${ROLLOUT_RESULT_DIR}"
)
if [ -n "${REWARD_KEY}" ]; then
   ROLLOUT_ARGS+=(--reward-key "${REWARD_KEY}")
fi
if [ -n "${APPLY_CHAT_TEMPLATE_KWARGS}" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --qkv-format thd
   --calculate-per-token-loss
   --micro-batch-size 1
   --no-rope-fusion
)

ALGORITHM_ARGS=(
   --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
   --kl-coef 0
   --use-kl-loss
   --kl-loss-type low_var_kl
   --kl-loss-coef 0
   --entropy-coef 0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --seed 1234
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)
if [ -n "${TRAIN_DATA_DIR:-}" ]; then
   MISC_ARGS+=(--save-debug-train-data "${TRAIN_DATA_DIR}/{rollout_id}_{rank}.pt")
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${TENSORBOARD_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 4], "rollout": [1, 4]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGORITHM_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/${RUN_ID}-${now}.log"
