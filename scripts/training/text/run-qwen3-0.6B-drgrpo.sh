#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B synchronous colocate script for GRPO/Dr.GRPO correctness tests.
#
# Usage:
#   ADVANTAGE_ESTIMATOR=dr_grpo CONTEXT_PARALLEL_SIZE=4 NUM_GPUS=4 \
#     bash scripts/training/text/run-qwen3-0.6B-drgrpo.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-dr_grpo}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
DATA_PAD_SIZE_MULTIPLIER="${DATA_PAD_SIZE_MULTIPLIER:-128}"
RUN_ID="${RUN_ID:-qwen3-0.6b-${ADVANTAGE_ESTIMATOR}-cp${CONTEXT_PARALLEL_SIZE}-gpu${NUM_GPUS}}"
APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS:-}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to a complete Qwen3-0.6B HF checkpoint}"
PROMPT_SET="${PROMPT_SET:?PROMPT_SET must point to a math JSONL dataset}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/log}"
ROLLOUT_RESULT_DIR="${ROLLOUT_RESULT_DIR:-${OUTPUT_DIR}/rollout_result}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"
DETERMINISTIC_MODE="${DETERMINISTIC_MODE:-0}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-0}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${MODEL_PATH}}"
ACTOR_LOAD="${ACTOR_LOAD:-}"
ENTROPY_COEF="${ENTROPY_COEF:-0}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0}"
USE_OPSM="${USE_OPSM:-0}"
OPSM_DELTA="${OPSM_DELTA:-0.01}"
USE_OPD="${USE_OPD:-0}"
OPD_TEACHER_LOAD="${OPD_TEACHER_LOAD:-}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --ref-load "${REF_MODEL_PATH}"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)
if [ -n "${ACTOR_LOAD}" ]; then
   CKPT_ARGS+=(--load "${ACTOR_LOAD}")
fi
if [ "${SAVE_CHECKPOINT}" = "1" ]; then
   CKPT_ARGS+=(--save "${OUTPUT_DIR}/checkpoint" --save-interval "${SAVE_INTERVAL}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rollout-seed 42

   --rm-type dapo
   --reward-key score

   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   --rollout-temperature 1

   --global-batch-size 16
   --log-passrate
   --skip-eval-before-train
   --rollout-result-dir "${ROLLOUT_RESULT_DIR}"
)
if [ -n "${APPLY_CHAT_TEMPLATE_KWARGS}" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --qkv-format thd
   --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER}"

   --calculate-per-token-loss
   --micro-batch-size "${MICRO_BATCH_SIZE}"
)
if [ "${USE_DYNAMIC_BATCH_SIZE}" = "1" ]; then
   PERF_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
fi

ALGORITHM_ARGS=(
   --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
   --kl-coef 0
   --kl-loss-coef "${KL_LOSS_COEF}"
   --kl-loss-type low_var_kl
   --entropy-coef "${ENTROPY_COEF}"
   --eps-clip 0.2
   --eps-clip-high 0.28
)
if [ "${KL_LOSS_COEF}" != "0" ]; then
   ALGORITHM_ARGS+=(--use-kl-loss)
fi
if [ "${USE_OPSM}" = "1" ]; then
   ALGORITHM_ARGS+=(--use-opsm --opsm-delta "${OPSM_DELTA}")
fi
if [ "${USE_OPD}" = "1" ]; then
   ALGORITHM_ARGS+=(
      --use-opd
      --opd-type megatron
      --opd-teacher-load "${OPD_TEACHER_LOAD}"
      --opd-kl-coef 0
      --opd-loss-coef 0.01
      --opd-only-reward
      --opd-token-selection student_sampled
      --opd-log-prob-top-k 0
      --opd-kl-type low_var_kl
   )
fi

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
   --rollout-num-gpus-per-engine "${NUM_GPUS}"
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)
if [ "${DETERMINISTIC_MODE}" = "1" ]; then
   MISC_ARGS+=(--deterministic-mode)
fi

DEBUG_ARGS=()
if [ -n "${DUMP_DETAILS:-}" ]; then
   DEBUG_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi
if [ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
   DEBUG_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"actor\": [1, ${NUM_GPUS}], \"rollout\": [1, ${NUM_GPUS}]}" \
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
   "${MISC_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/${RUN_ID}-${now}.log"
