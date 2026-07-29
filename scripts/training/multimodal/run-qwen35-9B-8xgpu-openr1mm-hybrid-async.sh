#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xGPU async training script (ray-job mode).
# The Ray cluster is managed externally — do NOT kill ray or start a new cluster.
#
# Usage:
#   bash scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh [hybrid-async|sync]

set -ex
set -o pipefail

MODE=${1:-"hybrid-async"}

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
MODEL_CONFIG_FILE="${MODEL_CONFIG_FILE:-${MODEL_CONFIG_DIR}/qwen35-9B.sh}"
if [ ! -f "${MODEL_CONFIG_FILE}" ]; then
    echo "MODEL_CONFIG_FILE does not exist: ${MODEL_CONFIG_FILE}" >&2
    exit 2
fi
source "${MODEL_CONFIG_FILE}"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/fully_async_openr1mm}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
MODEL_RUN_NAME="${MODEL_RUN_NAME:-qwen35-9b}"
MODEL_CHECKPOINT_DIR="${MODEL_CHECKPOINT_DIR:-${MODEL_DIR}/${MODEL_NAME}}"
REFERENCE_CHECKPOINT_DIR="${REFERENCE_CHECKPOINT_DIR:-${MODEL_CHECKPOINT_DIR}}"

CHECKPOINT_SAVE="${CHECKPOINT_SAVE:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${EXP_DIR}/${MODEL_NAME}_mcore_8xgpu/}"
CHECKPOINT_SAVE_INTERVAL="${CHECKPOINT_SAVE_INTERVAL:-100}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
ROLLOUT_RESULT_DIR="${ROLLOUT_RESULT_DIR:-}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-}"

HYBRID_PIPELINE_FORWARD="${HYBRID_PIPELINE_FORWARD:-0}"
HYBRID_PIPELINE_TRACE_DIR="${HYBRID_PIPELINE_TRACE_DIR:-}"
HYBRID_PIPELINE_FETCH_TIMEOUT_S="${HYBRID_PIPELINE_FETCH_TIMEOUT_S:-600}"
NUM_ITERS_PER_TRAIN_UPDATE="${NUM_ITERS_PER_TRAIN_UPDATE:-2}"
SGLANG_DETERMINISTIC_INFERENCE="${SGLANG_DETERMINISTIC_INFERENCE:-0}"
SEED="${SEED:-}"
ROLLOUT_SEED="${ROLLOUT_SEED:-}"

ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-10240}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-2048}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-12288}"
ACTOR_MAX_TOKENS_PER_GPU="${ACTOR_MAX_TOKENS_PER_GPU:-12288}"
HYBRID_ACTOR_GPUS="${HYBRID_ACTOR_GPUS:-4}"
HYBRID_ROLLOUT_GPUS="${HYBRID_ROLLOUT_GPUS:-4}"
SYNC_GPUS="${SYNC_GPUS:-8}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"

require_positive_integer() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got ${value}" >&2
        exit 2
    fi
}

require_fraction() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
        echo "${name} must be greater than 0 and at most 1, got ${value}" >&2
        exit 2
    fi
}

case "${MODE}" in
  hybrid-async|sync) ;;
  *)
    echo "MODE must be hybrid-async or sync, got ${MODE}" >&2
    exit 2
    ;;
esac
case "${HYBRID_PIPELINE_FORWARD}" in
  0|1) ;;
  *)
    echo "HYBRID_PIPELINE_FORWARD must be 0 or 1, got ${HYBRID_PIPELINE_FORWARD}" >&2
    exit 2
    ;;
esac
case "${SGLANG_DETERMINISTIC_INFERENCE}" in
  0|1) ;;
  *)
    echo "SGLANG_DETERMINISTIC_INFERENCE must be 0 or 1, got ${SGLANG_DETERMINISTIC_INFERENCE}" >&2
    exit 2
    ;;
esac
case "${CHECKPOINT_SAVE}" in
  0|1) ;;
  *)
    echo "CHECKPOINT_SAVE must be 0 or 1, got ${CHECKPOINT_SAVE}" >&2
    exit 2
    ;;
esac
if [ "${MODE}" != "hybrid-async" ] && {
    [ "${HYBRID_PIPELINE_FORWARD}" = "1" ] || [ -n "${HYBRID_PIPELINE_TRACE_DIR}" ];
}; then
    echo "Hybrid pipeline forward/trace options require MODE=hybrid-async" >&2
    exit 2
fi

for item in \
    "NUM_ROLLOUT:${NUM_ROLLOUT}" \
    "HYBRID_PIPELINE_FETCH_TIMEOUT_S:${HYBRID_PIPELINE_FETCH_TIMEOUT_S}" \
    "NUM_ITERS_PER_TRAIN_UPDATE:${NUM_ITERS_PER_TRAIN_UPDATE}" \
    "ROLLOUT_MAX_RESPONSE_LEN:${ROLLOUT_MAX_RESPONSE_LEN}" \
    "ROLLOUT_MAX_PROMPT_LEN:${ROLLOUT_MAX_PROMPT_LEN}" \
    "ROLLOUT_MAX_CONTEXT_LEN:${ROLLOUT_MAX_CONTEXT_LEN}" \
    "ACTOR_MAX_TOKENS_PER_GPU:${ACTOR_MAX_TOKENS_PER_GPU}" \
    "HYBRID_ACTOR_GPUS:${HYBRID_ACTOR_GPUS}" \
    "HYBRID_ROLLOUT_GPUS:${HYBRID_ROLLOUT_GPUS}" \
    "SYNC_GPUS:${SYNC_GPUS}" \
    "ROLLOUT_NUM_GPUS_PER_ENGINE:${ROLLOUT_NUM_GPUS_PER_ENGINE}"; do
    require_positive_integer "${item%%:*}" "${item#*:}"
done
if [ "${HYBRID_PIPELINE_FORWARD}" = "1" ] && (( NUM_ITERS_PER_TRAIN_UPDATE < 2 )); then
    echo "HYBRID_PIPELINE_FORWARD requires NUM_ITERS_PER_TRAIN_UPDATE >= 2" >&2
    exit 2
fi
if [ "${CHECKPOINT_SAVE}" = "1" ]; then
    require_positive_integer "CHECKPOINT_SAVE_INTERVAL" "${CHECKPOINT_SAVE_INTERVAL}"
    require_positive_integer "MAX_ACTOR_CKPT_TO_KEEP" "${MAX_ACTOR_CKPT_TO_KEEP}"
fi
if (( ROLLOUT_MAX_CONTEXT_LEN < ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN )); then
    echo "ROLLOUT_MAX_CONTEXT_LEN must cover prompt + response limits" >&2
    exit 2
fi
if (( HYBRID_ROLLOUT_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
    echo "HYBRID_ROLLOUT_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE" >&2
    exit 2
fi
if (( SYNC_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
    echo "SYNC_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE" >&2
    exit 2
fi
if (( HYBRID_ACTOR_GPUS % 4 != 0 || SYNC_GPUS % 4 != 0 )); then
    echo "actor GPU counts must be multiples of TP(2) * CP(2)" >&2
    exit 2
fi
if [ "${HYBRID_PIPELINE_FORWARD}" = "1" ] && [ "${HYBRID_ACTOR_GPUS}" != "4" ]; then
    echo "HYBRID_PIPELINE_FORWARD currently requires TP=2, CP=2, DP=1 (4 actor GPUs)" >&2
    exit 2
fi
require_fraction "SGLANG_MEM_FRACTION_STATIC" "${SGLANG_MEM_FRACTION_STATIC}"

if [ "${CHECKPOINT_SAVE}" = "0" ]; then
    ROLLOUT_RESULT_DIR="${ROLLOUT_RESULT_DIR:-${EXP_DIR}/rollout_result}"
    TENSORBOARD_DIR="${TENSORBOARD_DIR:-${EXP_DIR}/tensorboard_log}"
fi
if [ -n "${TENSORBOARD_DIR}" ]; then
    export TENSORBOARD_DIR
    RUNTIME_ENV_JSON_INPUT="${RUNTIME_ENV_JSON:-}"
    if [ -z "${RUNTIME_ENV_JSON_INPUT}" ]; then
        RUNTIME_ENV_JSON_INPUT='{}'
    fi
    RUNTIME_ENV_JSON="$(
        python3 - "${RUNTIME_ENV_JSON_INPUT}" "${TENSORBOARD_DIR}" <<'PY'
import json
import sys

runtime_env = json.loads(sys.argv[1])
if not isinstance(runtime_env, dict):
    raise SystemExit("RUNTIME_ENV_JSON must decode to an object")
env_vars = runtime_env.setdefault("env_vars", {})
if not isinstance(env_vars, dict):
    raise SystemExit("RUNTIME_ENV_JSON env_vars must be an object")
env_vars["TENSORBOARD_DIR"] = sys.argv[2]
print(json.dumps(runtime_env, separators=(",", ":"), sort_keys=True))
PY
    )"
fi

HYBRID_RESOURCE="{\"actor\": [1, ${HYBRID_ACTOR_GPUS}], \"rollout\": [1, ${HYBRID_ROLLOUT_GPUS}]}"
SYNC_RESOURCE="{\"actor\": [1, ${SYNC_GPUS}], \"rollout\": [1, ${SYNC_GPUS}]}"
if [ "${MODE}" = "hybrid-async" ]; then
    RUN_GPU_COUNT=$((HYBRID_ACTOR_GPUS + HYBRID_ROLLOUT_GPUS))
else
    RUN_GPU_COUNT="${SYNC_GPUS}"
fi

HYBRID_PIPELINE_ARGS=()
if [ "${HYBRID_PIPELINE_FORWARD}" = "1" ]; then
    HYBRID_PIPELINE_ARGS+=(
        --hybrid-pipeline-forward
        --hybrid-pipeline-fetch-timeout-s "${HYBRID_PIPELINE_FETCH_TIMEOUT_S}"
    )
fi
if [ -n "${HYBRID_PIPELINE_TRACE_DIR}" ]; then
    HYBRID_PIPELINE_ARGS+=(
        --hybrid-pipeline-trace-dir "${HYBRID_PIPELINE_TRACE_DIR}"
    )
fi

REPRO_ARGS=()
if [ -n "${SEED}" ]; then
    REPRO_ARGS+=(--seed "${SEED}")
fi
if [ -n "${ROLLOUT_SEED}" ]; then
    REPRO_ARGS+=(--rollout-seed "${ROLLOUT_SEED}")
fi

DEBUG_ARGS=()
if [ -n "${SAVE_DEBUG_ROLLOUT_DATA:-}" ]; then
    DEBUG_ARGS+=(--save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA}")
fi
if [ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
    DEBUG_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi
if [ -n "${SAVE_DEBUG_TRAIN_DATA:-}" ]; then
    DEBUG_ARGS+=(--save-debug-train-data "${SAVE_DEBUG_TRAIN_DATA}")
fi
if [ "${SGLANG_DETERMINISTIC_INFERENCE}" = "1" ]; then
    DEBUG_ARGS+=(--sglang-enable-deterministic-inference)
fi

printf '%s\n' \
    "MODE=${MODE}" \
    "NUM_ROLLOUT=${NUM_ROLLOUT}" \
    "HYBRID_PIPELINE_FORWARD=${HYBRID_PIPELINE_FORWARD}" \
    "HYBRID_PIPELINE_TRACE_DIR=${HYBRID_PIPELINE_TRACE_DIR}" \
    "HYBRID_PIPELINE_FETCH_TIMEOUT_S=${HYBRID_PIPELINE_FETCH_TIMEOUT_S}" \
    "NUM_ITERS_PER_TRAIN_UPDATE=${NUM_ITERS_PER_TRAIN_UPDATE}" \
    "SEED=${SEED}" \
    "ROLLOUT_SEED=${ROLLOUT_SEED}" \
    "SGLANG_DETERMINISTIC_INFERENCE=${SGLANG_DETERMINISTIC_INFERENCE}" \
    "MODEL_CONFIG_FILE=${MODEL_CONFIG_FILE}" \
    "MODEL_CHECKPOINT_DIR=${MODEL_CHECKPOINT_DIR}" \
    "REFERENCE_CHECKPOINT_DIR=${REFERENCE_CHECKPOINT_DIR}" \
    "CHECKPOINT_SAVE=${CHECKPOINT_SAVE}" \
    "CHECKPOINT_DIR=${CHECKPOINT_DIR}" \
    "ROLLOUT_RESULT_DIR=${ROLLOUT_RESULT_DIR}" \
    "TENSORBOARD_DIR=${TENSORBOARD_DIR}" \
    "ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU}" \
    "HYBRID_RESOURCE=${HYBRID_RESOURCE}" \
    "SYNC_RESOURCE=${SYNC_RESOURCE}" \
    "RUN_GPU_COUNT=${RUN_GPU_COUNT}"


CKPT_ARGS=(
   --hf-checkpoint "${MODEL_CHECKPOINT_DIR}"
   --ref-load "${REFERENCE_CHECKPOINT_DIR}"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)
if [ "${CHECKPOINT_SAVE}" = "1" ]; then
    CKPT_ARGS+=(
        --save "${CHECKPOINT_DIR}"
        --save-interval "${CHECKPOINT_SAVE_INTERVAL}"
        --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP}"
    )
fi

OUTPUT_ARGS=()
if [ -n "${ROLLOUT_RESULT_DIR}" ]; then
    OUTPUT_ARGS+=(--rollout-result-dir "${ROLLOUT_RESULT_DIR}")
fi
if [ -n "${TENSORBOARD_DIR}" ]; then
    OUTPUT_ARGS+=(--tensorboard-dir "${TENSORBOARD_DIR}")
fi

PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet}"

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type openr1mm
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-temperature 0.8
   --global-batch-size 256
   --multimodal-keys '{"image":"image"}'
   --system-prompt "${SYSTEM_PROMPT}"
   --use-streaming-dataset
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss
   # --micro-batch-size 16
   # --qkv-format bshd
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${ACTOR_MAX_TOKENS_PER_GPU}"
   --no-rope-fusion
)

GRPO_ARGS=(
   # --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

)

WANDB_ARGS=(
   --use-tensorboard
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${MODEL_RUN_NAME}-GRPO-gpu${RUN_GPU_COUNT}-${MODE}-${now}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)


LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"
if [ "${MODE}" = "hybrid-async" ]; then
     ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 -m relax.entrypoints.train \
        --resource "${HYBRID_RESOURCE}" \
   --max-staleness 2 \
        --num-data-storage-units 1 \
        --num-iters-per-train-update "${NUM_ITERS_PER_TRAIN_UPDATE}" \
         --balance-data \
        --hybrid \
        "${HYBRID_PIPELINE_ARGS[@]}" \
        "${REPRO_ARGS[@]}" \
        "${DEBUG_ARGS[@]}" \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${OUTPUT_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_DIR}/${MODEL_RUN_NAME}-GRPO-gpu${RUN_GPU_COUNT}-hybrid-async-${now}.log"
else
    ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
         --runtime-env-json="${RUNTIME_ENV_JSON}" \
         -- python3 -m relax.entrypoints.train \
         --resource "${SYNC_RESOURCE}" \
         --max-staleness 0 \
         --num-data-storage-units 1 \
         --colocate \
         --use-health-check \
         --balance-data \
         "${REPRO_ARGS[@]}" \
         "${DEBUG_ARGS[@]}" \
         "${MODEL_ARGS[@]}" \
         "${CKPT_ARGS[@]}" \
         "${OUTPUT_ARGS[@]}" \
         "${ROLLOUT_ARGS[@]}" \
         "${OPTIMIZER_ARGS[@]}" \
         "${GRPO_ARGS[@]}" \
         "${WANDB_ARGS[@]}" \
         "${PERF_ARGS[@]}" \
         "${SGLANG_ARGS[@]}" \
         "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_DIR}/${MODEL_RUN_NAME}-GRPO-gpu${RUN_GPU_COUNT}-fully-sync-${now}.log"
fi
