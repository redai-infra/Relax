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
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"
# source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/fully_async_openr1mm}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

HYBRID_PIPELINE_FORWARD="${HYBRID_PIPELINE_FORWARD:-0}"
HYBRID_PIPELINE_TRACE_DIR="${HYBRID_PIPELINE_TRACE_DIR:-}"
HYBRID_PIPELINE_FETCH_TIMEOUT_S="${HYBRID_PIPELINE_FETCH_TIMEOUT_S:-600}"
SGLANG_DETERMINISTIC_INFERENCE="${SGLANG_DETERMINISTIC_INFERENCE:-0}"
SEED="${SEED:-}"
ROLLOUT_SEED="${ROLLOUT_SEED:-}"

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
if [ "${MODE}" != "hybrid-async" ] && {
    [ "${HYBRID_PIPELINE_FORWARD}" = "1" ] || [ -n "${HYBRID_PIPELINE_TRACE_DIR}" ];
}; then
    echo "Hybrid pipeline forward/trace options require MODE=hybrid-async" >&2
    exit 2
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
    "SEED=${SEED}" \
    "ROLLOUT_SEED=${ROLLOUT_SEED}" \
    "SGLANG_DETERMINISTIC_INFERENCE=${SGLANG_DETERMINISTIC_INFERENCE}"


CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/Qwen3.5-9B
   # --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-4B-Instruct
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   # --ref-load ${MODEL_DIR}/Qwen3-VL-4B-Instruct
   # --load ${EXP_DIR}/Qwen3.5-9B_mcore_8xgpu/
   --save ${EXP_DIR}/Qwen3.5-9B_mcore_8xgpu/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type openr1mm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 10240
   --rollout-max-prompt-len 2048
   --rollout-max-context-len 12288
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
   --max-tokens-per-gpu 12288
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
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen35-9b-GRPO-gpu8-${MODE}-${now}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
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
        --resource '{"actor": [1, 4], "rollout": [1, 4]}'\
   --max-staleness 2 \
        --num-data-storage-units 1 \
        --num-iters-per-train-update 2  \
         --balance-data \
        --hybrid \
        "${HYBRID_PIPELINE_ARGS[@]}" \
        "${REPRO_ARGS[@]}" \
        "${DEBUG_ARGS[@]}" \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_DIR}/qwen35-9b-GRPO-gpu8-hybrid-async-${now}.log"
else
    ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
         --runtime-env-json="${RUNTIME_ENV_JSON}" \
         -- python3 -m relax.entrypoints.train \
         --resource '{"actor": [1, 8], "rollout": [1, 8]}'\
         --max-staleness 0 \
         --num-data-storage-units 1 \
         --colocate \
         --use-health-check \
         --balance-data \
         "${REPRO_ARGS[@]}" \
         "${DEBUG_ARGS[@]}" \
         "${MODEL_ARGS[@]}" \
         "${CKPT_ARGS[@]}" \
         "${ROLLOUT_ARGS[@]}" \
         "${OPTIMIZER_ARGS[@]}" \
         "${GRPO_ARGS[@]}" \
         "${WANDB_ARGS[@]}" \
         "${PERF_ARGS[@]}" \
         "${SGLANG_ARGS[@]}" \
         "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_DIR}/qwen35-9b-GRPO-gpu8-fully-sync-${now}.log"
fi
