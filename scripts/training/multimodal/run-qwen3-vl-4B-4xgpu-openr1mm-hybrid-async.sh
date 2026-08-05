#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-VL-4B 4xGPU hybrid-async multimodal validation script.
# Same recipe as run-qwen3-vl-4B-2xgpu-openr1mm-hybrid-async.sh, scaled to
# 2 actor GPUs + 2 rollout GPUs. Actor uses TP=2 (PP=CP=1) to exercise the
# TP-gather path in DCS's actor->rollout weight push (all_gather_param);
# rollout stays 1 GPU per SGLang engine (2 engine replicas).
#
# Set ENABLE_OPTIMIZED=1 to append both benchmark optimizations:
# --hybrid-weights-backuper-on-gpu and --dedup-multimodal-preprocess.
# ENABLE_DEDUP=1 remains available for testing dedup in isolation.
#
# Usage:
#   bash scripts/training/multimodal/run-qwen3-vl-4B-4xgpu-openr1mm-hybrid-async.sh
#   ENABLE_OPTIMIZED=1 bash scripts/training/multimodal/run-qwen3-vl-4B-4xgpu-openr1mm-hybrid-async.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/hybrid_openr1mm_4gpu_4b}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../exps}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/../../../model}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../../../data}"
NUM_ROLLOUT="${NUM_ROLLOUT:=30}"
RUN_NAME="${RUN_NAME:-qwen3-vl-4b-GRPO-gpu4-hybrid-async-${now}}"
LOG_DIR="${LOG_DIR:-log}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-4B-Instruct
   --ref-load ${MODEL_DIR}/Qwen3-VL-4B-Instruct
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)

PROMPT_SET=${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001-converted.parquet

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type openr1mm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-max-prompt-len 2048
   --rollout-max-context-len 6144
   --rollout-temperature 0.8
   --global-batch-size 64
   --multimodal-keys '{"image":"image"}'
   --system-prompt "${SYSTEM_PROMPT}"
   --use-streaming-dataset
)

# Actor: TP=2/PP=1/CP=1 across its 2 GPUs (DP=1).
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --calculate-per-token-loss
   --use-dynamic-batch-size
   --max-tokens-per-gpu 6144
   --no-rope-fusion
   --recompute-granularity selective
)

GRPO_ARGS=(
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
   --use-clearml
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name ${RUN_NAME}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
)
if [ "${ENABLE_OPTIMIZED:-0}" = "1" ]; then
    MISC_ARGS+=(--hybrid-weights-backuper-on-gpu)
fi
if [ "${ENABLE_OPTIMIZED:-0}" = "1" ] || [ "${ENABLE_DEDUP:-0}" = "1" ]; then
    MISC_ARGS+=(--dedup-multimodal-preprocess)
fi


mkdir -p "${LOG_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 2], "rollout": [1, 2]}'\
   --max-staleness 2 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 1 \
   --balance-data \
   --hybrid \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
