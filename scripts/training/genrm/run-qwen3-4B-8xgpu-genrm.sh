#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# GenRM + Async Demo Script
# This script combines GenRM (Generative Reward Model) with async training mode
#
# Usage:
#   bash scripts/training/genrm/run-qwen3-4B-8xgpu-genrm.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/genrm-async}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B/
   --ref-load ${MODEL_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

# ============================================
# ROLLOUT ARGS - Using dapo-genrm rm-type
# ============================================
ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   # Use dapo-genrm instead of dapo
   --rm-type dapo-genrm
   --reward-key score

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --use-fault-tolerance
)

EVAL_ARGS=(
   --skip-eval-before-train
   --log-passrate
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   #--micro-batch-size 16 # avoid OOM
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen3-4b-GRPO-GenRM-async-gpu8-${now}
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-genrm-async-test
   # --wandb-key ${WANDB_KEY}
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


# ============================================
# Resource Configuration:
# 4 GPU for rollout + 4 GPU for genRM + 8 GPU for training (colocated)
# ============================================

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 2], "rollout": [1, 3], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0], "genrm": [1, 1]}' \
   --fully-async \
   --ref-actor-config '{"tensor_model_parallel_size": 1, "max_tokens_per_gpu": 16384, "sequence_parallel": false, "only_load_weight": true}' \
   --genrm-model-path ${MODEL_DIR}/Qwen3-VL-30B-A3B-Instruct/ \
   --genrm-num-gpus-per-engine 1 \
   --genrm-engine-config '{"max_context_len": 10240}' \
   --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}' \
   --max-staleness 2 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-4b-GRPO-GenRM-async-gpu8-${now}.log
