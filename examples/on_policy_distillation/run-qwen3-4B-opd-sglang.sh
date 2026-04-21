#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# On-Policy Distillation Example: Qwen3-4B distilled from Qwen3-VL-8B-Instruct
# This script demonstrates OPD with SGLang teacher mode

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/opd}"
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../exps}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

# ============================================
# IMPORTANT: Start teacher SGLang server first
# ============================================
# In a separate terminal, start the teacher SGLang server:
#
nohup python -m sglang.launch_server \
    --model-path ${MODEL_DIR}/Qwen3-VL-8B-Instruct/ \
    --context-length 10240  \
    --tp-size 4 \
    --port 30010  \
    --mem-fraction-static 0.1 \
    --log-requests-level 2 --log-requests > /tmp/sglang.log 2>&1 &
#
# Or use a different port and set TEACHER_PORT env var:
#   TEACHER_PORT=30010 bash examples/on_policy_distillation/run-qwen3-4B-opd-sglang.sh
# ============================================

# Teacher SGLang server configuration
TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-30010}"
OPD_TEACHER_TIMEOUT_S="${OPD_TEACHER_TIMEOUT_S:-30}"
OPD_LOG_PROB_TOP_K="${OPD_LOG_PROB_TOP_K:-10}"


CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3-4B/
   --ref-load ${EXP_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
   # --load ${EXP_DIR}/Qwen3-4B_mcore_8xgpu/
   # --save ${EXP_DIR}/Qwen3-4B_mcore_8xgpu/
   # --save-interval 100
)

PROMPT_SET=${EXP_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
   --use-fault-tolerance
)

# OPD Configuration (SGLang teacher mode)
# The framework automatically fetches teacher log-probs from the SGLang server
# during rollout. No need to set --custom-rm-path or --custom-reward-post-process-path.
# Users can freely use their own custom reward functions alongside OPD.
OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --opd-teacher-timeout-s ${OPD_TEACHER_TIMEOUT_S}
   --opd-log-prob-top-k ${OPD_LOG_PROB_TOP_K}
   --rm-url http://${TEACHER_HOST}:${TEACHER_PORT}/generate
)

EVAL_ARGS=(
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${EXP_DIR}/aime-2024/aime-2024.jsonl
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
   --tb-experiment-name qwen3-4b-OPD-sglang-${now}
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
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

echo "Waiting for teacher SGLang server at ${TEACHER_HOST}:${TEACHER_PORT}..."
echo "Please ensure teacher server is running before proceeding!"

MODE=${MODE:-"sync"}

if [ ${MODE} == "sync" ]; then
    ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
        -- python3 relax/entrypoints/train.py \
        --resource '{"actor": [1, 8], "rollout": [1, 8]}'\
        --max-staleness 0 \
        --num-data-storage-units 1 \
        --colocate \
        ${MODEL_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${ROLLOUT_ARGS[@]} \
        ${OPD_ARGS[@]} \
        ${OPTIMIZER_ARGS[@]} \
        ${GRPO_ARGS[@]} \
        ${WANDB_ARGS[@]} \
        ${PERF_ARGS[@]} \
        ${EVAL_ARGS[@]} \
        ${SGLANG_ARGS[@]} \
        ${MISC_ARGS[@]}  2>&1 | tee qwen3-4b-OPD-sglang-sync-${now}.log
elif [ ${MODE} == "async" ]; then
    ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
        -- python3 relax/entrypoints/train.py \
        --resource '{"actor": [1, 2], "rollout": [1, 4], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}'\
        --max-staleness 2 \
        --num-data-storage-units 1 \
        --num-iters-per-train-update 8 \
        --ref-actor-config '{"tensor_model_parallel_size": 1, "max_tokens_per_gpu": 16384, "sequence_parallel": false, "only_load_weight": true}' \
        --fully-async \
        ${MODEL_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${ROLLOUT_ARGS[@]} \
        ${OPD_ARGS[@]} \
        ${OPTIMIZER_ARGS[@]} \
        ${GRPO_ARGS[@]} \
        ${WANDB_ARGS[@]} \
        ${PERF_ARGS[@]} \
        ${EVAL_ARGS[@]} \
        ${SGLANG_ARGS[@]} \
        ${MISC_ARGS[@]}  2>&1 | tee qwen3-4b-OPD-sglang-async-${now}.log
else
    echo "Unknown MODE: ${MODE}. Please set MODE to 'sync' or 'async'."
    exit 1
fi
