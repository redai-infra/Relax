#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B 1xGPU REINFORCE++ quickstart — GSM8K train, AIME-2024 eval.
#
# Colocate mode: actor and rollout time-share the same GPU.
#
# train_iters = NUM_ROLLOUT × ROLLOUT_BATCH_SIZE × N_SAMPLES / GLOBAL_BATCH_SIZE
#             = 100 × 4 × 8 / 16 = 200 steps
#
# Dataset: openai/gsm8k (~7.5K problems)
#   Requires JSONL conversion before use; see beginner-task.md for the conversion script.
#   Expects: $DATA_DIR/gsm8k/train.jsonl with fields {question, answer}
#
# Usage:
#   bash examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-pp.sh
#
# Key overridable env vars:
#   MODEL_DIR  - dir containing Qwen3-0.6B/
#   DATA_DIR   - dir containing gsm8k/train.jsonl and aime-2024/aime-2024.jsonl
#
# Metrics to watch in ClearML:
#   rollout/raw_reward   — accuracy 0/1 (expect ~0.5–0.7 initial on GSM8K, rising over training)
#   train/pg_loss        — policy gradient loss (non-zero and decreasing)
#   train/grad_norm      — gradient norm (stable, not exploding)
#   eval/aime-2024-pass@1 — AIME-2024 passrate every 10 steps
#   NOTE: REINFORCE++ uses a discounted Monte-Carlo return with no group
#         baseline and a per-token k1-style KL penalty (--kl-coef 0.001)
#         folded into the return (paper §3.1); advantages are whitened
#         (--normalize-advantages) and hover near 0, while rollout/raw_reward
#         reflects accuracy. Watch train/ppo_kl for the KL magnitude.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/beginner-task}"
MODEL_DIR="${MODEL_DIR:-/your/model}"
DATA_DIR="${DATA_DIR:-/your/data}"

NUM_ROLLOUT="${NUM_ROLLOUT:=100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=4}"
N_SAMPLES="${N_SAMPLES:=8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:=16}"
# train_iters = 100 × 4 × 8 / 16 = 200

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-0.6B
    --ref-load ${MODEL_DIR}/Qwen3-0.6B
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
    --prompt-data ${DATA_DIR}/gsm8k/train.jsonl
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle

    --rm-type math

    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
    --n-samples-per-prompt ${N_SAMPLES}
    --rollout-max-response-len 2048
    --rollout-temperature 1

    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --balance-data
    --use-fault-tolerance
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --log-probs-max-tokens-per-gpu 8192
)

GRPO_ARGS=(
    --advantage-estimator reinforce_plus_plus --normalize-advantages
    # Paper convention (arXiv:2501.03262 §3.1): per-token k1-style KL penalty
    # folded into the discounted return via --kl-coef (here 0.001).
    --kl-coef 0.001
    --kl-loss-type k1
    --entropy-coef 0.00
    --eps-clip 0.2

    --use-rollout-logprobs
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
    # ~55% for training; SGLang uses 45%
    --sglang-mem-fraction-static 0.45
)

WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name qwen3-0.6b-REINFORCEpp-gsm8k-1xgpu-${now}
)

EVAL_ARGS=(
    --skip-eval-before-train
    --eval-interval 10
    --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
    --eval-input-key prompt
    --eval-label-key label
    --n-samples-per-eval-prompt 4
    --eval-max-response-len 2048
    --log-passrate
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-0.6b-REINFORCEpp-gsm8k-1xgpu-${now}.log
