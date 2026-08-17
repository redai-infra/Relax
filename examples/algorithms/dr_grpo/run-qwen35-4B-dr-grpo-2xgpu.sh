#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -ex
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-4B.sh"

USE_DRGRPO="${USE_DRGRPO:-1}"
PROJECT_NAME="${PROJECT_NAME:=Relax/dr-grpo/math}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/Loop/ROLL_loop/data/math_deepmath_deal.jsonl}"
EVAL_DATA="${EVAL_DATA:-${DATA_DIR}/G-OPD/data/aime24/test.jsonl}"
MODEL_PATH="${MODEL_DIR}/Qwen3.5-4B"

CHECKPOINT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key ground_truth
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": true}'
    --rollout-shuffle
    --rm-type math
    --num-rollout 200
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --rollout-max-response-len 16384
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size 512
    --wandb-always-use-train-step
    --reward-num-workers 2
    --use-wandb
    --no-use-metrics-service
    --wandb-group "${PROJECT_NAME}"
)

ALGORITHM_ARGS=(
    --advantage-estimator grpo
    --calculate-per-token-loss
    --kl-coef 0.0
    --kl-loss-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
)
if [ "${USE_DRGRPO}" = 1 ]; then
    ALGORITHM_ARGS+=(
        --disable-grpo-std-normalization
        --pg-loss-aggregation seq-mean-token-sum-norm
    )
fi

EVAL_ARGS=(
    --log-passrate
    --log-correct-samples
    --skip-eval-before-train
    --eval-interval 5
    --eval-prompt-data aime "${EVAL_DATA}"
    --eval-input-key problem
    --eval-label-key answer
    --apply-chat-template-kwargs '{"enable_thinking": true}'
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 16384
    --eval-temperature 1.0
    --eval-top-p 1.0
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --grad-reduce-in-bf16
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --no-rope-fusion
    --use-dynamic-batch-size
    --max-tokens-per-gpu 20384
    --log-probs-max-tokens-per-gpu 16384
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.45
    --sglang-chunked-prefill-size 4096
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --recompute-loss-function
)

exec python3 -m relax.entrypoints.train \
    --seed 1234 \
    --resource '{"actor": [1, 2], "rollout": [1, 2], "advantages": [1, 0]}' \
    --colocate \
    --offload \
    --wandb-project "${PROJECT_NAME}" \
    "${MODEL_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ALGORITHM_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}"
