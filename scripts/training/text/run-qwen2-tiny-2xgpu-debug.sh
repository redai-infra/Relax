#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Tiny local 2xGPU training smoke script for ROCm/CUDA validation.
# It generates a tiny Qwen2 HF checkpoint plus synthetic rollout data locally,
# then runs one debug_train_only training step on 2 GPUs.
#
# Usage:
#   NUM_GPUS=2 bash scripts/training/text/run-qwen2-tiny-2xgpu-debug.sh

set -ex
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../entrypoint/device_env.sh"

SELECTED_GPUS="$(relax_select_top_gpus_by_free_mem 2)"
if [ -n "${SELECTED_GPUS}" ]; then
    relax_export_visible_devices "${SELECTED_GPUS}"
fi

export RAY_TMPDIR="${RAY_TMPDIR:=/tmp/ray-relax-smoke-$$}"
export RELAX_SERVE_PORT="${RELAX_SERVE_PORT:=18080}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

DATE="$(date +%Y%m%d_%H%M%S)"
NUM_GPUS="${NUM_GPUS:=2}"
ASSET_DIR="${RELAX_SMOKE_ASSET_DIR:=/tmp/relax-tiny-qwen2-smoke}"
WORK_DIR="${RELAX_SMOKE_WORK_DIR:=${ASSET_DIR}/run-${DATE}}"
mkdir -p "${WORK_DIR}" log

python3 "${SCRIPT_DIR}/../../tools/create_tiny_qwen2_smoke_assets.py" --output-dir "${ASSET_DIR}"

HF_MODEL_DIR="${ASSET_DIR}/hf_model"
DEBUG_ROLLOUT_PATH="${ASSET_DIR}/debug_rollout/{rollout_id}.pt"

MODEL_ARGS=(
   --swiglu
   --num-layers 2
   --hidden-size 128
   --ffn-hidden-size 256
   --num-attention-heads 4
   --group-query-attention
   --num-query-groups 2
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization LayerNorm
   --norm-epsilon 1e-5
   --rotary-base 10000
   --vocab-size 256
   --kv-channels 32
   --untie-embeddings-and-output-weights
   --seq-length 128
)

TRAIN_ARGS=(
   --debug-train-only
   --resource '{"actor": [1, 2]}'
   --num-gpus-per-node 2
   --actor-num-gpus-per-node 2
   --rollout-batch-size 2
   --n-samples-per-prompt 2
   --global-batch-size 4
   --micro-batch-size 1
   --num-rollout 1
   --load-debug-rollout-data "${DEBUG_ROLLOUT_PATH}"
   --save-debug-train-data "${WORK_DIR}/train_dump/{rollout_id}_{rank}.pt"
   --save "${WORK_DIR}/ckpt"
   --save-interval 1
   --async-save
   --transformer-impl local
   --hf-checkpoint "${HF_MODEL_DIR}"
   --advantage-estimator grpo
   --optimizer adam
   --lr 1e-4
   --lr-decay-style constant
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.95
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --attention-backend unfused
   --no-masked-softmax-fusion
   --no-rope-fusion
   --no-bias-swiglu-fusion
   --no-bias-dropout-fusion
   --train-memory-margin-bytes 0
)

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   "${MODEL_ARGS[@]}" \
   "${TRAIN_ARGS[@]}" 2>&1 | tee "log/qwen2-tiny-debug-gpu2-${DATE}.log"
