#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Real-model 2xGPU training smoke for Meta-Llama-3-8B-Instruct on ROCm/CUDA hosts.

set -ex
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../entrypoint/device_env.sh"

SELECTED_GPUS="$(relax_select_top_gpus_by_free_mem 2)"
if [ -n "${SELECTED_GPUS}" ]; then
    relax_export_visible_devices "${SELECTED_GPUS}"
fi

export RAY_TMPDIR="${RAY_TMPDIR:=/tmp/ray-relax-llama3-smoke-$$}"
export RELAX_SERVE_PORT="${RELAX_SERVE_PORT:=18080}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/llama3-8B.sh"

DATE="$(date +%Y%m%d_%H%M%S)"
ASSET_DIR="${RELAX_SMOKE_ASSET_DIR:=/tmp/relax-real-llama3-smoke}"
WORK_DIR="${RELAX_SMOKE_WORK_DIR:=${ASSET_DIR}/run-${DATE}}"
REAL_HF_MODEL_DIR="${REAL_HF_MODEL_DIR:=/mnt/dcgpuval/models/meta-llama/Meta-Llama-3-8B-Instruct}"

mkdir -p "${WORK_DIR}" log

if [ ! -f "${REAL_HF_MODEL_DIR}/config.json" ]; then
    echo "REAL_HF_MODEL_DIR does not point to a HuggingFace checkpoint: ${REAL_HF_MODEL_DIR}" >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/../../tools/create_tiny_qwen2_smoke_assets.py" --output-dir "${ASSET_DIR}"
DEBUG_ROLLOUT_PATH="${ASSET_DIR}/debug_rollout/{rollout_id}.pt"

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
   --hf-checkpoint "${REAL_HF_MODEL_DIR}"
   --ref-load "${REAL_HF_MODEL_DIR}"
   --megatron-to-hf-mode bridge
   --load "${REAL_HF_MODEL_DIR}"
   --model-name llama
   --transformer-impl local
   --advantage-estimator grpo
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.95
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --qkv-format bshd
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
   "${TRAIN_ARGS[@]}" 2>&1 | tee "log/llama3-8b-real-debug-gpu2-${DATE}.log"
