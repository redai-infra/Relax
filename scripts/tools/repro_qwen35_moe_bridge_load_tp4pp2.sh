#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Reproduce Qwen3.5-35B-A3B Megatron Bridge HF-to-Megatron loading with 8 GPUs:
#   TP=4, PP=2, DP=1
#
# Usage:
#   HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B bash scripts/tools/repro_qwen35_moe_bridge_load_tp4pp2.sh

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RELAX_REPRO_PROFILE_DIR=${RELAX_REPRO_PROFILE_DIR:-"/tmp/relax/profile"}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${REPO_ROOT}/scripts/models}"

export PYTHONPATH=$REPO_ROOT:$PYTHONPATH

source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
WORLD_GPUS=$((NNODES * NPROC_PER_NODE))

export RELAX_REPRO_PROFILE="${RELAX_REPRO_PROFILE:-0}"
export RELAX_REPRO_PROFILE_DIR="${RELAX_REPRO_PROFILE_DIR:-${REPO_ROOT}/log/megatron_bridge_profile/$(date +%Y%m%d-%H%M%S)}"

TORCHRUN_ARGS=(
   --nnodes "${NNODES}"
   --node_rank "${NODE_RANK}"
   --nproc_per_node "${NPROC_PER_NODE}"
   --master_addr "${MASTER_ADDR}"
   --master_port "${MASTER_PORT}"
)

REPRO_ARGS=(
   --debug-train-only
   --resource "{\"actor\": [${NNODES}, ${WORLD_GPUS}]}"
   --num-gpus-per-node "${NPROC_PER_NODE}"
   --actor-num-nodes "${NNODES}"
   --actor-num-gpus-per-node "${NPROC_PER_NODE}"

   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${HF_CHECKPOINT}"
   --ref-actor-config '{}'
   --load "${LOAD_CHECKPOINT:-${HF_CHECKPOINT}}"
   --megatron-to-hf-mode bridge

   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-1}"
   --expert-tensor-parallel-size 1

   --micro-batch-size 1
   --global-batch-size 1
   --num-rollout 1
   --rollout-batch-size 1
   --n-samples-per-prompt 1

   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --no-rope-fusion
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

if [[ "${RELAX_REPRO_ONLY_LOAD_WEIGHT:-1}" == "1" ]]; then
   export RELAX_REPRO_ROLE="${RELAX_REPRO_ROLE:-reference}"
   REPRO_ARGS+=(--only-load-weight)
fi

cd "${REPO_ROOT}"
torchrun "${TORCHRUN_ARGS[@]}" \
   "${REPO_ROOT}/scripts/tools/repro_megatron_bridge_load.py" \
   "${MODEL_ARGS[@]}" \
   "${REPRO_ARGS[@]}" \
   "$@"
