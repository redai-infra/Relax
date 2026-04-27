#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -ex
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
RUN_ID="${RUN_ID:-qwen35-9b-dapo-math-direct-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/${RUN_ID}}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

DEFAULT_MASTER_ADDR="$(hostname -I | awk '{print $1}')"

export MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/assets/exps}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MASTER_ADDR="${MASTER_ADDR:-${DEFAULT_MASTER_ADDR:-127.0.0.1}}"
export RAY_ADDRESS="${RAY_ADDRESS:-${MASTER_ADDR}:6380}"
export RELAX_SERVE_PORT="${RELAX_SERVE_PORT:-18080}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_DEBUG="${NVTE_DEBUG:-1}"
export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-2}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export MEGATRON="${MEGATRON:-/root/Megatron-LM/}"
export RELAX="${RELAX:-${REPO_ROOT}}"
export PYTHONPATH="${RELAX}:${MEGATRON}:${PYTHONPATH:-}"
export MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${REPO_ROOT}/scripts/models}"

source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

now=$(date "+%Y-%m-%d-%H:%M:%S")
PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${MODEL_DIR}"
NUM_ROLLOUT="${NUM_ROLLOUT:=2}"

CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3.5-9B
   --ref-load ${EXP_DIR}/Qwen3.5-9B
   --megatron-to-hf-mode bridge
   --load ${EXP_DIR}/Qwen3-9B_mcore_8xgpu/
   --save ${EXP_DIR}/Qwen3-9B_mcore_8xgpu/
   --save-interval 50
   --max-actor-ckpt-to-keep 1
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
   --rollout-batch-size 1
   --n-samples-per-prompt 2
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 4
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${EXP_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --micro-batch-size 1
   --qkv-format bshd
   --no-rope-fusion
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
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen35-9B-8x-direct-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
   --disable-jit-fuser
   --train-env-vars '{"TORCHDYNAMO_DISABLE": "1"}'
)

python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 4], "rollout": [1, 2], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}' \
   --max-staleness 2 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 1 \
   --ref-actor-config '{"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 1, "max_tokens_per_gpu": 10240, "sequence_parallel": false, "only_load_weight": true}' \
   --fully-async \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "direct-train-${now}.log"
