#!/usr/bin/env bash

set -ex
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
RUN_ID="${RUN_ID:-qwen3-4b-dapo-math-direct-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/${RUN_ID}}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

export MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/assets/exps}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export RAY_ADDRESS="${RAY_ADDRESS:-10.235.26.199:6380}"
export MASTER_ADDR="${MASTER_ADDR:-10.235.26.199}"
export RELAX_SERVE_PORT="${RELAX_SERVE_PORT:-18081}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_DEBUG="${NVTE_DEBUG:-1}"
export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-2}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export MEGATRON="${MEGATRON:-/root/Megatron-LM/}"
export RELAX="${RELAX:-${REPO_ROOT}}"
export PYTHONPATH="${RELAX}:${MEGATRON}:${RELAX}:${PYTHONPATH:-}"
export MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${REPO_ROOT}/scripts/models}"

source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

now=$(date "+%Y-%m-%d-%H:%M:%S")
PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${MODEL_DIR}"
NUM_ROLLOUT="${NUM_ROLLOUT:=4}"

CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3-4B/
   --ref-load ${EXP_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
   --save ${EXP_DIR}/Qwen3-4B_mcore_4xgpu/
   --save-interval 100
)

PROMPT_SET=${EXP_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --use-streaming-dataset
   --streaming-buffer-size 10000
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 0.8
   --global-batch-size 16
   --use-fault-tolerance
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --micro-batch-size 1
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

TRACKING_ARGS=(
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-4b-4x-direct-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --qkv-format bshd
   --attention-backend auto
   --disable-jit-fuser
   --train-env-vars '{"TORCHDYNAMO_DISABLE": "1"}'
)

python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 1], "rollout": [1, 1], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}' \
   --max-staleness 1 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 2 \
   --ref-actor-config '{"tensor_model_parallel_size": 1, "max_tokens_per_gpu": 9216, "sequence_parallel": false, "only_load_weight": true}' \
   --fully-async \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${TRACKING_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "direct-train-${now}.log"
