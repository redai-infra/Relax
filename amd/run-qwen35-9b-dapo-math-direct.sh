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

export MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/assets/exps}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-Qwen/Qwen3.5-9B}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-${SCRIPT_DIR}/assets/hf-models}"
export HF_TRAIN_DATASET_PATH="${HF_TRAIN_DATASET_PATH:-zhuzilin/dapo-math-17k/dapo-math-17k.jsonl}"
export HF_EVAL_DATASET_PATH="${HF_EVAL_DATASET_PATH:-zhuzilin/aime-2024/aime-2024.jsonl}"
export HF_DATASET_DIR="${HF_DATASET_DIR:-${SCRIPT_DIR}/assets/hf-datasets}"
if command -v rocm-smi >/dev/null 2>&1; then
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
    unset CUDA_VISIBLE_DEVICES
    unset ROCR_VISIBLE_DEVICES
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    unset HIP_VISIBLE_DEVICES
    unset ROCR_VISIBLE_DEVICES
fi
export NUM_GPUS="${NUM_GPUS:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export QWEN35_SOCKET_IFNAME="${QWEN35_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${QWEN35_SOCKET_IFNAME}}"
export RAY_ADDRESS="${RAY_ADDRESS:-${MASTER_ADDR}:6379}"
export RELAX_SERVE_PORT="${RELAX_SERVE_PORT:-8000}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-1}"
export NVTE_DEBUG="${NVTE_DEBUG:-1}"
export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-2}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export MEGATRON="${MEGATRON:-/root/Megatron-LM/}"
export RELAX="${RELAX:-${REPO_ROOT}}"
export PYTHONPATH="${RELAX}:${MEGATRON}:${PYTHONPATH:-}"
export MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${REPO_ROOT}/scripts/models}"
if [ -z "${RELAX_RESOURCE:-}" ]; then
    RELAX_RESOURCE="$(
        python3 - <<'PY'
import json

print(
    json.dumps(
        {
            "actor": [1, 4],
            "rollout": [1, 2],
            "reference": [1, 1],
            "actor_fwd": [1, 1],
            "advantages": [1, 0],
        }
    )
)
PY
    )"
    export RELAX_RESOURCE
fi

source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

resolve_hf_model_path() {
    local repo_id="$1"
    local repo_name="${repo_id##*/}"
    local local_dir="${HF_MODEL_DIR}/${repo_name}"

    if [ ! -f "${local_dir}/config.json" ]; then
        mkdir -p "${local_dir}"
        hf download "${repo_id}" --local-dir "${local_dir}" >&2
    fi

    if [ ! -f "${local_dir}/config.json" ]; then
        echo "HF model config was not found after download: ${repo_id} -> ${local_dir}" >&2
        exit 1
    fi

    printf "%s" "${local_dir}"
}

resolve_hf_dataset_file() {
    local repo_file="$1"
    local repo_id="${repo_file%/*}"
    local filename="${repo_file##*/}"
    local repo_name="${repo_id##*/}"
    local local_dir="${HF_DATASET_DIR}/${repo_name}"
    local local_file="${local_dir}/${filename}"

    if [ ! -f "${local_file}" ]; then
        mkdir -p "${local_dir}"
        hf download --repo-type dataset "${repo_id}" --include "${filename}" --local-dir "${local_dir}" >&2
    fi

    if [ ! -f "${local_file}" ]; then
        echo "HF dataset file was not found after download: ${repo_file} -> ${local_file}" >&2
        exit 1
    fi

    printf "%s" "${local_file}"
}

now=$(date "+%Y-%m-%d-%H:%M:%S")
PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${MODEL_DIR}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"
MODEL_PATH="$(resolve_hf_model_path "${HF_MODEL_PATH}")"
PROMPT_SET="$(resolve_hf_dataset_file "${HF_TRAIN_DATASET_PATH}")"
EVAL_PROMPT_SET="$(resolve_hf_dataset_file "${HF_EVAL_DATASET_PATH}")"
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-${EXP_DIR}/Qwen3-9B_mcore_8xgpu}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_PATH}
   --ref-load ${MODEL_PATH}
   --megatron-to-hf-mode bridge
   --load ${LOCAL_CKPT_DIR}/
   --save ${LOCAL_CKPT_DIR}/
   --save-interval 50
   --max-actor-ckpt-to-keep 1
)

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
   --rollout-max-response-len 2000
   --rollout-temperature 1
   --global-batch-size 256
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${EVAL_PROMPT_SET}
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 2000
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size ${ACTOR_TP:-4}
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
   --rollout-num-gpus-per-engine 2
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
)

mkdir -p log
RAY_JOB_ARGS=()
if [ -n "${WORKING_DIR:-}" ]; then
   RAY_JOB_ARGS+=(--working-dir "${WORKING_DIR}")
fi
if [ -n "${RUNTIME_ENV_JSON:-}" ]; then
   RAY_JOB_ARGS+=(--runtime-env-json="${RUNTIME_ENV_JSON}")
fi

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP:-${MASTER_ADDR}}:${RAY_DASHBOARD_PORT:-8265}" \
   "${RAY_JOB_ARGS[@]}" \
   -- python3 -m relax.entrypoints.train \
   --resource "${RELAX_RESOURCE}" \
   --max-staleness 2 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 32 \
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
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen35-9B-GRPO-gpu16-async-${now}.log"
