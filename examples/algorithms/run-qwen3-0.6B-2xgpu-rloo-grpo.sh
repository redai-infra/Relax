#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Matched 2xGPU Qwen3-0.6B GSM8K experiment for synchronous GRPO/RLOO.
#
# Usage:
#   ALGORITHM=grpo NUM_ROLLOUT=10 bash examples/algorithms/run-qwen3-0.6B-2xgpu-rloo-grpo.sh
#   ALGORITHM=rloo NUM_ROLLOUT=10 bash examples/algorithms/run-qwen3-0.6B-2xgpu-rloo-grpo.sh

set -ex
set -o pipefail

ALGORITHM="${ALGORITHM:-rloo}"
if [[ "${ALGORITHM}" != "grpo" && "${ALGORITHM}" != "rloo" ]]; then
    echo "ALGORITHM must be grpo or rloo, got ${ALGORITHM}" >&2
    exit 2
fi

now=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
VENV_ROOT="${RELAX_ROOT}/.venv"
export PATH="${VENV_ROOT}/bin:${PATH}"
export MEGATRON="${MEGATRON:-${VENV_ROOT}/Megatron-LM}"
export PYTHONPATH="${RELAX_ROOT}:${MEGATRON}:${PYTHONPATH:-}"
NVIDIA_SITE="${VENV_ROOT}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_SITE}/cudnn/lib:${NVIDIA_SITE}/cu13/lib:${NVIDIA_SITE}/nccl/lib:${NVIDIA_SITE}/nvshmem/lib:${NVIDIA_SITE}/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS=2
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/rloo}"
EXP_DIR="${EXP_DIR:-${RELAX_ROOT}/output/relax-onboarding}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES="${N_SAMPLES:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
KL_COEF="${KL_COEF:-0.01}"
SEED="${SEED:-1234}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
if (( CONTEXT_PARALLEL_SIZE > 1 )); then
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-local}"
    QKV_FORMAT="${QKV_FORMAT:-bshd}"
else
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-local}"
    QKV_FORMAT="${QKV_FORMAT:-bshd}"
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B"
    --ref-load "${MODEL_DIR}/Qwen3-0.6B"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/gsm8k/train.jsonl"
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
    --use-fault-tolerance
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --qkv-format "${QKV_FORMAT}"
    --micro-batch-size 1
)
if (( CONTEXT_PARALLEL_SIZE > 1 )); then
    PERF_ARGS+=(--calculate-per-token-loss)
fi

if [[ "${ALGORITHM}" == "rloo" ]]; then
    ALGO_ARGS=(
        --advantage-estimator rloo
        --kl-coef "${KL_COEF}"
        --kl-loss-type k1
        --entropy-coef 0.0
    )
else
    ALGO_ARGS=(
        --advantage-estimator grpo
        --use-kl-loss
        --kl-loss-coef "${KL_COEF}"
        --kl-loss-type k1
        --entropy-coef 0.0
        --eps-clip 0.2
    )
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --optimizer-offload-fraction 1.0
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.45
)

TRACKING_ARGS=(
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "qwen3-0.6b-${ALGORITHM}-gsm8k-2x5090-${now}"
)

MISC_ARGS=(
    --seed "${SEED}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --spec local
    --attention-backend "${ATTENTION_BACKEND}"
)
if [[ "${ATTENTION_BACKEND}" == "local" ]]; then
    MISC_ARGS+=(--no-masked-softmax-fusion)
fi

mkdir -p "${RELAX_ROOT}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 2], "rollout": [1, 2]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --num-gpus-per-node 2 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${RELAX_ROOT}/log/qwen3-0.6b-${ALGORITHM}-gsm8k-2x5090-${now}.log"
