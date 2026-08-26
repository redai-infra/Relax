#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Two-GPU Hybrid-async Task 22 recipe. Variants isolate zero-KL reference
# pruning from interval-two weight publication.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." >/dev/null 2>&1 && pwd)"

export RELAX_ROOT="${REPO_ROOT}"
export RELAX="${REPO_ROOT}"
source "${WORKSPACE_ROOT}/relax.env"
source "${WORKSPACE_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/scripts/models/qwen3-0.6B.sh"

build_runtime_env_json() {
    python - <<'PY'
import json
import os

runtime_env = {
    "worker_process_setup_hook": "relax.utils.logging_utils.install_asyncio_noise_filter",
    "env_vars": {
        "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "CUDACXX": os.environ.get("CUDACXX", ""),
        "CUDNN_HOME": os.environ.get("CUDNN_HOME", ""),
        "NCCL_HOME": os.environ.get("NCCL_HOME", ""),
        "CPATH": os.environ.get("CPATH", ""),
        "LIBRARY_PATH": os.environ.get("LIBRARY_PATH", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "24"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "24"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "24"),
        "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", "0"),
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": os.environ.get(
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "32"
        ),
        "NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", "1"),
        "SGLANG_HEALTH_CHECK_TIMEOUT": os.environ.get("SGLANG_HEALTH_CHECK_TIMEOUT", "180"),
        "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": os.environ.get("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", ""),
        "NVTE_USE_CUTLASS_GROUPED_GEMM": os.environ.get("NVTE_USE_CUTLASS_GROUPED_GEMM", "1"),
        "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK": os.environ.get(
            "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", "1"
        ),
        "NVTE_CUDA_ARCHS": os.environ.get("NVTE_CUDA_ARCHS", "120;120-real"),
        "INDEXER_ROPE_NEOX_STYLE": os.environ.get("INDEXER_ROPE_NEOX_STYLE", "0"),
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE", "1"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": os.environ.get(
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python"
        ),
    },
}
print(json.dumps(runtime_env))
PY
}

MODEL_PATH="${MODEL_PATH:-${HOME}/model/Qwen3-0.6B}"
PROMPT_DATA="${PROMPT_DATA:-}"
TASK22_VARIANT="${TASK22_VARIANT:-baseline}"
NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
RUN_ID="${RUN_ID:-1}"
SEED="${SEED:-$((20260801 + RUN_ID))}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
TASK22_JOB_ID="${TASK22_JOB_ID:-task22-${TASK22_VARIANT}-run-${RUN_ID}}"
if [[ -z "${RUNTIME_ENV_JSON:-}" || "${RUNTIME_ENV_JSON}" == "{}" ]]; then
    RUNTIME_ENV_JSON="$(build_runtime_env_json)"
fi

case "${TASK22_VARIANT}" in
    baseline)
        UPDATE_WEIGHT_BUFFER_SIZE="${UPDATE_WEIGHT_BUFFER_SIZE:-536870912}"
        MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
        LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-8192}"
        UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
        ENABLE_ZERO_KL_REFERENCE=1
        ;;
    zero_kl)
        UPDATE_WEIGHT_BUFFER_SIZE="${UPDATE_WEIGHT_BUFFER_SIZE:-536870912}"
        MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
        LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-8192}"
        UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
        ENABLE_ZERO_KL_REFERENCE=0
        ;;
    optimized)
        UPDATE_WEIGHT_BUFFER_SIZE="${UPDATE_WEIGHT_BUFFER_SIZE:-536870912}"
        MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
        LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-8192}"
        UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-2}"
        ENABLE_ZERO_KL_REFERENCE=0
        ;;
    *)
        echo "TASK22_VARIANT must be baseline, zero_kl, or optimized, got ${TASK22_VARIANT}" >&2
        exit 2
        ;;
esac

if ! [[ "${UPDATE_WEIGHT_BUFFER_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "UPDATE_WEIGHT_BUFFER_SIZE must be a positive integer, got ${UPDATE_WEIGHT_BUFFER_SIZE}" >&2
    exit 2
fi
if ! [[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || ((NUM_ROLLOUT > 20)); then
    echo "NUM_ROLLOUT must be an integer in [1, 20], got ${NUM_ROLLOUT}" >&2
    exit 2
fi
if ! [[ "${MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TOKENS_PER_GPU must be a positive integer, got ${MAX_TOKENS_PER_GPU}" >&2
    exit 2
fi
if ! [[ "${LOG_PROBS_MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LOG_PROBS_MAX_TOKENS_PER_GPU must be a positive integer, got ${LOG_PROBS_MAX_TOKENS_PER_GPU}" >&2
    exit 2
fi
if ! [[ "${UPDATE_WEIGHTS_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "UPDATE_WEIGHTS_INTERVAL must be a positive integer, got ${UPDATE_WEIGHTS_INTERVAL}" >&2
    exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != *,* ]] || [[ "${CUDA_VISIBLE_DEVICES}" == *,*,* ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly two GPU IDs" >&2
    exit 2
fi
if [[ ! -s "${MODEL_PATH}/model.safetensors" ]]; then
    echo "Missing model checkpoint: ${MODEL_PATH}/model.safetensors" >&2
    exit 1
fi
if [[ -z "${PROMPT_DATA}" ]]; then
    echo "PROMPT_DATA must point to a prepared JSONL dataset" >&2
    exit 1
fi
if [[ ! -s "${PROMPT_DATA}" ]]; then
    echo "Missing prompt data: ${PROMPT_DATA}" >&2
    exit 1
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)
if ((ENABLE_ZERO_KL_REFERENCE)); then
    CKPT_ARGS+=(--ref-load "${MODEL_PATH}")
fi
GRPO_ARGS=(
    --advantage-estimator grpo
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
)
if ((ENABLE_ZERO_KL_REFERENCE)); then
    GRPO_ARGS+=(--use-kl-loss)
fi

TRAIN_CMD=(
    python3 -m relax.entrypoints.train
    --resource '{"actor": [1, 1], "rollout": [1, 1]}'
    --max-staleness 2
    --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}"
    --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE}"
    --num-data-storage-units 1
    --num-iters-per-train-update 1
    --hybrid
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --seed "${SEED}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --use-fault-tolerance
    --balance-data
    --skip-eval-before-train
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}"
    "${GRPO_ARGS[@]}"
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --log-passrate
    --no-use-metrics-service
    --tb-project-name Relax/task22-qwen3-0.6b
    --tb-experiment-name "task22-${TASK22_VARIANT}-run-${RUN_ID}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

printf 'Task 22 command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "${TASK22_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

ray job submit \
    --no-wait \
    --log-style=record \
    --log-color=false \
    --address="${RAY_DASHBOARD_ADDRESS}" \
    --submission-id="${TASK22_JOB_ID}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    --working-dir="${REPO_ROOT}" \
    -- "${TRAIN_CMD[@]}"
