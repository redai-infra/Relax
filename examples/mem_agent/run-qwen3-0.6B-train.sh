#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the frozen Qwen3-0.6B BF16 checkpoint.}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the screened Task 36 pilot data directory.}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the Task 36 experiment output directory.}"
SAVE_DIR="${SAVE_DIR:-${RUN_ROOT}/checkpoints}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/pilot-train.jsonl}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-${DATA_DIR}/pilot-selection.manifest.json}"
NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
GPU_ID="${GPU_ID:-0}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-12}"
RUN_NAME="${RUN_NAME:-mem-agent-qwen3-0.6b-pilot}"
LOAD_PATH="${LOAD_PATH:-}"
START_ROLLOUT_ID="${START_ROLLOUT_ID:-0}"

[[ -f "${TRAIN_DATA}" ]] || {
  echo "Missing Pass@N-screened pilot data: ${TRAIN_DATA}" >&2
  exit 1
}
[[ -f "${SELECTION_MANIFEST}" ]] || {
  echo "Missing selection manifest: ${SELECTION_MANIFEST}; baseline screening must run before training." >&2
  exit 1
}
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing model config: ${MODEL_PATH}/config.json" >&2; exit 1; }
((START_ROLLOUT_ID >= 0 && START_ROLLOUT_ID < NUM_ROLLOUT)) || {
  echo "START_ROLLOUT_ID must be in [0, NUM_ROLLOUT)." >&2
  exit 1
}
if ((START_ROLLOUT_ID > 0)) && [[ -z "${LOAD_PATH}" ]]; then
  echo "A positive START_ROLLOUT_ID requires LOAD_PATH." >&2
  exit 1
fi
if [[ -n "${LOAD_PATH}" && ! -d "${LOAD_PATH}" ]]; then
  echo "Missing resume checkpoint directory: ${LOAD_PATH}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tensorboard" "${SAVE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NUM_GPUS=1
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# PPIO exposes many logical CPUs under a much smaller cgroup pids.max. Limit
# only `ray start`; all other Ray CLI calls retain their original arguments.
ray() {
  if [[ "${1:-}" == "start" ]]; then
    command ray "$@" --num-cpus "${RAY_NUM_CPUS}"
  else
    command ray "$@"
  fi
}
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
  # The shared entrypoint treats several unset networking overrides as empty,
  # but reads them without nounset-safe expansion. Suspend only `-u` while it
  # is sourced; `-e`/pipefail remain active, and strict mode is restored before
  # any Task36 argument construction or job submission.
  set +u
  source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
  set -u
fi
unset -f ray
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

RESUME_ARGS=()
if [[ -n "${LOAD_PATH}" ]]; then
  # --num-rollout remains the total target. ReLax restores optimizer/model
  # state from LOAD_PATH and continues at this explicit rollout id. The trend
  # gate and full pilot intentionally use different total rollout targets, so
  # keep the new constant-LR scheduler horizon while loading the saved state.
  RESUME_ARGS+=(
    --load "${LOAD_PATH}"
    --start-rollout-id "${START_ROLLOUT_ID}"
    --override-opt-param-scheduler
  )
fi

NOW="$(date '+%Y%m%dT%H%M%S%z')"
LOG_FILE="${RUN_ROOT}/logs/${RUN_NAME}-${NOW}.log"
GPU_LOG="${RUN_ROOT}/logs/${RUN_NAME}-${NOW}-gpu.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,nounits \
  --loop=5 >"${GPU_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
trap 'kill -TERM "${GPU_MONITOR_PID}" 2>/dev/null || true; wait "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT INT TERM

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -m relax.entrypoints.train \
  --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_PATH}" \
  "${RESUME_ARGS[@]}" \
  --ref-load "${MODEL_PATH}" \
  --megatron-to-hf-mode bridge \
  --warm-hf-checkpoint-page-cache \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --max-actor-ckpt-to-keep 4 \
  --prompt-data "${TRAIN_DATA}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --custom-generate-function-path examples.mem_agent.rollout.generate \
  --custom-rm-path examples.mem_agent.reward.reward_func \
  --custom-convert-samples-to-train-data-path examples.mem_agent.convert.convert_samples \
  --custom-config-path "${SCRIPT_DIR}/config-pilot-qwen3-0.6b.yaml" \
  --reward-key score \
  --reward-num-workers 2 \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --rollout-max-response-len 128 \
  --rollout-max-context-len 1536 \
  --rollout-temperature 1.0 \
  --rollout-top-p 1.0 \
  --rollout-seed 42 \
  --rollout-shuffle \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --balance-data \
  --log-passrate \
  --use-rollout-logprobs \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef 0.001 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.3 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 1536 \
  --log-probs-max-tokens-per-gpu 1536 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.35 \
  --seed 1234 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --skip-eval-before-train \
  --max-staleness 0 \
  --num-data-storage-units 1 \
  --colocate \
  --use-health-check \
  --use-metrics-service \
  --tb-project-name Relax/task36-mem-agent-0.6b \
  --tb-experiment-name "${RUN_NAME}-${NOW}" \
  2>&1 | tee "${LOG_FILE}"

ray job list >"${RUN_ROOT}/logs/${RUN_NAME}-${NOW}-ray-job-list.txt" 2>&1

# Keep the durable text log, exact CSV points and a dependency-free SVG curve
# together. Missing/duplicated/conflicting rollout ids make this step fail.
python3 "${SCRIPT_DIR}/summarize_reward.py" \
  --log-file "${LOG_FILE}" \
  --rollout-result-dir "${SAVE_DIR}/rollout_result/train" \
  --output "${RUN_ROOT}/training-reward.summary.json" \
  --csv-output "${RUN_ROOT}/training-reward.csv" \
  --svg-output "${RUN_ROOT}/training-reward.svg" \
  --expected-steps "$((NUM_ROLLOUT - START_ROLLOUT_ID))" \
  --expected-start "${START_ROLLOUT_ID}" \
  --window-size 5
