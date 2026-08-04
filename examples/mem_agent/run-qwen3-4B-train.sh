#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
  source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the frozen Qwen3-4B snapshot directory.}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the prepared MemAgent data directory.}"
SAVE_DIR="${SAVE_DIR:?Set SAVE_DIR to the checkpoint output directory.}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.jsonl}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
RUN_NAME="${RUN_NAME:-mem-agent-qwen3-4b}"

[[ -f "${TRAIN_DATA}" ]] || { echo "Missing training data: ${TRAIN_DATA}" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing model config: ${MODEL_PATH}/config.json" >&2; exit 1; }
mkdir -p "${SAVE_DIR}" "${RELAX_ROOT}/logs"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --ref-load "${MODEL_PATH}"
  --megatron-to-hf-mode bridge
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
  --max-actor-ckpt-to-keep 3
)

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --custom-generate-function-path examples.mem_agent.rollout.generate
  --custom-rm-path examples.mem_agent.reward.reward_func
  --custom-convert-samples-to-train-data-path examples.mem_agent.convert.convert_samples
  --custom-config-path "${SCRIPT_DIR}/config.yaml"
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len 1024
  # Each request is an independent <=2K chunk + <=1K memory turn. An 8K
  # engine limit covers the real request envelope while retaining VIME's 9K
  # per-GPU packing budget and sample-mean loss semantics.
  --rollout-max-context-len 8192
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-seed 42
  --rollout-shuffle
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.001
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.3
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 9216
  --log-probs-max-tokens-per-gpu 32768
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
  --seed 1234
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --skip-eval-before-train
  --max-staleness 0
  --num-data-storage-units 1
  --colocate
  --use-health-check
)

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
  ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -m relax.entrypoints.train \
  --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee "${RELAX_ROOT}/logs/${RUN_NAME}.log"
