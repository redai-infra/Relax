#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DATA_DIR="${DATA_DIR:?Set DATA_DIR.}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH.}"
SAVE_DIR="${SAVE_DIR:?Set SAVE_DIR.}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR.}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
if ((NUM_ROLLOUT <= 0)); then
  echo "NUM_ROLLOUT must be positive." >&2
  exit 1
fi

export DATA_DIR MODEL_PATH SAVE_DIR RESULTS_DIR NUM_ROLLOUT

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  bash "${SCRIPT_DIR}/prepare-data.sh"
fi
bash "${SCRIPT_DIR}/run-qwen3-4B-train.sh"

export CHECKPOINT_DIR="${SAVE_DIR}"
# ReLax numbers checkpoints from zero, so a two-step pipeline produces
# iter_0000001 while the frozen 100-step recipe produces iter_0000099.
printf -v LAST_ROLLOUT_ID "%07d" "$((NUM_ROLLOUT - 1))"
export CHECKPOINT_TAG="${CHECKPOINT_TAG:-iter_${LAST_ROLLOUT_ID}}"
export HF_OUTPUT_DIR="${HF_OUTPUT_DIR:-${SAVE_DIR}-HF/${CHECKPOINT_TAG}}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
bash "${SCRIPT_DIR}/convert-to-hf.sh"

export MODEL_PATH="${HF_OUTPUT_DIR}"
bash "${SCRIPT_DIR}/run-eval.sh"
