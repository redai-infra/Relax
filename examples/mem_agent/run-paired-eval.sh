#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_MODEL_PATH="${VIME_MODEL_PATH:?Set VIME_MODEL_PATH to the official or equivalently trained VIME checkpoint.}"
RELAX_MODEL_PATH="${RELAX_MODEL_PATH:?Set RELAX_MODEL_PATH to the converted ReLax checkpoint.}"
TOKENIZER_PATH="${TOKENIZER_PATH:?Set TOKENIZER_PATH to the frozen Qwen3-4B tokenizer snapshot.}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the prepared frozen evaluation data.}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR for raw and summary results.}"
LENGTHS="${LENGTHS:-50 200 800}"

mkdir -p "${RESULTS_DIR}/vime" "${RESULTS_DIR}/relax"

# Start and stop each checkpoint independently while preserving every other
# serving and evaluation variable. run-eval.sh writes both raw JSONL records
# and summary JSON, so the final comparison remains sample-auditable.
MODEL_PATH="${VIME_MODEL_PATH}" RUN_NAME=vime RESULTS_DIR="${RESULTS_DIR}/vime" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" DATA_DIR="${DATA_DIR}" LENGTHS="${LENGTHS}" \
  bash "${SCRIPT_DIR}/run-eval.sh"
MODEL_PATH="${RELAX_MODEL_PATH}" RUN_NAME=relax RESULTS_DIR="${RESULTS_DIR}/relax" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" DATA_DIR="${DATA_DIR}" LENGTHS="${LENGTHS}" \
  bash "${SCRIPT_DIR}/run-eval.sh"

COMPARE_ARGS=()
for length in ${LENGTHS}; do
  COMPARE_ARGS+=(
    --pair "ruler-hqa-${length}"
    "${RESULTS_DIR}/vime/vime-ruler-hqa-${length}.summary.json"
    "${RESULTS_DIR}/relax/relax-ruler-hqa-${length}.summary.json"
  )
done
python3 "${SCRIPT_DIR}/compare_results.py" \
  "${COMPARE_ARGS[@]}" \
  --metric sub_em_pct \
  --tolerance-pp 3.0 \
  --output "${RESULTS_DIR}/vime-vs-relax.json"
