#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the frozen Qwen3-4B snapshot.}"
VIME_MODEL_PATH="${VIME_MODEL_PATH:?Set VIME_MODEL_PATH to the official or equivalently trained VIME checkpoint.}"
RELAX_MODEL_PATH="${RELAX_MODEL_PATH:?Set RELAX_MODEL_PATH to the converted ReLax checkpoint.}"
TOKENIZER_PATH="${TOKENIZER_PATH:?Set TOKENIZER_PATH to the frozen Qwen3-4B tokenizer snapshot.}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the prepared frozen evaluation data.}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR for raw and summary results.}"
LENGTHS="${LENGTHS:-50 200 800}"

mkdir -p "${RESULTS_DIR}/base" "${RESULTS_DIR}/vime" "${RESULTS_DIR}/relax"

# Start and stop each checkpoint independently while preserving every other
# serving and evaluation variable. Frozen base also uses recurrent mode: the
# policy weights are the only variable in the trained-vs-base acceptance.
# run-eval.sh retains raw JSONL so every aggregate remains sample-auditable.
MODEL_PATH="${BASE_MODEL_PATH}" RUN_NAME=base RESULTS_DIR="${RESULTS_DIR}/base" MODE=recurrent \
  TOKENIZER_PATH="${TOKENIZER_PATH}" DATA_DIR="${DATA_DIR}" LENGTHS="${LENGTHS}" \
  bash "${SCRIPT_DIR}/run-eval.sh"
MODEL_PATH="${VIME_MODEL_PATH}" RUN_NAME=vime RESULTS_DIR="${RESULTS_DIR}/vime" MODE=recurrent \
  TOKENIZER_PATH="${TOKENIZER_PATH}" DATA_DIR="${DATA_DIR}" LENGTHS="${LENGTHS}" \
  bash "${SCRIPT_DIR}/run-eval.sh"
MODEL_PATH="${RELAX_MODEL_PATH}" RUN_NAME=relax RESULTS_DIR="${RESULTS_DIR}/relax" MODE=recurrent \
  TOKENIZER_PATH="${TOKENIZER_PATH}" DATA_DIR="${DATA_DIR}" LENGTHS="${LENGTHS}" \
  bash "${SCRIPT_DIR}/run-eval.sh"

COMPARE_ARGS=()
for length in ${LENGTHS}; do
  COMPARE_ARGS+=(
    --pair "ruler-hqa-${length}"
    "${RESULTS_DIR}/vime/vime-ruler-hqa-${length}.summary.json"
    "${RESULTS_DIR}/relax/relax-ruler-hqa-${length}.summary.json"
    --baseline-pair "ruler-hqa-${length}" sub_em_pct
    "${RESULTS_DIR}/base/base-ruler-hqa-${length}.summary.json"
    "${RESULTS_DIR}/relax/relax-ruler-hqa-${length}.summary.json"
  )
done
COMPARE_ARGS+=(
  --baseline-pair hotpotqa-dev boxed_em_pct
  "${RESULTS_DIR}/base/base-hotpotqa-dev.summary.json"
  "${RESULTS_DIR}/relax/relax-hotpotqa-dev.summary.json"
)
python3 "${SCRIPT_DIR}/compare_results.py" \
  "${COMPARE_ARGS[@]}" \
  --metric sub_em_pct \
  --tolerance-pp 3.0 \
  --output "${RESULTS_DIR}/acceptance-comparison.json"
