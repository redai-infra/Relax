#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
NORMALIZED_DATA="${NORMALIZED_DATA:?Set NORMALIZED_DATA to the frozen normalized HotpotQA JSONL.}"
TOKENIZER_PATH="${TOKENIZER_PATH:?Set TOKENIZER_PATH to the frozen Qwen3-0.6B tokenizer.}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the formal Task 36 pilot directory.}"
TRAIN_CANDIDATE_COUNT="${TRAIN_CANDIDATE_COUNT:-4000}"
SMOKE_COUNT="${SMOKE_COUNT:-48}"
DIAGNOSTIC_COUNT="${DIAGNOSTIC_COUNT:-128}"
HELDOUT_COUNT="${HELDOUT_COUNT:-500}"

mkdir -p "${DATA_DIR}"
python3 "${SCRIPT_DIR}/prepare_pilot_data.py" freeze \
  --input "${NORMALIZED_DATA}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --train-candidates-output "${DATA_DIR}/formal-train-candidates.jsonl" \
  --smoke-output "${DATA_DIR}/formal-smoke-candidates.jsonl" \
  --diagnostic-output "${DATA_DIR}/formal-diagnostic.jsonl" \
  --heldout-output "${DATA_DIR}/formal-heldout.jsonl" \
  --manifest "${DATA_DIR}/formal-static-splits.manifest.json" \
  --chunk-tokens 512 \
  --min-chunks 2 \
  --max-chunks 4 \
  --train-candidate-count "${TRAIN_CANDIDATE_COUNT}" \
  --smoke-count "${SMOKE_COUNT}" \
  --diagnostic-count "${DIAGNOSTIC_COUNT}" \
  --heldout-count "${HELDOUT_COUNT}" \
  --seed 42

echo "Frozen mutually disjoint Task 36 smoke, train candidates, diagnostic and held-out IDs under ${DATA_DIR}."
