#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the Task 36 pilot data directory.}"
TOKENIZER_PATH="${TOKENIZER_PATH:?Set TOKENIZER_PATH to the frozen Qwen3-0.6B tokenizer.}"
SOURCE_DATA="${SOURCE_DATA:-${DATA_DIR}/train.jsonl}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-24}"

mkdir -p "${DATA_DIR}"
if [[ ! -f "${SOURCE_DATA}" ]]; then
  # Download only the training parquet before GPU allocation. The long RULER
  # files are not needed for the short 0.6B pilot candidate screen.
  python3 "${SCRIPT_DIR}/prepare_data.py" \
    --hf-file hotpotqa_train_32k.parquet \
    --output "${SOURCE_DATA}" \
    --repo-id BytedTsinghua-SIA/hotpotqa \
    --revision 27275ff4fee67ac0acb6478e405e7ac07efbdc1a \
    --cache-dir "${DATA_DIR}/hf-cache" \
    --manifest "${DATA_DIR}/source-manifest.json"
fi

python3 "${SCRIPT_DIR}/prepare_pilot_data.py" candidates \
  --input "${SOURCE_DATA}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --output "${DATA_DIR}/pilot-candidates.jsonl" \
  --manifest "${DATA_DIR}/pilot-candidates.manifest.json" \
  --chunk-tokens 512 \
  --min-chunks 2 \
  --max-chunks 4 \
  --candidate-count "${CANDIDATE_COUNT}" \
  --seed 42

echo "Prepared ${DATA_DIR}/pilot-candidates.jsonl before GPU allocation."
