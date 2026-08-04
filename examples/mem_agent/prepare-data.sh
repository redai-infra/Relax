#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the MemAgent data output directory.}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${DATA_DIR}/hf-cache}"
MANIFEST="${DATA_DIR}/artifact_manifest.json"
REPO_ID="BytedTsinghua-SIA/hotpotqa"
REVISION="27275ff4fee67ac0acb6478e405e7ac07efbdc1a"

mkdir -p "${DATA_DIR}"

convert_file() {
  local source_file="$1"
  local output_file="$2"
  python3 "${SCRIPT_DIR}/prepare_data.py" \
    --hf-file "${source_file}" \
    --output "${DATA_DIR}/${output_file}" \
    --repo-id "${REPO_ID}" \
    --revision "${REVISION}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --manifest "${MANIFEST}"
}

convert_file hotpotqa_train_32k.parquet train.jsonl
convert_file hotpotqa_dev.parquet dev.jsonl
for length in 50 200 800; do
  convert_file "eval_${length}.json" "eval_${length}.jsonl"
done

echo "Prepared frozen MemAgent data under ${DATA_DIR}"
