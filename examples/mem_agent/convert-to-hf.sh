#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the frozen base model directory.}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the ReLax checkpoint root.}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-iter_0000099}"
HF_OUTPUT_DIR="${HF_OUTPUT_DIR:-${CHECKPOINT_DIR}-HF/${CHECKPOINT_TAG}}"

python3 "${RELAX_ROOT}/scripts/tools/convert_torch_dist_to_hf_bridge.py" \
  --input-dir "${CHECKPOINT_DIR}/${CHECKPOINT_TAG}" \
  --output-dir "${HF_OUTPUT_DIR}" \
  --origin-hf-dir "${MODEL_PATH}"

echo "Converted checkpoint: ${HF_OUTPUT_DIR}"
