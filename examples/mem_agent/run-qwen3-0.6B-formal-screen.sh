#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Batch-screen only the pre-frozen training candidate IDs. Diagnostic and
# held-out files never enter this command, preventing baseline-outcome leakage.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the frozen Qwen3-0.6B checkpoint.}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the frozen formal pilot directory.}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to the formal screening output directory.}"
GPU_ID="${GPU_ID:-0}"
SERVE_PORT="${SERVE_PORT:-30000}"
SAMPLES_PER_ITEM="${SAMPLES_PER_ITEM:-8}"
TRAIN_COUNT="${TRAIN_COUNT:-1000}"
CONCURRENCY="${CONCURRENCY:-32}"
TRAIN_CANDIDATES="${DATA_DIR}/formal-train-candidates.jsonl"
STATIC_MANIFEST="${DATA_DIR}/formal-static-splits.manifest.json"

[[ -f "${TRAIN_CANDIDATES}" ]] || { echo "Missing ${TRAIN_CANDIDATES}" >&2; exit 1; }
[[ -f "${STATIC_MANIFEST}" ]] || { echo "Missing ${STATIC_MANIFEST}" >&2; exit 1; }
mkdir -p "${RESULTS_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"
export PYTHONPATH="${RELAX_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SERVER_LOG="${RESULTS_DIR}/qwen3-0.6b-formal-screen-server.log"
GPU_LOG="${RESULTS_DIR}/qwen3-0.6b-formal-screen-gpu.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,nounits \
  --loop=5 >"${GPU_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${SERVE_PORT}" \
  --api-key EMPTY \
  --tp-size 1 \
  --context-length 1536 \
  --mem-fraction-static 0.55 \
  --trust-remote-code \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill -TERM "${SERVER_PID}" "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "SGLang exited early; see ${SERVER_LOG}" >&2
    exit 1
  fi
  if curl --noproxy '*' -fsS "http://127.0.0.1:${SERVE_PORT}/health" >/dev/null; then
    break
  fi
  sleep 5
done
curl --noproxy '*' -fsS "http://127.0.0.1:${SERVE_PORT}/health" >/dev/null

python3 "${SCRIPT_DIR}/eval_ruler_hqa.py" \
  --data-file "${TRAIN_CANDIDATES}" \
  --model "${MODEL_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --output-dir "${RESULTS_DIR}" \
  --run-name qwen3-0.6b-formal-train-candidates-pass8 \
  --mode recurrent \
  --base-url "http://127.0.0.1:${SERVE_PORT}/v1" \
  --api-key EMPTY \
  --samples-per-item "${SAMPLES_PER_ITEM}" \
  --seed 42 \
  --disable-thinking \
  --temperature 1.0 \
  --top-p 1.0 \
  --chunk-tokens 512 \
  --max-memory-tokens 128 \
  --max-final-tokens 64 \
  --max-chunks 4 \
  --max-input-tokens 1472 \
  --server-max-model-len 1536 \
  --concurrency "${CONCURRENCY}"

python3 "${SCRIPT_DIR}/prepare_pilot_data.py" select \
  --candidates "${TRAIN_CANDIDATES}" \
  --baseline-records "${RESULTS_DIR}/qwen3-0.6b-formal-train-candidates-pass8.jsonl" \
  --train-output "${DATA_DIR}/formal-train.jsonl" \
  --eval-output "${DATA_DIR}/formal-screen-unused-eval.jsonl" \
  --manifest "${DATA_DIR}/formal-selection.manifest.json" \
  --samples-per-item "${SAMPLES_PER_ITEM}" \
  --train-count "${TRAIN_COUNT}" \
  --eval-count 0 \
  --seed 42

[[ "$(wc -l < "${DATA_DIR}/formal-train.jsonl")" -eq "${TRAIN_COUNT}" ]] || {
  echo "Formal training row count did not match ${TRAIN_COUNT}." >&2
  exit 1
}
echo "Selected ${TRAIN_COUNT} independent training questions; diagnostic/held-out IDs remained outcome-independent."
