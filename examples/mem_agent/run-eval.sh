#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the Hugging Face checkpoint to evaluate.}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the prepared MemAgent data directory.}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to the evaluation output directory.}"
RUN_NAME="${RUN_NAME:-$(basename "${MODEL_PATH}")}"
MODE="${MODE:-recurrent}"
LENGTHS="${LENGTHS:-50 200 800}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8000}"
TP="${TP:-1}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"

mkdir -p "${RESULTS_DIR}"
SERVER_LOG="${RESULTS_DIR}/${RUN_NAME}.server.log"

vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size "${TP}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
  --trust-remote-code \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM server exited early; see ${SERVER_LOG}" >&2
    exit 1
  fi
  if curl -fsS "http://${SERVE_HOST}:${SERVE_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 5
done
curl -fsS "http://${SERVE_HOST}:${SERVE_PORT}/v1/models" >/dev/null

run_eval() {
  local data_file="$1"
  local suffix="$2"
  # Fixed VIME's Python evaluator has a 512 fallback, but its official
  # run-eval.sh sources _common.sh, which exports MEM_MAX_CHUNKS=64. Pin the
  # effective official-run value explicitly instead of relying on inheritance.
  python3 "${SCRIPT_DIR}/eval_ruler_hqa.py" \
    --data-file "${data_file}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --output-dir "${RESULTS_DIR}" \
    --run-name "${RUN_NAME}-${suffix}" \
    --mode "${MODE}" \
    --base-url "http://${SERVE_HOST}:${SERVE_PORT}/v1" \
    --temperature 0.7 \
    --top-p 0.95 \
    --chunk-tokens 2048 \
    --max-memory-tokens 1024 \
    --max-final-tokens 256 \
    --max-chunks 64 \
    --server-max-model-len "${MAX_MODEL_LEN}" \
    --concurrency "${CONCURRENCY}"
}

run_eval "${DATA_DIR}/dev.jsonl" hotpotqa-dev
for length in ${LENGTHS}; do
  run_eval "${DATA_DIR}/eval_${length}.jsonl" "ruler-hqa-${length}"
done
