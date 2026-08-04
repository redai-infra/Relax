#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a converted Qwen3-0.6B checkpoint.}"
TOKENIZER_PATH="${TOKENIZER_PATH:?Set TOKENIZER_PATH to the frozen base tokenizer.}"
EVAL_DATA="${EVAL_DATA:?Set EVAL_DATA to the frozen pilot-eval.jsonl.}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to the checkpoint evaluation output directory.}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to an immutable checkpoint/run label.}"
GPU_ID="${GPU_ID:-0}"
SERVE_PORT="${SERVE_PORT:-30000}"
SAMPLES_PER_ITEM="${SAMPLES_PER_ITEM:-8}"
CONCURRENCY="${CONCURRENCY:-2}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"

mkdir -p "${RESULTS_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"

SERVER_LOG="${RESULTS_DIR}/${RUN_NAME}.server.log"
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
trap 'kill -TERM "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT INT TERM

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
  --data-file "${EVAL_DATA}" \
  --model "${MODEL_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --output-dir "${RESULTS_DIR}" \
  --run-name "${RUN_NAME}" \
  --mode recurrent \
  --base-url "http://127.0.0.1:${SERVE_PORT}/v1" \
  --api-key EMPTY \
  --samples-per-item "${SAMPLES_PER_ITEM}" \
  --seed 4242 \
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

if [[ -n "${BASELINE_SUMMARY}" ]]; then
  [[ -f "${BASELINE_SUMMARY}" ]] || { echo "Missing baseline summary: ${BASELINE_SUMMARY}" >&2; exit 1; }
  python3 "${SCRIPT_DIR}/compare_results.py" \
    --baseline-pair pilot-boxed-em boxed_em_pct "${BASELINE_SUMMARY}" "${RESULTS_DIR}/${RUN_NAME}.summary.json" \
    --baseline-pair pilot-pass-at-n pass_at_n_pct "${BASELINE_SUMMARY}" "${RESULTS_DIR}/${RUN_NAME}.summary.json" \
    --output "${RESULTS_DIR}/${RUN_NAME}.vs-baseline.json"
fi
