#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the frozen Qwen3-0.6B BF16 checkpoint.}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the Task 36 pilot data directory.}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to the Task 36 baseline output directory.}"
GPU_ID="${GPU_ID:-0}"
SERVE_PORT="${SERVE_PORT:-30000}"
SAMPLES_PER_ITEM="${SAMPLES_PER_ITEM:-8}"
CONCURRENCY="${CONCURRENCY:-2}"

[[ -f "${DATA_DIR}/pilot-candidates.jsonl" ]] || {
  echo "Missing pilot candidates; run prepare-pilot-candidates.sh before starting a GPU." >&2
  exit 1
}
mkdir -p "${RESULTS_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"
# The evaluator is executed by file path, so Python otherwise places only
# examples/mem_agent on sys.path and cannot import examples.mem_agent.*.
export PYTHONPATH="${RELAX_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SERVER_LOG="${RESULTS_DIR}/qwen3-0.6b-baseline-server.log"
GPU_LOG="${RESULTS_DIR}/qwen3-0.6b-baseline-gpu.csv"
# Keep low-frequency GPU telemetry beside the raw model outputs so startup,
# OOM and idle/hang diagnoses do not depend on an external dashboard.
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
  --data-file "${DATA_DIR}/pilot-candidates.jsonl" \
  --model "${MODEL_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --output-dir "${RESULTS_DIR}" \
  --run-name qwen3-0.6b-untrained-pass8 \
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
  --candidates "${DATA_DIR}/pilot-candidates.jsonl" \
  --baseline-records "${RESULTS_DIR}/qwen3-0.6b-untrained-pass8.jsonl" \
  --train-output "${DATA_DIR}/pilot-train.jsonl" \
  --eval-output "${DATA_DIR}/pilot-eval.jsonl" \
  --manifest "${DATA_DIR}/pilot-selection.manifest.json" \
  --samples-per-item "${SAMPLES_PER_ITEM}" \
  --train-count 8 \
  --eval-count 4 \
  --seed 42

# Re-evaluate the disjoint diagnostic split while the same untrained server is
# still resident. This is the baseline compared with trained checkpoints; the
# larger candidate run exists only for Pass@N screening.
python3 "${SCRIPT_DIR}/eval_ruler_hqa.py" \
  --data-file "${DATA_DIR}/pilot-eval.jsonl" \
  --model "${MODEL_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --output-dir "${RESULTS_DIR}" \
  --run-name qwen3-0.6b-untrained-heldout-pass8 \
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

echo "Baseline and non-degenerate pilot split completed under ${RESULTS_DIR} and ${DATA_DIR}."
