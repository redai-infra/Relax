#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Three-stage orchestrator for scripts/debug/compare_sglang_megatron_dotsocr_packed.py.
# Stage 1 (single proc):  HF + SGLang dumps for sample A (mm) and B (text-only).
# Stage 2 (torchrun):     distributed Megatron packed forward, dumps per sample.
# Stage 3 (single proc):  pairwise comparison.
#
# Each stage runs in its own python process — required because SGLang's
# subprocess model conflicts with torchrun's MASTER_ADDR/RANK env vars, and
# Megatron's NCCL world conflicts with SGLang's per-engine process group.
#
# Usage:
#   bash scripts/debug/run-compare-dotsocr-packed.sh [extra args...]
#   STAGES=front bash ... -- # run only stage 1
#   STAGES="megatron compare" bash ... -- # skip the front stage
#
# Environment overrides:
#   MODEL_DIR  - base directory holding rednote-hilab/dots.mocr (default: $EXP_DIR)
#   EXP_DIR    - exps root (default: ./exps)
#   HF_CKPT    - explicit checkpoint path (overrides MODEL_DIR)
#   DUMP_DIR   - per-side logprob dumps (default: /tmp/relax_dotsocr_debug_packed)
#   DTYPE      - bf16 (default) | fp16 | fp32
#   TP / PP / CP - parallelism (default 2/2/2; product must equal NPROC)
#   NPROC      - GPUs for stage 2 (default 8)
#   MASTER_PORT - rendezvous port for torchrun (default 29503)
#   STAGES     - space-separated subset of "front megatron compare" (default: all three)
#   SKIP_HF / SKIP_SGLANG - "1" to pass --skip-hf / --skip-sglang in stage 1.

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/compare_sglang_megatron_dotsocr_packed.py"

EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/rednote-hilab/dots.mocr}"
DUMP_DIR="${DUMP_DIR:-/tmp/relax_dotsocr_debug_packed}"
DTYPE="${DTYPE:-bf16}"

TP="${TP:-2}"
PP="${PP:-2}"
CP="${CP:-2}"
MAX_GPUS="${MAX_GPUS:-8}"
STAGES="${STAGES:-front megatron compare}"

# NPROC is always TP*PP*CP (no DP in this debug script). Capping at physical
# GPU count prevents torchrun from over-subscribing the box.
NPROC=$((TP * PP * CP))
if [ "${NPROC}" -gt "${MAX_GPUS}" ]; then
    echo "TP*PP*CP=${NPROC} exceeds MAX_GPUS=${MAX_GPUS}" >&2
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29503}"

# SGLang refuses to launch under torch 2.9.1 + cudnn < 9.15 unless this is set.
# Our debug script doesn't exercise Conv3d so the warned-about bug doesn't apply.
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

# SGLang's deep_gemm wrapper asserts CUDA_HOME at import time. When the
# system has no /usr/local/cuda but torch ships its own CUDA libs via the
# nvidia-cuda-runtime wheel, point CUDA_HOME there. Skip if user already set it.
if [ -z "${CUDA_HOME:-}" ]; then
    _NV_CUDA_RT="$(python3 -c 'import nvidia.cuda_runtime, os; print(os.path.dirname(nvidia.cuda_runtime.__file__))' 2>/dev/null || true)"
    if [ -n "${_NV_CUDA_RT}" ] && [ -d "${_NV_CUDA_RT}" ]; then
        export CUDA_HOME="${_NV_CUDA_RT}"
        echo "[run-compare] CUDA_HOME=${CUDA_HOME} (auto-detected from nvidia.cuda_runtime wheel)"
    fi
fi

COMMON_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --dtype "${DTYPE}"
    --dump-dir "${DUMP_DIR}"
    --tp-size "${TP}"
    --pp-size "${PP}"
    --cp-size "${CP}"
)
FRONT_ARGS=()
[ "${SKIP_HF:-0}" = "1" ]     && FRONT_ARGS+=(--skip-hf)
[ "${SKIP_SGLANG:-0}" = "1" ] && FRONT_ARGS+=(--skip-sglang)

mkdir -p "${DUMP_DIR}"

run_stage() {
    local name="$1"; shift
    echo
    echo "========================================"
    echo "[$(date +%H:%M:%S)] STAGE: ${name}"
    echo "========================================"
    "$@"
}

for stage in ${STAGES}; do
    case "${stage}" in
        front|hf|sglang)
            run_stage "${stage}" \
                python3 "${PY_SCRIPT}" --side "${stage}" "${COMMON_ARGS[@]}" "${FRONT_ARGS[@]}" "$@"
            ;;
        megatron)
            run_stage "megatron(torchrun nproc=${NPROC} tp=${TP} pp=${PP} cp=${CP})" \
                torchrun --nproc-per-node="${NPROC}" --nnodes=1 \
                    --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
                    "${PY_SCRIPT}" --side megatron "${COMMON_ARGS[@]}" "$@"
            ;;
        compare)
            run_stage "compare" \
                python3 "${PY_SCRIPT}" --side compare "${COMMON_ARGS[@]}" "$@"
            ;;
        *)
            echo "unknown stage '${stage}' (allowed: front hf sglang megatron compare)" >&2
            exit 1
            ;;
    esac
done
