#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Default environment configuration for local single-node development.
# This script handles process cleanup, environment setup, and Ray cluster startup.
# It is designed to be *sourced* by run-*.sh scripts when no external entrypoint
# (spmd-multinode.sh or ray-job.sh) has been used.
#
# When an existing Ray cluster is detected (RAY_ADDRESS set and `ray status` OK),
# this script delegates to `ray-job.sh` (source mode) instead of starting a new
# local Ray head node.
#
# Usage (from a run script):
#   source scripts/entrypoint/local.sh
#
# Environment variables:
#   NUM_GPUS               - Number of GPUs to use (optional, auto-detect from visible device envs)
#   CUDA_VISIBLE_DEVICES   - Comma-separated GPU IDs (e.g., "0,1,2,3" → 4 GPUs)
#   ROCR_VISIBLE_DEVICES   - ROCm visible GPU IDs
#   HIP_VISIBLE_DEVICES    - HIP visible GPU IDs
#   MASTER_ADDR            - Head node IP address (default: 127.0.0.1)
#   MEGATRON               - Path to Megatron-LM (default: /root/Megatron-LM/)
#   RELAX                  - Path to Relax project (default: ../../)

# Guard: skip if already sourced by another entrypoint
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

_LOCAL_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${_LOCAL_SH_DIR}/device_env.sh"

# ── delegate to ray-job.sh when inside an existing Ray cluster ─────────────
# When RAY_ADDRESS is set AND `ray status` succeeds, we're already part of an
# externally-managed Ray cluster. Skip local Ray startup / process cleanup and
# fall through to ray-job.sh (source mode) for env setup.
if [ -n "${RAY_ADDRESS:-}" ] && timeout 5 ray status >/dev/null 2>&1; then
    echo "=== Detected existing Ray cluster (RAY_ADDRESS=$RAY_ADDRESS); delegating to ray-job.sh ==="
    # shellcheck source=./ray-job.sh
    source "${_LOCAL_SH_DIR}/ray-job.sh"
    return 0 2>/dev/null || exit 0
fi

set -eo pipefail

# ── process cleanup ─────────────────────────────────────────────────────────
echo "=== Cleaning up stale processes ==="
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

set -x

# ── environment setup ───────────────────────────────────────────────────────
unset MASTER_ADDR 2>/dev/null || true
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export RELAX=${RELAX:-${_LOCAL_SH_DIR}/../../}
export PYTHONPATH=${RELAX}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${_LOCAL_SH_DIR}/../models"

# ── fast interconnect detection ────────────────────────────────────────────
export HAS_NVLINK="$(relax_detect_fast_interconnect)"
if [ -n "$NCCL_NVLS_ENABLE" ] && [ "$NCCL_NVLS_ENABLE" -eq 0 ]; then
    export HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK"

# ── GPU count detection ───────────────────────────────────────────────────────
# Priority: NUM_GPUS env > visible device envs > default 8
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS="$(relax_gpu_count_from_env_or_default 8)"
fi

# Ray temp dir: prefer caller-provided RAY_TMPDIR so repeated container smokes
# do not collide on persisted session metadata under the default /tmp/ray path.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
mkdir -p "${RAY_TMPDIR}"

# ── Ray cluster startup (single node) ──────────────────────────────────────
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
echo "Starting Ray head node: MASTER_ADDR=$MASTER_ADDR, NUM_GPUS=$NUM_GPUS"

ray start --head \
    --temp-dir "${RAY_TMPDIR}" \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

# ── set entrypoint mode ────────────────────────────────────────────────────
export RELAX_ENTRYPOINT_MODE="local"

_VISIBLE_DEVICE_ENV_JSON=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _VISIBLE_DEVICE_ENV_JSON="
   \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES}\",
"
elif [ -n "${ROCR_VISIBLE_DEVICES:-}" ] || [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    _VISIBLE_DEVICE_ENV_JSON="
   \"CUDA_VISIBLE_DEVICES\": \"${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}\",
"
fi

# Runtime env for single-node (empty, env inherited from Ray cluster)
export RUNTIME_ENV_JSON="{
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RELAX_SERVE_PORT\": \"${RELAX_SERVE_PORT:-8000}\",
${_VISIBLE_DEVICE_ENV_JSON}   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
}
}"

echo "=== Local environment ready ==="
