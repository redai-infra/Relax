#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Default environment configuration for local single-node development.
# This script handles process cleanup, environment setup, and Ray cluster startup.
# It is designed to be *sourced* by run-*.sh scripts when no external entrypoint
# (spmd-multinode.sh or ray-job-npu.sh) has been used.
#
# When an existing Ray cluster is detected (RAY_ADDRESS set and `ray status` OK),
# this script delegates to `ray-job-npu.sh` (source mode) instead of starting a new
# local Ray head node.
#
# Usage (from a run script):
#   source scripts/entrypoint/local-npu.sh
#
# Environment variables:
#   ASCEND_RT_VISIBLE_DEVICES   - Comma-separated NPU IDs (e.g., "0,1,2,3" → 4 NPUs)
#   MASTER_ADDR                 - Head node IP address (default: 127.0.0.1)
#   MEGATRON                    - Path to Megatron-LM (default: /root/Megatron-LM/)
#   RELAX                       - Path to Relax project (default: ../../)

# Guard: skip if already sourced by another entrypoint
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

_LOCAL_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ── delegate to ray-job.sh when inside an existing Ray cluster ─────────────
# When RAY_ADDRESS is set AND `ray status` succeeds, we're already part of an
# externally-managed Ray cluster. Skip local Ray startup / process cleanup and
# fall through to ray-job.sh (source mode) for env setup.
if [ -n "${RAY_ADDRESS:-}" ] && timeout 5 ray status >/dev/null 2>&1; then
    echo "=== Detected existing Ray cluster (RAY_ADDRESS=$RAY_ADDRESS); delegating to ray-job.sh ==="
    # shellcheck source=./ray-job.sh
    source "${_LOCAL_SH_DIR}/ray-job-npu.sh"
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
export MEGATRON_BRIDGE_SRC=${MEGATRON_BRIDGE_SRC:-/root/Megatron-Bridge/src/}
export MINDSPEED=${MINDSPEED:-/root/MindSpeed/}
export RELAX=${RELAX:-${_LOCAL_SH_DIR}/../../}
export PYTHONPATH=${RELAX}:${MEGATRON_BRIDGE_SRC}:${MINDSPEED}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${_LOCAL_SH_DIR}/../models"

# ── Ray cluster startup (single node) ──────────────────────────────────────
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

# ── set entrypoint mode ────────────────────────────────────────────────────
export RELAX_ENTRYPOINT_MODE="local"

# Runtime env for single-node (empty, env inherited from Ray cluster)
export RUNTIME_ENV_JSON="{
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\"
}
}"

echo "=== Local environment ready ==="
