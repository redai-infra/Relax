#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Entrypoint for SPMD multi-node tasks.
# This script handles process cleanup, environment setup, multi-node Ray cluster
# formation (head + worker nodes), and then delegates to the actual training script
# via ray job submit.
#
# Usage:
#   bash scripts/entrypoint/spmd-multinode.sh <run-script> [extra-args...]
#
# Example:
#   bash scripts/entrypoint/spmd-multinode.sh scripts/training/text/run-qwen3-4B-16xgpu.sh
#
# Environment variables (required for multi-node):
#   MASTER_ADDR   - Hostname/IP of the head node (compared with POD_NAME to decide role)
#   POD_NAME      - Current pod's hostname
#   HOST_IP       - Current node's IP address for Ray binding
#   WORLD_SIZE    - Number of nodes (default: 2)
#
# Environment variables (optional):
#   NUM_GPUS      - Number of GPUs per node (default: 8)
#   MEGATRON      - Path to Megatron-LM (default: /root/Megatron-LM/)
#   RELAX         - Path to Relax project (default: ../../)

set -eo pipefail

# ── argument parsing ────────────────────────────────────────────────────────
RUN_SCRIPT="${1:-}"
if [ -z "$RUN_SCRIPT" ]; then
    echo "Usage: $0 <run-script> [extra-args...]" >&2
    exit 1
fi
shift  # remaining args are extra overrides

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
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${DIR}/device_env.sh"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export RELAX=${RELAX:-${DIR}/../../}
export PYTHONPATH=${RELAX}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${DIR}/../models"

# ── fast interconnect detection ────────────────────────────────────────────
export HAS_NVLINK="$(relax_detect_fast_interconnect)"
if [ -n "$NCCL_NVLS_ENABLE" ] && [ "$NCCL_NVLS_ENABLE" -eq 0 ]; then
    export HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK"

# ── multi-node parameters ──────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${WORLD_SIZE:-2}"

# ── head node vs worker node ───────────────────────────────────────────────
if [ "$MASTER_ADDR" = "$POD_NAME" ]; then
    # ── HEAD NODE ───────────────────────────────────────────────────────────
    echo "=== Head node: starting Ray cluster ==="
    ray start --head \
        --node-ip-address "${HOST_IP}" \
        --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

    sleep 5

    # Wait for all worker nodes to join
    while true; do
        ray_status_output=$(ray status)
        gpu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*GPU)' | head -n 1)
        echo "Current GPU count: $gpu_count"
        gpu_count_int=$(echo "$gpu_count" | awk '{print int($1)}')
        device_count=$((gpu_count_int / ${NUM_GPUS}))

        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster is ready with $device_count devices (from $gpu_count GPU resources)."
            ray status
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done

    # Delegate to the training script
    echo "=== Launching training script: $RUN_SCRIPT ==="
    export RELAX_ENTRYPOINT_MODE="spmd-multinode"

    # Runtime env for multi-node (includes MASTER_ADDR)
    export RUNTIME_ENV_JSON="{
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"ROCR_VISIBLE_DEVICES\": \"${ROCR_VISIBLE_DEVICES:-}\",
   \"HIP_VISIBLE_DEVICES\": \"${HIP_VISIBLE_DEVICES:-}\",
   \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES:-}\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
   \"MASTER_ADDR\": \"${HOST_IP}\"
}
}"
    exec bash "$RUN_SCRIPT" "$@"
else
    # ── WORKER NODE ─────────────────────────────────────────────────────────
    echo "=== Worker node: joining Ray cluster at ${MASTER_ADDR}:6379 ==="
    while true; do
        ray start \
            --address="${MASTER_ADDR}:6379" \
            --num-gpus "${NUM_GPUS}" \
            --node-ip-address "${HOST_IP}" \
            --disable-usage-stats \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=8265

        sleep 5
        ray status
        if [ $? -eq 0 ]; then
            echo "Successfully connected to the Ray cluster!"
            break
        else
            echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
        fi
    done

    # Worker nodes block indefinitely (training runs on head node)
    echo "=== Worker node ready, waiting for training to complete ==="
    sleep inf
fi
