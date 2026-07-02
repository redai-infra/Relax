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
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export RELAX=${RELAX:-${DIR}/../../}
export PYTHONPATH=${RELAX}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${DIR}/../models"

# ── NVLink detection ────────────────────────────────────────────────────────
if [ -e /dev/xpuctrl ]; then
    NVLINK_COUNT=0
else
    NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
fi
if [ "$NVLINK_COUNT" -gt 0 ]; then
    export HAS_NVLINK=1
else
    export HAS_NVLINK=0
fi
if [ -n "$NCCL_NVLS_ENABLE" ] && [ "$NCCL_NVLS_ENABLE" -eq 0 ]; then
    export HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

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
    NVSHMEM_LIB_PATH="${NVSHMEM_LIB_PATH:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib}"
    CURRENT_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${NVSHMEM_LIB_PATH}"

    # Cap OMP/MKL/OpenBLAS threads (default 24) to avoid CPU oversubscription when colocating multiple Ray actors per node.
    export RUNTIME_ENV_JSON="{
\"worker_process_setup_hook\": \"relax.utils.logging_utils.install_asyncio_noise_filter\",
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"OMP_NUM_THREADS\": \"${OMP_NUM_THREADS:-24}\",
   \"MKL_NUM_THREADS\": \"${MKL_NUM_THREADS:-24}\",
   \"OPENBLAS_NUM_THREADS\": \"${OPENBLAS_NUM_THREADS:-24}\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
   \"MASTER_ADDR\": \"${HOST_IP}\",
   \"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK\": \"${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-32}\",
   \"NVSHMEM_DISABLE_NCCL\": \"${NVSHMEM_DISABLE_NCCL:-1}\",
   \"SGLANG_HEALTH_CHECK_TIMEOUT\": \"${SGLANG_HEALTH_CHECK_TIMEOUT:-180}\",
   \"INDEXER_ROPE_NEOX_STYLE\": \"${INDEXER_ROPE_NEOX_STYLE:-0}\",
   \"NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME\": \"${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${NCCL_SOCKET_IFNAME}}\",
   \"NVTE_USE_CUTLASS_GROUPED_GEMM\": \"${NVTE_USE_CUTLASS_GROUPED_GEMM:-1}\",
   \"NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK\": \"${NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK:-1}\",
   \"LD_LIBRARY_PATH\": \"${CURRENT_LD_LIBRARY_PATH}\"

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
