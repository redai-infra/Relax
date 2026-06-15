#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Environment configuration and Ray cluster startup for local / multi-node training.
# This script handles process cleanup, environment setup, and Ray cluster startup.
# It is designed to be *sourced* by run-*.sh scripts when no external entrypoint
# (spmd-multinode.sh or ray-job.sh) has been used.
#
# Multi-node support:
#   When NUM_GPUS > GPUS_PER_NODE (default 8), this script automatically starts a
#   multi-node Ray cluster by SSH'ing to worker nodes and launching Ray workers
#   inside their containers.
#
# When an existing Ray cluster is detected (RAY_ADDRESS set and `ray status` OK),
# this script delegates to `ray-job.sh` (source mode) instead of starting a new
# local Ray head node.
#
# Usage (from a run script):
#   source scripts/entrypoint/local.sh
#
# Environment variables:
#   NUM_GPUS               - Total GPUs across all nodes (e.g., 16 for 2-node × 8)
#   GPUS_PER_NODE          - GPUs per node (default: 8)
#   HEAD_IP                - Head node IP (default: auto-detect via hostname -I)
#   WORKER_NODES           - Comma-separated worker IPs (default: 10.0.0.44)
#   CONTAINER_NAME         - Docker container name on worker nodes (default: same as hostname)
#   CONDA_ENV_ACTIVATE     - Conda activation command (default: auto-detect)
#   CUDA_VISIBLE_DEVICES   - GPU device list (default: 0,1,2,3,4,5,6,7)
#   MASTER_ADDR            - Alias for HEAD_IP
#   MEGATRON               - Path to Megatron-LM (default: /workspace/Megatron-LM/)
#   RELAX                  - Path to Relax project (default: ../../)

# Guard: skip if already sourced by another entrypoint
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

_LOCAL_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ── delegate to ray-job.sh when inside an existing Ray cluster ─────────────
if [ -n "${RAY_ADDRESS:-}" ] && timeout 5 ray status >/dev/null 2>&1; then
    echo "=== Detected existing Ray cluster (RAY_ADDRESS=$RAY_ADDRESS); delegating to ray-job.sh ==="
    # shellcheck source=./ray-job.sh
    source "${_LOCAL_SH_DIR}/ray-job.sh"
    return 0 2>/dev/null || exit 0
fi

# ── Kunlun XPU / Ray compatibility env vars ──────────────────────────────────
# On Kunlun (torch_xmlir) machines:
#   - pynvml detects 0 GPUs (no real NVIDIA), but torch.cuda.is_available()=True
#   - Ray's get_accelerator_ids_for_accelerator_resource crashes with IndexError
#     if CUDA_VISIBLE_DEVICES is not set at ray start time
#   - RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 prevents Ray from overriding
#     CUDA_VISIBLE_DEVICES per-task (which would fail since pynvml sees 0 GPUs)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ── process cleanup (local node) ─────────────────────────────────────────────
echo "=== Cleaning up stale processes ==="
# NOTE: use "python.*xxx" patterns to avoid pkill matching our own bash process
pkill -9 -f "python.*sglang" 2>/dev/null || true
sleep 1
ray stop --force 2>/dev/null || true
pkill -9 -f '^ray::' 2>/dev/null || true
pkill -9 -f "python.*relax.entrypoints.train" 2>/dev/null || true
sleep 2

# ── XCCL shared memory cleanup (local node) ──────────────────────────────────
# XCCL uses shared memory segments with permission 666; stale segments from
# previous runs cause "shmget failed: errno=17 (File exists)" and can lead to
# XCCL communication hangs (kl3_all_reduce / kl3_group_send_recv timeout).
echo "=== Cleaning XCCL shared memory (local) ==="
ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do ipcrm -m "$shmid" 2>/dev/null; done || true

set -x

# ── environment setup ───────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/workspace/Megatron-LM/}
export RELAX=${RELAX:-${_LOCAL_SH_DIR}/../../../}
export PYTHONPATH=${RELAX}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX}/scripts/models}"

# ── GPU / node topology detection ─────────────────────────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS=${GPUS_PER_NODE}
fi

# Head IP: prefer HEAD_IP > MASTER_ADDR > auto-detect
if [ -n "${HEAD_IP:-}" ]; then
    export MASTER_ADDR="${HEAD_IP}"
elif [ -n "${MASTER_ADDR:-}" ]; then
    export MASTER_ADDR="${MASTER_ADDR}"
else
    # Auto-detect: first non-loopback IPv4
    MASTER_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -z "${MASTER_ADDR}" ] || [ "${MASTER_ADDR}" = "127.0.0.1" ]; then
        MASTER_ADDR="127.0.0.1"
    fi
    export MASTER_ADDR
fi

# Determine number of nodes
NUM_NODES=$(( (NUM_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))

echo "=== Cluster topology: NUM_GPUS=${NUM_GPUS}, GPUS_PER_NODE=${GPUS_PER_NODE}, NUM_NODES=${NUM_NODES}, HEAD=${MASTER_ADDR} ==="

# ── Conda activation command (for SSH remote execution) ──────────────────────
CONDA_ENV_ACTIVATE=${CONDA_ENV_ACTIVATE:-". /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda"}

# ── Container name for worker docker exec ────────────────────────────────────
CONTAINER_NAME=${CONTAINER_NAME:-"wangzhenpeng01_sglang_relax"}

# ── Multi-node: clean worker processes ────────────────────────────────────────
if [ "${NUM_NODES}" -gt 1 ]; then
    WORKER_NODES=${WORKER_NODES:-"10.10.0.52"}

    echo "=== Cleaning worker nodes: ${WORKER_NODES} ==="
    IFS=',' read -ra _WORKERS <<< "${WORKER_NODES}"
    for _worker_ip in "${_WORKERS[@]}"; do
        echo "--- Cleaning ${_worker_ip} ---"
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "root@${_worker_ip}" \
            "docker exec ${CONTAINER_NAME} bash -c '
                pkill -9 -f \"python.*sglang\" 2>/dev/null || true;
                ray stop --force 2>/dev/null || true;
                pkill -9 -f \"^ray::\" 2>/dev/null || true;
                pkill -9 -f \"python.*relax.entrypoints.train\" 2>/dev/null || true;
                echo \"=== Cleaning XCCL shared memory (worker) ===\";
                ipcs -m | awk \"\\$4 == 666 {print \\$2}\" | xargs -r -I{} ipcrm -m {} 2>/dev/null || true
            '" 2>/dev/null || echo "WARNING: Failed to clean ${_worker_ip}"
    done
    sleep 2
fi

# ── Start Ray head node ───────────────────────────────────────────────────────
echo "=== Starting Ray head node: MASTER_ADDR=${MASTER_ADDR}, GPUS_PER_NODE=${GPUS_PER_NODE} ==="

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

# ── Multi-node: start worker nodes ───────────────────────────────────────────
if [ "${NUM_NODES}" -gt 1 ]; then
    echo "=== Starting worker nodes: ${WORKER_NODES} ==="
    IFS=',' read -ra _WORKERS <<< "${WORKER_NODES}"
    for _worker_ip in "${_WORKERS[@]}"; do
        echo "--- Starting Ray worker on ${_worker_ip} ---"
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@${_worker_ip}" \
            "docker exec ${CONTAINER_NAME} bash -c '
                ${CONDA_ENV_ACTIVATE} &&
                export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 &&
                export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 &&
                ray stop --force 2>/dev/null;
                sleep 1;
                ray start \
                    --address=${MASTER_ADDR}:6379 \
                    --node-ip-address ${_worker_ip} \
                    --num-gpus ${GPUS_PER_NODE} \
                    --disable-usage-stats
            '" || { echo "ERROR: Failed to start worker on ${_worker_ip}"; exit 1; }
    done

    # ── Verify cluster has expected GPUs ──────────────────────────────────────
    echo "=== Verifying cluster (expecting ${NUM_GPUS} GPUs across ${NUM_NODES} nodes) ==="
    sleep 3
    _actual_gpus=$(ray status 2>/dev/null | grep -oP '[0-9.]+/\K[0-9.]+(?=\s+GPU)' | head -1 | cut -d. -f1)
    if [ "${_actual_gpus:-0}" -lt "${NUM_GPUS}" ]; then
        echo "WARNING: Expected ${NUM_GPUS} GPUs but found ${_actual_gpus:-0}. Retrying in 5s..."
        sleep 5
        _actual_gpus=$(ray status 2>/dev/null | grep -oP '[0-9.]+/\K[0-9.]+(?=\s+GPU)' | head -1 | cut -d. -f1)
        if [ "${_actual_gpus:-0}" -lt "${NUM_GPUS}" ]; then
            echo "ERROR: Cluster has only ${_actual_gpus:-0}/${NUM_GPUS} GPUs. Check worker connectivity."
            ray status
            exit 1
        fi
    fi
    echo "=== Cluster verified: ${_actual_gpus} GPUs ready ==="
fi

# ── set entrypoint mode ──────────────────────────────────────────────────────
export RELAX_ENTRYPOINT_MODE="local"

echo "=== Local environment ready (${NUM_NODES} node(s), ${NUM_GPUS} GPUs) ==="