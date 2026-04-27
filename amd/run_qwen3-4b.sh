#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# Edit this block directly when changing the smoke configuration.
export MODEL_DIR="${SCRIPT_DIR}/assets/exps"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NUM_GPUS="4"
export MASTER_ADDR="10.235.26.199"
export RAY_PORT="6380"
export RAY_DASHBOARD_PORT="8266"
export RAY_MIN_WORKER_PORT="30000"
export RAY_MAX_WORKER_PORT="65000"
export RAY_NODE_MANAGER_PORT="6381"
export RAY_OBJECT_MANAGER_PORT="6382"
export RAY_RUNTIME_ENV_AGENT_PORT="6383"
export RAY_DASHBOARD_AGENT_LISTEN_PORT="6384"
export RAY_DASHBOARD_AGENT_GRPC_PORT="6385"
export RAY_TMPDIR="/tmp/ray-qwen3-4b"
export RELAX_SERVE_PORT="18081"
export MEGATRON="/root/Megatron-LM/"
export RELAX="${REPO_ROOT}"
export RUN_ID="qwen3-4b-dapo-math-te-debug-$(date +%Y%m%d-%H%M%S)"

# Runtime diagnostics for the current ROCm TransformerEngine backend issue.
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export NVTE_DEBUG="1"
export NVTE_DEBUG_LEVEL="2"
export RAY_DEDUP_LOGS="0"
export TORCHDYNAMO_DISABLE="1"
export PYTHONUNBUFFERED="1"

export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
export PYTHONPATH="${RELAX}:${MEGATRON}:${RELAX}:${PYTHONPATH:-}"

cleanup_stale_processes() {
    echo "=== Cleaning stale Relax/Ray/SGLang processes ==="
    timeout 30 ray stop --force 2>/dev/null || true

    pkill -9 -f "relax.entrypoints.train" 2>/dev/null || true
    pkill -9 -f "sglang" 2>/dev/null || true
    pkill -9 -f "MegatronTrainRayActor" 2>/dev/null || true
    pkill -9 -f "SGLang" 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    pkill -9 -f "default_worker.py" 2>/dev/null || true
    pkill -9 -f "runtime_env_agent" 2>/dev/null || true
    pkill -9 -f "dashboard_agent" 2>/dev/null || true
    pkill -9 -f "dashboard.py" 2>/dev/null || true
    pkill -9 -f "log_monitor.py" 2>/dev/null || true
    pkill -9 -f "monitor.py" 2>/dev/null || true

    sleep 3

    if [[ "${RAY_TMPDIR}" == /tmp/ray-qwen3-4b* ]]; then
        rm -rf "${RAY_TMPDIR}"
    fi
    mkdir -p "${RAY_TMPDIR}"
}

start_ray_head() {
    echo "=== Starting Ray head ${RAY_ADDRESS} with ${NUM_GPUS} GPUs ==="
    ray start --head \
        --node-ip-address="${MASTER_ADDR}" \
        --port="${RAY_PORT}" \
        --num-gpus="${NUM_GPUS}" \
        --temp-dir="${RAY_TMPDIR}" \
        --disable-usage-stats \
        --include-dashboard=true \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
        --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
        --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
        --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
        --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
        --min-worker-port="${RAY_MIN_WORKER_PORT}" \
        --max-worker-port="${RAY_MAX_WORKER_PORT}"

    for _ in $(seq 1 30); do
        if ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
            ray status --address="${RAY_ADDRESS}"
            return 0
        fi
        sleep 1
    done

    echo "Ray did not become ready at ${RAY_ADDRESS}" >&2
    return 1
}

cleanup_stale_processes
start_ray_head

echo "=== Launching Qwen3-4B DAPO-Math direct runner ==="
cd "${REPO_ROOT}"
exec bash "${SCRIPT_DIR}/run-qwen3-4b-dapo-math-direct.sh"
