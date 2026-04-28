#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -Eeuo pipefail

ulimit -n 1048576
pkill -9 python 2>/dev/null || true
pkill -9 python3 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/entrypoint/device_env.sh"

select_available_gpus() {
    local min_free_vram_gb="$1"
    local requested_gpus="${2:-}"

    python3 - "${min_free_vram_gb}" "${requested_gpus}" <<'PY'
import os
import sys

threshold_gb = float(sys.argv[1])
threshold = threshold_gb * 1024**3
requested = int(sys.argv[2]) if sys.argv[2] else None
supported_counts = sorted(
    int(item) for item in os.environ.get("QWEN35_SUPPORTED_GPU_COUNTS", "4,6,8").split(",") if item
)
if requested is not None and requested not in supported_counts:
    print(
        f"[qwen35-gpu-select] Requested {requested} GPUs, but supported profiles are {supported_counts}",
        file=sys.stderr,
    )
    sys.exit(2)

for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ.pop(key, None)

import torch

if not torch.cuda.is_available():
    print("[qwen35-gpu-select] torch.cuda is not available", file=sys.stderr)
    sys.exit(2)

selected = []
for index in range(torch.cuda.device_count()):
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    print(
        f"[qwen35-gpu-select] GPU {index}: free={free_gb:.1f}GB total={total_gb:.1f}GB",
        file=sys.stderr,
    )
    if free_bytes >= threshold:
        selected.append(str(index))

if requested is None:
    target = max((count for count in supported_counts if count <= len(selected)), default=0)
else:
    target = requested

if target <= 0:
    print(
        f"[qwen35-gpu-select] Need at least {supported_counts[0]} GPUs with >= {threshold_gb:.1f}GB free, "
        f"but only found {len(selected)}: {','.join(selected) or '<none>'}",
        file=sys.stderr,
    )
    sys.exit(2)

if len(selected) < target:
    print(
        f"[qwen35-gpu-select] Need {target} GPUs with >= {threshold_gb:.1f}GB free, "
        f"but only found {len(selected)}: {','.join(selected) or '<none>'}",
        file=sys.stderr,
    )
    sys.exit(2)

print(
    f"[qwen35-gpu-select] Selected {target} GPUs from {len(selected)} eligible GPUs "
    f"(supported={supported_counts})",
    file=sys.stderr,
)
print(",".join(selected[:target]))
PY
}

# Edit this block directly when changing the smoke configuration.
export MODEL_DIR="${SCRIPT_DIR}/assets/exps"
export HF_MODEL_PATH="${HF_MODEL_PATH:-Qwen/Qwen3.5-9B}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-${SCRIPT_DIR}/assets/hf-models}"
export HF_TRAIN_DATASET_PATH="${HF_TRAIN_DATASET_PATH:-zhuzilin/dapo-math-17k/dapo-math-17k.jsonl}"
export HF_EVAL_DATASET_PATH="${HF_EVAL_DATASET_PATH:-zhuzilin/aime-2024/aime-2024.jsonl}"
export QWEN35_SUPPORTED_GPU_COUNTS="${QWEN35_SUPPORTED_GPU_COUNTS:-4,8}"
export QWEN35_MIN_FREE_VRAM_GB="${QWEN35_MIN_FREE_VRAM_GB:-150}"
if [ -n "${QWEN35_VISIBLE_DEVICES:-}" ]; then
    SELECTED_GPUS="${QWEN35_VISIBLE_DEVICES}"
else
    if ! SELECTED_GPUS="$(select_available_gpus "${QWEN35_MIN_FREE_VRAM_GB}" "${QWEN35_NUM_GPUS:-}")"; then
        echo "=== Not enough free GPUs for Qwen3.5 smoke; exiting without starting Ray ==="
        exit 0
    fi
fi
export HIP_VISIBLE_DEVICES="${SELECTED_GPUS}"
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export NUM_GPUS="$(python3 - <<'PY'
import os
print(len([x for x in os.environ["HIP_VISIBLE_DEVICES"].split(",") if x]))
PY
)"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export QWEN35_SOCKET_IFNAME="${QWEN35_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${QWEN35_SOCKET_IFNAME}}"
export RAY_PORT="6379"
export RAY_DASHBOARD_PORT="8265"
export RAY_MIN_WORKER_PORT="30000"
export RAY_MAX_WORKER_PORT="65000"
export RAY_NODE_MANAGER_PORT="6381"
export RAY_OBJECT_MANAGER_PORT="6382"
export RAY_RUNTIME_ENV_AGENT_PORT="6383"
export RAY_DASHBOARD_AGENT_LISTEN_PORT="6384"
export RAY_DASHBOARD_AGENT_GRPC_PORT="6385"
export RAY_TMPDIR="/tmp/ray-qwen35-9b"
export RELAX_SERVE_PORT="8000"
export TENSORBOARD_DIR="${SCRIPT_DIR}/tensorboard/qwen35-9b"
export MEGATRON="/root/Megatron-LM/"
export RELAX="${REPO_ROOT}"
export RUN_ID="qwen35-9b-dapo-math-te-debug-$(date +%Y%m%d-%H%M%S)"

if [ "${NUM_GPUS}" -ge 8 ]; then
    export ACTOR_GPUS="${ACTOR_GPUS:-4}"
    export ACTOR_TP="${ACTOR_TP:-4}"
    export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
else
    export ACTOR_GPUS="${ACTOR_GPUS:-1}"
    export ACTOR_TP="${ACTOR_TP:-1}"
    export ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
fi
export REFERENCE_GPUS="${REFERENCE_GPUS:-1}"
export ACTOR_FWD_GPUS="${ACTOR_FWD_GPUS:-1}"
if [ -n "${QWEN35_RESOURCE:-}" ]; then
    export RELAX_RESOURCE="${QWEN35_RESOURCE}"
else
    RELAX_RESOURCE="$(
        python3 - <<'PY'
import json
import os

resource = {
    "actor": [1, int(os.environ["ACTOR_GPUS"])],
    "rollout": [1, int(os.environ["ROLLOUT_GPUS"])],
    "reference": [1, int(os.environ["REFERENCE_GPUS"])],
    "actor_fwd": [1, int(os.environ["ACTOR_FWD_GPUS"])],
    "advantages": [1, 0],
}
print(json.dumps(resource))
PY
    )"
    export RELAX_RESOURCE
fi

echo "=== Selected GPUs: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} ==="
echo "=== Resource plan: NUM_GPUS=${NUM_GPUS}, ACTOR_GPUS=${ACTOR_GPUS}, ACTOR_TP=${ACTOR_TP}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, REFERENCE_GPUS=${REFERENCE_GPUS}, ACTOR_FWD_GPUS=${ACTOR_FWD_GPUS} ==="

# Keep the 9B smoke aligned with the verified 4B ROCm settings.
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES="1"
export NVTE_DEBUG="1"
export NVTE_DEBUG_LEVEL="2"
export RAY_DEDUP_LOGS="0"
export TORCHDYNAMO_DISABLE="1"
export PYTHONUNBUFFERED="1"

export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
export PYTHONPATH="${RELAX}:${MEGATRON}:${PYTHONPATH:-}"
export HOST_IP="${MASTER_ADDR}"
export HAS_NVLINK="$(relax_detect_fast_interconnect)"
export RUNTIME_ENV_JSON="{
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RELAX_SERVE_PORT\": \"${RELAX_SERVE_PORT}\",
   \"HIP_VISIBLE_DEVICES\": \"${HIP_VISIBLE_DEVICES}\",
   \"RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES\": \"${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES}\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
}
}"

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

    if [[ "${RAY_TMPDIR}" == /tmp/ray-qwen35-9b* ]]; then
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

echo "=== Launching Qwen3.5-9B DAPO-Math direct runner ==="
cd "${REPO_ROOT}"
exec bash "${SCRIPT_DIR}/run-qwen35-9b-dapo-math-direct.sh"
