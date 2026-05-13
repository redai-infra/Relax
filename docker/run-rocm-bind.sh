#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

IMAGE="${IMAGE:-relax:rocm-gfx942}"
CONTAINER_NAME="${CONTAINER_NAME:-relax_rocm_bind}"
RELAX_DIR="${RELAX_DIR:-${REPO_ROOT}}"
MODEL_DIR="${MODEL_DIR:-/mnt/dcgpuval/models}"
NETWORK_MODE="${NETWORK_MODE:-bridge}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required but not found in PATH" >&2
    exit 1
fi

if [ ! -d "${RELAX_DIR}" ]; then
    echo "RELAX_DIR does not exist: ${RELAX_DIR}" >&2
    exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
    echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "Container already exists: ${CONTAINER_NAME}" >&2
    echo "Remove it manually or set CONTAINER_NAME to a new value." >&2
    exit 1
fi

exec docker run -d \
    --name "${CONTAINER_NAME}" \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --security-opt label=disable \
    --ipc=host \
    --network "${NETWORK_MODE}" \
    -v "${RELAX_DIR}:/root/Relax" \
    -v "${MODEL_DIR}:/mnt/dcgpuval/models:ro" \
    "${IMAGE}" \
    sleep infinity
