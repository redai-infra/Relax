#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# Build target arch for sgl-kernel. Default gfx942 (MI300X); set
# GPU_ARCH=gfx950 for MI355/MI350. AITER is prebuilt for both archs so
# the resulting image can still run on either host.
GPU_ARCH="${GPU_ARCH:-gfx942}"
IMAGE_TAG="${IMAGE_TAG:-relax:rocm-${GPU_ARCH}}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required but not found in PATH" >&2
    exit 1
fi

exec env DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" docker build \
    -f "${SCRIPT_DIR}/Dockerfile.rocm" \
    --target relax \
    --build-arg GPU_ARCH="${GPU_ARCH}" \
    -t "${IMAGE_TAG}" \
    "$@" \
    "${REPO_ROOT}"
