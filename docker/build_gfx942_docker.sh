#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required but not found in PATH" >&2
    exit 1
fi

exec env DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" docker build \
    -f "${SCRIPT_DIR}/Dockerfile.rocm" \
    --target relax \
    --build-arg GPU_ARCH=gfx942 \
    -t relax:rocm-gfx942 \
    "$@" \
    "${REPO_ROOT}"
