#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

usage() {
    cat <<'EOF'
Usage:
  start_r2e_gym_remote.sh \
    --gym-host <routable-local-ip> \
    --data-dir <prepared-r2e-directory> \
    --mode <golden|train> \
    [--callback-host <relax-callback-host>] \
    [--image <nemo-gym-image>] \
    [--repo-dir <docker-host-relax-checkout>] \
    [--proxy <http-proxy-url>] \
    [--callback-proxy <http-proxy-url>] \
    [--callback-timeout-s <positive-integer>] \
    [--max-concurrency <positive-integer>] \
    [--verbose] \
    [--container-name <name>]

golden mode uses the Gym host as its non-contacted callback allowlist entry.
train mode requires --callback-host to exactly match the host in Relax's callback URL.
EOF
}

GYM_HOST=""
R2E_DATA_DIR=""
R2E_GYM_MODE=""
RELAX_CALLBACK_HOST=""
NEMO_GYM_IMAGE="${NEMO_GYM_IMAGE:-relax-nemo-gym:r2e-dev}"
NEMO_GYM_CONTAINER="${NEMO_GYM_CONTAINER:-nemo-gym-r2e-local}"
NEMO_GYM_HTTP_PROXY="${NEMO_GYM_HTTP_PROXY:-${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}}"
NEMO_GYM_NO_PROXY="${NEMO_GYM_NO_PROXY:-${no_proxy:-${NO_PROXY:-}}}"
NEMO_GYM_CALLBACK_PROXY="${NEMO_GYM_CALLBACK_PROXY:-}"
NEMO_GYM_CALLBACK_TIMEOUT_S="${NEMO_GYM_CALLBACK_TIMEOUT_S:-600}"
NEMO_GYM_START_TIMEOUT_S="${NEMO_GYM_START_TIMEOUT_S:-1800}"
GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS:-8}"
R2E_GYM_MAX_CONCURRENCY="${R2E_GYM_MAX_CONCURRENCY:-16}"
NEMO_GYM_VERBOSE="${NEMO_GYM_VERBOSE:-0}"
RELAX_REPO_ROOT="${RELAX_REPO_ROOT:-$(cd -- "${EXAMPLE_DIR}/../.." &>/dev/null && pwd)}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gym-host)
            GYM_HOST="${2:-}"
            shift 2
            ;;
        --data-dir)
            R2E_DATA_DIR="${2:-}"
            shift 2
            ;;
        --mode)
            R2E_GYM_MODE="${2:-}"
            shift 2
            ;;
        --callback-host)
            RELAX_CALLBACK_HOST="${2:-}"
            shift 2
            ;;
        --image)
            NEMO_GYM_IMAGE="${2:-}"
            shift 2
            ;;
        --repo-dir)
            RELAX_REPO_ROOT="${2:-}"
            shift 2
            ;;
        --proxy)
            NEMO_GYM_HTTP_PROXY="${2:-}"
            shift 2
            ;;
        --callback-proxy)
            NEMO_GYM_CALLBACK_PROXY="${2:-}"
            shift 2
            ;;
        --callback-timeout-s)
            NEMO_GYM_CALLBACK_TIMEOUT_S="${2:-}"
            shift 2
            ;;
        --max-concurrency)
            R2E_GYM_MAX_CONCURRENCY="${2:-}"
            shift 2
            ;;
        --verbose)
            NEMO_GYM_VERBOSE=1
            shift
            ;;
        --container-name)
            NEMO_GYM_CONTAINER="${2:-}"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "${GYM_HOST}" ] || [ -z "${R2E_DATA_DIR}" ] || [ -z "${R2E_GYM_MODE}" ]; then
    usage >&2
    exit 2
fi
if [[ "${GYM_HOST}" == *"://"* ]] || [[ "${GYM_HOST}" == *":"* ]]; then
    echo "--gym-host must be a bare routable host or IP without scheme or port" >&2
    exit 2
fi
case "${R2E_DATA_DIR}" in
    /*) ;;
    *)
        echo "--data-dir must be an absolute path" >&2
        exit 2
        ;;
esac
case "${RELAX_REPO_ROOT}" in
    /*) ;;
    *)
        echo "--repo-dir must be an absolute path" >&2
        exit 2
        ;;
esac
case "${R2E_GYM_MODE}" in
    golden)
        R2E_GYM_VERIFY_GOLDEN_PATCH=1
        RELAX_CALLBACK_HOST="${RELAX_CALLBACK_HOST:-${GYM_HOST}}"
        ;;
    train)
        R2E_GYM_VERIFY_GOLDEN_PATCH=0
        if [ -z "${RELAX_CALLBACK_HOST}" ]; then
            echo "train mode requires --callback-host" >&2
            exit 2
        fi
        ;;
    *)
        echo "--mode must be golden or train" >&2
        exit 2
        ;;
esac
if [[ "${RELAX_CALLBACK_HOST}" == *"://"* ]] || [[ "${RELAX_CALLBACK_HOST}" == *":"* ]]; then
    echo "--callback-host must be the exact bare host from the Relax callback URL" >&2
    exit 2
fi
if ! [[ "${NEMO_GYM_START_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NEMO_GYM_START_TIMEOUT_S must be a positive integer" >&2
    exit 2
fi
if ! [[ "${GYM_RAY_NUM_CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GYM_RAY_NUM_CPUS must be a positive integer" >&2
    exit 2
fi
if ! [[ "${R2E_GYM_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-concurrency must be a positive integer" >&2
    exit 2
fi
if [ "${NEMO_GYM_VERBOSE}" != "0" ] && [ "${NEMO_GYM_VERBOSE}" != "1" ]; then
    echo "NEMO_GYM_VERBOSE must be 0 or 1" >&2
    exit 2
fi
if ! [[ "${NEMO_GYM_CALLBACK_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--callback-timeout-s must be a positive integer" >&2
    exit 2
fi

command -v docker >/dev/null 2>&1 || {
    echo "docker is required" >&2
    exit 2
}
command -v curl >/dev/null 2>&1 || {
    echo "curl is required" >&2
    exit 2
}
if docker container inspect "${NEMO_GYM_CONTAINER}" >/dev/null 2>&1; then
    echo "Removing existing container ${NEMO_GYM_CONTAINER}..."
    docker rm -f "${NEMO_GYM_CONTAINER}" >/dev/null
fi

docker image inspect "${NEMO_GYM_IMAGE}" >/dev/null
test -s "${R2E_DATA_DIR}/r2e_gym_train.jsonl" || {
    echo "Missing prepared dataset: ${R2E_DATA_DIR}/r2e_gym_train.jsonl" >&2
    exit 2
}
find "${R2E_DATA_DIR}/sif" -maxdepth 1 -type f -name "*.sif" -size +0c -print -quit | grep -q . || {
    echo "No prepared SIF found under ${R2E_DATA_DIR}/sif" >&2
    exit 2
}

callback_allowlist="${RELAX_CALLBACK_HOST},${GYM_HOST},127.0.0.1"
container_no_proxy="127.0.0.1,localhost,${GYM_HOST},${RELAX_CALLBACK_HOST}"
if [ -n "${NEMO_GYM_NO_PROXY}" ]; then
    container_no_proxy="${container_no_proxy},${NEMO_GYM_NO_PROXY}"
fi

r2e_setup_volume="${NEMO_GYM_CONTAINER}-r2e-setup"
openhands_setup_volume="${NEMO_GYM_CONTAINER}-openhands-setup"

docker volume create "${r2e_setup_volume}" >/dev/null
docker volume create "${openhands_setup_volume}" >/dev/null

echo "Creating remote R2E-Gym container:"
echo "  container=${NEMO_GYM_CONTAINER}"
echo "  image=${NEMO_GYM_IMAGE}"
echo "  mode=${R2E_GYM_MODE}"
echo "  gym_url=http://${GYM_HOST}:28100"
echo "  callback_host=${RELAX_CALLBACK_HOST}"
echo "  data_dir=${R2E_DATA_DIR}"
echo "  repo_dir=${RELAX_REPO_ROOT}"
echo "  max_concurrency=${R2E_GYM_MAX_CONCURRENCY}"
echo "  callback_timeout_s=${NEMO_GYM_CALLBACK_TIMEOUT_S}"
if [ -n "${NEMO_GYM_CALLBACK_PROXY}" ]; then
    echo "  callback_proxy=enabled"
else
    echo "  callback_proxy=disabled"
fi
echo "  verbose=${NEMO_GYM_VERBOSE}"

docker create \
    --name "${NEMO_GYM_CONTAINER}" \
    --privileged \
    --network host \
    --shm-size 16g \
    --mount "type=bind,source=${RELAX_REPO_ROOT},target=/opt/relax-integration,readonly" \
    --volume "${R2E_DATA_DIR}:${R2E_DATA_DIR}" \
    --volume "${r2e_setup_volume}:/opt/nemo-gym/responses_api_agents/swe_agents/swe_r2e_gym_setup" \
    --volume "${openhands_setup_volume}:/opt/nemo-gym/responses_api_agents/swe_agents/swe_openhands_setup" \
    --env GYM_HOST="${GYM_HOST}" \
    --env GYM_BIND_HOST="${GYM_HOST}" \
    --env GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS}" \
    --env RAY_CLI="/opt/nemo-gym/.venv/bin/ray" \
    --env R2E_GYM_CLUSTER_PYTHON="/opt/nemo-gym/.venv/bin/python" \
    --env R2E_GYM_MODE="${R2E_GYM_MODE}" \
    --env R2E_GYM_MAX_CONCURRENCY="${R2E_GYM_MAX_CONCURRENCY}" \
    --env NEMO_GYM_VERBOSE="${NEMO_GYM_VERBOSE}" \
    --env NEMO_GYM_CALLBACK_ALLOWED_HOSTS="${callback_allowlist}" \
    --env NEMO_GYM_CALLBACK_PROXY="${NEMO_GYM_CALLBACK_PROXY}" \
    --env NEMO_GYM_CALLBACK_TIMEOUT_S="${NEMO_GYM_CALLBACK_TIMEOUT_S}" \
    --env R2E_GYM_DATA="${R2E_DATA_DIR}/r2e_gym_train.jsonl" \
    --env R2E_GYM_SIF_DIR="${R2E_DATA_DIR}/sif" \
    --env R2E_GYM_VERIFY_GOLDEN_PATCH="${R2E_GYM_VERIFY_GOLDEN_PATCH}" \
    --env HTTP_PROXY="${NEMO_GYM_HTTP_PROXY}" \
    --env HTTPS_PROXY="${NEMO_GYM_HTTP_PROXY}" \
    --env NO_PROXY="${container_no_proxy}" \
    --env http_proxy="${NEMO_GYM_HTTP_PROXY}" \
    --env https_proxy="${NEMO_GYM_HTTP_PROXY}" \
    --env no_proxy="${container_no_proxy}" \
    "${NEMO_GYM_IMAGE}" \
    bash -lc '
        set -euo pipefail
        /opt/nemo-gym/.venv/bin/ray start \
            --head \
            --node-ip-address="${GYM_HOST}" \
            --port=6381 \
            --num-cpus="${GYM_RAY_NUM_CPUS}" \
            --include-dashboard=true \
            --dashboard-port=28265 \
            --disable-usage-stats \
            --temp-dir=/tmp/ray-nemo-r2e
        export RAY_ADDRESS="${GYM_HOST}:6381"
        local_args=(
            --data-dir "$(dirname "${R2E_GYM_DATA}")"
            --mode "${R2E_GYM_MODE}"
            --max-concurrency "${R2E_GYM_MAX_CONCURRENCY}"
        )
        if [ "${NEMO_GYM_VERBOSE}" = "1" ]; then
            local_args+=(--verbose)
        fi
        exec bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/r2e-gym/start_r2e_gym_local.sh \
            "${local_args[@]}"
    ' >/dev/null

docker start "${NEMO_GYM_CONTAINER}" >/dev/null

echo "Waiting for http://${GYM_HOST}:28100/readyz ..."
deadline=$((SECONDS + NEMO_GYM_START_TIMEOUT_S))
next_progress=$SECONDS
while [ "${SECONDS}" -lt "${deadline}" ]; do
    if ! docker container inspect --format "{{.State.Running}}" "${NEMO_GYM_CONTAINER}" | grep -q true; then
        echo "R2E-Gym container exited before becoming ready." >&2
        docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
        exit 1
    fi
    if ready_json="$(curl --noproxy "*" -fsS "http://${GYM_HOST}:28100/readyz" 2>/dev/null)" &&
        printf '%s' "${ready_json}" | grep -q '"ready":true'; then
        printf '%s\n' "${ready_json}"
        echo "R2E-Gym is ready at http://${GYM_HOST}:28100"
        exit 0
    fi
    if [ "${SECONDS}" -ge "${next_progress}" ]; then
        echo "Still starting; inspect with: docker logs -f ${NEMO_GYM_CONTAINER}"
        next_progress=$((SECONDS + 30))
    fi
    sleep 2
done

echo "Timed out waiting for R2E-Gym readiness." >&2
docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
exit 1
