#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Per-session WebShop agent launcher. Relax spawns this once per rollout session
# via `--agent-command`, injecting RELAX_BASE_URL / RELAX_SESSION_ID /
# RELAX_INPUT_JSON / RELAX_OUTPUT_JSON.
#
# WebShop's env is a several-GB shared asset, so instead of loading it per session
# (ALFWorld-style) we lazily start ONE env server per node (app/server.py) and
# connect to it over localhost (DESIGN.md §3, §6.4). An flock guarantees exactly
# one server per node no matter how many sessions launch concurrently.
#
# CONDA_HOME / WEBSHOP_CONDA_ENV / WEBSHOP_HOME / WEBSHOP_PORT / WEBSHOP_MAX_TURNS
# are forwarded from the training script via --agent-env (multi-node safe).

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Some nodes have a global http_proxy/https_proxy that routes EVEN 127.0.0.1 and
# internal cluster IPs through an external (squid) proxy -> "Connection refused"
# to the local WebShop env server and to the internal LLM endpoint, so agent
# sessions launch but can never connect (stuck at launch_submitted, 0 warming).
# The agent only talks to localhost + internal IPs, so bypass the proxy entirely.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"

CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
export WEBSHOP_HOME="${WEBSHOP_HOME:-/root/WebShop}"
export WEBSHOP_PORT="${WEBSHOP_PORT:-36001}"
export WEBSHOP_URL="http://127.0.0.1:${WEBSHOP_PORT}"
READY_TIMEOUT_S="${WEBSHOP_SERVER_READY_TIMEOUT_S:-1200}"
# How long a session waits to ACQUIRE the bootstrap lock before treating it as
# stale. A legit first-starter holds the lock only while the server boots, so
# this tracks READY_TIMEOUT_S; a wedged/orphaned holder (e.g. left behind by a
# dead launcher daemon) can then delay a cold start by at most this long instead
# of hanging the node forever.
LOCK_WAIT_S="${WEBSHOP_LOCK_WAIT_S:-${READY_TIMEOUT_S}}"

# ---------------------------------------------------------------------------
# Lazy per-node server bootstrap: flock so only the first session on this node
# starts the server; the rest just wait for /health to go ready.
# ---------------------------------------------------------------------------
_health_ok() { curl -fs "${WEBSHOP_URL}/health" 2>/dev/null | grep -q '"ready": *true'; }

# True when nothing is bound to WEBSHOP_PORT (best-effort: if lsof is absent we
# optimistically assume free).
_port_free() { ! lsof -ti "tcp:${WEBSHOP_PORT}" 2>/dev/null | grep -q .; }

# Clear any stale WebShop server squatting the port from a prior (crashed) job.
# server.py is nohup'd and outlives the Ray job, so leftovers accumulate across
# restarts. A one-shot `lsof -ti | kill` (the old logic) was not enough: kill -9
# is async, so the new server could bind() before the kernel released the socket
# (EADDRINUSE), and a defunct/zombie holder could linger. So we kill BOTH by name
# and by port, then WAIT for the socket to actually be released before returning.
_reclaim_port() {
    pkill -9 -f "app\.server" 2>/dev/null || true
    pkill -9 -f "run_webshop_server\.sh" 2>/dev/null || true
    local pids
    if pids="$(lsof -ti "tcp:${WEBSHOP_PORT}" 2>/dev/null)" && [ -n "${pids}" ]; then
        echo "[run_agent_app] reclaiming port ${WEBSHOP_PORT} from stale pid(s) [${pids}]" >&2
        kill -9 ${pids} 2>/dev/null || true
    fi
    command -v fuser >/dev/null 2>&1 && fuser -k "${WEBSHOP_PORT}/tcp" 2>/dev/null || true
    local i
    for i in $(seq 1 20); do
        _port_free && return 0
        sleep 0.5
    done
    echo "[run_agent_app] warning: port ${WEBSHOP_PORT} still occupied after reclaim" >&2
}

# Serialized server bootstrap. Runs while holding the flock (or after a bounded
# flock timeout, when the holder is presumed stale). Double-checks health first
# so we never start a second server behind an already-healthy one, then reclaims
# the port and starts a server, retrying if it dies before becoming ready (e.g.
# a transient bind race) instead of polling a corpse for the whole timeout.
_bootstrap_server() {
    if _health_ok; then
        return 0
    fi
    local attempt srv_pid deadline
    for attempt in 1 2 3; do
        _reclaim_port
        echo "[run_agent_app] starting WebShop server on ${WEBSHOP_URL} (attempt ${attempt}) ..."
        # 200<&- closes the inherited lock fd in the backgrounded server so the
        # long-lived process doesn't keep holding the flock forever; otherwise
        # every later session blocks on `flock` for as long as the server runs.
        nohup bash "${SCRIPT_DIR}/run_webshop_server.sh" \
            >"/tmp/webshop-server-${WEBSHOP_PORT}.log" 2>&1 200<&- &
        srv_pid=$!
        echo "${srv_pid}" >"/tmp/webshop-server-${WEBSHOP_PORT}.pid"
        deadline=$((SECONDS + READY_TIMEOUT_S))
        until _health_ok; do
            if ! kill -0 "${srv_pid}" 2>/dev/null; then
                echo "[run_agent_app] server (pid ${srv_pid}) died before ready" \
                    "(see /tmp/webshop-server-${WEBSHOP_PORT}.log); retrying" >&2
                break
            fi
            if [ "${SECONDS}" -ge "${deadline}" ]; then
                echo "[run_agent_app] server not ready after ${READY_TIMEOUT_S}s; see /tmp/webshop-server-${WEBSHOP_PORT}.log" >&2
                return 1
            fi
            sleep 3
        done
        if _health_ok; then
            echo "[run_agent_app] WebShop server ready."
            return 0
        fi
    done
    echo "[run_agent_app] WebShop server failed to become ready after retries" >&2
    return 1
}

# Fast path: once the server is healthy, every later session skips the flock
# entirely — no lock contention, and a wedged/stale lock holder left behind by a
# dead launcher daemon can NEVER block new sessions while a healthy server
# exists. (That orphaned-holder-never-releases-the-flock case hung a whole node:
# with this check, sessions see the still-healthy server and never touch the
# lock.)
if ! _health_ok; then
    (
        # Bounded acquisition: a stale holder (e.g. orphaned by a dead launcher
        # daemon) must not block this session forever. On timeout, treat the lock
        # as stale and bootstrap anyway; the port-reclaim in _bootstrap_server
        # makes a racing double-start self-healing.
        flock -w "${LOCK_WAIT_S}" 200 \
            || echo "[run_agent_app] flock not acquired in ${LOCK_WAIT_S}s; assuming stale lock, bootstrapping anyway" >&2
        _bootstrap_server
    ) 200>"/tmp/webshop-server-${WEBSHOP_PORT}.lock" || exit 1
fi

# Activate the conda env for the agent process itself (imports httpx / openai).
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${WEBSHOP_CONDA_ENV:-relax-opd-webshop}"

# Policy endpoint (SGLang). Trailing-slash strip avoids a double-slash 404 if any
# client builds f"{base_url}/chat/completions".
export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

exec python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
