#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Trailing-slash strip avoids the nemo_gym-style 404 trap
# (memory id f3c6b412): OpenAI SDK normalises, but if any client
# does f"{base_url}/chat/completions" you get a double-slash 404.
export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

# Per-session stdout+stderr capture, written DIRECTLY to a persistent
# per-session file keyed by AGENT_DEBUG_LOG_DIR (set by the training launcher
# to log/agent/${TIMESTAMP}). ALL logs are kept regardless of exit code — the
# whole point, since Relax's own tmpdir capture (runtime.py:358/397) is
# dropped on both silent-success and clean exit, making hang/perf debugging
# impossible. A fresh ${TIMESTAMP} dir per run keeps logs from piling up.
#
# Do NOT reintroduce `tee` process substitution here (the old
# `exec > >(tee ...) 2> >(tee ...)` form). Each session spawned two `tee`
# children; because this shell then `exec`s into python, those tee's are
# orphaned when python exits/is-killed and — since the container's PID 1 is a
# non-reaping `sleep` — become permanent zombies, each holding a host PID
# slot. That leak reached tens of thousands of zombies and fed host-level
# PID/thread exhaustion, surfacing as `pthread_create`/`fork` EAGAIN crashes
# across the job. A plain append redirect keeps full per-session logs with
# ZERO extra processes. (Functional agent output goes to RELAX_OUTPUT_JSON,
# not stdout, so nothing is lost by not mirroring to Relax's stream capture.)
if [ -n "${AGENT_DEBUG_LOG_DIR:-}" ]; then
    mkdir -p "${AGENT_DEBUG_LOG_DIR}"
    AGENT_LOG_FILE="${AGENT_DEBUG_LOG_DIR}/${RELAX_SESSION_ID:-unknown}.log"
    exec >> "${AGENT_LOG_FILE}" 2>&1
fi

# Host-side jupyter_client is needed by apptainer_jupyter_backend.py:200
# (the ZMQ client that talks to the in-SIF ipykernel). Some Ray worker
# nodes' base Python doesn't have it — caught at session creation time
# and crashes the agent. Cheap idempotent guard: import-check first,
# pip-install only on miss (~50ms when present).
python -c "import jupyter_client" 2>/dev/null || pip install --quiet --no-input "jupyter_client>=8"

exec python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
