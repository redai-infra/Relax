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

# Per-session stdout+stderr capture. Relax captures the subprocess streams
# into a tmpdir/command.log and only surfaces it when the agent exits with
# code != 0 (relax/agentic/pipeline/runtime.py:358); the tmpdir is then
# cleaned up (runtime.py:397), so silent-success failures (subprocess exits
# 0 but never produced a chat IR) leave no trace, and successful sessions
# leave no trace either — both make hang/perf debugging impossible.
#
# Mirrors nemo_gym_agentic/scripts/run_agent_app.sh: tee both streams to a
# persistent per-session file keyed by AGENT_DEBUG_LOG_DIR (set by the
# training launcher to log/agent/${TIMESTAMP}) and keep ALL logs regardless
# of exit code. The training script picks a fresh ${TIMESTAMP} directory
# per run so logs don't pile up across runs.
if [ -n "${AGENT_DEBUG_LOG_DIR:-}" ]; then
    mkdir -p "${AGENT_DEBUG_LOG_DIR}"
    AGENT_LOG_FILE="${AGENT_DEBUG_LOG_DIR}/${RELAX_SESSION_ID:-unknown}.log"
    exec > >(tee -a "${AGENT_LOG_FILE}") 2> >(tee -a "${AGENT_LOG_FILE}" >&2)
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
