#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_BASE_URL="${RELAX_BASE_URL:?RELAX_BASE_URL is required}"
export OPENAI_API_KEY="${RELAX_SESSION_ID:?RELAX_SESSION_ID is required}"

exec "${ALFWORLD_PYTHON:-python3}" -m examples.graphgpo.rollout_agent \
    --input-json "${RELAX_INPUT_JSON:?RELAX_INPUT_JSON is required}" \
    --output-json "${RELAX_OUTPUT_JSON:?RELAX_OUTPUT_JSON is required}"
