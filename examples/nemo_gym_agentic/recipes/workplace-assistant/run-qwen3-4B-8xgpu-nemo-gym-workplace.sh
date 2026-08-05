#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU Workplace Assistant agentic training smoke.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

export NEMO_GYM_ENVIRONMENT="${NEMO_GYM_ENVIRONMENT:-workplace_assistant}"
export NEMO_GYM_CONFIG="${NEMO_GYM_CONFIG:-workplace-assistant-v1}"
export NEMO_GYM_SOURCE_DATA="${NEMO_GYM_SOURCE_DATA:-/data/nemo-gym/workplace_assistant_train.jsonl}"
export NEMO_GYM_DATA_LIMIT="${NEMO_GYM_DATA_LIMIT:-32}"
export NEMO_GYM_NUM_ROLLOUT="${NEMO_GYM_NUM_ROLLOUT:-3}"
export NEMO_GYM_N_SAMPLES_PER_PROMPT="${NEMO_GYM_N_SAMPLES_PER_PROMPT:-1}"
export NEMO_GYM_GLOBAL_BATCH_SIZE="${NEMO_GYM_GLOBAL_BATCH_SIZE:-${NEMO_GYM_N_SAMPLES_PER_PROMPT}}"
export NEMO_GYM_ROLLOUT_MAX_RESPONSE_LEN="${NEMO_GYM_ROLLOUT_MAX_RESPONSE_LEN:-2048}"
export NEMO_GYM_ROLLOUT_MAX_CONTEXT_LEN="${NEMO_GYM_ROLLOUT_MAX_CONTEXT_LEN:-8192}"
export NEMO_GYM_TOOL_CALL_PARSER="${NEMO_GYM_TOOL_CALL_PARSER:-qwen}"
export EXP_DIR="${EXP_DIR:-${PWD}/outputs/nemo-gym-workplace}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/dev/nemo-gym}"
export EXP_NAME="${EXP_NAME:-workplace-assistant-qwen3-4b-8xgpu}"

exec bash "${EXAMPLE_DIR}/scripts/run-qwen3-4B-8xgpu-nemo-gym.sh"
