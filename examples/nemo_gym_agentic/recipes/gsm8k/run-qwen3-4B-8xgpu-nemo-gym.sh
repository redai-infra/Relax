#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU GSM8K NeMo Gym agentic training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

export NEMO_GYM_ENVIRONMENT="${NEMO_GYM_ENVIRONMENT:-gsm8k}"
export NEMO_GYM_CONFIG="${NEMO_GYM_CONFIG:-gsm8k-v1}"
export NEMO_GYM_SOURCE_DATA="${NEMO_GYM_SOURCE_DATA:-/data/nemo-gym/gsm8k_benchmark.jsonl}"
export EXP_DIR="${EXP_DIR:-${PWD}/outputs/nemo-gym-gsm8k}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/dev/nemo-gym}"
export EXP_NAME="${EXP_NAME:-gsm8k-qwen3-4b-8xgpu}"

exec bash "${EXAMPLE_DIR}/scripts/run-qwen3-4B-8xgpu-nemo-gym.sh"
