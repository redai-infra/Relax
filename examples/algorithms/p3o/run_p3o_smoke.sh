#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

CONFIG="${1:-p3o_on_policy}"
case "${CONFIG}" in
    p3o_on_policy)
        export TASK40_ALGORITHM=p3o
        export TASK40_BEHAVIOR_MISMATCH=0
        ;;
    grpo_on_policy)
        export TASK40_ALGORITHM=grpo
        export TASK40_BEHAVIOR_MISMATCH=0
        ;;
    p3o_temperature_1p2)
        export TASK40_ALGORITHM=p3o
        export TASK40_BEHAVIOR_MISMATCH=1
        ;;
    grpo_temperature_1p2)
        export TASK40_ALGORITHM=grpo
        export TASK40_BEHAVIOR_MISMATCH=1
        ;;
    *)
        echo "Unknown smoke config: ${CONFIG}" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export TASK40_MODE=smoke
source "${SCRIPT_DIR}/common_a100x4.sh"
task40_run
