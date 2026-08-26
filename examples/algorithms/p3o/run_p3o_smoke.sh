#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

CONFIG="${1:-p3o_on_policy}"
case "${CONFIG}" in
    p3o_on_policy)
        export P3O_ALGORITHM=p3o
        export P3O_ENABLE_TEMPERATURE_OVERRIDE=0
        export P3O_UPDATE_WEIGHTS_INTERVAL=1
        ;;
    grpo_on_policy)
        export P3O_ALGORITHM=grpo
        export P3O_ENABLE_TEMPERATURE_OVERRIDE=0
        export P3O_UPDATE_WEIGHTS_INTERVAL=1
        ;;
    p3o_temperature_0p6)
        export P3O_ALGORITHM=p3o
        export P3O_ENABLE_TEMPERATURE_OVERRIDE=1
        export P3O_BEHAVIOR_TEMPERATURE=0.6
        export P3O_UPDATE_WEIGHTS_INTERVAL=1
        ;;
    grpo_temperature_0p6)
        export P3O_ALGORITHM=grpo
        export P3O_ENABLE_TEMPERATURE_OVERRIDE=1
        export P3O_BEHAVIOR_TEMPERATURE=0.6
        export P3O_UPDATE_WEIGHTS_INTERVAL=1
        ;;
    p3o_temperature_1p2)
        export P3O_ALGORITHM=p3o
        export P3O_ENABLE_TEMPERATURE_OVERRIDE=1
        export P3O_BEHAVIOR_TEMPERATURE=1.2
        export P3O_UPDATE_WEIGHTS_INTERVAL=1
        ;;
    grpo_temperature_1p2)
        export P3O_ALGORITHM=grpo
        export P3O_ENABLE_TEMPERATURE_OVERRIDE=1
        export P3O_BEHAVIOR_TEMPERATURE=1.2
        export P3O_UPDATE_WEIGHTS_INTERVAL=1
        ;;
    p3o_periodic_sync_interval_3)
        export P3O_ALGORITHM=p3o
        export P3O_ENABLE_TEMPERATURE_OVERRIDE=0
        export P3O_UPDATE_WEIGHTS_INTERVAL=3
        ;;
    grpo_periodic_sync_interval_3)
        export P3O_ALGORITHM=grpo
        export P3O_ENABLE_TEMPERATURE_OVERRIDE=0
        export P3O_UPDATE_WEIGHTS_INTERVAL=3
        ;;
    *)
        echo "Unknown smoke config: ${CONFIG}" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export P3O_MODE=smoke
source "${SCRIPT_DIR}/common_a100x4.sh"
P3O_run
