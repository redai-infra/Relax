#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export P3O_ALGORITHM=grpo
export P3O_ENABLE_TEMPERATURE_OVERRIDE=0
export P3O_UPDATE_WEIGHTS_INTERVAL=1
source "${SCRIPT_DIR}/common_a100x4.sh"
P3O_run
