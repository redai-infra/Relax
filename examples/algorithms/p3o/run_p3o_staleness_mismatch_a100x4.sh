#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export TASK40_ALGORITHM=p3o
export TASK40_BEHAVIOR_MISMATCH=1
export TASK40_MAX_STALENESS="${TASK40_MAX_STALENESS:-2}"
source "${SCRIPT_DIR}/common_a100x4.sh"
task40_run
