#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Task 22 four-GPU Carry-Aware Cross-Version KV Continuation entrypoint:
#   Actor TP2 + two independent one-GPU Rollout engines.
# Enables cross-version KV cache continuation, carry-aware oversampling,
# and in-place DCS publication while leaving admission, priority scheduling,
# and work-aware placement disabled.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export NUM_GPUS=4
export NUM_ROLLOUT="${NUM_ROLLOUT:-11}"
export TASK22_EXPERIMENT_ARM=dcs_carry_aware_kv_continuation_on

exec "$SCRIPT_DIR/run_clean_main_calibration.sh"
