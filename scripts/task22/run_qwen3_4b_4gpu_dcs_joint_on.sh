#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Task 22 four-GPU joint ON entrypoint:
#   Actor TP2 + two independent one-GPU Rollout engines.
# The clean calibration runner owns the full workload and evidence contract.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export NUM_GPUS=4
export NUM_ROLLOUT="${NUM_ROLLOUT:-11}"
export TASK22_EXPERIMENT_ARM=dcs_joint_on

exec "$SCRIPT_DIR/run_clean_main_calibration.sh"
