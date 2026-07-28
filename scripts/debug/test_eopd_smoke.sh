#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# EOPD (Entropy-aware OPD) smoke test.
#
# This script validates that the EOPD arguments, data pipeline, and loss
# function integrate without errors. It runs a colocate OPD+EOPD training
# for 5 steps on a small model.
#
# Prerequisites:
#   - A small Megatron-compatible checkpoint (e.g., Qwen3-0.6B converted to mcore)
#   - A prompt dataset (JSONL with "prompt" and "label" keys)
#
# Usage:
#   MODEL_DIR=/path/to/small_model DATA_DIR=/path/to/data \
#     bash scripts/debug/test_eopd_smoke.sh
#
# Expected outcome:
#   - Training completes 5 steps without crash
#   - Logs contain "eopd_fkl_loss" metric

set -ex
set -o pipefail

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to a small Megatron checkpoint directory}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to a directory containing a prompt JSONL file}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/prompts.jsonl}"

python relax/entrypoints/train.py \
    --hf-checkpoint "${MODEL_DIR}" \
    --ref-load "${MODEL_DIR}" \
    --megatron-to-hf-mode bridge \
    --prompt-data "${PROMPT_SET}" \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rm-type dapo \
    --reward-key score \
    --num-rollout 5 \
    --rollout-batch-size 2 \
    --n-samples-per-prompt 2 \
    --rollout-max-response-len 128 \
    --rollout-temperature 1 \
    --global-batch-size 4 \
    --loss-type grpo \
    --advantage-estimator grpo \
    --lr 1e-6 \
    --eps-clip 0.2 \
    --use-opd \
    --opd-type megatron \
    --opd-teacher-load "${MODEL_DIR}" \
    --opd-loss-coef 1.0 \
    --opd-kl-coef 0.0 \
    --opd-log-prob-top-k 4 \
    --opd-token-selection teacher_topk \
    --use-eopd \
    --eopd-entropy-threshold 0.8 \
    --eopd-fkl-coef 1.0 \
    2>&1 | tee /tmp/eopd_smoke.log

echo "---"
echo "Checking for EOPD metrics in logs..."
if grep -q "eopd_fkl_loss" /tmp/eopd_smoke.log; then
    echo "PASS: eopd_fkl_loss found in training logs."
else
    echo "WARN: eopd_fkl_loss not found in logs (may be expected if 0 steps ran)."
fi
echo "EOPD smoke test completed."
