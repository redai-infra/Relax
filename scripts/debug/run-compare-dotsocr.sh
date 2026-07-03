#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Single-GPU helper that runs scripts/debug/compare_sglang_megatron_dotsocr.py
# against the same dots.mocr checkpoint used by run-dotsocr2-8xgpu.sh.
#
# Usage:
#   # multimodal (image + prompt)
#   bash scripts/debug/run-compare-dotsocr.sh <image_path_or_url> [user_prompt] [extra args...]
#   # text-only — pass empty string as first arg
#   bash scripts/debug/run-compare-dotsocr.sh "" "What is 2+3?" [extra args...]
#
# Environment overrides:
#   MODEL_DIR  - base directory holding rednote-hilab/dots.mocr (default: $EXP_DIR)
#   EXP_DIR    - exps root used by the training script family (default: ./exps)
#   HF_CKPT    - explicit checkpoint path (overrides MODEL_DIR computation)
#   DUMP_DIR   - where per-side logprob dumps land (default: /tmp/relax_dotsocr_debug)
#   DTYPE      - bf16 (default) | fp16 | fp32

set -ex
set -o pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <image_path_or_url|''> [user_prompt] [extra args...]" >&2
    echo "  pass '' as the image to run text-only mode." >&2
    exit 1
fi

IMAGE_PATH="$1"
shift
USER_PROMPT="${1:-Describe this image.}"
if [ "$#" -gt 0 ]; then shift; fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/rednote-hilab/dots.mocr/}"
DUMP_DIR="${DUMP_DIR:-/tmp/relax_dotsocr_debug}"
DTYPE="${DTYPE:-bf16}"

# Constrain to a single GPU; the script asserts tp=pp=cp=ep=1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Keep megatron mpu's nccl init quiet and deterministic on one rank.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29503}"
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

IMAGE_ARGS=()
if [ -n "${IMAGE_PATH}" ]; then
    IMAGE_ARGS=(--image "${IMAGE_PATH}")
fi

python3 "${SCRIPT_DIR}/compare_sglang_megatron_dotsocr.py" \
    --hf-checkpoint "${HF_CKPT}" \
    --prompt "${USER_PROMPT}" \
    --dtype "${DTYPE}" \
    --dump-dir "${DUMP_DIR}" \
    "${IMAGE_ARGS[@]}" \
    "$@"
