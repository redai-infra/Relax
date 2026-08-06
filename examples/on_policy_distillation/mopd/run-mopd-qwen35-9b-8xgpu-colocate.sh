#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Multi-Teacher OPD (MOPD), single-node 8 GPU colocate:
#   student : Qwen3.5-9B, actor(8) + rollout(4) colocate
#   teacher : one per data_source, shares the actor GPU pool (4 GPUs total)
#     dapo-math-17k       -> Qwen3.5-9B-dapo-teacher-hf (text, TP=2)
#     multimodal-open-r1  -> Qwen3.5-9B-vl-teacher-hf   (VL,   TP=2)
#   data    : dapo-math-17k + multimodal-open-r1, merged by prepare_data.py
#
# Where the teachers come from: both are Qwen3.5-9B self-distillation checkpoints,
# each trained on its own domain, then converted mcore->HF (SGLang loads HF):
#   Qwen3.5-9B-dapo-teacher-hf : scripts/training/text/run-qwen35-9B-8xgpu.sh (dapo-math)
#   Qwen3.5-9B-vl-teacher-hf   : scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-async.sh (openr1mm)
#
# Pure distillation: --opd-only-reward zeroes the task reward; only the OPD KL
# term trains the student toward the two teachers.
# Constraint (enforced): rollout_gpus + teacher_gpus == actor_gpus (4 + 4 == 8).
#
# Usage:
#   bash run-mopd-qwen35-9b-8xgpu-colocate.sh

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
export RELAX_OPD_PREEXPANDED_PATCH=1
# Forward the patch flag into the Ray runtime env so remote-node teachers see it.
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}RELAX_OPD_PREEXPANDED_PATCH"

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/recipes/mopd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mopd-qwen35-9b-8xgpu-${now}}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-9B}"
TEXT_TEACHER_MODEL_NAME="${TEXT_TEACHER_MODEL_NAME:-Qwen3.5-9B-dapo-teacher-hf}"
VL_TEACHER_MODEL_NAME="${VL_TEACHER_MODEL_NAME:-Qwen3.5-9B-vl-teacher-hf}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/MOPD/train.parquet}"
# Derive eval set from PROMPT_SET's directory so overriding PROMPT_SET alone is
# enough. Use the small balanced subset for fast monitoring.
EVAL_SET="${EVAL_SET:-${PROMPT_SET%/*}/test_small.parquet}"

ACTOR_GPUS="${ACTOR_GPUS:-8}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
TEACHER_GPUS="${TEACHER_GPUS:-4}"
TEACHER_NUM_GPUS_PER_ENGINE="${TEACHER_NUM_GPUS_PER_ENGINE:-2}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/${STUDENT_MODEL_NAME}/"
   --megatron-to-hf-mode bridge
   # --save "${EXP_DIR}/save/mopd-${STUDENT_MODEL_NAME}/"
   # --save-interval 200
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --metadata-key extra_info
   --apply-chat-template
   # NOTE: --rollout-shuffle is intentionally OMITTED. prepare_data.py already
   # (a) shuffles within each data_source and (b) interleaves the two sources so
   # every contiguous 64-sample window (== one DP rank's per-step slice) holds
   # both text and image samples. --rollout-shuffle re-permutes the whole epoch
   # and destroys that interleaving, letting a DP-local step draw zero image
   # samples — which leaves the student's vision-encoder params with no gradient
   # and crashes Megatron grad-sync ("0/98 params have grad available"). Keep the
   # deterministic interleaved order instead.

   --multimodal-keys '{"image":"images"}'

   --rm-type mopd
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 128
   --n-samples-per-prompt 1
   --rollout-max-prompt-len 2048
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128

   --log-passrate
   --use-fault-tolerance
   --use-streaming-dataset
)

# Teacher routes: data_source (stamped by prepare_data.py) -> HF checkpoint path.
TEACHER_ROUTES="{\"dapo-math-17k\":\"${MODEL_DIR}/${TEXT_TEACHER_MODEL_NAME}/\",\"multimodal-open-r1\":\"${MODEL_DIR}/${VL_TEACHER_MODEL_NAME}/\"}"

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-only-reward
   --opd-kl-coef 1.0
   --opd-teacher-key data_source
   --opd-teacher-routes "${TEACHER_ROUTES}"
   --teacher-num-gpus-per-engine "${TEACHER_NUM_GPUS_PER_ENGINE}"
   --teacher-sglang-mem-fraction-static "${TEACHER_MEM_FRACTION:-0.7}"
   --teacher-sglang-chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
   --teacher-sglang-max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-128}"
   --teacher-sglang-disable-cuda-graph
   --opd-token-selection student_sampled
   --opd-log-prob-min-clamp -10.0
   --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-300}"
   --opd-teacher-image-key images
   --use-rollout-logprobs
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-distributed-optimizer
   # --overlap-grad-reduce / --overlap-param-gather are intentionally DISABLED.
   # With overlap on, Megatron fires each grad bucket's async all-reduce only once
   # EVERY param in that bucket has a gradient (via backward hooks). This student is
   # multimodal: its vision-encoder params (the "98-param" bucket) get no gradient on
   # steps whose batch happens not to exercise the vision tower, so that bucket never
   # completes, the all-reduce is never issued, and finish_grad_sync asserts
   # "Communication call has not been issued for this bucket (0/98 params have grad
   # available)" — a data-dependent crash at random steps (observed at steps 3/8/1).
   # Without overlap, finish_grad_sync issues a synchronous all-reduce for every
   # bucket unconditionally (see Megatron param_and_grad_buffer.py:682), so a
   # conditionally-unused module is harmless (its grad is simply zero that step).
   --accumulate-allreduce-grads-in-fp32
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXPERIMENT_NAME}"
)

EVAL_ARGS=(
   --skip-eval-before-train
   --eval-prompt-data mopd-9b "${EVAL_SET}"
   --eval-global-batch-size 128
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

# Student colocate on the full actor pool: TP=4, PP=1, DP=2 (actor 8 GPU).
# Mirrors the proven text recipe scripts/training/text/run-qwen35-9B-8xgpu.sh.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --log-probs-max-tokens-per-gpu 40960
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
   --sglang-disable-cuda-graph
)

RESOURCE_JSON="{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}"

python3 -m relax.entrypoints.train \
    --resource "${RESOURCE_JSON}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --offload \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "mopd-qwen35-9b-${now}.log"
