#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-Omni-30B-A3B 16xGPU (2-node) colocate training script.
#
# Usage:
#   bash scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-omni-30B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/omni}"
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3-Omni-30B-A3B-Instruct
   --ref-load ${EXP_DIR}/Qwen3-Omni-30B-A3B-Instruct
   --megatron-to-hf-mode bridge
)

SYSTEM_PROMPT="Please think about this question as if you were a human pondering deeply, carefully considering both the visual and audio information before answering, engaging in an internal dialogue using expressions such as let me think, wait, hmm, oh I see, or let's break it down, including self-reflection or verification in the reasoning process, providing the detailed reasoning between the <think> </think> tags, and finally giving only the single option letter (e.g., A, B, C, D, etc.) as the final answer within the <answer> </answer> tags."

PROMPT_SET=${EXP_DIR}/AVQA/AVQA_R1/train/omni_rl_format_train_convert.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type multiple_choice
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   # --rollout-max-prompt-len 2048
   --rollout-temperature 0.8
   --global-batch-size 512
   --balance-data
   --use-fault-tolerance
   --system-prompt "${SYSTEM_PROMPT}"
   --multimodal-keys '{"image":"image","audio":"audio"}'
)

EVAL_ARGS=(
   --log-passrate
   --eval-interval 20
   --eval-prompt-data avqa ${EXP_DIR}/AVQA/AVQA_R1/valid/small_valid.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 2048
   --eval-top-p 0.7
)

PERF_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1
   --micro-batch-size 2 # avoid OOM
   # --use-dynamic-batch-size
   # --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --no-rope-fusion
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen3-omni-30B-image_audio-gpu16-${now}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address=${RAY_ADDRESS:-"http://127.0.0.1:8265"} \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 16], "rollout": [1, 16]}'\
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-omni-30B-image_audio-gpu16-${now}.log
