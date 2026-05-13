#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# GLM5-744B-A40B 128xGPU colocate training script.
#
# Usage:
#   bash scripts/training/text/run-glm5-744B-A40B-128xgpu.sh

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
source "${MODEL_CONFIG_DIR}/glm5-744B-A40B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"



CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/GLM-5/
   --ref-load ${EXP_DIR}/GLM-5/
   --megatron-to-hf-mode bridge
   # --load ${EXP_DIR}/GLM_ckpt/
   --save ${EXP_DIR}/GLM_ckpt/
   --save-interval 50
   --no-save-optim
   --no-save-rng
   --no-load-optim
   --no-load-rng
)

PROMPT_SET=${EXP_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1
   --global-batch-size 64
   --balance-data
   --use-fault-tolerance
   --rollout-health-check-timeout 120
)

EVAL_ARGS=(
   --skip-eval-before-train
   --log-passrate
   --eval-interval 20
   --eval-prompt-data aime ${EXP_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 32768
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --decoder-last-pipeline-num-layers 18
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1

   # Enable Context Parallelism for longer sequences (requires fused DSAMLASelfAttention)
   --context-parallel-size 2

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
   --data-pad-size-multiplier 4096
   --log-probs-chunk-size 1024

   # use deepep for megatron
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --moe-router-dtype fp32
   --calculate-per-token-loss

   # --use-pytorch-profiler
   # --profile-step-start 1
   # --profile-step-end 2
   # --profile-with-stack
   # --tensorboard-dir /tmp/tensorboard/
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
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
   # --no-rope-fusion
   --no-pin-cpu-grads
   --no-pin-cpu-params
)


SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 64
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-ep-size 64
   --sglang-dp-size 64
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head

   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode auto
   --sglang-load-format dummy

   # # mtp
   # --sglang-speculative-algorithm EAGLE
   # --sglang-speculative-num-steps 3
   # --sglang-speculative-eagle-topk 1
   # --sglang-speculative-num-draft-tokens 4

   # dsa
   --sglang-page-size 64
   --sglang-nsa-decode-backend flashmla_sparse
   --sglang-nsa-prefill-backend flashmla_sparse
   --sglang-attention-backend nsa
   --sglang-cuda-graph-max-bs 8

   --sglang-max-running-requests 512
   --sglang-chunked-prefill-size 131072
   --sglang-watchdog-timeout 3600
)


WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name GLM5-744B-A40B-128xgpu-${now}
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
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
   --update-weight-buffer-size $(( 1024 * 1024 * 1024 )) \

)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 128], "rollout": [1, 128]}'\
   --max-staleness 0 \
   --num-data-storage-units 16 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/GLM-5-744B-A40B-GRPO-gpu128-${now}.log
