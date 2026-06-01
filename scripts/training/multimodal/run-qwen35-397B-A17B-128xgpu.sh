#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-397B-A17B 128xGPU (16-node) fully sync training script for multimodal-open-r1 dataset.
#
# Usage:
#   bash scripts/training/multimodal/run-qwen35-397B-A17B-128xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-397B-A17B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/openr1mm}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-397B-A17B/
   --ref-load ${MODEL_DIR}/Qwen3.5-397B-A17B/
   --megatron-to-hf-mode bridge

   --load ${EXP_DIR}/Qwen3.5-397B-A17B_mcore_128xgpu/
   --save ${EXP_DIR}/Qwen3.5-397B-A17B_mcore_128xgpu/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
   --no-save-optim
   --no-save-rng
   --no-load-optim
   --no-load-rng
)

PROMPT_SET=${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet
SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type openr1mm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-max-prompt-len 2048
   --rollout-temperature 1
   --global-batch-size 256
   --use-fault-tolerance
   --balance-data
   --rollout-health-check-timeout 120
   --system-prompt "${SYSTEM_PROMPT}"
   --multimodal-keys '{"image":"image"}'
   --use-streaming-dataset
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 4
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1
   --decoder-first-pipeline-num-layers 11
   --decoder-last-pipeline-num-layers 5

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-distributed-optimizer

   --calculate-per-token-loss
   --use-dynamic-batch-size
   --vision-dp-when-cp
   --vision-dp-when-tp
   --max-tokens-per-gpu 16384
   --log-probs-max-tokens-per-gpu 32768

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
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
   --lr 5e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   # NOTE(wuhuan): to avoid algorithm performance degradation
   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
   --update-weight-buffer-size $(( 4 * 512 * 1024 * 1024 )) \

)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
   --sglang-mem-fraction-static 0.7
   # dp attention
   --sglang-enable-dp-attention
   --sglang-dp-size 32
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
   --sglang-ep-size 32
   --sglang-load-format dummy

   --sglang-cuda-graph-max-bs 8
   --sglang-server-concurrency 1024
   --sglang-watchdog-timeout 3600
   --sglang-enable-nan-detection
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen35-397B-A17B-128x-sync-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # gated delta net does not support flash attention backend
   --attention-backend flash
)
RUNTIME_ENV_JSON=$(python3 -c '
import json, os
d = json.loads(os.environ["RUNTIME_ENV_JSON"])
d.setdefault("env_vars", {}).update({
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
})
print(json.dumps(d))
')
export RUNTIME_ENV_JSON


mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 128], "rollout": [1, 128]}' \
   --num-data-storage-units 16 \
   --colocate \
   --max-staleness 0 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-397B-A17B-MM-GRPO-gpu128-sync-${now}.log
