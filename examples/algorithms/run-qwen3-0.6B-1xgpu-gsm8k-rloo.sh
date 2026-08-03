#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B 1xGPU RLOO training script on GSM8K (colocate mode).
#
# RLOO (REINFORCE Leave-One-Out, https://arxiv.org/abs/2402.14740) replaces
# GRPO's whole-group baseline with a leave-one-out baseline:
#
#     A_i = R_i - mean(R_j for j != i)
#
# The baseline no longer depends on the sample it scores, which makes it
# unbiased, and no standard-deviation scaling is applied. Everything else
# matches GRPO *except the objective*: RLOO uses unclipped REINFORCE
#   L_i = -sg(A_i) * log pi(y_i)
# rather than PPO-Clip, so --eps-clip does not apply.
#
# One optimizer step per rollout is deliberate: RLOO has no importance-ratio
# correction, so it assumes the sampling policy equals the training policy.
#
# Usage:
#   MODEL_DIR=/path/to/models DATA_DIR=/path/to/data \
#     bash examples/algorithms/run-qwen3-0.6B-1xgpu-gsm8k-rloo.sh
#
# train_iters = NUM_ROLLOUT x ROLLOUT_BATCH_SIZE x N_SAMPLES / GLOBAL_BATCH_SIZE

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/rloo}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"

# openai/gsm8k stores the answer as "<full derivation> #### 72", but --rm-type math
# scores against the bare number. Normalize once into a side-by-side parquet;
# skipping this step silently yields a reward of 0 on every sample.
GSM8K_RAW="${GSM8K_RAW:-${DATA_DIR}/gsm8k/main/train-00000-of-00001.parquet}"
GSM8K_CLEAN="${GSM8K_CLEAN:-${DATA_DIR}/gsm8k/main/train_clean.parquet}"
if [ ! -f "${GSM8K_CLEAN}" ]; then
    python3 - <<EOF
import pandas as pd

df = pd.read_parquet("${GSM8K_RAW}")
df["answer"] = df["answer"].str.split("####").str[-1].str.strip()
df.to_parquet("${GSM8K_CLEAN}", index=False)
print(f"Wrote {len(df)} rows to ${GSM8K_CLEAN}")
EOF
fi

NUM_ROLLOUT="${NUM_ROLLOUT:=100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=4}"
N_SAMPLES="${N_SAMPLES:=8}"
# ROLLOUT_BATCH_SIZE * N_SAMPLES == GLOBAL_BATCH_SIZE, i.e. exactly ONE optimizer
# step per rollout. This is a correctness requirement for RLOO, not a tuning
# choice: RLOO is unclipped REINFORCE with no importance-ratio term, so a second
# step within the same rollout trains at updated weights against log-probs
# sampled from the old ones, with nothing correcting the mismatch. PPO-Clip
# tolerates this via the ratio; RLOO does not. Enforced at startup by
# validate_rloo_args, so overriding these to imply two steps fails fast.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:=32}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-0.6B
   --ref-load ${MODEL_DIR}/Qwen3-0.6B
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
   --prompt-data ${GSM8K_CLEAN}
   --input-key question
   --label-key answer
   --apply-chat-template
   --rollout-shuffle

   --rm-type math

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   # RLOO needs at least 2 samples per prompt to form a leave-one-out baseline;
   # a group of 1 has no baseline and is assigned a zero advantage.
   --n-samples-per-prompt ${N_SAMPLES}
   --rollout-max-response-len 2048
   --rollout-temperature 1

   # Scaling to more data-parallel ranks: the mini rollout batch must divide
   # dp_size, so DP=8 needs --micro-batch-size 8 and --global-batch-size 64.
   # Measured on 8xH100; DP<=4 works with the defaults below.
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
   --use-fault-tolerance
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --calculate-per-token-loss
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 8192
)

RLOO_ARGS=(
   --advantage-estimator rloo
   # KL loss is computed but weighted at 0, so train/kl_loss stays observable as a
   # drift diagnostic without entering the objective. Raise the coefficient to use it.
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   # No --eps-clip: RLOO is unclipped REINFORCE, so the clip margins are inert
   # and train/pg_clipfrac stays 0. Keeping them here would only mislead.

   --use-rollout-logprobs
   # Reward normalization is on by default and is what builds the leave-one-out
   # baseline; --disable-rewards-normalization turns RLOO into plain REINFORCE.
   # --disable-grpo-std-normalization has no effect here: the std branch is
   # GRPO-family only, so there is nothing to disable for RLOO.
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # ~55% of HBM stays with training; SGLang takes the remaining 45%
   --sglang-mem-fraction-static 0.45
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-0.6b-RLOO-gsm8k-1xgpu-${now}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${RLOO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee log/qwen3-0.6b-RLOO-gsm8k-1xgpu-${now}.log
