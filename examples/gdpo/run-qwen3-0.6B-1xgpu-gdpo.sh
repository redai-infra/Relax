#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B single-GPU GDPO training on GSM8K.
#
# GDPO (arXiv 2601.05242) standardizes each reward component within its prompt
# group before combining them. That keeps two things GRPO's single
# standardization of the summed reward loses: the relative strength between
# groups (per-group standardization forces every group to unit variance), and
# the balance between components of very different scale. It does NOT rescue a
# group whose components sum to a constant -- those cancel to zero under equal
# weights, same as GRPO. See examples/gdpo/README.md.
# The reward function in reward_gdpo.py returns both components;
# --gdpo-reward-keys names them.
#
# Usage:
#   bash examples/gdpo/run-qwen3-0.6B-1xgpu-gdpo.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=relax-gdpo}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=20}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-0.6B
   --ref-load ${MODEL_DIR}/Qwen3-0.6B
   --load ${EXP_DIR}/Qwen3-0.6B_mcore_gdpo/
   --save ${EXP_DIR}/Qwen3-0.6B_mcore_gdpo/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
   --megatron-to-hf-mode bridge
)

# NOTE: the format instruction belongs in the prompt text, not in
# --system-prompt. relax/utils/data/data_utils.py:181 builds the system message
# with multimodal list content (`content: [{"type": "text", ...}]`), which a
# text-only chat template such as Qwen3-0.6B's cannot render:
#   TypeError: can only concatenate str (not "list") to str
# Prepare the dataset so each question already asks for the <think>/<answer>
# tags; see examples/gdpo/README.md.
ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/gsm8k/train.jsonl
   --input-key question
   --label-key answer
   --apply-chat-template
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1.0
   # 4 * 8 == 32 just keeps this example minimal. GDPO's step 3 stays aligned
   # with Eq. 6 regardless: _whiten_by_segment splits any merged rollout back
   # into per-optimizer-batch segments and whitens each on its own, so
   # num_rollout_minis > 1 does NOT require rollout_batch_size * n == gbs.
   --global-batch-size 32
)

# The two reward components come from examples/gdpo/reward_gdpo.py.
# --reward-key selects the scalar used for metrics and the raw_reward column;
# --gdpo-reward-keys names the components GDPO standardizes independently.
GDPO_ARGS=(
   --advantage-estimator gdpo
   --gdpo-reward-keys correctness format
   --gdpo-reward-weights 1.0 1.0
   --custom-rm-path examples.gdpo.reward_gdpo.reward_func
   --reward-key score
   --eps-clip 0.2
   --kl-coef 0.00
   --entropy-coef 0.00
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

WANDB_ARGS=(
   --use-tensorboard
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-0.6b-gdpo-gpu1-${now}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    --balance-data \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GDPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee log/qwen3-0.6b-gdpo-gpu1-${now}.log
