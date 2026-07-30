#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -ex
set -o pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPU=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)

###############################################################################
#                                 ENVIRONMENT                                 #
###############################################################################

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

# DeepEyes sends pre-tokenized input_ids that SGLang's stock Qwen-VL processor
# mishandles; --deepeyes-qwen-vl-patch applies the runtime monkey-patch at
# SGLang engine start-up (replaces the old ``cp qwen_vl.py /sgl-workspace/...``
# overlay). Default off (stock SGLang behavior). Set DEEPEYES_PATCH=1 to enable
# the patch — needed for correct DeepEyes multi-turn rollout (pre-tokenized
# input_ids). See relax/backends/sglang/patches/qwen_vl_patch.py.
DEEPEYES_PATCH="${DEEPEYES_PATCH:-0}"

###############################################################################
#                                    DIRS                                     #
###############################################################################

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/deepeyes_r3}"
EXP_NAME="qwen3vl-4B-deepeyes-r3-${TIMESTAMP}"

# Require MODEL_DIR, DATA_DIR, SAVE_DIR from environment or set defaults
MODEL_DIR="${MODEL_DIR:=/apdcephfs_hldy/share_304318596/weiyangguo/models}"
DATA_DIR="${DATA_DIR:=/apdcephfs/private_weiyangguo/Agent-Omni/Relax-Task/Relax/datasets}"
SAVE_DIR="${SAVE_DIR:=/apdcephfs/private_weiyangguo/Agent-Omni/Relax-Task/Relax/outputs}"
mkdir -p ${SAVE_DIR}

###############################################################################
#                              JUDGE MODEL API                                #
###############################################################################

# LLM judge: source the remote OpenAI-compatible judge service (no local sglang
# server). Override DEEPEYES_JUDGE_BASE_URL / DEEPEYES_JUDGE_API_KEY /
# DEEPEYES_JUDGE_MODELS before running to point at a different endpoint.
# Fall back to the local sglang judge by sourcing sglang_judge_service.sh instead.
source "${SCRIPT_DIR}/openai_judge_service.sh"

###############################################################################
#                                  MODEL CONFIG                               #
###############################################################################

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-4B-Instruct
    --ref-load ${MODEL_DIR}/Qwen3-VL-4B-Instruct
    --save ${SAVE_DIR}/Qwen3-VL-4B-Instruct-Checkpoint
    --megatron-to-hf-mode bridge
    --save-interval 100
    --max-actor-ckpt-to-keep 3
    # --load ${SAVE_DIR}/Qwen3-VL-4B-Instruct-Checkpoint
)

###############################################################################
#                                  DATASETS                                   #
###############################################################################

TRAIN_FILES=(
    "'${DATA_DIR}/deepeyes-v1/data_0.1.2_visual_toolbox_v2.parquet@[0:5000]'"
    "'${DATA_DIR}/deepeyes-v1/data_v0.8_visual_toolbox_v2.parquet@[0:5000]'"
)
TEST_FILES=("${DATA_DIR}/deepeyes-v1/data_thinklite_reasoning_acc.parquet@[0:256]")
PROMPT_SET="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

###############################################################################
#                               ROLLOUT CONFIG                                #
###############################################################################

NUM_ROLLOUT="${NUM_ROLLOUT:=2000}"

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key reward_model
    --multimodal-keys '{"image":"images"}'
    --reward-key score
    --metadata-key extra_info
    --apply-chat-template
    --custom-generate-function-path examples.deepeyes.rollout.generate
    --custom-rm-path examples.deepeyes.reward_deepeyes.reward_func
    --custom-config-path examples/deepeyes/deepeyes_config.yaml
    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size 32
    --micro-batch-size 1
    --n-samples-per-prompt 8
    --rollout-max-response-len 2048
    --rollout-max-prompt-len 2048
    --rollout-temperature 1
    --global-batch-size 256
    --use-fault-tolerance
    --rollout-shuffle
)
###############################################################################
#                                EVAL CONFIG                                  #
###############################################################################

EVAL_ARGS=(
    --skip-eval-before-train
    --eval-interval 100
    --eval-prompt-data vstar ${TEST_FILES}
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 2048
    --eval-top-p 0.7
)

###############################################################################
#                              ALGORITHM CONFIG                               #
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --eps-clip-c 3
    --use-tis
)

###############################################################################
#                              OPTIMIZER CONFIG                               #
###############################################################################

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

###############################################################################
#                               SGLANG CONFIG                                 #
###############################################################################

SGLANG_ARGS=(
    --sglang-mem-fraction-static 0.8
)

# Apply the DeepEyes Qwen-VL processor patch unless DEEPEYES_PATCH=0.
if [ "${DEEPEYES_PATCH}" = "1" ]; then
    SGLANG_ARGS+=(--deepeyes-qwen-vl-patch)
fi

###############################################################################
#                               LOGGING CONFIG                                #
###############################################################################

LOG_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
    # SwanLab: opt-in via USE_SWANLAB=1 (see the USE_SWANLAB block below).
    # Requires `pip install swanlab` (and `swanlab[dashboard]` for local mode).
    # --dump-details dump_details_8k_0204
    # --use-wandb
    # --wandb-project slime-dev
    # --wandb-group qwen3-4B-test
    # --wandb-key ${WANDB_KEY}
)

# SwanLab: opt-in via USE_SWANLAB=1. Requires `pip install swanlab` in the image.
# Override project/run name via SWANLAB_PROJECT / SWANLAB_EXPERIMENT_NAME; api key
# via SWANLAB_API_KEY (or let swanlab read it from env).
USE_SWANLAB="${USE_SWANLAB:-0}"
if [ "${USE_SWANLAB}" = "1" ]; then
    LOG_ARGS+=(
        --use-swanlab
        --swanlab-project "${SWANLAB_PROJECT:-${PROJECT_NAME}}"
        --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-${EXP_NAME}}"
    )
    if [ -n "${SWANLAB_API_KEY:-}" ]; then
        LOG_ARGS+=(--swanlab-api-key "${SWANLAB_API_KEY}")
    fi
    if [ -n "${SWANLAB_MODE:-}" ]; then
        LOG_ARGS+=(--swanlab-mode "${SWANLAB_MODE}")
    fi
fi

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################

MEGATRON_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --max-tokens-per-gpu 8192
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-dynamic-batch-size
)

###############################################################################
#                              RESOURCE CONFIG                                #
###############################################################################

RAY_RESOURCE_ARGS=(
    --rollout-num-gpus-per-engine 1
    --resource '{"actor": [1, 4], "rollout": [1, 4]}'
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
)

###############################################################################
#                                 LAUNCH JOB                                  #
###############################################################################

mkdir -p logs

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    -- python3 relax/entrypoints/train.py \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee logs/${EXP_NAME}.log
