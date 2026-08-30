#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source scripts/models/qwen3-8B.sh
source examples/on_policy_distillation/sdpo/env.sh

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

export RELAX_OPD_PER_POS_TOKEN_IDS=1

student_model="${STUDENT_MODEL_PATH:?Set STUDENT_MODEL_PATH}"
teacher_model="${TEACHER_MODEL_PATH:-${student_model}}"
data_path="${DATA_PATH:-${SDPO_DATA_ROOT:?Set SDPO_DATA_ROOT}/toolalpaca/train.jsonl}"
eval_path="${EVAL_PATH:-${SDPO_DATA_ROOT:?Set SDPO_DATA_ROOT}/toolalpaca/eval.jsonl}"
now="$(date '+%Y-%m-%d-%H:%M:%S')"
experiment_name="${EXPERIMENT_NAME:-sdpo-toolalpaca-${now}}"

CKPT_ARGS=(
    --hf-checkpoint "${student_model}"
    --megatron-to-hf-mode bridge
    --attention-backend flash
)

ROLLOUT_ARGS=(
    --prompt-data "${data_path}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --apply-chat-template
    --group-rm
    --custom-rm-path examples.on_policy_distillation.sdpo.reward.score
    --reward-key score
    --num-rollout 5000
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --global-batch-size 256
    --rollout-max-prompt-len 2048
    --rollout-max-response-len 8192
    --rollout-temperature 1.0
    --use-fault-tolerance
)

EVAL_ARGS=(
    --eval-interval 5
    --eval-prompt-data toolalpaca "${eval_path}"
    --n-samples-per-eval-prompt 16
)

OPD_ARGS=(
    --use-opd
    --opd-feedback-class "relax.utils.opd.sdpo.feedback.GoldenAnswerSDPOFeedback"
    --opd-type sglang
    --teacher-hf-checkpoint "${teacher_model}"
    --teacher-num-gpus-per-engine 1
    --teacher-sglang-mem-fraction-static 0.5
    --teacher-sglang-enable-weights-cpu-backup
    --opd-loss-coef 1.0
    --opd-kl-coef 0.0
    --opd-disable-rl-reward
    --opd-token-selection student_topk
    --opd-log-prob-top-k 16
    --opd-kl-type jsd
    --opd-jsd-alpha 0.5
    --opd-norm-mode tail
    --opd-teacher-timeout-s 600
    --use-rollout-logprobs
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --clip-grad 1.0
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --context-parallel-size 1
    --pipeline-model-parallel-size 1
    --calculate-per-token-loss
    --no-masked-softmax-fusion
    --optimizer-cpu-offload
    --selective-offload
    --use-precision-aware-optimizer
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --qkv-format bshd
    --micro-batch-size 1
)

SGLANG_ARGS=(
    --rollout-num-gpus 3
    --rollout-num-gpus-per-engine 1
    --sglang-load-format dummy
    --sglang-mem-fraction-static 0.45
)

MISC_ARGS=(
    --resource '{"actor": [1, 4], "rollout": [1, 3], "teacher": [1, 1]}'
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
    --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}'
    --use-health-check
    --actor-num-gpus-per-node 4
    --num-gpus-per-node 4
    --tb-experiment-name "${experiment_name}"
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project relax-sdpo
    --wandb-group "${WANDB_RUN_GROUP:-sdpo-toolalpaca}"
    --wandb-key "${WANDB_API_KEY}"
)

exec python -m relax.entrypoints.train \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${EVAL_ARGS[@]}" \
    "${OPD_ARGS[@]}" "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
