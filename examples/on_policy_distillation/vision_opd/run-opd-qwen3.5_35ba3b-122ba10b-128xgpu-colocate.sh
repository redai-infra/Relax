#!/bin/bash
# =============================================================================
# 普通 OPD（无特权）· 16 机 128 卡 · colocate-sync · 学生 Qwen3.5-35B-A3B (VLM) / 教师 Qwen3.5-122B-A10B (VLM)
# ---------------------------------------------------------------------------
# 与同目录的 run-vision-opd-*.sh（特权 OPSD / Vision-OPD）区别：
#   * Vision-OPD：teacher 看 bbox 裁剪图（特权视图），靠 --opd-teacher-image-key bbox_images。
#   * 本脚本（普通 OPD）：teacher **无特权**，看与 student 完全相同的文本和图片。
#     因此 **不设** 任何 teacher 特权字段（--opd-teacher-image-key / --opd-teacher-prompt-key）。
#     此时 OPD rollout 自动回退到 sample.tokens（student 的完整 prompt token，含图），
#     teacher 与 student 输入逐 token 对齐 —— 这才是标准单 teacher OPD。
#
# 数据：用 dataset/prepare_data.py 处理问一问 SFT 对话得到的 train.relax.jsonl，
#       schema 为 {messages, images, label, metadata}（无 bbox_images）。
#       数据集含部分纯文本样本（images: []），prepare_data.py 已校验
#       <image> 占位符数 == images 数，纯文本样本可正常处理。
#
# Colocate topology:
#   - rollout phase: rollout 占 bundle 0..119，teacher 占 bundle 120..127，并行
#   - actor train  : rollout + teacher offload，actor 在全部 128 卡上训练
#   - 三者共享同一个 128-bundle Placement Group, --colocate --max-staleness 0
#
# 启动（128 卡 = 16 节点 × 8 卡，多节点必须设 WORLD_SIZE=16）：
#   export WORLD_SIZE=16
#   export MODEL_DIR=... DATA_DIR=...
#   bash scripts/entrypoint/spmd-multinode.sh \
#     examples/on_policy_distillation/vision_opd/run-opd-qwen3.5_35ba3b-122ba10b-128xgpu_colocate.sh
# =============================================================================

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
export RELAX_OPD_PREEXPANDED_PATCH=1

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/recipes/opd}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"
# OPD_PRESET: student_sampled_reverse_kl_adv | teacher_topk_jsd_loss | student_topk_forward_kl_loss
OPD_PRESET="${OPD_PRESET:-student_sampled_reverse_kl_adv}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.6}"

echo "EXP_DIR: ${EXP_DIR}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "DATA_DIR: ${DATA_DIR}"
echo "OPD_PRESET: ${OPD_PRESET}"
echo "TEACHER_MEM_FRACTION: ${TEACHER_MEM_FRACTION}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-35B-A3B}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen3.5-122B-A10B}"

ROLLOUT_GPUS="${ROLLOUT_GPUS:-64}"
TEACHER_GPUS="${TEACHER_GPUS:-64}"
ACTOR_GPUS="${ACTOR_GPUS:-128}"

SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/opd-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-v0}"
mkdir -p "${SAVE_DIR}"
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --load ${SAVE_DIR}
   --save-interval 50
   --max-actor-ckpt-to-keep 2
)

# 由 dataset/prepare_data.py 生成的普通 OPD 数据（schema: messages/images/label/metadata）
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/train.relax.jsonl}"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key messages
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle

   # student 与 teacher 都看 images（同图，无特权）。
   --multimodal-keys '{"image":"images"}'
   --image-min-token-num 64                  # = shortest_edge / 32²
   --image-max-token-num ${IMG_MAX_TOKEN:-16384}               # = longest_edge  / 32²

   # pure OPSD：reward 被 --opd-disable-rl-reward 清零，但 RewardWorker 不支持 rm_type='none'；
   # random 不依赖 label 内容，做占位。
   --rm-type random

   --num-rollout              ${NUM_ROLLOUT}
   --rollout-batch-size       32
   --n-samples-per-prompt     8
   --rollout-max-prompt-len   ${ROLLOUT_MAX_PROMP:-16384}
   --rollout-max-response-len 1024
   --rollout-temperature      1

   --rollout-result-dir ${SAVE_DIR}/opd-traces-${now}

   --global-batch-size 256
   --use-fault-tolerance
   --use-streaming-dataset
   --balance-data
   --mm-processor-pool-size ${MM_PROCESSOR_POOL_SIZE:-32}
   --rollout-health-check-interval ${ROLLOUT_HEALTH_CHECK_INTERVAL:-60}
   --rollout-health-check-timeout ${ROLLOUT_HEALTH_CHECK_TIMEOUT:-120}
   --rollout-health-check-first-wait ${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-120}
   --rollout-health-check-max-consecutive-failures ${ROLLOUT_HEALTH_CHECK_MAX_CONSECUTIVE_FAILURES:-5}
)


OPD_ARGS=(
   --use-opd
   --opd-type sglang

   --teacher-hf-checkpoint ${MODEL_DIR}/${TEACHER_MODEL_NAME}/
   --warm-hf-checkpoint-page-cache

   --teacher-sglang-mem-fraction-static ${TEACHER_MEM_FRACTION}
   --teacher-sglang-chunked-prefill-size ${TEACHER_CHUNKED_PREFILL_SIZE:-8192}
   --teacher-sglang-max-running-requests ${TEACHER_MAX_RUNNING_REQUESTS:-64}
   --teacher-sglang-disable-cuda-graph
   --teacher-sglang-max-prefill-tokens 16384
   --teacher-num-gpus-per-engine 8
)

case "${OPD_PRESET}" in
   student_sampled_reverse_kl_adv)
      OPD_ARGS+=(
         --opd-kl-coef 1.0
         --opd-loss-coef 0.0
         --opd-kl-type reverse_kl
         --opd-token-selection student_sampled
         --use-rollout-logprobs
      )
      ;;
   teacher_topk_jsd_loss)
      OPD_ARGS+=(
         --opd-kl-coef 0.0
         --opd-loss-coef 1.0
         --opd-kl-type jsd
         --opd-jsd-alpha 0.5
         --opd-token-selection teacher_topk
         --opd-log-prob-top-k 100
      )
      ;;
   student_topk_forward_kl_loss)
      OPD_ARGS+=(
         --opd-kl-coef 0.0
         --opd-loss-coef 1.0
         --opd-kl-type forward_kl
         --opd-token-selection student_topk
         --opd-log-prob-top-k 16
      )
      ;;
   *)
      echo "Unknown OPD_PRESET: ${OPD_PRESET}" >&2
      echo "Supported OPD_PRESET values: student_sampled_reverse_kl_adv, teacher_topk_jsd_loss, student_topk_forward_kl_loss" >&2
      exit 1
      ;;
esac

OPD_ARGS+=(
   # 关闭 base RL outcome reward → pure OPSD
   --opd-disable-rl-reward

   # 普通 OPD：teacher 无特权，不设 --opd-teacher-image-key / --opd-teacher-prompt-key。
   # teacher 自动复用 student 的完整 prompt token（同文本同图）。

   --opd-is-clip              2.0
   --opd-teacher-timeout-s    6000
)

EVAL_ARGS=()

GRPO_ARGS=(
   # pure OPSD 下 advantages 被清零，下面 PPO 相关参数实际是 no-op，仅保留默认。
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.3
   # --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --lr-warmup-iters 10

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer

   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${ACTOR_MAX_TOKENS_PER_GPU:-20480}
)


SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static ${STUDENT_MEM_FRACTION:-0.7}
   --sglang-load-format dummy
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-enable-weights-cpu-backup
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name    ${PROJECT_NAME}
   --tb-experiment-name opd-128xgpu-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-colocate-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

mkdir -p log

if [ -z "${RAY_DASHBOARD:-}" ]; then
    if [ -n "${RAY_ADDRESS:-}" ]; then
        RAY_DASHBOARD="http://${RAY_ADDRESS%%:*}:8265"
    else
        RAY_DASHBOARD="http://${HOST_IP:-127.0.0.1}:8265"
    fi
fi

# Resource:
#   actor   128 卡 -> 整个 128-bundle shared PG
#   rollout 120 卡 -> 自动取 PG 的 bundle 0..119
#   teacher   8 卡 -> 框架内 slice PG 的 bundle 120..127
# Constraint: actor[1] >= rollout[1] + teacher[1] = 128.
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}" \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPD_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   2>&1 | tee log/opd-128xgpu-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-colocate-${now}.log
