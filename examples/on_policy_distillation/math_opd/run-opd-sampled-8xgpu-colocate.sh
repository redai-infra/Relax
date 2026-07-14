

#!/bin/bash
set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
# # topk 
# export RELAX_OPD_PER_POS_TOKEN_IDS=1 
now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/recipes/opd_baseline}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=50}"

ROLLOUT_BATCH_SIZE=1024          
ROLLOUT_N_GROUPS=1              
ROLLOUT_RESP_LENGTH=16384     
EVAL_ROLLOUT_RESP_LENGTH=16384   

OPD_KL_COEF=1.0
OPD_LOSS_COEF=0.0
OPD_KL_TYPE=reverse_kl
OPD_TOKEN_SELECTION=student_sampled

STUDENT_MODEL_NAME=Qwen3-4B
TEACHER_MODEL_NAME=Qwen3-4B-Non-Thinking-RL-Math-Step500

ROLLOUT_GPUS=4
TEACHER_GPUS=4
ACTOR_GPUS=8

EXP_NAME=opd-baseline-sampled-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-opd_${OPD_KL_TYPE}_${OPD_TOKEN_SELECTION}_klcoef${OPD_KL_COEF}_rollout_${ROLLOUT_BATCH_SIZE}_${ROLLOUT_N_GROUPS}_${ROLLOUT_RESP_LENGTH}_${ROLLOUT_TEMPERATURE}_${ROLLOUT_TOP_P}_eval_rollout_${EVAL_ROLLOUT_RESP_LENGTH}_${EVAL_ROLLOUT_TEMPERATURE}_${EVAL_ROLLOUT_TOP_P}-${now}
SAVE_DIR=${EXP_DIR}/save/${EXP_NAME}
mkdir -p "${SAVE_DIR}"
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --save-interval 2000
)

PROMPT_SET=${DATA_DIR}/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.jsonl
ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking": false}'
   --rollout-shuffle

   --rm-type math

   --num-rollout              ${NUM_ROLLOUT}
   --rollout-batch-size       ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt     ${ROLLOUT_N_GROUPS}
   --rollout-max-prompt-len   2048
   --rollout-max-response-len ${ROLLOUT_RESP_LENGTH}
   --rollout-temperature      1.0
   --rollout-top-p 1.0
   --global-batch-size $((ROLLOUT_BATCH_SIZE * ROLLOUT_N_GROUPS))
   --use-fault-tolerance
   --use-streaming-dataset

   --rollout-health-check-interval 60
   --rollout-health-check-timeout 120
   --rollout-health-check-max-consecutive-failures 3
   --rollout-health-check-first-wait 300
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data aime2024 ${DATA_DIR}/G-OPD-Training-Data/AIME2024/test.jsonl
   --n-samples-per-eval-prompt 32
   --eval-max-response-len ${EVAL_ROLLOUT_RESP_LENGTH}
   --eval-top-p 1.0
   --eval-temperature 1
)

OPD_ARGS=(
   --use-opd
   --opd-type sglang

   --teacher-hf-checkpoint ${MODEL_DIR}/${TEACHER_MODEL_NAME}/
   --warm-hf-checkpoint-page-cache

   --teacher-sglang-mem-fraction-static 0.7
   --teacher-num-gpus-per-engine 4
   --teacher-sglang-disable-cuda-graph

   --opd-kl-coef ${OPD_KL_COEF}
   --opd-loss-coef ${OPD_LOSS_COEF}
   --opd-kl-type ${OPD_KL_TYPE}
   --opd-token-selection ${OPD_TOKEN_SELECTION}

   --opd-teacher-timeout-s ${OPD_TEACHER_TIMEOUT_S:-6000}
   --use-rollout-logprobs
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.3
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE:-2}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --calculate-per-token-loss
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${ACTOR_MAX_TOKENS_PER_GPU:-16384}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
   --sglang-mem-fraction-static ${STUDENT_MEM_FRACTION:-0.7}
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-max-running-requests 64
)


WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name    ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}
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
   2>&1 | tee log/opd-baseline-sampled-8xgpu-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-colocate-${now}.log
