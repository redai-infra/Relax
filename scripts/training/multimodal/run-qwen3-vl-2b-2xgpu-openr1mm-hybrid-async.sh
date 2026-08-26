set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-2B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/hybrid_openr1mm_2gpu}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../exps}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/../../../model}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../../../data}"
NUM_ROLLOUT="${NUM_ROLLOUT:=20}"


CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-2B-Instruct
   --ref-load ${MODEL_DIR}/Qwen3-VL-2B-Instruct
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save ${EXP_DIR}/Qwen3-VL-2B-Instruct_mcore_2xgpu/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001-converted.parquet

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type openr1mm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 2048
   --rollout-max-prompt-len 1024
   --rollout-max-context-len 3072
   --rollout-temperature 0.8
   --global-batch-size 32
   --multimodal-keys '{"image":"image"}'
   --system-prompt "${SYSTEM_PROMPT}"
   --use-streaming-dataset
)

# Same per-GPU config as the 1xGPU colocate baseline: TP=1 PP=1 CP=1.
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --calculate-per-token-loss
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --no-rope-fusion
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
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
   --clip-grad 1.0
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-clearml
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-vl-2b-GRPO-gpu2-hybrid-async-${now}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
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
     --resource '{"actor": [1, 1], "rollout": [1, 1]}'\
     --max-staleness 2 \
     --num-data-storage-units 1 \
     --num-iters-per-train-update 2 \
     --balance-data \
     --hybrid \
     "${MODEL_ARGS[@]}" \
     "${CKPT_ARGS[@]}" \
     "${ROLLOUT_ARGS[@]}" \
     "${OPTIMIZER_ARGS[@]}" \
     "${GRPO_ARGS[@]}" \
     "${WANDB_ARGS[@]}" \
     "${PERF_ARGS[@]}" \
     "${SGLANG_ARGS[@]}" \
     "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-vl-2b-GRPO-gpu2-hybrid-async-${now}.log
