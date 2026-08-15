#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
[ -f "${SCRIPT_DIR}/env.sh" ] && source "${SCRIPT_DIR}/env.sh"

METHOD="${METHOD:-graphgpo}"
case "${METHOD}" in
    grpo|gigpo|graphgpo) ;;
    *)
        echo "ERROR: METHOD must be grpo, gigpo, or graphgpo (got ${METHOD})." >&2
        exit 2
        ;;
esac
ENABLE_EVAL="${ENABLE_EVAL:-1}"
case "${ENABLE_EVAL}" in
    0|1) ;;
    *)
        echo "ERROR: ENABLE_EVAL must be 0 or 1 (got ${ENABLE_EVAL})." >&2
        exit 2
        ;;
esac
EPISODE_WEIGHTING="${EPISODE_WEIGHTING:-trajectory_once}"
case "${EPISODE_WEIGHTING}" in
    reference_cross_steps|trajectory_once) ;;
    *)
        echo "ERROR: EPISODE_WEIGHTING must be reference_cross_steps or trajectory_once (got ${EPISODE_WEIGHTING})." >&2
        exit 2
        ;;
esac

SEED="${SEED:-0}"
NUM_GPUS="${NUM_GPUS:-2}"
OUTER_EPOCHS="${OUTER_EPOCHS:-150}"
TASK_GROUPS="${TASK_GROUPS:-16}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.50}"
MODEL_REVISION="${MODEL_REVISION:-775b11afaf83e0dc75bd5abaf90133e47b3ec082}"
EXPECTED_EXTERNAL_IMAGE_DIGEST="${EXPECTED_EXTERNAL_IMAGE_DIGEST:-ghcr.io/redai-infra/relaxrl@sha256:3fa8ce578acda6c829b83016bde42c38fa892681e4f36ca330f545616fe578e2}"

positive_integer_values=(TASK_GROUPS GROUP_SIZE GLOBAL_BATCH_SIZE MAX_STEPS)
for name in "${positive_integer_values[@]}"; do
    if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer (got ${!name})." >&2
        exit 2
    fi
done
if [[ ! "${SGLANG_MEM_FRACTION_STATIC}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
    echo "ERROR: SGLANG_MEM_FRACTION_STATIC must be greater than 0 and at most 1 (got ${SGLANG_MEM_FRACTION_STATIC})." >&2
    exit 2
fi
ROLLOUT_SAMPLE_COUNT=$((TASK_GROUPS * GROUP_SIZE))
if ((ROLLOUT_SAMPLE_COUNT % GLOBAL_BATCH_SIZE != 0)); then
    echo "ERROR: TASK_GROUPS * GROUP_SIZE must be divisible by GLOBAL_BATCH_SIZE." >&2
    exit 2
fi

ALFWORLD_CONFIG_PATH="${ALFWORLD_CONFIG_PATH:-${SCRIPT_DIR}/configs/alfworld_qwen2_5_1_5b.yaml}"
DATA_ARTIFACT_DIR="${DATA_ARTIFACT_DIR:-}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ARTIFACT_DIR:+${DATA_ARTIFACT_DIR}/train.prompts.jsonl}}"
EVAL_DATA="${EVAL_DATA:-${DATA_ARTIFACT_DIR:+${DATA_ARTIFACT_DIR}/eval_in_distribution.prompts.jsonl}}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATA_ARTIFACT_DIR:+${DATA_ARTIFACT_DIR}/train.manifest.json}}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${DATA_ARTIFACT_DIR:+${DATA_ARTIFACT_DIR}/eval_in_distribution.manifest.json}}"
PREPARE_LOCK="${PREPARE_LOCK:-${DATA_ARTIFACT_DIR:+${DATA_ARTIFACT_DIR}/prepare.lock.json}}"
DEPENDENCY_LOCK="${DEPENDENCY_LOCK:-}"
MODEL_LOCK="${MODEL_LOCK:-}"
ALFWORLD_PYTHON="${ALFWORLD_PYTHON:-python3}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
SAVE_DIR="${SAVE_DIR:-}"
LOAD_DIR="${LOAD_DIR:-}"

if [ "${ENABLE_EVAL}" = "1" ] && { [ -z "${EVAL_DATA}" ] || [ -z "${EVAL_MANIFEST}" ]; }; then
    echo "ERROR: EVAL_DATA and EVAL_MANIFEST must be set when ENABLE_EVAL=1." >&2
    exit 2
fi

required_values=(
    ALFWORLD_DATA
    HF_CHECKPOINT
    SAVE_DIR
    TRAIN_DATA
    TRAIN_MANIFEST
    PREPARE_LOCK
    DEPENDENCY_LOCK
    MODEL_LOCK
)
for name in "${required_values[@]}"; do
    if [ -z "${!name:-}" ]; then
        echo "ERROR: ${name} must be set." >&2
        exit 2
    fi
done

required_files=(
    "${ALFWORLD_CONFIG_PATH}"
    "${TRAIN_DATA}"
    "${TRAIN_MANIFEST}"
    "${PREPARE_LOCK}"
    "${DEPENDENCY_LOCK}"
    "${MODEL_LOCK}"
)
if [ "${ENABLE_EVAL}" = "1" ]; then
    required_files+=("${EVAL_DATA}" "${EVAL_MANIFEST}")
fi
for path in "${required_files[@]}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}" >&2
        exit 2
    fi
done
if [ ! -d "${HF_CHECKPOINT}" ]; then
    echo "ERROR: HF_CHECKPOINT must be a local model snapshot directory: ${HF_CHECKPOINT}" >&2
    exit 2
fi
if [ -n "${LOAD_DIR}" ] && [ ! -f "${LOAD_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: LOAD_DIR must be a native checkpoint root with latest_checkpointed_iteration.txt: ${LOAD_DIR}" >&2
    exit 2
fi

PREFLIGHT_COMMAND=(
    python3 -m examples.graphgpo.preflight
    --prepare-lock "${PREPARE_LOCK}"
    --alfworld-data-root "${ALFWORLD_DATA}"
    --model-lock "${MODEL_LOCK}"
    --checkpoint "${HF_CHECKPOINT}"
    --max-steps "${MAX_STEPS}"
    --model-revision "${MODEL_REVISION}"
    --task-groups "${TASK_GROUPS}"
    --group-size "${GROUP_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --split-artifact train "${TRAIN_DATA}" "${TRAIN_MANIFEST}"
)
if [ "${ENABLE_EVAL}" = "1" ]; then
    PREFLIGHT_COMMAND+=(
        --split-artifact eval_in_distribution "${EVAL_DATA}" "${EVAL_MANIFEST}"
    )
fi
"${PREFLIGHT_COMMAND[@]}"

export GRAPHGPO_METHOD="${METHOD}"
export GRAPHGPO_EXPECTED_GROUP_SIZE="${GROUP_SIZE}"
export GRAPHGPO_OMEGA="${OMEGA:-0.1}"
export GRAPHGPO_GAMMA="${GAMMA:-0.95}"
export GRAPHGPO_BETA="${BETA:-1.0}"
export GRAPHGPO_BETA_EPISODE="${BETA_EPISODE:-1.0}"
export GRAPHGPO_EPISODE_WEIGHTING="${EPISODE_WEIGHTING}"
if [ -n "${GRAPHGPO_DIAGNOSTICS_JSONL:-}" ]; then
    export GRAPHGPO_DIAGNOSTICS_JSONL
fi
export RELAX_AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE="${MAX_STEPS}"

TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
EXP_NAME="graphgpo-${METHOD}-qwen2.5-1.5b-s${SEED}-${TIMESTAMP}"
RUN_DIR="${SAVE_DIR}/runs/${EXP_NAME}"
CHECKPOINT_DIR="${SAVE_DIR}/checkpoints/${METHOD}/seed-${SEED}"

MODEL_ARGS=(
    --swiglu
    --num-layers 28
    --hidden-size 1536
    --ffn-hidden-size 8960
    --num-attention-heads 12
    --group-query-attention
    --num-query-groups 2
    --use-rotary-position-embeddings
    --disable-bias-linear
    --add-qkv-bias
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --vocab-size 151936
    --kv-channels 128
)

RESOURCE_ARGS=(
    --resource "{\"actor\":[1,${NUM_GPUS}],\"rollout\":[1,${NUM_GPUS}]}"
    --max-staleness 0
    --num-data-storage-units 1
    --use-health-check
    --colocate
)

CHECKPOINT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${HF_CHECKPOINT}"
    --save "${CHECKPOINT_DIR}"
    --megatron-to-hf-mode bridge
    --save-interval 10
    --max-actor-ckpt-to-keep 2
)
if [ -n "${LOAD_DIR}" ]; then
    CHECKPOINT_ARGS+=(--load "${LOAD_DIR}")
fi

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key messages
    --metadata-key metadata
    --use-agentic-rollout
    --agent-command "bash ${SCRIPT_DIR}/run_agent_app.sh"
    --agent-cwd "${PROJECT_ROOT}"
    --agent-env
    "ALFWORLD_CONFIG_PATH=${ALFWORLD_CONFIG_PATH}"
    "ALFWORLD_DATA=${ALFWORLD_DATA}"
    "ALFWORLD_PYTHON=${ALFWORLD_PYTHON}"
    --agentic-custom-advantage-path examples.graphgpo.custom_advantage.compute_custom_advantage
    --custom-rm-path examples.graphgpo.reward.reward_func
    --group-rm
    --num-rollout "${OUTER_EPOCHS}"
    --rollout-batch-size "${TASK_GROUPS}"
    --n-samples-per-prompt "${GROUP_SIZE}"
    --rollout-max-prompt-len 2048
    --rollout-max-response-len 512
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size 32
    --rollout-shuffle
    --use-streaming-dataset
    --agentic-prepare-pool-size 0
)

ALGORITHM_ARGS=(
    --advantage-estimator grpo
    --kl-coef 0
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type low_var_kl
    --eps-clip 0.2
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
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MEGATRON_ARGS=(
    --tensor-model-parallel-size "${NUM_GPUS}"
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    # TP ranks compile independently, so cold Dynamo/Triton progress can skew
    # around collectives.  Disable compilation before train actors import
    # Megatron, and keep lazily imported JIT-fused kernels on the eager path.
    --train-env-vars '{"TORCH_COMPILE_DISABLE":"1"}'
    --disable-jit-fuser
    --sequence-parallel
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --seed "${SEED}"
)

EVAL_ARGS=()
if [ "${ENABLE_EVAL}" = "1" ]; then
    EVAL_ARGS=(
        --eval-interval 5
        --eval-prompt-data alfworld "${EVAL_DATA}"
        --eval-input-key messages
        --n-samples-per-eval-prompt 1
        --eval-max-response-len 512
        --eval-temperature 0.4
        --eval-top-p 1.0
        --eval-top-k -1
        --custom-eval-rollout-log-function-path examples.graphgpo.eval_logger.log_eval_rollout_data
    )
fi

LOG_ARGS=(
    --tb-project-name "${SAVE_DIR}/tensorboard"
    --tb-experiment-name "${EXP_NAME}"
)

TRAIN_COMMAND=(
    python3 relax/entrypoints/train.py
    "${RESOURCE_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${CHECKPOINT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${ALGORITHM_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MEGATRON_ARGS[@]}"
    "${EVAL_ARGS[@]}"
    "${LOG_ARGS[@]}"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'ray job submit --address=http://127.0.0.1:8265 -- '
    printf '%q ' "${TRAIN_COMMAND[@]}"
    printf '\n'
    exit 0
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    set +u
    source "${PROJECT_ROOT}/scripts/entrypoint/local.sh"
    set -u
fi

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"
{
    echo "method=${METHOD}"
    echo "enable_eval=${ENABLE_EVAL}"
    echo "episode_weighting=${GRAPHGPO_EPISODE_WEIGHTING}"
    echo "sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}"
    if [ -n "${GRAPHGPO_DIAGNOSTICS_JSONL:-}" ]; then
        echo "graph_diagnostics_jsonl=enabled"
    else
        echo "graph_diagnostics_jsonl=disabled"
    fi
    echo "seed=${SEED}"
    echo "model_revision=${MODEL_REVISION}"
    echo "expected_external_image_digest=${EXPECTED_EXTERNAL_IMAGE_DIGEST}"
    echo "image_verification_scope=external_executor_required"
    echo "alfworld_config_sha256=$(sha256sum "${ALFWORLD_CONFIG_PATH}" | cut -d' ' -f1)"
    echo "train_manifest_sha256=$(sha256sum "${TRAIN_MANIFEST}" | cut -d' ' -f1)"
    if [ "${ENABLE_EVAL}" = "1" ]; then
        echo "eval_manifest_sha256=$(sha256sum "${EVAL_MANIFEST}" | cut -d' ' -f1)"
    fi
    echo "prepare_lock_sha256=$(sha256sum "${PREPARE_LOCK}" | cut -d' ' -f1)"
    echo "dependency_lock_sha256=$(sha256sum "${DEPENDENCY_LOCK}" | cut -d' ' -f1)"
    echo "model_lock_sha256=$(sha256sum "${MODEL_LOCK}" | cut -d' ' -f1)"
} >"${RUN_DIR}/run_lock.env"
{
    printf 'ray job submit --address=http://127.0.0.1:8265 -- '
    printf '%q ' "${TRAIN_COMMAND[@]}"
    printf '\n'
} >"${RUN_DIR}/expanded_command.sh"

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    -- "${TRAIN_COMMAND[@]}" \
    2>&1 | tee "${RUN_DIR}/train.log"
