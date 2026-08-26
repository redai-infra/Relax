#!/usr/bin/env bash

# Task40 Batch 3 command package: BF16 + THD + MBS1 only.

set -euo pipefail

if [[ "${1:-}" == "--in-container" ]]; then
    TOPOLOGY="${2:?usage: run_forward_dump.sh --in-container <dp4cp1|dp2cp2|dp2cp1>}"
    case "${TOPOLOGY}" in
        dp4cp1) ACTOR_WORLD_SIZE=4; CONTEXT_PARALLEL_SIZE=1 ;;
        dp2cp2) ACTOR_WORLD_SIZE=4; CONTEXT_PARALLEL_SIZE=2 ;;
        dp2cp1) ACTOR_WORLD_SIZE=2; CONTEXT_PARALLEL_SIZE=1 ;;
        *) echo "unsupported topology: ${TOPOLOGY}" >&2; exit 2 ;;
    esac

    : "${P3O_STEP0_FIXTURE:?P3O_STEP0_FIXTURE must be set}"
    export P3O_ALGORITHM=p3o
    export P3O_ENABLE_TEMPERATURE_OVERRIDE=0
    export P3O_UPDATE_WEIGHTS_INTERVAL=1

    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
    source "${REPO_ROOT}/examples/algorithms/p3o/common_a100x4.sh"

    eval "$(declare -f P3O_build_args | sed '1s/P3O_build_args/P3O_build_args_base/')"

    replace_train_arg() {
        local flag="$1"
        local value="$2"
        local index
        for ((index = 0; index < ${#P3O_TRAIN_ARGS[@]}; index++)); do
            if [[ "${P3O_TRAIN_ARGS[index]}" == "${flag}" ]]; then
                P3O_TRAIN_ARGS[index + 1]="${value}"
                return 0
            fi
        done
        echo "required argument not found: ${flag}" >&2
        return 3
    }

    P3O_build_args() {
        P3O_build_args_base
        replace_train_arg --resource "{\"actor\":[1,${ACTOR_WORLD_SIZE}],\"rollout\":[1,4]}"
        replace_train_arg --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
        replace_train_arg --lr 1e-6
        P3O_TRAIN_ARGS+=(
            --load-debug-rollout-data "${P3O_STEP0_FIXTURE}"
            --custom-megatron-init-path scripts.local_p3o_task40.p3o_forward_dump.configure
            --custom-megatron-before-log-prob-hook-path scripts.local_p3o_task40.p3o_forward_dump.before_log_prob
            --custom-megatron-before-train-step-hook-path scripts.local_p3o_task40.p3o_forward_dump.before_train_step
            --dump-details "${P3O_OUTPUT_ROOT}/${P3O_CONFIG_NAME}/seed_${P3O_SEED}/${P3O_RUN_ID}/debug"
        )
    }

    P3O_CONFIG_NAME="${TOPOLOGY}"
    set +e
    P3O_run
    TRAIN_EXIT_CODE=$?
    set -e
    FINALIZE_EXIT_CODE=0
    if [[ -n "${P3O_RUN_DIR:-}" && -d "${P3O_RUN_DIR}" ]]; then
        python3 scripts/local_p3o_task40/p3o_forward_dump.py finalize-manifest \
            --run-dir "${P3O_RUN_DIR}" || FINALIZE_EXIT_CODE=$?
    else
        FINALIZE_EXIT_CODE=1
    fi
    if [[ "${TRAIN_EXIT_CODE}" -ne 0 ]]; then
        exit "${TRAIN_EXIT_CODE}"
    fi
    exit "${FINALIZE_EXIT_CODE}"
fi

TOPOLOGY="${1:?usage: run_forward_dump.sh <dp4cp1|dp2cp2|dp2cp1> <run-id>}"
RUN_ID="${2:?usage: run_forward_dump.sh <dp4cp1|dp2cp2|dp2cp1> <run-id>}"
case "${TOPOLOGY}" in
    dp4cp1|dp2cp2|dp2cp1) ;;
    *) echo "unsupported topology: ${TOPOLOGY}" >&2; exit 2 ;;
esac

INFRA=/lustre/home/sztu_camdt_zhanghua/jimaomo/infra
REPO="${INFRA}/Relax"
IMAGE="${INFRA}/images/relaxrl-dev-20260715-8325919e.sif"
CAMPAIGN="${INFRA}/Output/task40/task40_cp_forward_diag_20260817_6c7a3d2"
FIXTURE="${INFRA}/Output/task40/task40_p0_cluster_20260814_6c7a3d2/fixtures/step0_rollout.pt"
LOG_DIR="${CAMPAIGN}/logs"
mkdir -p "${LOG_DIR}"

set +e
apptainer exec --nv --bind /lustre:/lustre "${IMAGE}" env \
    P3O_MODE=smoke \
    P3O_MODEL_CONFIG="${REPO}/scripts/local_p3o_task40/qwen2p5_1p5b.sh" \
    P3O_MODEL_ROTARY_BASE=1000000 \
    P3O_MODEL_DIR="${INFRA}/Qwen2.5-1.5B-Instruct" \
    P3O_TRAIN_DATA="${INFRA}/gsm8k/main/train_clean.parquet" \
    P3O_INPUT_KEY=question \
    P3O_LABEL_KEY=answer \
    P3O_RM_TYPE=openr1mm \
    P3O_OUTPUT_ROOT="${CAMPAIGN}/runs/step0_dump" \
    P3O_MEGATRON_DIR=/root/Megatron-LM \
    P3O_RAY_DASHBOARD=http://127.0.0.1:8265 \
    P3O_NUM_ROLLOUT=1 \
    P3O_ROLLOUT_BATCH_SIZE=4 \
    P3O_N_SAMPLES=16 \
    P3O_GLOBAL_BATCH_SIZE=64 \
    P3O_MICRO_BATCH_SIZE=1 \
    P3O_MAX_RESPONSE_LEN=4096 \
    P3O_ESS_SCOPE=step \
    P3O_KL_MODE=proxy_safe \
    P3O_SEED=42 \
    P3O_PIPELINE_MODEL_PARALLEL_SIZE=1 \
    P3O_ACTIVATION_RECOMPUTE=0 \
    P3O_LOG_PROBS_CHUNK_SIZE=1024 \
    P3O_ROLLOUT_SHUFFLE=0 \
    P3O_DETERMINISTIC_INFERENCE=1 \
    P3O_CLEAR_RUNTIME_PROXIES=1 \
    P3O_STEP0_FIXTURE="${FIXTURE}" \
    P3O_RUN_ID="${RUN_ID}" \
    bash -lc "cd '${REPO}' && bash scripts/local_p3o_task40/run_forward_dump.sh --in-container '${TOPOLOGY}'" \
    2>&1 | tee "${LOG_DIR}/step0_dump_${TOPOLOGY}_${RUN_ID}.launcher.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e
printf '%s\n' "${EXIT_CODE}" >"${LOG_DIR}/step0_dump_${TOPOLOGY}_${RUN_ID}.launcher_exit_code.txt"
exit "${EXIT_CODE}"
