#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Fixed 4-GPU Qwen3-4B 2x2 experiment for the PR211 × PR242 integration:
#   interval1_kv_off  interval2_kv_off
#   interval1_kv_on   interval2_kv_on

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
TASK22_PYTHON="${TASK22_PYTHON:?Set TASK22_PYTHON to an absolute executable Python path}"
TASK22_ARM="${TASK22_ARM:?Set TASK22_ARM to one of the four integration arms}"
NUM_ROLLOUT="${NUM_ROLLOUT:-21}"
TASK22_LOCAL_RUN_DIR="${TASK22_LOCAL_RUN_DIR:?Set a fresh absolute local-disk run directory}"
TASK22_PERSIST_DIR="${TASK22_PERSIST_DIR:?Set a fresh absolute persistent artifact directory}"
TASK22_EXPECTED_GIT_SHA="${TASK22_EXPECTED_GIT_SHA:?Set the exact integration commit SHA}"

if [[ "$TASK22_PYTHON" != /* || ! -x "$TASK22_PYTHON" ]]; then
    echo "TASK22_PYTHON must be an executable absolute path" >&2
    exit 2
fi
if ! [[ "$NUM_ROLLOUT" =~ ^[0-9]+$ ]] || ((NUM_ROLLOUT < 3 || NUM_ROLLOUT > 21)); then
    echo "NUM_ROLLOUT must be an integer in [3, 21]" >&2
    exit 2
fi
for output_dir in "$TASK22_LOCAL_RUN_DIR" "$TASK22_PERSIST_DIR"; do
    [[ "$output_dir" == /* ]] || { echo "Artifact directories must be absolute" >&2; exit 2; }
    [[ ! -e "$output_dir" ]] || { echo "Artifact directory already exists: $output_dir" >&2; exit 2; }
done
if [[ "$TASK22_LOCAL_RUN_DIR" != /root/task22-local/runs/* ]]; then
    echo "TASK22_LOCAL_RUN_DIR must be under /root/task22-local/runs" >&2
    exit 2
fi
if [[ "$TASK22_PERSIST_DIR" != /root/autodl-fs/task22/* ]]; then
    echo "TASK22_PERSIST_DIR must be under /root/autodl-fs/task22" >&2
    exit 2
fi

ACTUAL_GIT_SHA="$(git -C "$REPO" rev-parse HEAD)"
if [[ "$ACTUAL_GIT_SHA" != "$TASK22_EXPECTED_GIT_SHA" ]]; then
    echo "Git SHA mismatch: $ACTUAL_GIT_SHA != $TASK22_EXPECTED_GIT_SHA" >&2
    exit 3
fi
if [[ -n "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Task22 integration experiment requires a clean worktree" >&2
    exit 3
fi

UPDATE_WEIGHTS_INTERVAL=""
CROSS_VERSION_KV="0"
CROSS_VERSION_KV_MAX_GAP=""
case "$TASK22_ARM" in
    interval1_kv_off)
        UPDATE_WEIGHTS_INTERVAL="1"
        ;;
    interval2_kv_off)
        UPDATE_WEIGHTS_INTERVAL="2"
        ;;
    interval1_kv_on)
        UPDATE_WEIGHTS_INTERVAL="1"
        CROSS_VERSION_KV="1"
        CROSS_VERSION_KV_MAX_GAP="2"
        ;;
    interval2_kv_on)
        UPDATE_WEIGHTS_INTERVAL="2"
        CROSS_VERSION_KV="1"
        CROSS_VERSION_KV_MAX_GAP="1"
        ;;
    *)
        echo "Unknown TASK22_ARM=$TASK22_ARM" >&2
        exit 2
        ;;
esac

mkdir -p "$TASK22_LOCAL_RUN_DIR"
RUN_START_EPOCH="$("$TASK22_PYTHON" -c 'import time; print(time.time())')"
GPU_MONITOR_PID=""

finish_run() {
    exit_code=$?
    trap - EXIT
    set +e
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
        wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    fi
    run_end_epoch="$("$TASK22_PYTHON" -c 'import time; print(time.time())')"
    "$TASK22_PYTHON" - "$TASK22_LOCAL_RUN_DIR" "$TASK22_PERSIST_DIR" \
        "$RUN_START_EPOCH" "$run_end_epoch" "$exit_code" <<'PY'
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
start = float(sys.argv[3])
end = float(sys.argv[4])
exit_code = int(sys.argv[5])
analysis_path = source / "analysis_pr211_integration.json"
analysis_verdict = None
if analysis_path.is_file():
    try:
        analysis_verdict = json.loads(analysis_path.read_text(encoding="utf-8")).get("verdict")
    except (OSError, json.JSONDecodeError):
        analysis_verdict = "MALFORMED"
status = "INVALID_INPUT" if analysis_verdict == "INVALID_INPUT" else ("SUCCEEDED" if exit_code == 0 else "FAILED")
(source / "run_lifecycle.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "run_start_epoch": start,
            "run_end_epoch": end,
            "run_elapsed_seconds": end - start,
            "exit_code": exit_code,
            "status": status,
            "analysis_verdict": analysis_verdict,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
(source / "STATUS").write_text(
    "INVALID_INPUT\n"
    if status == "INVALID_INPUT"
    else ("SUCCEEDED\n" if status == "SUCCEEDED" else f"FAILED exit_code={exit_code}\n")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


files = []
for candidate in sorted(source.rglob("*")):
    if candidate.is_file() and candidate.name != "artifact_manifest.json":
        files.append(
            {
                "path": str(candidate.relative_to(source)),
                "size": candidate.stat().st_size,
                "sha256": sha256(candidate),
            }
        )
(source / "artifact_manifest.json").write_text(
    json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if destination.exists():
    raise SystemExit(f"persistent destination unexpectedly exists: {destination}")
destination.parent.mkdir(parents=True, exist_ok=True)
staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
shutil.copytree(source, staging)
for item in files:
    copied = staging / item["path"]
    if copied.stat().st_size != item["size"] or sha256(copied) != item["sha256"]:
        raise SystemExit(f"persistent artifact verification failed: {item['path']}")
(staging / "PERSISTED").write_text("verified\n", encoding="utf-8")
staging.rename(destination)
PY
    finalize_code=$?
    if [[ "$finalize_code" != "0" ]]; then
        echo "Task22 artifact finalization failed with exit code $finalize_code" >&2
        exit 6
    fi
    exit "$exit_code"
}
trap finish_run EXIT
printf '%s\n' "PREPARING" > "$TASK22_LOCAL_RUN_DIR/STATUS"

export NUM_GPUS=4
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export RAY_DEDUP_LOGS=0
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0a}"
export TASK22_ARM
if ((CROSS_VERSION_KV)); then
    export RELAX_CROSS_VERSION_KV_ABORT_RETRY_INTERVAL_SECONDS="0.5"
    export RELAX_CROSS_VERSION_KV_ABORT_TIMEOUT_SECONDS="15"
    export RELAX_CROSS_VERSION_KV_PROTECTED_DRAIN_TIMEOUT_SECONDS="600"
else
    unset RELAX_CROSS_VERSION_KV_ABORT_RETRY_INTERVAL_SECONDS
    unset RELAX_CROSS_VERSION_KV_ABORT_TIMEOUT_SECONDS
    unset RELAX_CROSS_VERSION_KV_PROTECTED_DRAIN_TIMEOUT_SECONDS
fi

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "$REPO/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

EXP_DIR="${EXP_DIR:?Set EXP_DIR}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR}"
MODEL_PATH="$MODEL_DIR/Qwen3-4B"
PROMPT_DATA="$DATA_DIR/dapo-math-17k/dapo-math-17k.jsonl"
[[ -s "$MODEL_PATH/config.json" ]] || { echo "Missing Qwen3-4B model at $MODEL_PATH" >&2; exit 3; }
[[ -s "$PROMPT_DATA" ]] || { echo "Missing DAPO-MATH dataset at $PROMPT_DATA" >&2; exit 3; }

if [[ -z "${RUNTIME_ENV_JSON:-}" ]]; then
    RUNTIME_ENV_JSON="{}"
fi
RUNTIME_ENV_JSON="$("$TASK22_PYTHON" - "$RUNTIME_ENV_JSON" <<'PY'
import json
import os
import sys

runtime = json.loads(sys.argv[1])
env = runtime.setdefault("env_vars", {})
for name in (
    "FLASHINFER_CUDA_ARCH_LIST",
    "NCCL_NVLS_ENABLE",
    "NCCL_SOCKET_IFNAME",
    "RAY_DEDUP_LOGS",
    "RELAX_CROSS_VERSION_KV_ABORT_RETRY_INTERVAL_SECONDS",
    "RELAX_CROSS_VERSION_KV_ABORT_TIMEOUT_SECONDS",
    "RELAX_CROSS_VERSION_KV_PROTECTED_DRAIN_TIMEOUT_SECONDS",
    "TASK22_ARM",
):
    if name in os.environ:
        env[name] = os.environ[name]
print(json.dumps(runtime, sort_keys=True, separators=(",", ":")))
PY
)"

TRAIN_ARGS=(
    --resource '{"actor": [1, 2], "rollout": [1, 2]}'
    --max-staleness 2
    --num-data-storage-units 1
    --num-iters-per-train-update 1
    --update-weights-interval "$UPDATE_WEIGHTS_INTERVAL"
    --hybrid
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$MODEL_PATH"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
    --prompt-data "$PROMPT_DATA"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo
    --reward-key score
    --num-rollout "$NUM_ROLLOUT"
    --rollout-batch-size 8
    --n-samples-per-prompt 8
    --rollout-max-response-len 8192
    --rollout-temperature 1
    --rollout-seed 42
    --global-batch-size 64
    --partial-rollout
    --partial-rollout-max-aborted-count 2
    --mask-offpolicy-in-partial-rollout
    --over-sampling-batch-size 16
    --use-fault-tolerance
    --balance-data
    --advantage-estimator grpo
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --sglang-show-time-cost
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --sglang-cuda-graph-max-bs 64
    --sglang-disable-piecewise-cuda-graph
    --use-clearml
    --use-metrics-service
    --tb-project-name Relax/task22/pr211-integration
    --tb-experiment-name "task22-${TASK22_ARM}-$(date '+%Y%m%d-%H%M%S')"
    --skip-eval-before-train
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --seed 1234
)
if ((CROSS_VERSION_KV)); then
    TRAIN_ARGS+=(
        --enable-cross-version-kv-continuation
        --cross-version-kv-max-gap "$CROSS_VERSION_KV_MAX_GAP"
        --use-slime-router
        --hybrid-dcs-weight-sync
        --hybrid-weights-backuper-on-gpu
        --checkpoint-engine-backend nccl
        --targeted-retirement-timeout-seconds 15
    )
fi

TRAIN_ARGS_JSON="$("$TASK22_PYTHON" - "${TRAIN_ARGS[@]}" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:], separators=(",", ":")))
PY
)"
"$TASK22_PYTHON" "$REPO/scripts/task22/validate_pr211_integration_contract.py" \
    --arm "$TASK22_ARM" \
    --num-rollout "$NUM_ROLLOUT" \
    --train-args-json "$TRAIN_ARGS_JSON" \
    | tee "$TASK22_LOCAL_RUN_DIR/preflight.log"

export ACTUAL_GIT_SHA MODEL_PATH PROMPT_DATA NUM_ROLLOUT UPDATE_WEIGHTS_INTERVAL
export CROSS_VERSION_KV CROSS_VERSION_KV_MAX_GAP TRAIN_ARGS_JSON RUNTIME_ENV_JSON
"$TASK22_PYTHON" - "$TASK22_LOCAL_RUN_DIR/run_contract.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


model_config = Path(os.environ["MODEL_PATH"]) / "config.json"
dataset = Path(os.environ["PROMPT_DATA"])
num_rollout = int(os.environ["NUM_ROLLOUT"])
train_argv = json.loads(os.environ["TRAIN_ARGS_JSON"])


def option(flag: str) -> str:
    positions = [index for index, item in enumerate(train_argv) if item == flag or item.startswith(f"{flag}=")]
    if len(positions) != 1:
        raise SystemExit(f"{flag} must appear exactly once, found {len(positions)}")
    index = positions[0]
    item = train_argv[index]
    if item.startswith(f"{flag}="):
        return item.split("=", maxsplit=1)[1]
    if index + 1 >= len(train_argv) or train_argv[index + 1].startswith("--"):
        raise SystemExit(f"{flag} has no value")
    return train_argv[index + 1]


resource = json.loads(option("--resource"))
rollout_total_gpus = int(resource["rollout"][1])
rollout_gpus_per_engine = int(option("--rollout-num-gpus-per-engine"))
if rollout_total_gpus <= 0 or rollout_gpus_per_engine <= 0:
    raise SystemExit("Rollout GPU counts must be positive")
if rollout_total_gpus % rollout_gpus_per_engine:
    raise SystemExit("Rollout total GPUs must be divisible by GPUs per engine")
payload = {
    "schema_version": 1,
    "git_sha": os.environ["ACTUAL_GIT_SHA"],
    "arm": os.environ["TASK22_ARM"],
    "num_rollout": num_rollout,
    # The formal 21-step primary window was predeclared as 3..17 after review.
    # Shorter runs are diagnostic probes and must not manufacture a headline.
    "headline": {"logical_step_lo": 3, "logical_step_hi": 17} if num_rollout == 21 else None,
    "update_weights_interval": int(os.environ["UPDATE_WEIGHTS_INTERVAL"]),
    "cross_version_kv": os.environ["CROSS_VERSION_KV"] == "1",
    "cross_version_kv_max_gap": (
        int(os.environ["CROSS_VERSION_KV_MAX_GAP"]) if os.environ["CROSS_VERSION_KV_MAX_GAP"] else None
    ),
    "train_argv": train_argv,
    "expected_topology": {
        "rollout_total_gpus": rollout_total_gpus,
        "rollout_gpus_per_engine": rollout_gpus_per_engine,
        "rollout_receiver_count": rollout_total_gpus // rollout_gpus_per_engine,
        "group_world_size": 1 + rollout_total_gpus,
    },
    # Do not persist arbitrary runtime env values: callers may inject service
    # credentials into RUNTIME_ENV_JSON.  Only attest the experiment controls.
    "runtime_env_attestation": {
        key: value
        for key, value in json.loads(os.environ["RUNTIME_ENV_JSON"]).get("env_vars", {}).items()
        if key
        in {
            "FLASHINFER_CUDA_ARCH_LIST",
            "NCCL_NVLS_ENABLE",
            "NCCL_SOCKET_IFNAME",
            "RAY_DEDUP_LOGS",
            "RELAX_CROSS_VERSION_KV_ABORT_RETRY_INTERVAL_SECONDS",
            "RELAX_CROSS_VERSION_KV_ABORT_TIMEOUT_SECONDS",
            "RELAX_CROSS_VERSION_KV_PROTECTED_DRAIN_TIMEOUT_SECONDS",
            "TASK22_ARM",
        }
    },
    "model_config_sha256": sha256(model_config),
    "dataset": {"path": str(dataset), "size": dataset.stat().st_size, "sha256": sha256(dataset)},
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "${TASK22_DRY_RUN:-0}" == "1" ]]; then
    printf '%s\n' "DRY_RUN_SUCCEEDED" > "$TASK22_LOCAL_RUN_DIR/STATUS"
    exit 0
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "$GPU_COUNT" == "4" ]] || { echo "Expected exactly 4 visible GPUs, found $GPU_COUNT" >&2; exit 3; }
(
    printf '%s\n' "timestamp,index,uuid,memory.total.MiB,memory.used.MiB,utilization.gpu.percent,power.draw.W"
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,uuid,memory.total,memory.used,utilization.gpu,power.draw \
            --format=csv,noheader,nounits
        sleep 20
    done
) > "$TASK22_LOCAL_RUN_DIR/nvidia_smi_20s.csv" 2>&1 &
GPU_MONITOR_PID=$!
printf '%s\n' "RUNNING" > "$TASK22_LOCAL_RUN_DIR/STATUS"

"$TASK22_PYTHON" -m ray.scripts.scripts job submit \
    --address="http://127.0.0.1:8265" \
    --working-dir "$REPO" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- "$TASK22_PYTHON" -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "$TASK22_LOCAL_RUN_DIR/driver.log"

"$TASK22_PYTHON" "$REPO/scripts/task22/analyze_pr211_integration_run.py" \
    --run-contract "$TASK22_LOCAL_RUN_DIR/run_contract.json" \
    --driver-log "$TASK22_LOCAL_RUN_DIR/driver.log" \
    --output "$TASK22_LOCAL_RUN_DIR/analysis_pr211_integration.json"
