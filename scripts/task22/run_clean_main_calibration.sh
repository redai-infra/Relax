#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Task22 zero-KL instrumented baseline on an unmodified scheduling policy:
#   2 Actor GPUs + 2 persistent single-GPU rollout engines, hybrid async.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${TASK22_PYTHON:?Set TASK22_PYTHON to an absolute executable launcher}"
if [[ "$PYTHON_BIN" != /* || ! -x "$PYTHON_BIN" ]]; then
    echo "TASK22_PYTHON must be an executable absolute path" >&2
    exit 2
fi
PYTHON_BIN_DIR="$(dirname -- "$PYTHON_BIN")"
export PATH="$PYTHON_BIN_DIR${PATH:+:$PATH}"
command -v ray >/dev/null 2>&1 || {
    echo "ray CLI must be installed beside TASK22_PYTHON or otherwise available on PATH" >&2
    exit 2
}
if [[ -n "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Task22 clean-main calibration requires a clean worktree" >&2
    exit 4
fi

export NUM_GPUS="${NUM_GPUS:-4}"
[[ "$NUM_GPUS" == "4" ]] || { echo "NUM_GPUS must be 4" >&2; exit 4; }
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export RAY_DEDUP_LOGS=0
TASK22_LOCAL_RUN_DIR="${TASK22_LOCAL_RUN_DIR:?Set a fresh absolute local-disk run directory}"
TASK22_PERSIST_DIR="${TASK22_PERSIST_DIR:?Set a fresh absolute persistent artifact directory}"
[[ "$TASK22_LOCAL_RUN_DIR" == /* ]] || { echo "TASK22_LOCAL_RUN_DIR must be absolute" >&2; exit 4; }
[[ "$TASK22_PERSIST_DIR" == /* ]] || { echo "TASK22_PERSIST_DIR must be absolute" >&2; exit 4; }
[[ ! -e "$TASK22_LOCAL_RUN_DIR" ]] || { echo "TASK22_LOCAL_RUN_DIR must not already exist" >&2; exit 4; }
[[ ! -e "$TASK22_PERSIST_DIR" ]] || { echo "TASK22_PERSIST_DIR must not already exist" >&2; exit 4; }
"$PYTHON_BIN" - "$TASK22_LOCAL_RUN_DIR" "$TASK22_PERSIST_DIR" <<'PY'
import sys
from pathlib import Path

local = Path(sys.argv[1]).resolve()
persistent = Path(sys.argv[2]).resolve()
network_root = Path("/root/autodl-fs").resolve()
if local == persistent or local in persistent.parents or persistent in local.parents:
    raise SystemExit("local and persistent artifact directories must not be equal, ancestors, or descendants")
if local == network_root or network_root in local.parents:
    raise SystemExit("TASK22_LOCAL_RUN_DIR must not resolve inside /root/autodl-fs")
PY
mkdir -p "$TASK22_LOCAL_RUN_DIR"
export TASK22_CALIBRATION_DIR="$TASK22_LOCAL_RUN_DIR"
export TASK22_LOCAL_RUN_DIR TASK22_PERSIST_DIR

export TASK22_RUN_START_EPOCH="$("$PYTHON_BIN" -c 'import time; print(time.time())')"
GPU_MONITOR_PATH="$TASK22_LOCAL_RUN_DIR/nvidia_smi_20s.csv"
GPU_MONITOR_PID=""

finish_run() {
    exit_code=$?
    trap - EXIT
    set +e
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
        wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    fi
    export TASK22_RUN_END_EPOCH="$("$PYTHON_BIN" -c 'import time; print(time.time())')"
    export TASK22_RUN_EXIT_CODE="$exit_code"
    "$PYTHON_BIN" - "$TASK22_LOCAL_RUN_DIR/run_lifecycle.json" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
start = float(os.environ["TASK22_RUN_START_EPOCH"])
end = float(os.environ["TASK22_RUN_END_EPOCH"])
instance_start = os.environ.get("TASK22_INSTANCE_START_EPOCH")
exit_code = int(os.environ["TASK22_RUN_EXIT_CODE"])
payload = {
    "schema_version": 1,
    "run_start_epoch": start,
    "run_end_epoch": end,
    "run_elapsed_seconds": end - start,
    "exit_code": exit_code,
    "status": "SUCCEEDED" if exit_code == 0 else "FAILED",
    "instance_start_epoch": float(instance_start) if instance_start else None,
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as output:
    json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
    output.write("\n")
    output.flush()
    os.fsync(output.fileno())
os.replace(temporary, path)
PY
    lifecycle_exit_code=$?
    if [[ "$exit_code" == "0" ]]; then
        printf '%s\n' "SUCCEEDED" > "$TASK22_LOCAL_RUN_DIR/STATUS"
    else
        printf '%s\n' "FAILED exit_code=$exit_code" > "$TASK22_LOCAL_RUN_DIR/STATUS"
    fi
    status_exit_code=$?
    "$PYTHON_BIN" - "$TASK22_LOCAL_RUN_DIR" "$TASK22_PERSIST_DIR" <<'PY'
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
if destination.exists():
    raise SystemExit(f"persistent destination unexpectedly exists: {destination}")


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
    json.dumps({"schema_version": 1, "files": files}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
destination.parent.mkdir(parents=True, exist_ok=True)
staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
shutil.copytree(source, staging)
for item in files:
    copied = staging / item["path"]
    if not copied.is_file() or copied.stat().st_size != item["size"] or sha256(copied) != item["sha256"]:
        raise SystemExit(f"persistent artifact verification failed: {item['path']}")
(staging / "PERSISTED").write_text("verified\n", encoding="utf-8")
staging.rename(destination)
print(f"TASK22_ARTIFACT_PERSISTENCE verdict=PASS files={len(files)} destination={destination}")
PY
    persist_exit_code=$?
    if [[ "$lifecycle_exit_code" != "0" || "$status_exit_code" != "0" || "$persist_exit_code" != "0" ]]; then
        echo "Task22 finalization failed lifecycle=$lifecycle_exit_code status=$status_exit_code persist=$persist_exit_code" >&2
        exit 6
    fi
    exit "$exit_code"
}
trap finish_run EXIT

printf '%s\n' "RUNNING" > "$TASK22_LOCAL_RUN_DIR/STATUS"
(
    printf '%s\n' "timestamp,index,uuid,memory.total.MiB,memory.used.MiB,memory.free.MiB,utilization.gpu.percent,utilization.memory.percent,power.draw.W"
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
            --format=csv,noheader,nounits
        sleep 20
    done
) > "$GPU_MONITOR_PATH" 2>&1 &
GPU_MONITOR_PID=$!

# These values are part of the evidence contract.  Do not allow inherited shell
# state to silently redirect or thin the scheduler heartbeat stream.
export SGLANG_LOG_SCHEDULER_STATUS_TARGET="stdout"
export SGLANG_LOG_SCHEDULER_STATUS_INTERVAL="1.0"
export RELAX_RID_ONLY_REQUEST_LOGGING="1"
TASK22_EXPERIMENT_ARM="${TASK22_EXPERIMENT_ARM:-baseline}"
export TASK22_EXPERIMENT_ARM
unset RELAX_SYNC_INTENT_TTL_SECONDS
unset RELAX_SYNC_INTENT_WINDOW_GROUPS
unset RELAX_SYNC_INTENT_QUIESCE_MULTIPLIER
unset RELAX_SYNC_INTENT_QUIESCE_FLOOR_SECONDS
unset RELAX_SYNC_INTENT_ABORT_RETRY_INTERVAL_SECONDS
unset RELAX_SYNC_INTENT_ABORT_TIMEOUT_SECONDS
unset RELAX_SYNC_INTENT_PROTECTED_DRAIN_TIMEOUT_SECONDS
case "$TASK22_EXPERIMENT_ARM" in
    baseline)
        export RELAX_SYNC_INTENT_POLICY="0"
        ;;
    a3_workaware_on)
        export RELAX_SYNC_INTENT_POLICY="1"
        export RELAX_SYNC_INTENT_TTL_SECONDS="600"
        export RELAX_SYNC_INTENT_WINDOW_GROUPS="16"
        export RELAX_SYNC_INTENT_QUIESCE_MULTIPLIER="1.25"
        export RELAX_SYNC_INTENT_QUIESCE_FLOOR_SECONDS="2.0"
        export RELAX_SYNC_INTENT_ABORT_RETRY_INTERVAL_SECONDS="0.5"
        export RELAX_SYNC_INTENT_ABORT_TIMEOUT_SECONDS="15"
        export RELAX_SYNC_INTENT_PROTECTED_DRAIN_TIMEOUT_SECONDS="600"
        ;;
    *)
        echo "Unknown TASK22_EXPERIMENT_ARM=$TASK22_EXPERIMENT_ARM" >&2
        exit 4
        ;;
esac
# RTX PRO 6000 Blackwell Server Edition reports SM120.  FlashInfer 0.6.11
# cannot infer that target under the image's CUDA 12.8 toolchain, but accepts
# the explicit forward-compatible architecture spelling used by the prior
# successful Task22 runs.
export FLASHINFER_CUDA_ARCH_LIST="12.0a"

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "$REPO/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

EXP_DIR="${EXP_DIR:?Set EXP_DIR}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR}"
NUM_ROLLOUT="${NUM_ROLLOUT:-11}"
[[ "$NUM_ROLLOUT" == "11" ]] || { echo "NUM_ROLLOUT must be 11" >&2; exit 4; }
WORKING_DIR="$REPO"
DRIVER_LOG_PATH="$TASK22_LOCAL_RUN_DIR/driver.log"
export TASK22_EXPECTED_GIT_SHA="${TASK22_EXPECTED_GIT_SHA:?Set the exact pushed calibration commit SHA}"

# Do not spell the default as `${RUNTIME_ENV_JSON:-{}}`: Bash treats the first
# `}` as the parameter-expansion terminator and appends the second one when the
# variable is already set, corrupting otherwise valid JSON.
if [[ -z "${RUNTIME_ENV_JSON:-}" ]]; then
    RUNTIME_ENV_JSON="{}"
fi
RUNTIME_ENV_JSON="$("$PYTHON_BIN" - "$RUNTIME_ENV_JSON" "$WORKING_DIR" <<'PY'
import json
import os
import sys

runtime = json.loads(sys.argv[1])
runtime["working_dir"] = sys.argv[2]
env = runtime.setdefault("env_vars", {})
for name in (
    "FLASHINFER_CUDA_ARCH_LIST",
    "NCCL_NVLS_ENABLE",
    "NCCL_SOCKET_IFNAME",
    "RAY_DEDUP_LOGS",
    "RELAX_RID_ONLY_REQUEST_LOGGING",
    "RELAX_SYNC_INTENT_POLICY",
    "RELAX_SYNC_INTENT_TTL_SECONDS",
    "RELAX_SYNC_INTENT_WINDOW_GROUPS",
    "RELAX_SYNC_INTENT_QUIESCE_MULTIPLIER",
    "RELAX_SYNC_INTENT_QUIESCE_FLOOR_SECONDS",
    "RELAX_SYNC_INTENT_ABORT_RETRY_INTERVAL_SECONDS",
    "RELAX_SYNC_INTENT_ABORT_TIMEOUT_SECONDS",
    "RELAX_SYNC_INTENT_PROTECTED_DRAIN_TIMEOUT_SECONDS",
    "SGLANG_LOG_SCHEDULER_STATUS_INTERVAL",
    "SGLANG_LOG_SCHEDULER_STATUS_TARGET",
    "TASK22_CALIBRATION_DIR",
    "TASK22_EXPERIMENT_ARM",
):
    if name in os.environ:
        env[name] = os.environ[name]
print(json.dumps(runtime, sort_keys=True, separators=(",", ":")))
PY
)"
export RUNTIME_ENV_JSON

CKPT_ARGS=(
    --hf-checkpoint "$MODEL_DIR/Qwen3-4B/"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)
ROLLOUT_ARGS=(
    --prompt-data "$DATA_DIR/dapo-math-17k/dapo-math-17k.jsonl"
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
)
PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
)
GRPO_ARGS=(
    --advantage-estimator grpo
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
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --sglang-show-time-cost
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --sglang-cuda-graph-max-bs 64
    --sglang-disable-piecewise-cuda-graph
)
if [[ "$TASK22_EXPERIMENT_ARM" == "a3_workaware_on" ]]; then
    SGLANG_ARGS+=(
        --use-slime-router
        --slime-router-work-aware
        --sglang-enable-priority-scheduling
        --sglang-disable-priority-preemption
        --sglang-default-priority-value 0
    )
fi
METRIC_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "Relax/task22/main-calibration"
    --tb-experiment-name "task22-main-calibration-$(date '+%Y%m%d-%H%M%S')"
)
MISC_ARGS=(
    --skip-eval-before-train
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --seed 1234
)

TRAIN_ARGS=(
    --resource '{"actor": [1, 2], "rollout": [1, 2]}'
    --max-staleness 2
    --num-data-storage-units 1
    --num-iters-per-train-update 1
    --update-weights-interval 1
    --hybrid
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${METRIC_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
)
if [[ "$TASK22_EXPERIMENT_ARM" == "a3_workaware_on" ]]; then
    TRAIN_ARGS+=(
        --enable-cross-version-kv-continuation
        --cross-version-kv-max-gap 2
    )
fi

export TASK22_GIT_SHA="$(git -C "$REPO" rev-parse HEAD)"
export TASK22_REPO="$REPO"
export TASK22_TRAIN_ARGS_JSON
TASK22_TRAIN_ARGS_JSON="$("$PYTHON_BIN" - "${TRAIN_ARGS[@]}" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:], separators=(",", ":")))
PY
)"
export TASK22_RUNTIME_ENV_JSON="$RUNTIME_ENV_JSON"
export MODEL_DIR DATA_DIR NUM_ROLLOUT TASK22_EXPERIMENT_ARM
TASK22_PYTHON="$PYTHON_BIN" MODEL_DIR="$MODEL_DIR" DATA_DIR="$DATA_DIR" \
    TASK22_TRAIN_ARGS_JSON="$TASK22_TRAIN_ARGS_JSON" TASK22_RUNTIME_ENV_JSON="$RUNTIME_ENV_JSON" \
    "$REPO/scripts/task22/preflight_zero_kl_instrumented_baseline.sh" \
    2>&1 | tee "$TASK22_LOCAL_RUN_DIR/formal_preflight.log"

"$PYTHON_BIN" - "$TASK22_CALIBRATION_DIR/run_contract.json" <<'PY'
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


path = Path(sys.argv[1])
model_root = Path(os.environ["MODEL_DIR"]) / "Qwen3-4B"
data_path = Path(os.environ["DATA_DIR"]) / "dapo-math-17k" / "dapo-math-17k.jsonl"
model_files = []
for candidate in sorted(model_root.iterdir()):
    if candidate.is_file():
        model_files.append(
            {
                "name": candidate.name,
                "size": candidate.stat().st_size,
                "sha256": sha256(candidate),
            }
        )

gpu_raw = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
gpu_inventory = []
for row in csv.reader(io.StringIO(gpu_raw)):
    if not row:
        continue
    index, uuid, name, memory_mib, driver = (field.strip() for field in row)
    gpu_inventory.append(
        {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(memory_mib),
            "driver_version": driver,
        }
    )
payload = {
    "schema_version": 1,
    "git_sha": os.environ["TASK22_GIT_SHA"],
    "experiment_arm": os.environ["TASK22_EXPERIMENT_ARM"],
    "python": os.environ["TASK22_PYTHON"],
    "headline": {"logical_step_lo": 2, "logical_step_hi": 9},
    "measurement_contract": {
        "purpose": (
            "zero-KL A3 work-aware ON"
            if os.environ["TASK22_EXPERIMENT_ARM"] == "a3_workaware_on"
            else "zero-KL instrumented baseline bottleneck calibration"
        ),
        "historical_wall_directly_comparable": False,
        "reason": "1s scheduler/permit/timing instrumentation enabled; compare only with the matched zero-KL baseline",
    },
    "train_argv": json.loads(os.environ["TASK22_TRAIN_ARGS_JSON"]),
    "artifacts": {
        "local_run_dir": os.environ["TASK22_LOCAL_RUN_DIR"],
        "persistent_dir": os.environ["TASK22_PERSIST_DIR"],
        "critical_path_writes": "local_run_dir_only",
        "persistence_timing": "after_training_exit",
    },
    "topology": {"actor_gpus": 2, "rollout_engines": 2, "rollout_gpus_per_engine": 1},
    "gpu_inventory": gpu_inventory,
    "workload": {
        "num_rollout": int(os.environ.get("NUM_ROLLOUT", "11")),
        "rollout_batch_size_groups": 8,
        "samples_per_prompt": 8,
        "global_batch_size_samples": 64,
        "max_response_tokens": 8192,
        "max_staleness": 2,
        "partial_rollout": True,
        "partial_rollout_max_aborted_count": 2,
        "mask_offpolicy_in_partial_rollout": True,
        "over_sampling_batch_size_groups": 16,
        "update_weights_interval": 1,
        "zero_kl_reference_forward": False,
        "sync_intent_policy": os.environ["TASK22_EXPERIMENT_ARM"] == "a3_workaware_on",
        "cross_version_kv_continuation": os.environ["TASK22_EXPERIMENT_ARM"] == "a3_workaware_on",
        "priority_scheduling": os.environ["TASK22_EXPERIMENT_ARM"] == "a3_workaware_on",
        "slime_router_work_aware": os.environ["TASK22_EXPERIMENT_ARM"] == "a3_workaware_on",
        "rollout_seed": 42,
        "train_seed": 1234,
    },
    "scheduler_status_interval_seconds": float(os.environ["SGLANG_LOG_SCHEDULER_STATUS_INTERVAL"]),
    "flashinfer_cuda_arch_list": os.environ["FLASHINFER_CUDA_ARCH_LIST"],
    "runtime_env": json.loads(os.environ["RUNTIME_ENV_JSON"]),
    "model": {
        "root": str(model_root),
        "config_sha256": sha256(model_root / "config.json"),
        "files": model_files,
    },
    "dataset": {
        "path": str(data_path),
        "size": data_path.stat().st_size,
        "sha256": sha256(data_path),
    },
}

repo = Path(os.environ["TASK22_REPO"]).resolve()
source_paths = {
    "relax_actor": repo / "relax/backends/megatron/actor.py",
    "relax_permit_observability": repo / "relax/engine/rollout/permit_observability.py",
    "relax_request_permit": repo / "relax/engine/rollout/request_permit.py",
    "relax_router": repo / "relax/engine/router/router.py",
    "relax_router_work_accounting": repo / "relax/engine/router/work_accounting.py",
    "relax_sglang_rollout": repo / "relax/engine/rollout/sglang_rollout.py",
}
try:
    import importlib.metadata
    import importlib.util

    import torch

    for label, module_name in {
        "flashinfer_jit_core": "flashinfer.jit.core",
        "sgl_kernel_elementwise": "sgl_kernel.elementwise",
        "sglang_req_time_stats": "sglang.srt.observability.req_time_stats",
        "sglang_scheduler_metrics": "sglang.srt.observability.scheduler_metrics_mixin",
        "sglang_scheduler_status": "sglang.srt.utils.scheduler_status_logger",
        "sglang_runtime_checker": "sglang.srt.managers.scheduler_runtime_checker_mixin",
        "sglang_request_logger": "sglang.srt.utils.request_logger",
        "megatron_frozen_linear": "megatron.core.tensor_parallel.layers",
    }.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve dependency module: {module_name}")
        source_paths[label] = Path(spec.origin).resolve()
    payload["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("flashinfer-python", "sglang", "torch", "ray")
    }
    payload["versions"]["torch_cuda"] = torch.version.cuda
except Exception as exc:
    raise SystemExit(f"failed to attest runtime dependencies: {exc}") from exc

payload["source_files"] = {
    label: {"path": str(source), "size": source.stat().st_size, "sha256": sha256(source)}
    for label, source in sorted(source_paths.items())
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as output:
    json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
    output.write("\n")
    output.flush()
    os.fsync(output.fileno())
os.replace(temporary, path)
PY

"$PYTHON_BIN" -m ray.scripts.scripts job submit \
    --address="http://127.0.0.1:8265" \
    --working-dir "$WORKING_DIR" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- "$PYTHON_BIN" -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "$DRIVER_LOG_PATH"
