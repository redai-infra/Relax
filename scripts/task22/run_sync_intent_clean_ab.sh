#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${TASK22_PYTHON:?Set TASK22_PYTHON to an absolute executable launcher}"
AB_ARM="${TASK22_AB_ARM:?Set TASK22_AB_ARM to off or on}"
LOCAL_RUN_DIR="${TASK22_LOCAL_RUN_DIR:?Set a fresh absolute local run directory}"
PERSIST_DIR="${TASK22_PERSIST_DIR:?Set a fresh absolute persistent artifact directory}"
EXPECTED_GIT_SHA="${TASK22_EXPECTED_GIT_SHA:?Set the exact clean-policy commit SHA}"

case "$AB_ARM" in
    off) export RELAX_SYNC_INTENT_POLICY=0 ;;
    on) export RELAX_SYNC_INTENT_POLICY=1 ;;
    *) echo "TASK22_AB_ARM must be off or on, got $AB_ARM" >&2; exit 2 ;;
esac

[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || {
    echo "TASK22_PYTHON must be an executable absolute path" >&2
    exit 2
}
[[ "$LOCAL_RUN_DIR" == /* && "$PERSIST_DIR" == /* ]] || {
    echo "run directories must be absolute" >&2
    exit 2
}
[[ ! -e "$LOCAL_RUN_DIR" && ! -e "$PERSIST_DIR" ]] || {
    echo "run directories must not already exist" >&2
    exit 2
}
"$PYTHON_BIN" - "$LOCAL_RUN_DIR" "$PERSIST_DIR" <<'PY'
import sys
from pathlib import Path

local = Path(sys.argv[1]).resolve()
persistent = Path(sys.argv[2]).resolve()
network_root = Path("/root/autodl-fs").resolve()
if local == persistent or local in persistent.parents or persistent in local.parents:
    raise SystemExit("local and persistent run directories must be independent")
if local == network_root or network_root in local.parents:
    raise SystemExit("local run directory must not be inside /root/autodl-fs")
PY
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_GIT_SHA" ]] || {
    echo "unexpected git SHA" >&2
    exit 2
}
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "clean A/B requires a clean worktree" >&2
    exit 2
}

export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export NUM_GPUS="${NUM_GPUS:-4}"
[[ "$NUM_GPUS" == "4" ]] || { echo "NUM_GPUS must be 4" >&2; exit 2; }
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export RAY_DEDUP_LOGS=0
export RELAX="$REPO"
export RELAX_SYNC_INTENT_TTL_SECONDS=600
export RELAX_SYNC_INTENT_WINDOW_GROUPS=16
export RELAX_SYNC_INTENT_QUIESCE_MULTIPLIER=1.25
export RELAX_SYNC_INTENT_QUIESCE_FLOOR_SECONDS=2.0
export FLASHINFER_CUDA_ARCH_LIST=12.0a

mkdir -p "$LOCAL_RUN_DIR"
export TASK22_LOCAL_RUN_DIR="$LOCAL_RUN_DIR"
printf '%s\n' RUNNING > "$LOCAL_RUN_DIR/STATUS"
RUN_START_EPOCH="$("$PYTHON_BIN" -c 'import time; print(time.time())')"

finish_run() {
    exit_code=$?
    trap - EXIT
    set +e
    RUN_END_EPOCH="$("$PYTHON_BIN" -c 'import time; print(time.time())')"
    "$PYTHON_BIN" - "$LOCAL_RUN_DIR/run_lifecycle.json" "$RUN_START_EPOCH" "$RUN_END_EPOCH" "$exit_code" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
start = float(sys.argv[2])
end = float(sys.argv[3])
exit_code = int(sys.argv[4])
path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "run_start_epoch": start,
            "run_end_epoch": end,
            "run_elapsed_seconds": end - start,
            "exit_code": exit_code,
            "status": "SUCCEEDED" if exit_code == 0 else "FAILED",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
    if [[ "$exit_code" == 0 ]]; then
        printf '%s\n' SUCCEEDED > "$LOCAL_RUN_DIR/STATUS"
    else
        printf 'FAILED exit_code=%s\n' "$exit_code" > "$LOCAL_RUN_DIR/STATUS"
    fi
    "$PYTHON_BIN" - "$LOCAL_RUN_DIR" "$PERSIST_DIR" <<'PY'
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])


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
destination.parent.mkdir(parents=True, exist_ok=True)
staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
shutil.copytree(source, staging)
for item in files:
    copied = staging / item["path"]
    if copied.stat().st_size != item["size"] or sha256(copied) != item["sha256"]:
        raise SystemExit(f"artifact verification failed: {item['path']}")
(staging / "PERSISTED").write_text("verified\n", encoding="utf-8")
staging.rename(destination)
PY
    persist_exit_code=$?
    if [[ "$persist_exit_code" != 0 ]]; then
        echo "artifact persistence failed" >&2
        exit 6
    fi
    exit "$exit_code"
}
trap finish_run EXIT

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "$REPO/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

EXP_DIR="${EXP_DIR:?Set EXP_DIR}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR}"
NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
[[ "$NUM_ROLLOUT" == "20" ]] || { echo "clean A/B requires NUM_ROLLOUT=20" >&2; exit 2; }

if [[ -z "${RUNTIME_ENV_JSON:-}" ]]; then
    RUNTIME_ENV_JSON="{}"
fi
RUNTIME_ENV_JSON="$("$PYTHON_BIN" - "$RUNTIME_ENV_JSON" "$REPO" <<'PY'
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
    "RELAX_SYNC_INTENT_POLICY",
    "RELAX_SYNC_INTENT_TTL_SECONDS",
    "RELAX_SYNC_INTENT_WINDOW_GROUPS",
    "RELAX_SYNC_INTENT_QUIESCE_MULTIPLIER",
    "RELAX_SYNC_INTENT_QUIESCE_FLOOR_SECONDS",
):
    env[name] = os.environ[name]
print(json.dumps(runtime, sort_keys=True, separators=(",", ":")))
PY
)"
export RUNTIME_ENV_JSON

CKPT_ARGS=(
    --hf-checkpoint "$MODEL_DIR/Qwen3-4B/"
    --ref-load "$MODEL_DIR/Qwen3-4B/"
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
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
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
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --sglang-cuda-graph-max-bs 64
    --sglang-disable-piecewise-cuda-graph
    --sglang-enable-priority-scheduling
    --sglang-disable-priority-preemption
    --sglang-default-priority-value 0
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
    --hybrid
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
)

export TASK22_AB_ARM="$AB_ARM"
export TASK22_GIT_SHA="$(git -C "$REPO" rev-parse HEAD)"
export TASK22_TRAIN_ARGS_JSON
TASK22_TRAIN_ARGS_JSON="$("$PYTHON_BIN" - "${TRAIN_ARGS[@]}" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:], separators=(",", ":")))
PY
)"
export TASK22_RUNTIME_ENV_JSON="$RUNTIME_ENV_JSON"
export MODEL_DIR DATA_DIR
"$PYTHON_BIN" - "$LOCAL_RUN_DIR/run_contract.json" <<'PY'
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


repo = Path(os.environ["RELAX"]).resolve()
model = Path(os.environ["MODEL_DIR"]) / "Qwen3-4B"
dataset = Path(os.environ["DATA_DIR"]) / "dapo-math-17k" / "dapo-math-17k.jsonl"
train_argv = json.loads(os.environ["TASK22_TRAIN_ARGS_JSON"])
runtime_env = json.loads(os.environ["TASK22_RUNTIME_ENV_JSON"])
runtime_env_common = json.loads(json.dumps(runtime_env))
runtime_env_common.get("env_vars", {}).pop("RELAX_SYNC_INTENT_POLICY", None)
source_paths = (
    repo / "relax/backends/megatron/actor.py",
    repo / "relax/components/rollout.py",
    repo / "relax/distributed/ray/rollout.py",
    repo / "relax/engine/rollout/sglang_rollout.py",
    repo / "relax/engine/rollout/sync_intent.py",
    repo / "relax/engine/rollout/sync_intent_rollout.py",
)
source_sha256 = {str(path.relative_to(repo)): sha256(path) for path in source_paths}
versions = {name: importlib.metadata.version(name) for name in ("ray", "sglang", "torch")}
megatron_spec = importlib.util.find_spec("megatron.core.tensor_parallel.layers")
if megatron_spec is None or megatron_spec.origin is None:
    raise SystemExit("cannot resolve Megatron tensor-parallel layers")
megatron_path = Path(megatron_spec.origin).resolve()
gpu_rows = []
gpu_output = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
for line in gpu_output.splitlines():
    index, uuid, name, memory_mib, driver = (field.strip() for field in line.split(",", maxsplit=4))
    gpu_rows.append(
        {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(memory_mib),
            "driver_version": driver,
        }
    )
comparison_payload = {
    "git_sha": os.environ["TASK22_GIT_SHA"],
    "train_argv": train_argv,
    "dataset_sha256": sha256(dataset),
    "model_config_sha256": sha256(model / "config.json"),
    "source_sha256": source_sha256,
    "versions": versions,
    "megatron_layers_sha256": sha256(megatron_path),
    "runtime_env_without_policy_flag": runtime_env_common,
    "topology": {
        "actor_gpus": 2,
        "rollout_engines": 2,
        "rollout_gpus_per_engine": 1,
    },
    "policy_parameters": {
        "ttl_seconds": os.environ["RELAX_SYNC_INTENT_TTL_SECONDS"],
        "window_groups": os.environ["RELAX_SYNC_INTENT_WINDOW_GROUPS"],
        "quiesce_multiplier": os.environ["RELAX_SYNC_INTENT_QUIESCE_MULTIPLIER"],
        "quiesce_floor_seconds": os.environ["RELAX_SYNC_INTENT_QUIESCE_FLOOR_SECONDS"],
    },
}
comparison_fingerprint = hashlib.sha256(
    json.dumps(comparison_payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
payload = {
    "schema_version": 1,
    "arm": os.environ["TASK22_AB_ARM"],
    "policy_enabled": os.environ["RELAX_SYNC_INTENT_POLICY"] == "1",
    "comparison_fingerprint": comparison_fingerprint,
    "comparison_payload": comparison_payload,
    "instrumentation": {
        "framework_standard_logs": True,
        "task22_calibration_timeline": False,
        "permit_jsonl": False,
        "scheduler_heartbeat": False,
        "gpu_high_frequency_sampling": False,
        "sglang_show_time_cost": False,
    },
    "gpu_inventory": gpu_rows,
}
Path(os.environ["TASK22_LOCAL_RUN_DIR"]).joinpath("run_contract.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

"$PYTHON_BIN" -m ray.scripts.scripts job submit \
    --address="http://127.0.0.1:8265" \
    --working-dir "$REPO" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- "$PYTHON_BIN" -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOCAL_RUN_DIR/driver.log"
