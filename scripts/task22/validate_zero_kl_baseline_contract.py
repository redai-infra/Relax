#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate the launch contract for the Task 22 zero-KL diagnostic baseline."""

from __future__ import annotations

import argparse
import json
from typing import Any


REQUIRED_FLAGS = (
    "--hybrid",
    "--partial-rollout",
    "--mask-offpolicy-in-partial-rollout",
    "--use-tis",
    "--sglang-show-time-cost",
    "--sequence-parallel",
    "--use-dynamic-batch-size",
    "--sglang-disable-piecewise-cuda-graph",
    "--skip-eval-before-train",
    "--apply-chat-template",
    "--rollout-shuffle",
    "--use-fault-tolerance",
    "--balance-data",
)

FORBIDDEN_FLAGS = (
    "--ref-load",
    "--ref-update-interval",
    "--use-kl-loss",
    "--kl-loss-coef",
    "--kl-loss-type",
    "--kl-coef",
    "--enable-cross-version-kv-continuation",
    "--cross-version-kv-max-gap",
    "--keep-old-actor",
    "--sglang-enable-priority-scheduling",
    "--sglang-disable-priority-preemption",
    "--sglang-default-priority-value",
    "--fully-async",
    "--colocate",
)

EXPECTED_VALUES = {
    "--num-rollout": "11",
    "--max-staleness": "2",
    "--num-data-storage-units": "1",
    "--num-iters-per-train-update": "1",
    "--rollout-batch-size": "8",
    "--n-samples-per-prompt": "8",
    "--rollout-max-response-len": "8192",
    "--rollout-temperature": "1",
    "--rollout-seed": "42",
    "--global-batch-size": "64",
    "--over-sampling-batch-size": "16",
    "--partial-rollout-max-aborted-count": "2",
    "--update-weights-interval": "1",
    "--advantage-estimator": "grpo",
    "--optimizer": "adam",
    "--tensor-model-parallel-size": "2",
    "--pipeline-model-parallel-size": "1",
    "--context-parallel-size": "1",
    "--expert-model-parallel-size": "1",
    "--expert-tensor-parallel-size": "1",
    "--max-tokens-per-gpu": "9216",
    "--rollout-num-gpus-per-engine": "1",
    "--sglang-mem-fraction-static": "0.8",
    "--sglang-attention-backend": "triton",
    "--sglang-sampling-backend": "pytorch",
    "--sglang-cuda-graph-max-bs": "64",
    "--attention-backend": "flash",
    "--seed": "1234",
}

EXPECTED_RESOURCE = {"actor": [1, 2], "rollout": [1, 2]}
DISABLED_VALUES = {"0", "false", "off", "no"}


class ContractError(ValueError):
    """Raised when the submitted launch no longer matches the baseline."""


def _positions(argv: list[str], flag: str) -> list[int]:
    return [index for index, item in enumerate(argv) if item == flag or item.startswith(f"{flag}=")]


def _require_once(argv: list[str], flag: str) -> int:
    positions = _positions(argv, flag)
    if len(positions) != 1:
        raise ContractError(f"{flag} must appear exactly once, found {len(positions)}")
    return positions[0]


def _value(argv: list[str], flag: str) -> str:
    index = _require_once(argv, flag)
    item = argv[index]
    if item.startswith(f"{flag}="):
        return item.split("=", maxsplit=1)[1]
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ContractError(f"{flag} is missing a value")
    return argv[index + 1]


def validate_contract(runtime: Any, argv: Any) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        raise ContractError("runtime env must be a JSON object")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ContractError("train argv must be a JSON string array")

    for flag in REQUIRED_FLAGS:
        _require_once(argv, flag)
    for flag in FORBIDDEN_FLAGS:
        if _positions(argv, flag):
            raise ContractError(f"forbidden zero-KL baseline flag: {flag}")
    for flag, expected in EXPECTED_VALUES.items():
        actual = _value(argv, flag)
        if actual != expected:
            raise ContractError(f"{flag} mismatch: {actual!r} != {expected!r}")

    try:
        resource = json.loads(_value(argv, "--resource"))
    except json.JSONDecodeError as error:
        raise ContractError(f"--resource is not valid JSON: {error}") from error
    if resource != EXPECTED_RESOURCE:
        raise ContractError(f"--resource mismatch: {resource!r} != {EXPECTED_RESOURCE!r}")

    env = runtime.get("env_vars")
    if not isinstance(env, dict) or not all(isinstance(key, str) for key in env):
        raise ContractError("runtime env lacks a string-keyed env_vars object")
    calibration_dir = env.get("TASK22_CALIBRATION_DIR")
    if not isinstance(calibration_dir, str) or not calibration_dir.startswith("/root/task22-local/runs/"):
        raise ContractError("TASK22_CALIBRATION_DIR must use local /root/task22-local/runs storage")
    if env.get("SGLANG_LOG_SCHEDULER_STATUS_INTERVAL") != "1.0":
        raise ContractError("scheduler status interval must be 1.0 seconds")
    if env.get("SGLANG_LOG_SCHEDULER_STATUS_TARGET") != "stdout":
        raise ContractError("scheduler status target must be stdout")
    if env.get("RELAX_RID_ONLY_REQUEST_LOGGING") != "1":
        raise ContractError("RELAX_RID_ONLY_REQUEST_LOGGING must be 1")

    policy_value = str(env.get("RELAX_SYNC_INTENT_POLICY", "0")).strip().lower()
    if policy_value not in DISABLED_VALUES:
        raise ContractError("RELAX_SYNC_INTENT_POLICY must be disabled")
    forbidden_env = sorted(
        key
        for key in env
        if key.startswith("RELAX_TASK22_")
        or (key.startswith("RELAX_SYNC_INTENT_") and key != "RELAX_SYNC_INTENT_POLICY")
    )
    if forbidden_env:
        raise ContractError(f"optimization environment must be absent: {forbidden_env}")

    return {
        "schema_version": 1,
        "purpose": "zero_kl_instrumented_baseline",
        "num_rollout": 11,
        "headline": {"logical_step_lo": 2, "logical_step_hi": 9},
        "resource": EXPECTED_RESOURCE,
        "max_response_tokens": 8192,
        "zero_kl_reference_forward": False,
        "sync_intent_policy": False,
        "cross_version_kv_continuation": False,
        "priority_scheduling": False,
        "update_weights_interval": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-env-json", required=True)
    parser.add_argument("--train-args-json", required=True)
    args = parser.parse_args()

    try:
        summary = validate_contract(
            json.loads(args.runtime_env_json),
            json.loads(args.train_args_json),
        )
    except (ContractError, json.JSONDecodeError) as error:
        raise SystemExit(f"TASK22_ZERO_KL_PREFLIGHT contract=FAIL reason={error}") from error
    print(f"TASK22_ZERO_KL_PREFLIGHT contract=PASS summary={json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
