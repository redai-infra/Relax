#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate the fixed Qwen3-4B Task 22 PR211 × PR242 experiment arms."""

from __future__ import annotations

import argparse
import json
from typing import Any


ARMS = {
    "interval1_kv_off": {"interval": "1", "kv": False, "max_gap": None},
    "interval2_kv_off": {"interval": "2", "kv": False, "max_gap": None},
    "interval1_kv_on": {"interval": "1", "kv": True, "max_gap": "2"},
    "interval2_kv_on": {"interval": "2", "kv": True, "max_gap": "1"},
}
COMMON_REQUIRED_FLAGS = (
    "--hybrid",
    "--partial-rollout",
    "--mask-offpolicy-in-partial-rollout",
    "--use-tis",
    "--use-fault-tolerance",
    "--balance-data",
    "--apply-chat-template",
    "--rollout-shuffle",
    "--sequence-parallel",
    "--use-dynamic-batch-size",
    "--sglang-show-time-cost",
    "--sglang-disable-piecewise-cuda-graph",
    "--skip-eval-before-train",
)
COMMON_FORBIDDEN_FLAGS = (
    "--use-kl-loss",
    "--kl-loss-coef",
    "--kl-coef",
    "--ref-load",
    "--keep-old-actor",
    "--use-rollout-logprobs",
    "--fully-async",
    "--colocate",
)
FIXED_VALUES = {
    "--max-staleness": "2",
    "--rollout-batch-size": "8",
    "--n-samples-per-prompt": "8",
    "--rollout-max-response-len": "8192",
    "--global-batch-size": "64",
    "--over-sampling-batch-size": "16",
    "--partial-rollout-max-aborted-count": "2",
    "--rollout-temperature": "1",
    "--rollout-seed": "42",
    "--seed": "1234",
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
}
KV_REQUIRED_FLAGS = (
    "--enable-cross-version-kv-continuation",
    "--use-slime-router",
    "--hybrid-dcs-weight-sync",
    "--hybrid-weights-backuper-on-gpu",
)


class ContractError(ValueError):
    """The submitted command does not match the paired experiment contract."""


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


def validate_contract(*, arm: str, num_rollout: int, argv: Any) -> dict[str, Any]:
    if arm not in ARMS:
        raise ContractError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ContractError("train argv must be a JSON string array")
    if num_rollout < 3 or num_rollout > 21:
        raise ContractError(f"num_rollout must be in [3, 21], got {num_rollout}")

    for flag in COMMON_REQUIRED_FLAGS:
        _require_once(argv, flag)
    for flag in COMMON_FORBIDDEN_FLAGS:
        if _positions(argv, flag):
            raise ContractError(f"forbidden common flag: {flag}")
    for flag, expected in FIXED_VALUES.items():
        actual = _value(argv, flag)
        if actual != expected:
            raise ContractError(f"{flag} mismatch: {actual!r} != {expected!r}")
    if _value(argv, "--num-rollout") != str(num_rollout):
        raise ContractError("--num-rollout does not match the declared run contract")
    if json.loads(_value(argv, "--resource")) != {"actor": [1, 2], "rollout": [1, 2]}:
        raise ContractError("--resource must allocate 2 Actor GPUs and 2 Rollout GPUs")

    spec = ARMS[arm]
    if _value(argv, "--update-weights-interval") != spec["interval"]:
        raise ContractError("--update-weights-interval does not match the selected arm")
    if spec["kv"]:
        for flag in KV_REQUIRED_FLAGS:
            _require_once(argv, flag)
        if _value(argv, "--cross-version-kv-max-gap") != spec["max_gap"]:
            raise ContractError("--cross-version-kv-max-gap does not match the selected arm")
        if _value(argv, "--checkpoint-engine-backend") != "nccl":
            raise ContractError("KV ON arms require --checkpoint-engine-backend nccl")
        if _value(argv, "--targeted-retirement-timeout-seconds") != "15":
            raise ContractError("KV ON arms require a 15 second targeted-retirement timeout")
    else:
        for flag in (
            *KV_REQUIRED_FLAGS,
            "--cross-version-kv-max-gap",
            "--checkpoint-engine-backend",
            "--targeted-retirement-timeout-seconds",
        ):
            if _positions(argv, flag):
                raise ContractError(f"KV OFF arm contains candidate flag: {flag}")

    effective_gap = int(spec["interval"]) * int(spec["max_gap"] or 0)
    if spec["kv"] and effective_gap > 2:
        raise ContractError("effective cross-version KV gap exceeds max_staleness")
    return {
        "schema_version": 1,
        "arm": arm,
        "num_rollout": num_rollout,
        "update_weights_interval": int(spec["interval"]),
        "cross_version_kv": bool(spec["kv"]),
        "cross_version_kv_max_gap": int(spec["max_gap"]) if spec["max_gap"] else None,
        "effective_actor_step_gap": effective_gap if spec["kv"] else None,
        "max_staleness": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--num-rollout", required=True, type=int)
    parser.add_argument("--train-args-json", required=True)
    args = parser.parse_args()
    try:
        summary = validate_contract(
            arm=args.arm,
            num_rollout=args.num_rollout,
            argv=json.loads(args.train_args_json),
        )
    except (ContractError, json.JSONDecodeError) as error:
        raise SystemExit(f"TASK22_INTEGRATION_PREFLIGHT verdict=FAIL reason={error}") from error
    print(f"TASK22_INTEGRATION_PREFLIGHT verdict=PASS summary={json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
