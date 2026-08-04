# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "task22" / "validate_zero_kl_baseline_contract.py"
SPEC = importlib.util.spec_from_file_location("validate_zero_kl_baseline_contract", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)

ContractError = VALIDATOR.ContractError
validate_contract = VALIDATOR.validate_contract


def _valid_runtime():
    return {
        "env_vars": {
            "TASK22_CALIBRATION_DIR": "/root/task22-local/runs/zero_kl_baseline",
            "SGLANG_LOG_SCHEDULER_STATUS_INTERVAL": "1.0",
            "SGLANG_LOG_SCHEDULER_STATUS_TARGET": "stdout",
            "RELAX_RID_ONLY_REQUEST_LOGGING": "1",
            "RELAX_SYNC_INTENT_POLICY": "0",
        }
    }


def _valid_argv():
    flags = list(VALIDATOR.REQUIRED_FLAGS)
    values = []
    for flag, value in VALIDATOR.EXPECTED_VALUES.items():
        values.extend([flag, value])
    values.extend(["--resource", '{"rollout":[1,2],"actor":[1,2]}'])
    return [*flags, *values]


def test_contract_accepts_short_instrumented_zero_kl_baseline() -> None:
    summary = validate_contract(_valid_runtime(), _valid_argv())

    assert summary["num_rollout"] == 11
    assert summary["headline"] == {"logical_step_lo": 2, "logical_step_hi": 9}
    assert summary["max_response_tokens"] == 8192
    assert summary["zero_kl_reference_forward"] is False
    assert summary["sync_intent_policy"] is False
    assert summary["cross_version_kv_continuation"] is False
    assert summary["priority_scheduling"] is False
    assert summary["update_weights_interval"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (("replace", "--num-rollout", "13"), "--num-rollout mismatch"),
        (("replace", "--rollout-max-response-len", "4096"), "--rollout-max-response-len mismatch"),
        (("replace", "--resource", '{"actor":[1,1],"rollout":[1,3]}'), "--resource mismatch"),
        (("append", "--ref-load", "/model"), "forbidden zero-KL baseline flag: --ref-load"),
        (("append", "--use-kl-loss", None), "forbidden zero-KL baseline flag: --use-kl-loss"),
        (
            ("append", "--enable-cross-version-kv-continuation", None),
            "forbidden zero-KL baseline flag: --enable-cross-version-kv-continuation",
        ),
        (
            ("append", "--sglang-enable-priority-scheduling", None),
            "forbidden zero-KL baseline flag: --sglang-enable-priority-scheduling",
        ),
        (("remove", "--partial-rollout", None), "--partial-rollout must appear exactly once"),
    ],
)
def test_contract_rejects_launch_drift(mutation, expected) -> None:
    argv = _valid_argv()
    operation, flag, replacement = mutation
    if operation == "replace":
        argv[argv.index(flag) + 1] = replacement
    elif operation == "append":
        argv.append(flag)
        if replacement is not None:
            argv.append(replacement)
    else:
        argv.remove(flag)

    with pytest.raises(ContractError, match=expected):
        validate_contract(_valid_runtime(), argv)


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("RELAX_SYNC_INTENT_POLICY", "1", "RELAX_SYNC_INTENT_POLICY must be disabled"),
        (
            "RELAX_SYNC_INTENT_WINDOW_GROUPS",
            "16",
            "optimization environment must be absent",
        ),
        (
            "RELAX_TASK22_SYNC_INTENT_GUARD",
            "1",
            "optimization environment must be absent",
        ),
        (
            "SGLANG_LOG_SCHEDULER_STATUS_INTERVAL",
            "20",
            "scheduler status interval must be 1.0 seconds",
        ),
        (
            "RELAX_RID_ONLY_REQUEST_LOGGING",
            "0",
            "RELAX_RID_ONLY_REQUEST_LOGGING must be 1",
        ),
    ],
)
def test_contract_rejects_runtime_drift(key, value, expected) -> None:
    runtime = copy.deepcopy(_valid_runtime())
    runtime["env_vars"][key] = value

    with pytest.raises(ContractError, match=expected):
        validate_contract(runtime, _valid_argv())


def test_contract_requires_local_calibration_writes() -> None:
    runtime = _valid_runtime()
    runtime["env_vars"]["TASK22_CALIBRATION_DIR"] = "/root/autodl-fs/task22/live"

    with pytest.raises(ContractError, match="must use local"):
        validate_contract(runtime, _valid_argv())
