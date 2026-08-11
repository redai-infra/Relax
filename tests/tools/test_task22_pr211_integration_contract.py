# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "task22" / "validate_pr211_integration_contract.py"
SPEC = importlib.util.spec_from_file_location("task22_integration_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def _common_argv(num_rollout: int = 21) -> list[str]:
    argv = [
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
        "--resource",
        '{"actor": [1, 2], "rollout": [1, 2]}',
        "--num-rollout",
        str(num_rollout),
    ]
    for flag, value in contract.FIXED_VALUES.items():
        argv.extend((flag, value))
    return argv


@pytest.mark.parametrize(
    ("arm", "interval", "max_gap"),
    [
        ("interval1_kv_off", "1", None),
        ("interval2_kv_off", "2", None),
        ("interval1_kv_on", "1", "2"),
        ("interval2_kv_on", "2", "1"),
    ],
)
def test_four_arms_have_a_valid_single_variable_contract(arm: str, interval: str, max_gap: str | None) -> None:
    argv = [*_common_argv(), "--update-weights-interval", interval]
    if max_gap is not None:
        argv.extend(contract.KV_REQUIRED_FLAGS)
        argv.extend(
            (
                "--cross-version-kv-max-gap",
                max_gap,
                "--checkpoint-engine-backend",
                "nccl",
                "--targeted-retirement-timeout-seconds",
                "15",
            )
        )

    summary = contract.validate_contract(arm=arm, num_rollout=21, argv=argv)

    assert summary["update_weights_interval"] == int(interval)
    assert summary["cross_version_kv"] is (max_gap is not None)


def test_combined_arm_rejects_unscaled_publication_gap() -> None:
    argv = [*_common_argv(), "--update-weights-interval", "2", *contract.KV_REQUIRED_FLAGS]
    argv.extend(
        (
            "--cross-version-kv-max-gap",
            "2",
            "--checkpoint-engine-backend",
            "nccl",
            "--targeted-retirement-timeout-seconds",
            "15",
        )
    )

    with pytest.raises(contract.ContractError, match="does not match"):
        contract.validate_contract(arm="interval2_kv_on", num_rollout=21, argv=argv)


def test_off_arm_rejects_candidate_flag_drift() -> None:
    argv = [*_common_argv(), "--update-weights-interval", "1", "--use-slime-router"]

    with pytest.raises(contract.ContractError, match="candidate flag"):
        contract.validate_contract(arm="interval1_kv_off", num_rollout=21, argv=argv)
