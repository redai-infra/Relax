# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
import copy
from argparse import Namespace
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_source_definitions(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    """Execute selected stdlib-only definitions without importing
    Ray/Megatron."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    definitions = [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    found = {node.name for node in definitions}
    if found != names:
        raise AssertionError(f"Missing source definitions in {path}: {sorted(names - found)}")

    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def _reward_waiting_group_type():
    namespace = _load_source_definitions(
        REPO_ROOT / "relax" / "agentic" / "pipeline" / "reward.py",
        {"RewardWaitingGroup"},
        {
            "Any": Any,
            "PendingExportUnit": object,
            "copy": copy,
            "dataclass": dataclass,
            "field": field,
        },
    )
    return namespace["RewardWaitingGroup"]


def _build_rollout_plan(args: Namespace):
    namespace = _load_source_definitions(
        REPO_ROOT / "relax" / "backends" / "megatron" / "data.py",
        {"RolloutMiniBatchPlan", "build_rollout_minibatch_plan"},
        {
            "Namespace": Namespace,
            "dataclass": dataclass,
        },
    )
    return namespace["build_rollout_minibatch_plan"](args, dp_size=1)


def _unit(
    name: str,
    *,
    weight_versions: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    return SimpleNamespace(
        name=name,
        sample=SimpleNamespace(
            metadata={"unit_name": name, **(metadata or {})},
            weight_versions=list(weight_versions or []),
        ),
    )


def _variable_turn_group():
    waiting_group = _reward_waiting_group_type()(expected_count=2)
    waiting_group.add_slot(
        slot_idx=0,
        units=[_unit("slot0_turn0"), _unit("slot0_turn1")],
    )
    waiting_group.add_slot(
        slot_idx=1,
        units=[_unit("slot1_turn0"), _unit("slot1_turn1"), _unit("slot1_turn2")],
    )
    assert waiting_group.is_complete()
    return waiting_group


def _fixed_n_args(*, global_batch_size: int) -> Namespace:
    return Namespace(
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        global_batch_size=global_batch_size,
        num_steps_per_rollout=1,
    )


def test_fixed_n_windows_leave_variable_turn_rows_undrained():
    """Two trajectory slots with 2/3 turns produce five rows, but only two are
    planned."""
    waiting_group = _variable_turn_group()
    materialized_units = waiting_group.materialized_units()
    assert len(materialized_units) == 5

    plan = _build_rollout_plan(_fixed_n_args(global_batch_size=2))
    planned_capacity = plan.num_rollout_minis * plan.mini_global_samples
    assert planned_capacity == 2

    remaining = deque(materialized_units)
    for _ in range(plan.num_rollout_minis):
        for _ in range(min(plan.mini_global_samples, len(remaining))):
            remaining.popleft()

    assert [unit.name for unit in remaining] == [
        "slot1_turn0",
        "slot1_turn1",
        "slot1_turn2",
    ]
    assert remaining  # The fixed number of windows cannot report this partition drained.


def test_fixed_n_plan_rejects_using_actual_variable_row_count_as_global_batch():
    """The existing plan cannot be repaired by setting global_batch_size to the
    five emitted rows."""
    try:
        _build_rollout_plan(_fixed_n_args(global_batch_size=5))
    except ValueError as exc:
        assert "fixed-n_samples rollout mini size" in str(exc)
    else:
        raise AssertionError("The fixed-n plan unexpectedly accepted five variable turn rows.")


def test_reward_metadata_injects_relax_captured_policy_version_without_overwriting_source():
    group = _reward_waiting_group_type()(expected_count=1)
    unit = _unit("turn_000", weight_versions=["7", "7"])
    group.add_slot(slot_idx=0, units=[unit])

    payload = group.metadata_by_slot(require_policy_version=True)

    assert payload[0]["turn_000"]["policy_version"] == "7"
    assert unit.sample.metadata["policy_version"] == "7"


def test_reward_metadata_policy_version_fallback_and_conflicts_fail_closed():
    group_type = _reward_waiting_group_type()

    fallback = group_type(expected_count=1)
    fallback_unit = _unit("turn_000", metadata={"start_rollout_id": 3})
    fallback.add_slot(slot_idx=0, units=[fallback_unit])
    assert fallback.metadata_by_slot(require_policy_version=True)[0]["turn_000"]["policy_version"] == "3"

    mixed = group_type(expected_count=1)
    mixed_unit = _unit("turn_000", weight_versions=["7", "8"])
    mixed.add_slot(slot_idx=0, units=[mixed_unit])
    try:
        mixed.metadata_by_slot(require_policy_version=True)
    except RuntimeError as exc:
        assert "multiple policy versions" in str(exc)
    else:
        raise AssertionError("mixed policy versions were accepted")
    assert "policy_version" not in mixed_unit.sample.metadata

    conflicting = group_type(expected_count=1)
    conflicting_unit = _unit(
        "turn_000",
        weight_versions=["7"],
        metadata={"policy_version": "8"},
    )
    conflicting.add_slot(slot_idx=0, units=[conflicting_unit])
    try:
        conflicting.metadata_by_slot(require_policy_version=True)
    except RuntimeError as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("a user-declared conflicting policy version was overwritten")
    assert conflicting_unit.sample.metadata["policy_version"] == "8"
