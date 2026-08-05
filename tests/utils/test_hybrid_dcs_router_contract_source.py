# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
from pathlib import Path


ARGUMENTS_PATH = Path(__file__).resolve().parents[2] / "relax" / "utils" / "arguments.py"
SGLANG_ROLLOUT_PATH = Path(__file__).resolve().parents[2] / "relax" / "engine" / "rollout" / "sglang_rollout.py"
KV_ROLLOUT_PATH = (
    Path(__file__).resolve().parents[2] / "relax" / "engine" / "rollout" / "cross_version_kv_rollout.py"
)
ROUTER_PATH = Path(__file__).resolve().parents[2] / "relax" / "engine" / "router" / "router.py"


def _hybrid_dcs_validation_source() -> str:
    tree = ast.parse(ARGUMENTS_PATH.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "slime_validate_args"
    )
    hybrid_dcs_guard = next(
        node
        for node in function.body
        if isinstance(node, ast.If) and "hybrid_dcs_weight_sync" in ast.unparse(node.test)
    )
    return ast.unparse(hybrid_dcs_guard)


def test_hybrid_dcs_with_cross_version_kv_keeps_router_requirement() -> None:
    source = _hybrid_dcs_validation_source()

    assert "use_slime_router" in source
    assert "targeted_retirement_timeout_seconds" in source


def test_removed_experimental_schedulers_are_absent_from_runtime_paths() -> None:
    arguments = ARGUMENTS_PATH.read_text(encoding="utf-8")
    sglang_rollout = SGLANG_ROLLOUT_PATH.read_text(encoding="utf-8")
    kv_rollout = KV_ROLLOUT_PATH.read_text(encoding="utf-8")
    router = ROUTER_PATH.read_text(encoding="utf-8")

    assert "--slime-router-work-aware" not in arguments
    assert "resolve_partition_request_priority" not in sglang_rollout
    assert "admission_policy_enabled" not in kv_rollout
    assert "work_accounting" not in router


def test_kv_entry_and_strict_fallback_preserve_final_experiment_path() -> None:
    sglang_rollout = SGLANG_ROLLOUT_PATH.read_text(encoding="utf-8")
    kv_rollout = KV_ROLLOUT_PATH.read_text(encoding="utf-8")

    assert "if cross_version_kv_enabled(args):" in sglang_rollout
    assert "sync_intent" not in sglang_rollout
    assert "await _submit_generate_tasks_debt_first(" in kv_rollout
    assert "1 if group_is_debt else 0" in kv_rollout
