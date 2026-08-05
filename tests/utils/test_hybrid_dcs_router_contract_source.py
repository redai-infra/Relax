# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
from pathlib import Path


ARGUMENTS_PATH = Path(__file__).resolve().parents[2] / "relax" / "utils" / "arguments.py"


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


def test_joint_hybrid_dcs_keeps_router_requirement_without_work_aware_dependency() -> None:
    source = _hybrid_dcs_validation_source()

    assert "use_slime_router" in source
    assert "targeted_retirement_timeout_seconds" in source
    assert "slime_router_work_aware" not in source
