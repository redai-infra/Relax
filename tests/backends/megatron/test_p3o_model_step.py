# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exception-safety tests for the P3O optimizer-step lifecycle."""

from __future__ import annotations

import ast
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.backends.megatron.model import _preserved_dynamic_cp_group


MODEL_PATH = Path(__file__).resolve().parents[3] / "relax" / "backends" / "megatron" / "model.py"


def test_p3o_model_step_restores_dynamic_cp_group_after_error():
    original_group = object()
    dynamic_group = object()
    inner = SimpleNamespace(pg_collection=SimpleNamespace(cp=original_group))
    wrapped = SimpleNamespace(module=inner)
    args = Namespace(dynamic_context_parallel=True)

    with pytest.raises(RuntimeError, match="stats pass failed"):
        with _preserved_dynamic_cp_group(args, [wrapped]):
            inner.pg_collection.cp = dynamic_group
            raise RuntimeError("stats pass failed")

    assert inner.pg_collection.cp is original_group


def test_p3o_model_step_guard_covers_stats_and_train_passes():
    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    train_one_step = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_one_step"
    )
    guard = next(
        node
        for node in ast.walk(train_one_step)
        if isinstance(node, ast.With)
        and any(
            isinstance(child, ast.Name) and child.id == "_preserved_dynamic_cp_group"
            for item in node.items
            for child in ast.walk(item.context_expr)
        )
    )
    guarded_calls = {
        child.func.id for child in ast.walk(guard) if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    assert "compute_p3o_step_context" in guarded_calls
    assert "forward_backward_func" in guarded_calls
