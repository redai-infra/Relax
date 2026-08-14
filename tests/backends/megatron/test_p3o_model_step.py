# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exception-safety tests for the P3O optimizer-step lifecycle."""

from __future__ import annotations

import ast
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.backends.megatron._megatron_stub import stubbed_megatron_modules


MODEL_PATH = Path(__file__).resolve().parents[3] / "relax" / "backends" / "megatron" / "model.py"

stream_dataloader = ModuleType("relax.utils.data.stream_dataloader")
stream_dataloader.StreamingTQIterator = object

with (
    patch.dict(sys.modules, {"relax.utils.data.stream_dataloader": stream_dataloader}),
    stubbed_megatron_modules(("megatron", "ray", "tensordict", "pybase64")),
):
    from relax.backends.megatron import model as model_module


_preserved_dynamic_cp_group = model_module._preserved_dynamic_cp_group


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


def test_p3o_model_step_rejects_global_zero_valid_tokens(monkeypatch):
    monkeypatch.setattr(model_module.mpu, "is_pipeline_last_stage", lambda **kwargs: True)
    monkeypatch.setattr(model_module.torch.distributed, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="no valid response tokens globally"):
        model_module._require_p3o_global_valid_tokens(
            torch.tensor(0.0),
            torch.device("cpu"),
        )


def test_p3o_model_step_allows_positive_global_valid_tokens(monkeypatch):
    monkeypatch.setattr(model_module.mpu, "is_pipeline_last_stage", lambda **kwargs: True)
    monkeypatch.setattr(model_module.torch.distributed, "is_available", lambda: False)

    model_module._require_p3o_global_valid_tokens(
        torch.tensor(2.0),
        torch.device("cpu"),
    )


def test_p3o_model_step_rejects_zero_step_context_before_training_schedule():
    with pytest.raises(RuntimeError, match="no valid response tokens globally"):
        model_module._require_p3o_step_context_tokens(SimpleNamespace(valid_token_count=torch.tensor(0.0)))

    model_module._require_p3o_step_context_tokens(SimpleNamespace(valid_token_count=torch.tensor(1.0)))


def test_p3o_model_step_finalizer_guard_skips_original_finalizer_on_zero_tokens(monkeypatch):
    original_calls = []
    config = SimpleNamespace(
        finalize_model_grads_func=lambda *args, **kwargs: original_calls.append((args, kwargs)),
    )
    model = [torch.nn.Linear(1, 1)]
    monkeypatch.setattr(model_module, "get_model_config", lambda _: config)
    monkeypatch.setattr(model_module.mpu, "is_pipeline_last_stage", lambda **kwargs: True)
    monkeypatch.setattr(model_module.torch.distributed, "is_available", lambda: False)
    original_finalizer = config.finalize_model_grads_func

    with pytest.raises(RuntimeError, match="no valid response tokens globally"):
        with model_module._p3o_valid_token_finalizer(Namespace(advantage_estimator="p3o"), model):
            config.finalize_model_grads_func(model, torch.tensor(0.0))

    assert original_calls == []
    assert config.finalize_model_grads_func is original_finalizer


def test_p3o_model_step_finalizer_guard_calls_original_and_restores_config(monkeypatch):
    original_calls = []
    config = SimpleNamespace(
        finalize_model_grads_func=lambda *args, **kwargs: original_calls.append((args, kwargs)),
    )
    model = [torch.nn.Linear(1, 1)]
    monkeypatch.setattr(model_module, "get_model_config", lambda _: config)
    monkeypatch.setattr(model_module.mpu, "is_pipeline_last_stage", lambda **kwargs: True)
    monkeypatch.setattr(model_module.torch.distributed, "is_available", lambda: False)
    original_finalizer = config.finalize_model_grads_func

    with model_module._p3o_valid_token_finalizer(Namespace(advantage_estimator="p3o"), model):
        config.finalize_model_grads_func(model, torch.tensor(2.0), pg_collection="pg")

    assert original_calls == [((model, torch.tensor(2.0)), {"pg_collection": "pg"})]
    assert config.finalize_model_grads_func is original_finalizer


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
    guarded_source = ast.dump(guard)
    assert "p3o_ess_scope" in guarded_source
    assert "micro-batch" in guarded_source
    assert "step" in guarded_source


def test_p3o_model_step_wraps_gradient_finalization_with_valid_token_guard():
    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    train_one_step = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_one_step"
    )
    p3o_finalizer_guard = next(
        node
        for node in ast.walk(train_one_step)
        if isinstance(node, ast.With)
        and any(
            isinstance(child, ast.Name) and child.id == "_p3o_valid_token_finalizer"
            for item in node.items
            for child in ast.walk(item.context_expr)
        )
    )
    guarded_calls = {
        child.func.id
        for child in ast.walk(p3o_finalizer_guard)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }

    assert "forward_backward_func" in guarded_calls
