# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for P3O's rollout behavior-policy sampling contract."""

import ast
import types
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest


SGLANG_ROLLOUT_PATH = Path(__file__).resolve().parents[3] / "relax" / "engine" / "rollout" / "sglang_rollout.py"


def _load_p3o_sampling_validator():
    """Extract the dependency-free validation helper from the rollout
    source."""
    tree = ast.parse(SGLANG_ROLLOUT_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_P3O_TRUNCATION_SAMPLING_KEYS" for target in node.targets
        )
    )
    validator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_p3o_behavior_sampling_params"
    )
    module = types.ModuleType("_p3o_sampling_contract")
    module.Any = Any
    module.Namespace = Namespace
    exec(
        compile(ast.Module(body=[assignment, validator], type_ignores=[]), str(SGLANG_ROLLOUT_PATH), "exec"),
        module.__dict__,
    )
    return module._validate_p3o_behavior_sampling_params


validate_p3o_behavior_sampling_params = _load_p3o_sampling_validator()


def test_p3o_sampling_contract_allows_only_untruncated_training_sampling():
    args = Namespace(advantage_estimator="p3o")

    validate_p3o_behavior_sampling_params(args, {"top_p": 1.0, "top_k": -1}, evaluation=False)

    for sampling_params in ({"top_p": 0.9}, {"top_k": 32}, {"min_p": 0.1}):
        with pytest.raises(ValueError, match="P3O behavior sampling"):
            validate_p3o_behavior_sampling_params(args, sampling_params, evaluation=False)


def test_p3o_sampling_contract_leaves_evaluation_and_non_p3o_unchanged():
    validate_p3o_behavior_sampling_params(
        Namespace(advantage_estimator="p3o"),
        {"top_p": 0.9, "min_p": 0.1},
        evaluation=True,
    )
    validate_p3o_behavior_sampling_params(
        Namespace(advantage_estimator="grpo"),
        {"top_p": 0.9, "min_p": 0.1},
        evaluation=False,
    )


def test_p3o_dispatch_requires_an_explicit_custom_generation_contract():
    tree = ast.parse(SGLANG_ROLLOUT_PATH.read_text(encoding="utf-8"))
    dispatch = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_dispatch_generate"
    )
    string_constants = {
        node.value for node in ast.walk(dispatch) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "p3o_behavior_logprob_contract" in string_constants
