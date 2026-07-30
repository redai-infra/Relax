# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the P3O configuration gates in ``arguments.py``.

Every rejection below guards a config that still *trains* -- it just silently
optimizes something other than the P3O objective (uncorrected ratio, per-
micro-batch denominator, double correction) or breaks the pre-pass replay
(FP8 amax history, dropout). A plausible loss curve is the failure mode, so
these are hard errors rather than warnings and are worth pinning.

``relax.utils.arguments`` pulls in the Megatron/Ray import chain, which is not
available in the unit-test environment, so the validator is extracted from the
module source by AST rather than imported.
"""

import ast
import types
from argparse import Namespace
from pathlib import Path

import pytest


ARGUMENTS_PATH = Path(__file__).resolve().parents[2] / "relax" / "utils" / "arguments.py"


def _load_validator():
    """Extract ``_validate_p3o_args`` without importing arguments.py."""
    tree = ast.parse(ARGUMENTS_PATH.read_text(encoding="utf-8"))
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_validate_p3o_args")
    module = types.ModuleType("_p3o_args")
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(ARGUMENTS_PATH), "exec"), module.__dict__)
    return module._validate_p3o_args


validate_p3o_args = _load_validator()


def _p3o_args(**overrides) -> Namespace:
    """A minimal P3O-valid config, with individual fields overridable."""
    config = dict(
        advantage_estimator="p3o",
        use_rollout_logprobs=True,
        calculate_per_token_loss=True,
        use_tis=False,
        true_on_policy_mode=False,
        use_critic=False,
        fp8=None,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        fully_async=False,
    )
    config.update(overrides)
    return Namespace(**config)


def test_p3o_arguments_accepts_a_valid_configuration():
    validate_p3o_args(_p3o_args())


@pytest.mark.parametrize(
    ("reason", "overrides"),
    [
        ("behavior policy would be undefined", dict(use_rollout_logprobs=False)),
        ("per-sample-mean reintroduces a micro-batch denominator", dict(calculate_per_token_loss=False)),
        ("TIS double-corrects the same mismatch", dict(use_tis=True)),
        ("on-policy mode has no ratio to correct", dict(true_on_policy_mode=True)),
        ("P3O is critic-free", dict(use_critic=True)),
        ("FP8 amax history breaks replay", dict(fp8="hybrid")),
        ("attention dropout breaks replay", dict(attention_dropout=0.1)),
        ("hidden dropout breaks replay", dict(hidden_dropout=0.1)),
        ("async streaming hides the window", dict(fully_async=True)),
    ],
)
def test_p3o_arguments_rejects_configs_that_change_the_objective(reason, overrides):
    with pytest.raises((AssertionError, ValueError)):
        validate_p3o_args(_p3o_args(**overrides))
