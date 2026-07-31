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
        get_mismatch_metrics=False,
        use_opsm=False,
        custom_pg_loss_reducer_function_path=None,
        enable_mtp_training=False,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        overlap_moe_expert_parallel_comm=False,
    )
    config.update(overrides)
    return Namespace(**config)


def test_p3o_arguments_accepts_a_valid_configuration():
    validate_p3o_args(_p3o_args())


def test_p3o_arguments_accepts_true_on_policy_scheduling():
    validate_p3o_args(_p3o_args(true_on_policy_mode=True))


@pytest.mark.parametrize(
    ("reason", "overrides"),
    [
        ("behavior policy would be undefined", dict(use_rollout_logprobs=False)),
        ("per-sample-mean reintroduces a micro-batch denominator", dict(calculate_per_token_loss=False)),
        ("TIS double-corrects the same mismatch", dict(use_tis=True)),
        ("P3O is critic-free", dict(use_critic=True)),
        ("FP8 amax history breaks replay", dict(fp8="hybrid")),
        ("attention dropout breaks replay", dict(attention_dropout=0.1)),
        ("hidden dropout breaks replay", dict(hidden_dropout=0.1)),
        ("async streaming hides the window", dict(fully_async=True)),
        ("mismatch metrics add an unverified extra forward", dict(get_mismatch_metrics=True)),
        ("OPSM changes the policy-gradient mask", dict(use_opsm=True)),
        (
            "custom reducer may change token-sum normalization",
            dict(custom_pg_loss_reducer_function_path="pkg.reducer"),
        ),
        ("MTP changes forward state between replay passes", dict(enable_mtp_training=True)),
        ("training routing replay changes the replayed forward", dict(use_routing_replay=True)),
        ("rollout routing replay changes the replayed forward", dict(use_rollout_routing_replay=True)),
        ("combined 1F1B bypasses the standard forward", dict(overlap_moe_expert_parallel_comm=True)),
    ],
)
def test_p3o_arguments_rejects_configs_that_change_the_objective(reason, overrides):
    with pytest.raises((AssertionError, ValueError)):
        validate_p3o_args(_p3o_args(**overrides))


def test_p3o_arguments_validate_after_effective_value_overrides():
    tree = ast.parse(ARGUMENTS_PATH.read_text(encoding="utf-8"))
    validator = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "slime_validate_args"
    )
    calls = [
        node
        for node in ast.walk(validator)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_validate_p3o_args"
    ]
    assert len(calls) == 1

    custom_config_if = next(
        node
        for node in ast.walk(validator)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "custom_config_path" for child in ast.walk(node.test)
        )
    )
    rollout_routing_if = next(
        node
        for node in ast.walk(validator)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "use_rollout_routing_replay"
            for child in ast.walk(node.test)
        )
    )
    assert calls[0].lineno > custom_config_if.end_lineno
    assert calls[0].lineno > rollout_routing_if.end_lineno
