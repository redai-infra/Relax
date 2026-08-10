# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def arguments_module(monkeypatch):
    router_pkg = ModuleType("sglang_router")
    launch_router = ModuleType("sglang_router.launch_router")
    launch_router.RouterArgs = object
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)

    sglang_arguments = ModuleType("relax.backends.sglang.arguments")
    sglang_arguments.sglang_parse_args = lambda: None
    sglang_arguments.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", sglang_arguments)

    device = ModuleType("relax.utils.device")
    device.get_dist_backend = lambda: "gloo"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    eval_config = ModuleType("relax.utils.training.eval_config")
    eval_config.EvalDatasetConfig = dict
    eval_config.build_eval_dataset_configs = lambda args, datasets_config, defaults: []
    eval_config.build_named_prompt_data_configs = lambda values: []
    eval_config.ensure_dataset_list = lambda values: values or []
    monkeypatch.setitem(sys.modules, "relax.utils.training.eval_config", eval_config)

    sys.modules.pop("relax.utils.arguments", None)
    module = importlib.import_module("relax.utils.arguments")
    yield module
    sys.modules.pop("relax.utils.arguments", None)


def _args(**overrides) -> SimpleNamespace:
    defaults = dict(
        advantage_estimator="dr_grpo",
        rollout_max_response_len=1024,
        fully_async=False,
        hybrid=False,
        rewards_normalization=True,
        kl_coef=0.0,
        calculate_per_token_loss=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize("response_budget", [0, True])
def test_dr_grpo_rejects_invalid_response_budget(arguments_module, response_budget):
    with pytest.raises(ValueError, match="positive integer"):
        arguments_module._validate_dr_grpo_args(_args(rollout_max_response_len=response_budget))


@pytest.mark.parametrize(("hybrid", "should_raise"), [(False, True), (True, False)])
def test_dr_grpo_validates_async_execution_mode(arguments_module, hybrid, should_raise):
    args = _args(fully_async=True, hybrid=hybrid)

    if should_raise:
        with pytest.raises(ValueError, match="pure --fully-async"):
            arguments_module._validate_dr_grpo_args(args)
    else:
        arguments_module._validate_dr_grpo_args(args)
        assert args.calculate_per_token_loss is True


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"rewards_normalization": False}, "disable-rewards-normalization"),
        ({"kl_coef": 0.01}, "does not apply reward-side KL"),
    ],
)
def test_dr_grpo_rejects_incompatible_semantics(arguments_module, overrides, error):
    with pytest.raises(ValueError, match=error):
        arguments_module._validate_dr_grpo_args(_args(**overrides))


def test_valid_dr_grpo_enables_per_token_loss(arguments_module):
    args = _args()

    arguments_module._validate_dr_grpo_args(args)

    assert args.calculate_per_token_loss is True
