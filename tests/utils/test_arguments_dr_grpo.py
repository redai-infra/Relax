# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
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


def test_pg_loss_aggregation_is_explicit(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    baseline = parser.parse_args([])
    dr_grpo = parser.parse_args(
        [
            "--disable-grpo-std-normalization",
            "--pg-loss-aggregation",
            "seq-mean-token-sum-norm",
            "--pg-loss-scale-factor",
            "1024",
            "--calculate-per-token-loss",
        ]
    )

    assert baseline.grpo_std_normalization is True
    assert baseline.pg_loss_aggregation == "seq-mean-token-mean"
    assert dr_grpo.grpo_std_normalization is False
    assert dr_grpo.pg_loss_aggregation == "seq-mean-token-sum-norm"
    assert dr_grpo.pg_loss_scale_factor == 1024
    assert dr_grpo.calculate_per_token_loss is True


@pytest.mark.parametrize(
    (
        "advantage_estimator",
        "pg_loss_scale_factor",
        "rollout_max_response_len",
        "calculate_per_token_loss",
        "fully_async",
        "match",
    ),
    [
        ("ppo", 1024, 2048, False, False, "requires --advantage-estimator grpo"),
        ("grpo", None, None, False, False, "requires a positive --pg-loss-scale-factor"),
        ("grpo", 1024, 2048, True, True, "does not support --fully-async"),
    ],
)
def test_pg_loss_aggregation_rejects_invalid_combinations(
    arguments_module,
    advantage_estimator,
    pg_loss_scale_factor,
    rollout_max_response_len,
    calculate_per_token_loss,
    fully_async,
    match,
):
    args = SimpleNamespace(
        pg_loss_aggregation="seq-mean-token-sum-norm",
        advantage_estimator=advantage_estimator,
        pg_loss_scale_factor=pg_loss_scale_factor,
        rollout_max_response_len=rollout_max_response_len,
        calculate_per_token_loss=calculate_per_token_loss,
        fully_async=fully_async,
    )

    with pytest.raises(ValueError, match=match):
        arguments_module._validate_pg_loss_aggregation(args)


class _FakeLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.warnings.append(msg)


def test_pg_loss_aggregation_warns_without_std_normalization_disabled(arguments_module, monkeypatch):
    args = SimpleNamespace(
        pg_loss_aggregation="seq-mean-token-sum-norm",
        advantage_estimator="grpo",
        pg_loss_scale_factor=1024,
        rollout_max_response_len=2048,
        calculate_per_token_loss=False,
        fully_async=False,
        grpo_std_normalization=True,
    )
    fake_logger = _FakeLogger()
    monkeypatch.setattr(arguments_module, "logger", fake_logger)

    arguments_module._validate_pg_loss_aggregation(args)

    assert any("--disable-grpo-std-normalization" in message for message in fake_logger.warnings)


def test_pg_loss_aggregation_is_silent_when_std_normalization_disabled(arguments_module, monkeypatch):
    args = SimpleNamespace(
        pg_loss_aggregation="seq-mean-token-sum-norm",
        advantage_estimator="grpo",
        pg_loss_scale_factor=1024,
        rollout_max_response_len=2048,
        calculate_per_token_loss=False,
        fully_async=False,
        grpo_std_normalization=False,
    )
    fake_logger = _FakeLogger()
    monkeypatch.setattr(arguments_module, "logger", fake_logger)

    arguments_module._validate_pg_loss_aggregation(args)

    assert all("--disable-grpo-std-normalization" not in message for message in fake_logger.warnings)


def test_explicit_pg_loss_scale_factor_overrides_rollout_max_response_length(arguments_module):
    args = SimpleNamespace(
        pg_loss_aggregation="seq-mean-token-sum-norm",
        advantage_estimator="grpo",
        pg_loss_scale_factor=1024,
        rollout_max_response_len=1536,
        calculate_per_token_loss=False,
        fully_async=False,
        grpo_std_normalization=False,
    )

    arguments_module._validate_pg_loss_aggregation(args)

    assert args.pg_loss_scale_factor == 1024
