# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils.megatron_peft_utils import build_mixture_lora_config, is_mixture_lora_enabled


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


def _args(**overrides):
    defaults = dict(
        lora_rank=16,
        lora_num_experts=4,
        lora_router_top_k=2,
        lora_router_temperature=1.0,
        lora_router_aux_loss_coef=0.01,
        lora_alpha=32,
        lora_target_modules=["linear_qkv", "linear_proj"],
        lora_merge_mode=False,
        lora_adapter_mode=False,
        fully_async=False,
        colocate=True,
        sglang_dp_size=1,
        sglang_tp_size=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_parser_defaults_preserve_single_lora_path(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    args = parser.parse_args([])

    assert args.lora_num_experts == 1
    assert args.lora_router_top_k is None
    assert args.lora_router_temperature is None
    assert args.lora_router_aux_loss_coef is None


def test_parser_accepts_explicit_mixture_lora_configuration(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    args = parser.parse_args(
        [
            "--lora-rank",
            "16",
            "--lora-num-experts",
            "4",
            "--lora-router-top-k",
            "2",
            "--lora-router-temperature",
            "0.8",
            "--lora-router-aux-loss-coef",
            "0.01",
        ]
    )

    assert args.lora_rank == 16
    assert args.lora_num_experts == 4
    assert args.lora_router_top_k == 2
    assert args.lora_router_temperature == 0.8
    assert args.lora_router_aux_loss_coef == 0.01


def test_valid_mixture_configuration_uses_shared_config(arguments_module):
    args = _args()

    arguments_module._validate_lora_args(args)
    config = build_mixture_lora_config(args)

    assert is_mixture_lora_enabled(args)
    assert config.num_experts == 4
    assert config.rank == 16
    assert config.top_k == 2
    assert config.temperature == 1.0
    assert config.aux_loss_coef == 0.01
    assert config.scale == 2.0
    assert args.lora_merge_mode is False
    assert args.lora_adapter_mode is False


def test_missing_mixture_values_are_reported_together(arguments_module):
    args = _args(
        lora_router_top_k=None,
        lora_router_temperature=None,
        lora_router_aux_loss_coef=None,
    )

    with pytest.raises(ValueError) as error:
        arguments_module._validate_lora_args(args)

    message = str(error.value)
    assert "--lora-router-top-k" in message
    assert "--lora-router-temperature" in message
    assert "--lora-router-aux-loss-coef" in message


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lora_num_experts": 0}, "lora-num-experts"),
        ({"lora_rank": 0}, "lora-rank"),
        ({"lora_router_top_k": 0}, "top_k"),
        ({"lora_router_top_k": 5}, "top_k"),
        ({"lora_router_temperature": 0.0}, "temperature"),
        ({"lora_router_temperature": float("inf")}, "temperature"),
        ({"lora_router_aux_loss_coef": -0.1}, "aux_loss_coef"),
        ({"lora_router_aux_loss_coef": float("inf")}, "aux_loss_coef"),
        ({"lora_target_modules": ["linear_qkv", "linear_qkv"]}, "duplicates"),
    ],
)
def test_mixture_configuration_rejects_invalid_values(arguments_module, overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        arguments_module._validate_lora_args(_args(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lora_merge_mode": True}, "merge-mode"),
        ({"lora_adapter_mode": True}, "adapter-mode"),
        ({"fully_async": True}, "fully-async"),
        ({"colocate": False}, "colocate"),
        ({"sglang_dp_size": 2}, "DP size 1"),
        ({"sglang_tp_size": 2}, "TP size 1"),
        ({"lora_target_modules": ["linear_fc1"]}, "linear_fc1"),
    ],
)
def test_mixture_configuration_rejects_unsupported_modes(arguments_module, overrides, message):
    with pytest.raises(ValueError, match=message):
        arguments_module._validate_lora_args(_args(**overrides))


def test_mixture_tp_size_can_be_derived_from_rollout_and_pp(arguments_module):
    args = _args(rollout_num_gpus_per_engine=2, sglang_pipeline_parallel_size=2)
    del args.sglang_tp_size

    arguments_module._validate_lora_args(args)


def test_single_lora_keeps_legacy_auto_merge_behavior(arguments_module):
    args = _args(
        lora_num_experts=1,
        lora_router_top_k=None,
        lora_router_temperature=None,
        lora_router_aux_loss_coef=None,
    )

    arguments_module._validate_lora_args(args)

    assert not is_mixture_lora_enabled(args)
    assert build_mixture_lora_config(args) is None
    assert args.lora_merge_mode is True
    assert args.lora_adapter_mode is False


def test_legacy_namespace_without_mixture_fields_is_unchanged(arguments_module):
    args = SimpleNamespace(lora_rank=8, lora_merge_mode=False, lora_adapter_mode=False, sglang_dp_size=1)

    arguments_module._validate_lora_args(args)

    assert args.lora_merge_mode is True


def test_single_lora_rejects_mixture_only_values(arguments_module):
    args = _args(
        lora_num_experts=1,
        lora_router_top_k=1,
        lora_router_temperature=None,
        lora_router_aux_loss_coef=None,
    )

    with pytest.raises(ValueError, match="lora-router-top-k"):
        arguments_module._validate_lora_args(args)
