# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

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


def _build_parser(arguments_module) -> argparse.ArgumentParser:
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-loss-scale",
        type=float,
        default=2**32,
        help="Megatron initial loss scale.",
    )
    parser.add_argument(
        "--min-loss-scale",
        type=float,
        default=1.0,
        help="Megatron minimum loss scale.",
    )
    parser.add_argument(
        "--use-precision-aware-optimizer",
        action="store_true",
        default=False,
        help="Megatron precision-aware optimizer.",
    )
    arguments_module.get_slime_extra_args_provider()(parser)
    return parser


def _precision_args(**overrides) -> SimpleNamespace:
    values = {
        "fp16": True,
        "bf16": False,
        "initial_loss_scale": None,
        "min_loss_scale": None,
        "use_precision_aware_optimizer": None,
        "store_param_remainders": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fp16_optimizer_options_reuse_megatron_actions_with_tristate_defaults(arguments_module):
    parser = _build_parser(arguments_module)

    args = parser.parse_args([])

    assert args.initial_loss_scale is None
    assert args.min_loss_scale is None
    assert args.use_precision_aware_optimizer is None
    assert args.store_param_remainders is None
    expected_help = {
        "--initial-loss-scale": "32768.0",
        "--min-loss-scale": "1.0",
        "--use-precision-aware-optimizer": "enables it",
        "--store-param-remainders": "disables it",
    }
    for option, fallback_text in expected_help.items():
        actions = [action for action in parser._actions if option in action.option_strings]
        assert len(actions) == 1
        assert "FP16" in actions[0].help
        assert fallback_text in actions[0].help


def test_fp16_optimizer_options_parse_explicit_positive_values(arguments_module):
    parser = _build_parser(arguments_module)

    args = parser.parse_args(
        [
            "--initial-loss-scale",
            "65536",
            "--min-loss-scale",
            "2",
            "--use-precision-aware-optimizer",
            "--store-param-remainders",
        ]
    )

    assert args.initial_loss_scale == 65536.0
    assert args.min_loss_scale == 2.0
    assert args.use_precision_aware_optimizer is True
    assert args.store_param_remainders is True


def test_fp16_optimizer_options_parse_explicit_negative_values(arguments_module):
    parser = _build_parser(arguments_module)

    args = parser.parse_args(
        [
            "--no-use-precision-aware-optimizer",
            "--no-store-param-remainders",
        ]
    )

    assert args.use_precision_aware_optimizer is False
    assert args.store_param_remainders is False


@pytest.mark.parametrize(
    ("argv", "dest", "expected"),
    [
        (
            ["--use-precision-aware-optimizer", "--no-use-precision-aware-optimizer"],
            "use_precision_aware_optimizer",
            False,
        ),
        (
            ["--no-use-precision-aware-optimizer", "--use-precision-aware-optimizer"],
            "use_precision_aware_optimizer",
            True,
        ),
        (
            ["--store-param-remainders", "--no-store-param-remainders"],
            "store_param_remainders",
            False,
        ),
        (
            ["--no-store-param-remainders", "--store-param-remainders"],
            "store_param_remainders",
            True,
        ),
    ],
)
def test_fp16_optimizer_boolean_options_honor_last_occurrence(arguments_module, argv, dest, expected):
    parser = _build_parser(arguments_module)

    args = parser.parse_args(argv)

    assert getattr(args, dest) is expected


def test_fp16_omission_applies_safe_fallbacks_and_warns_once(arguments_module, monkeypatch):
    warning = Mock()
    monkeypatch.setattr(arguments_module.logger, "warning", warning)
    args = _precision_args()

    arguments_module._normalize_precision_optimizer_args(args)

    assert args.initial_loss_scale == 32768.0
    assert args.min_loss_scale == 1.0
    assert args.use_precision_aware_optimizer is True
    assert args.store_param_remainders is False
    warning.assert_called_once()
    warning_text = " ".join(str(value) for value in warning.call_args.args)
    assert "--initial-loss-scale=32768.0" in warning_text
    assert "--min-loss-scale=1.0" in warning_text
    assert "--use-precision-aware-optimizer=True" in warning_text
    assert "--store-param-remainders=False" in warning_text

    arguments_module._normalize_precision_optimizer_args(args)

    warning.assert_called_once()


def test_fp16_partial_omission_warns_once_with_only_missing_fallbacks(arguments_module, monkeypatch):
    warning = Mock()
    monkeypatch.setattr(arguments_module.logger, "warning", warning)
    args = _precision_args(initial_loss_scale=8192.0, use_precision_aware_optimizer=False)

    arguments_module._normalize_precision_optimizer_args(args)

    assert args.initial_loss_scale == 8192.0
    assert args.min_loss_scale == 1.0
    assert args.use_precision_aware_optimizer is False
    assert args.store_param_remainders is False
    warning.assert_called_once()
    warning_text = " ".join(str(value) for value in warning.call_args.args)
    assert "--initial-loss-scale" not in warning_text
    assert "--use-precision-aware-optimizer" not in warning_text
    assert "--min-loss-scale=1.0" in warning_text
    assert "--store-param-remainders=False" in warning_text


def test_fp16_explicit_values_are_not_overwritten_or_warned(arguments_module, monkeypatch):
    warning = Mock()
    monkeypatch.setattr(arguments_module.logger, "warning", warning)
    args = _precision_args(
        initial_loss_scale=16384.0,
        min_loss_scale=4.0,
        use_precision_aware_optimizer=False,
        store_param_remainders=False,
    )

    arguments_module._normalize_precision_optimizer_args(args)

    assert vars(args) == {
        "fp16": True,
        "bf16": False,
        "initial_loss_scale": 16384.0,
        "min_loss_scale": 4.0,
        "use_precision_aware_optimizer": False,
        "store_param_remainders": False,
    }
    warning.assert_not_called()


@pytest.mark.parametrize("bf16", [False, True])
def test_non_fp16_omission_applies_megatron_defaults_without_warning(arguments_module, monkeypatch, bf16):
    warning = Mock()
    monkeypatch.setattr(arguments_module.logger, "warning", warning)
    args = _precision_args(fp16=False, bf16=bf16)

    arguments_module._normalize_precision_optimizer_args(args)

    assert args.initial_loss_scale == 2**32
    assert args.min_loss_scale == 1.0
    assert args.use_precision_aware_optimizer is False
    assert args.store_param_remainders is True
    warning.assert_not_called()


@pytest.mark.parametrize("bf16", [False, True])
def test_non_fp16_explicit_unused_scales_remain_compatible(arguments_module, monkeypatch, bf16):
    warning = Mock()
    monkeypatch.setattr(arguments_module.logger, "warning", warning)
    args = _precision_args(
        fp16=False,
        bf16=bf16,
        initial_loss_scale=0.0,
        min_loss_scale=-1.0,
        use_precision_aware_optimizer=False,
        store_param_remainders=True,
    )

    arguments_module._normalize_precision_optimizer_args(args)

    assert args.initial_loss_scale == 0.0
    assert args.min_loss_scale == -1.0
    warning.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value", "option"),
    [
        ("initial_loss_scale", "32768", "--initial-loss-scale"),
        ("initial_loss_scale", float("nan"), "--initial-loss-scale"),
        ("initial_loss_scale", float("inf"), "--initial-loss-scale"),
        ("initial_loss_scale", 0.0, "--initial-loss-scale"),
        ("initial_loss_scale", -1.0, "--initial-loss-scale"),
        ("min_loss_scale", "1", "--min-loss-scale"),
        ("min_loss_scale", float("nan"), "--min-loss-scale"),
        ("min_loss_scale", float("-inf"), "--min-loss-scale"),
        ("min_loss_scale", 0.0, "--min-loss-scale"),
        ("min_loss_scale", -1.0, "--min-loss-scale"),
    ],
)
def test_precision_optimizer_scales_reject_invalid_values(arguments_module, field, value, option):
    args = _precision_args(
        initial_loss_scale=32768.0,
        min_loss_scale=1.0,
        use_precision_aware_optimizer=True,
        store_param_remainders=False,
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match=option):
        arguments_module._normalize_precision_optimizer_args(args)


def test_precision_optimizer_rejects_minimum_scale_above_initial_scale(arguments_module):
    args = _precision_args(
        initial_loss_scale=8.0,
        min_loss_scale=16.0,
        use_precision_aware_optimizer=True,
        store_param_remainders=False,
    )

    with pytest.raises(ValueError, match="--min-loss-scale"):
        arguments_module._normalize_precision_optimizer_args(args)


def test_fp16_preserves_explicit_parameter_remainders(arguments_module):
    args = _precision_args(
        initial_loss_scale=32768.0,
        min_loss_scale=1.0,
        use_precision_aware_optimizer=True,
        store_param_remainders=True,
    )

    arguments_module._normalize_precision_optimizer_args(args)

    assert args.store_param_remainders is True
