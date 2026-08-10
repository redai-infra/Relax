# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


# The default CPU CI does not install Megatron. These tests run in the official
# Relax image, where they exercise the real parser and OptimizerConfig.
torch = pytest.importorskip("torch")
pytest.importorskip("megatron.training.arguments")

from relax.backends.megatron import arguments as megatron_arguments  # noqa: E402
from relax.backends.megatron import model  # noqa: E402
from relax.utils.arguments import (  # noqa: E402
    _add_fp16_optimizer_arguments,
    get_slime_extra_args_provider,
)


def _optimizer_args(fp16=True, **overrides):
    values = {
        "fp16": fp16,
        "loss_scale": None,
        "initial_loss_scale": None,
        "min_loss_scale": None,
        "use_precision_aware_optimizer": None,
        "store_param_remainders": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _capture_warnings(monkeypatch):
    messages = []

    def warning(message, *args):
        messages.append(message % args)

    monkeypatch.setattr(megatron_arguments.logger, "warning", warning)
    return messages


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], (None, None, None, None)),
        (
            [
                "--initial-loss-scale",
                "65536",
                "--min-loss-scale",
                "2",
                "--use-precision-aware-optimizer",
                "--store-param-remainders",
            ],
            (65536.0, 2.0, True, True),
        ),
        (
            ["--no-use-precision-aware-optimizer", "--no-store-param-remainders"],
            (None, None, False, False),
        ),
    ],
)
def test_fp16_optimizer_parser_preserves_unset_and_explicit_values(argv, expected):
    parser = argparse.ArgumentParser()
    _add_fp16_optimizer_arguments(parser)

    args = parser.parse_args(argv)

    assert (
        args.initial_loss_scale,
        args.min_loss_scale,
        args.use_precision_aware_optimizer,
        args.store_param_remainders,
    ) == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--fp16"], (None, None, None, None)),
        (
            [
                "--fp16",
                "--initial-loss-scale",
                "65536",
                "--min-loss-scale",
                "2",
                "--no-use-precision-aware-optimizer",
                "--store-param-remainders",
            ],
            (65536.0, 2.0, False, True),
        ),
        (
            [
                "--fp16",
                "--initial-loss-scale",
                "32768",
                "--min-loss-scale",
                "1",
                "--use-precision-aware-optimizer",
                "--no-store-param-remainders",
                "--initial-loss-scale",
                "65536",
                "--min-loss-scale",
                "2",
                "--no-use-precision-aware-optimizer",
                "--store-param-remainders",
            ],
            (65536.0, 2.0, False, True),
        ),
    ],
)
def test_megatron_parser_preserves_fp16_optimizer_values(monkeypatch, argv, expected):
    monkeypatch.setattr("sys.argv", ["test-fp16-parser", *argv])

    args = megatron_arguments._megatron_parse_args(
        extra_args_provider=get_slime_extra_args_provider(),
        ignore_unknown_args=True,
    )

    assert (
        args.initial_loss_scale,
        args.min_loss_scale,
        args.use_precision_aware_optimizer,
        args.store_param_remainders,
    ) == expected


def test_fp16_optimizer_unset_uses_legacy_fallback_and_warns_once(monkeypatch):
    args = _optimizer_args()
    warnings = _capture_warnings(monkeypatch)

    megatron_arguments._resolve_optimizer_precision_args(args)
    megatron_arguments._resolve_optimizer_precision_args(args)

    assert args.initial_loss_scale == 32768.0
    assert args.min_loss_scale == 1.0
    assert args.use_precision_aware_optimizer is True
    assert args.store_param_remainders is False
    assert len(warnings) == 1
    for option in (
        "--initial-loss-scale 32768.0",
        "--min-loss-scale 1.0",
        "--use-precision-aware-optimizer",
        "--no-store-param-remainders",
    ):
        assert option in warnings[0]


def test_fp16_optimizer_partial_explicit_values_are_preserved(monkeypatch):
    args = _optimizer_args(initial_loss_scale=65536.0, use_precision_aware_optimizer=False)
    warnings = _capture_warnings(monkeypatch)

    megatron_arguments._resolve_optimizer_precision_args(args)

    assert args.initial_loss_scale == 65536.0
    assert args.min_loss_scale == 1.0
    assert args.use_precision_aware_optimizer is False
    assert args.store_param_remainders is False
    assert len(warnings) == 1
    assert "--initial-loss-scale" not in warnings[0]
    assert "--min-loss-scale 1.0" in warnings[0]
    assert "--no-store-param-remainders" in warnings[0]


def test_fp16_optimizer_fully_explicit_values_do_not_warn(monkeypatch):
    args = _optimizer_args(
        initial_loss_scale=65536.0,
        min_loss_scale=2.0,
        use_precision_aware_optimizer=False,
        store_param_remainders=True,
    )
    warnings = _capture_warnings(monkeypatch)

    megatron_arguments._resolve_optimizer_precision_args(args)

    assert args.initial_loss_scale == 65536.0
    assert args.min_loss_scale == 2.0
    assert args.use_precision_aware_optimizer is False
    assert args.store_param_remainders is True
    assert warnings == []


def test_static_loss_scale_clears_inactive_dynamic_values(monkeypatch):
    args = _optimizer_args(
        loss_scale=1024.0,
        initial_loss_scale=0,
        min_loss_scale=float("nan"),
    )
    warnings = _capture_warnings(monkeypatch)

    megatron_arguments._resolve_optimizer_precision_args(args)

    assert args.initial_loss_scale is None
    assert args.min_loss_scale is None
    assert args.use_precision_aware_optimizer is True
    assert args.store_param_remainders is False
    assert len(warnings) == 1
    assert "--initial-loss-scale" not in warnings[0]
    assert "--min-loss-scale" not in warnings[0]
    assert "--use-precision-aware-optimizer" in warnings[0]
    assert "--no-store-param-remainders" in warnings[0]


@pytest.mark.parametrize(
    "loss_scale",
    [True, "1024", float("nan"), float("inf"), pytest.param(10**500, id="too-large"), 0.0, -1.0],
)
def test_static_loss_scale_rejects_invalid_values(loss_scale):
    args = _optimizer_args(loss_scale=loss_scale)

    with pytest.raises(ValueError, match="--loss-scale"):
        megatron_arguments._resolve_optimizer_precision_args(args)


def test_static_loss_scale_leaves_unset_dynamic_values_alone(monkeypatch):
    from megatron.core.optimizer import OptimizerConfig

    from relax.backends.megatron.model import _build_optimizer_config_kwargs

    args = _optimizer_args(
        loss_scale=1024.0,
        use_precision_aware_optimizer=False,
        store_param_remainders=True,
    )
    warnings = _capture_warnings(monkeypatch)

    megatron_arguments._resolve_optimizer_precision_args(args)
    config = OptimizerConfig(**_build_optimizer_config_kwargs(args))

    assert args.initial_loss_scale is None
    assert args.min_loss_scale is None
    assert config.loss_scale == 1024.0
    assert config.initial_loss_scale is None
    assert config.min_loss_scale is None
    assert warnings == []


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"initial_loss_scale": 0}, "--initial-loss-scale"),
        ({"initial_loss_scale": -1}, "--initial-loss-scale"),
        ({"initial_loss_scale": float("nan")}, "--initial-loss-scale"),
        ({"initial_loss_scale": float("inf")}, "--initial-loss-scale"),
        pytest.param(
            {"initial_loss_scale": 10**500},
            "--initial-loss-scale",
            id="initial-loss-scale-too-large",
        ),
        ({"initial_loss_scale": True}, "--initial-loss-scale"),
        ({"initial_loss_scale": "32768"}, "--initial-loss-scale"),
        ({"min_loss_scale": 0}, "--min-loss-scale"),
        ({"min_loss_scale": -1}, "--min-loss-scale"),
        ({"min_loss_scale": 65536, "initial_loss_scale": 32768}, "--min-loss-scale"),
        ({"use_precision_aware_optimizer": 1}, "--use-precision-aware-optimizer"),
        ({"store_param_remainders": 0}, "--store-param-remainders"),
    ],
)
def test_fp16_optimizer_invalid_values_fail_with_parameter_name(overrides, match):
    values = {
        "initial_loss_scale": 32768.0,
        "min_loss_scale": 1.0,
        "use_precision_aware_optimizer": True,
        "store_param_remainders": False,
    }
    values.update(overrides)
    args = _optimizer_args(**values)

    with pytest.raises(ValueError, match=match):
        megatron_arguments._resolve_optimizer_precision_args(args)


def test_bf16_unset_values_restore_native_megatron_defaults(monkeypatch):
    from megatron.core.optimizer import OptimizerConfig

    args = _optimizer_args(fp16=False)
    args.seq_length = None
    args.vocab_size = None
    args.padded_vocab_size = None
    args.tokenizer_model = "tokenizer"
    args.tokenizer_type = None
    warnings = _capture_warnings(monkeypatch)
    native_defaults = OptimizerConfig()

    megatron_arguments._set_default_megatron_args(args)

    assert args.bf16 is True
    for name in megatron_arguments._FP16_OPTIMIZER_FALLBACKS:
        assert getattr(args, name) == getattr(native_defaults, name)
    assert warnings == []


def test_non_fp16_explicit_values_are_preserved(monkeypatch):
    args = _optimizer_args(
        fp16=False,
        initial_loss_scale=16384.0,
        min_loss_scale=2.0,
        use_precision_aware_optimizer=True,
        store_param_remainders=False,
    )
    warnings = _capture_warnings(monkeypatch)

    megatron_arguments._resolve_optimizer_precision_args(args)

    assert args.initial_loss_scale == 16384.0
    assert args.min_loss_scale == 2.0
    assert args.use_precision_aware_optimizer is True
    assert args.store_param_remainders is False
    assert warnings == []


@pytest.mark.parametrize(
    ("field", "value", "option"),
    [
        ("use_precision_aware_optimizer", "false", "--use-precision-aware-optimizer"),
        ("store_param_remainders", 0, "--store-param-remainders"),
    ],
)
def test_non_fp16_optimizer_booleans_reject_non_boolean_values(field, value, option):
    args = _optimizer_args(
        fp16=False,
        use_precision_aware_optimizer=False,
        store_param_remainders=True,
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match=option):
        megatron_arguments._resolve_optimizer_precision_args(args)


def test_bf16_fallback_builds_valid_optimizer_config(monkeypatch):
    from megatron.core.optimizer import OptimizerConfig

    from relax.backends.megatron.model import _build_optimizer_config_kwargs

    args = _optimizer_args(fp16=False, bf16=True, params_dtype=torch.bfloat16)
    warnings = _capture_warnings(monkeypatch)

    megatron_arguments._resolve_optimizer_precision_args(args)
    config = OptimizerConfig(**_build_optimizer_config_kwargs(args))

    native_defaults = OptimizerConfig()
    for name in megatron_arguments._FP16_OPTIMIZER_FALLBACKS:
        assert getattr(config, name) == getattr(native_defaults, name)
    assert config.fp16 is False
    assert config.bf16 is True
    assert config.params_dtype is torch.bfloat16
    assert warnings == []


def test_model_optimizer_kwargs_keep_explicit_fp16_values(monkeypatch):
    from megatron.core.optimizer import OptimizerConfig

    from relax.backends.megatron.model import _build_optimizer_config_kwargs

    args = _optimizer_args(
        bf16=False,
        params_dtype=torch.float16,
        initial_loss_scale=65536.0,
        min_loss_scale=2.0,
        use_precision_aware_optimizer=False,
        store_param_remainders=True,
    )
    _capture_warnings(monkeypatch)

    kwargs = _build_optimizer_config_kwargs(args)
    config = OptimizerConfig(**kwargs)

    assert config.initial_loss_scale == 65536.0
    assert config.min_loss_scale == 2.0
    assert config.use_precision_aware_optimizer is False
    assert config.store_param_remainders is True
    assert config.fp16 is True
    assert config.bf16 is False
    assert config.params_dtype is torch.float16


def test_model_optimizer_kwargs_do_not_normalize_args(monkeypatch):
    from relax.backends.megatron.model import _build_optimizer_config_kwargs

    args = _optimizer_args()
    warnings = _capture_warnings(monkeypatch)
    original_values = vars(args).copy()

    kwargs = _build_optimizer_config_kwargs(args)

    assert vars(args) == original_values
    assert kwargs["initial_loss_scale"] is None
    assert kwargs["min_loss_scale"] is None
    assert kwargs["use_precision_aware_optimizer"] is None
    assert kwargs["store_param_remainders"] is None
    assert warnings == []


def test_legacy_fp16_fallback_builds_valid_optimizer_config(monkeypatch):
    from megatron.core.optimizer import OptimizerConfig

    from relax.backends.megatron.model import _build_optimizer_config_kwargs

    args = _optimizer_args(use_distributed_optimizer=True, optimizer="adam")
    warnings = _capture_warnings(monkeypatch)

    megatron_arguments._resolve_optimizer_precision_args(args)
    config = OptimizerConfig(**_build_optimizer_config_kwargs(args))

    assert config.initial_loss_scale == 32768.0
    assert config.min_loss_scale == 1.0
    assert config.use_precision_aware_optimizer is True
    assert config.store_param_remainders is False
    assert len(warnings) == 1


def test_fp16_recipe_explicitly_configures_values_and_allows_trailing_overrides():
    script = Path("scripts/training/text/run-qwen3-4B-fp16-8xgpu.sh").read_text(encoding="utf-8")

    for option in (
        "--initial-loss-scale 32768",
        "--min-loss-scale 1",
        "--use-precision-aware-optimizer",
        "--no-store-param-remainders",
    ):
        assert option in script
    assert script.index('"$@"') > script.index('"${MISC_ARGS[@]}"')


def test_build_optimizer_config_kwargs_does_not_rewrite_precision_values():
    args = Namespace(
        fp16=True,
        bf16=False,
        params_dtype=torch.float16,
        initial_loss_scale=65536.0,
        min_loss_scale=2.0,
        use_precision_aware_optimizer=False,
        store_param_remainders=True,
    )

    kwargs = model._build_optimizer_config_kwargs(args)

    assert kwargs["bf16"] is False
    assert kwargs["fp16"] is True
    assert kwargs["params_dtype"] is torch.float16
    assert kwargs["initial_loss_scale"] == 65536.0
    assert kwargs["min_loss_scale"] == 2.0
    assert kwargs["use_precision_aware_optimizer"] is False
    assert kwargs["store_param_remainders"] is True


def test_build_optimizer_config_kwargs_builds_real_fp16_optimizer_config():
    args = Namespace(
        fp16=True,
        bf16=False,
        params_dtype=torch.float16,
        initial_loss_scale=65536.0,
        min_loss_scale=2.0,
        use_precision_aware_optimizer=True,
        store_param_remainders=True,
        use_distributed_optimizer=True,
    )

    kwargs = model._build_optimizer_config_kwargs(args)
    config = model.OptimizerConfig(**kwargs)

    assert config.fp16 is True
    assert config.bf16 is False
    assert config.params_dtype is torch.float16
    assert config.initial_loss_scale == 65536.0
    assert config.min_loss_scale == 2.0
    assert config.use_precision_aware_optimizer is True
    assert config.store_param_remainders is True


def test_build_optimizer_config_kwargs_builds_real_bf16_optimizer_config():
    args = Namespace(
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        initial_loss_scale=131072.0,
        min_loss_scale=4.0,
        use_precision_aware_optimizer=True,
        store_param_remainders=True,
        use_distributed_optimizer=True,
    )

    kwargs = model._build_optimizer_config_kwargs(args)
    config = model.OptimizerConfig(**kwargs)

    assert config.bf16 is True
    assert config.fp16 is False
    assert config.params_dtype is torch.bfloat16
    assert config.initial_loss_scale == 131072.0
    assert config.min_loss_scale == 4.0
    assert config.use_precision_aware_optimizer is True
    assert config.store_param_remainders is True


def test_setup_model_and_optimizer_preserves_built_precision_config(monkeypatch):
    expected_kwargs = {
        "fp16": True,
        "bf16": False,
        "params_dtype": torch.float16,
        "initial_loss_scale": 65536.0,
        "min_loss_scale": 2.0,
        "use_precision_aware_optimizer": False,
        "store_param_remainders": True,
    }
    captured_kwargs = {}

    class CapturingOptimizerConfig:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.timers = object()

    monkeypatch.setattr(model, "get_model_provider_func", lambda _args, _role: "provider")
    monkeypatch.setattr(model, "wrap_model_provider_with_freeze", lambda provider, _args: provider)
    monkeypatch.setattr(model, "get_model", lambda *_args, **_kwargs: ["model"])
    monkeypatch.setattr(model, "ModelType", SimpleNamespace(encoder_or_decoder=object()))
    monkeypatch.setattr(model, "_build_optimizer_config_kwargs", lambda _args: expected_kwargs.copy())
    monkeypatch.setattr(model, "OptimizerConfig", CapturingOptimizerConfig)
    monkeypatch.setattr(model, "get_megatron_optimizer", lambda **_kwargs: "optimizer")
    monkeypatch.setattr(model, "get_optimizer_param_scheduler", lambda _args, _optimizer: "scheduler")

    args = Namespace(
        moe_use_upcycling=False,
        load="/checkpoint",
        pretrained_checkpoint=None,
        dynamic_context_parallel=False,
        context_parallel_size=1,
        only_load_weight=False,
        fp16=True,
        use_gloo_process_groups=False,
    )

    result = model.setup_model_and_optimizer(args)

    assert captured_kwargs == expected_kwargs
    assert result == (["model"], "optimizer", "scheduler")
