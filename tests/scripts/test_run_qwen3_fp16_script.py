# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "training" / "text" / "run-qwen3-4B-fp16-8xgpu.sh"


def _array_body(script: str, name: str) -> str:
    match = re.search(rf"{name}=\(\n(?P<body>.*?)\n\)", script, re.DOTALL)
    assert match is not None
    return match.group("body")


def _array_tokens(script: str, name: str) -> list[str]:
    active_lines = (line.split("#", 1)[0] for line in _array_body(script, name).splitlines())
    return shlex.split("\n".join(active_lines))


def test_fp16_optimizer_defaults_are_explicit() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    optimizer_args = _array_tokens(script, "OPTIMIZER_ARGS")

    assert optimizer_args[optimizer_args.index("--initial-loss-scale") + 1] == "32768"
    assert optimizer_args[optimizer_args.index("--min-loss-scale") + 1] == "1"
    assert "--use-precision-aware-optimizer" in optimizer_args
    assert "--no-store-param-remainders" in optimizer_args


def test_user_arguments_reach_validated_optimizer_config(tmp_path: Path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_ray = bin_dir / "ray"
    fake_ray.write_text('#!/bin/sh\nprintf \'%s\\000\' "$@" > "$RAY_CAPTURE"\n', encoding="utf-8")
    fake_ray.chmod(0o755)

    capture = tmp_path / "ray-argv"
    env = os.environ.copy()
    env.update(
        {
            "MODEL_CONFIG_DIR": str(SCRIPT.parents[2] / "models"),
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "RAY_CAPTURE": str(capture),
            "RELAX_ENTRYPOINT_MODE": "1",
            "RUNTIME_ENV_JSON": "{}",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    )
    user_args = [
        "--initial-loss-scale",
        "65536",
        "--min-loss-scale",
        "2",
        "--no-use-precision-aware-optimizer",
        "--store-param-remainders",
        "--probe",
        "value with spaces",
    ]

    subprocess.run(["bash", str(SCRIPT), *user_args], cwd=tmp_path, env=env, check=True)

    argv = [value.decode() for value in capture.read_bytes().split(b"\0") if value]
    command_separator = argv.index("--")
    default_scale = argv.index("--initial-loss-scale", command_separator)
    assert argv[command_separator + 1 : command_separator + 4] == ["python3", "-m", "relax.entrypoints.train"]
    assert argv[default_scale + 1] == "32768"
    assert argv[-len(user_args) :] == user_args
    assert default_scale < len(argv) - len(user_args)

    from relax.backends.megatron import model
    from relax.backends.megatron.arguments import megatron_parse_args
    from relax.backends.megatron.arguments import validate_args as megatron_validate_args
    from relax.utils.arguments import _normalize_precision_optimizer_args, get_slime_extra_args_provider

    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setattr(sys, "argv", ["probe", *argv[command_separator + 4 :]])
    args = megatron_parse_args(
        extra_args_provider=get_slime_extra_args_provider(),
        skip_hf_validate=True,
    )
    _normalize_precision_optimizer_args(args)
    args = megatron_validate_args(args)
    config = model.OptimizerConfig(**model._build_optimizer_config_kwargs(args))

    assert config.initial_loss_scale == 65536.0
    assert config.min_loss_scale == 2.0
    assert config.use_precision_aware_optimizer is False
    assert config.store_param_remainders is True
    assert config.fp16 is True
    assert config.bf16 is False
