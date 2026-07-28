# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "training" / "text" / "run-qwen3-4B-fp16-8xgpu.sh"


def _array_body(script: str, name: str) -> str:
    match = re.search(rf"{name}=\(\n(?P<body>.*?)\n\)", script, re.DOTALL)
    assert match is not None
    return match.group("body")


def test_fp16_optimizer_defaults_are_explicit() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    optimizer_args = _array_body(script, "OPTIMIZER_ARGS")

    assert "--initial-loss-scale 32768" in optimizer_args
    assert "--min-loss-scale 1" in optimizer_args
    assert "--use-precision-aware-optimizer" in optimizer_args
    assert "--no-store-param-remainders" in optimizer_args


def test_user_arguments_are_applied_last() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    misc_args = script.index('"${MISC_ARGS[@]}"')
    user_args = script.index('"$@"')
    redirection = script.index("2>&1", user_args)

    assert misc_args < user_args < redirection
