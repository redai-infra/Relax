# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator
from typing import Any

from relax.utils.doctor.option_audit import capture_registered_options, find_unknown_options
from relax.utils.doctor.runner import render_json, render_text, run_doctor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Relax training configuration without starting Ray, SGLang, GPU workers, "
            "or the training loop. Put training arguments after '--'."
        )
    )
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as errors so the command can be stricter in CI.",
    )
    parser.add_argument(
        "--doctor-skip-hf-validate",
        action="store_true",
        help="Append --skip-hf-validate to the training args before parsing, avoiding remote HF config access.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    doctor_argv, training_argv = _split_argv(raw_argv)
    options = build_parser().parse_args(doctor_argv)
    if options.doctor_skip_hf_validate and "--skip-hf-validate" not in training_argv:
        training_argv = [*training_argv, "--skip-hf-validate"]

    args = None
    parse_error = None
    with capture_registered_options() as registered_options:
        try:
            args = parse_training_args(training_argv)
        except SystemExit as exc:
            parse_error = f"argparse exited with code {exc.code}"
        except Exception as exc:  # noqa: BLE001 - all config failures should become diagnostics
            parse_error = f"{type(exc).__name__}: {exc}"
            try:
                args = parse_training_args(training_argv, validate=False)
            except SystemExit as fallback_exc:
                parse_error = f"{parse_error}; fallback argparse exited with code {fallback_exc.code}"
            except Exception as fallback_exc:  # noqa: BLE001 - preserve the original validation failure
                parse_error = f"{parse_error}; fallback parse failed: {type(fallback_exc).__name__}: {fallback_exc}"
    unknown_options = find_unknown_options(training_argv, registered_options) if args is not None else []

    report = run_doctor(
        argv=training_argv,
        args=args,
        parse_error=parse_error,
        unknown_options=unknown_options,
        strict_warnings=options.strict_warnings,
    )
    output = render_json(report) if options.format == "json" else render_text(report)
    print(output)
    return 0 if report.ok else 1


def parse_training_args(training_argv: list[str], *, validate: bool = True) -> Any:
    # Imported lazily so `python -m relax.entrypoints.doctor --help` does not
    # require Megatron/SGLang/Ray dependencies.
    from relax.utils.arguments import parse_args

    with _temporary_argv(["relax-doctor", *training_argv]):
        return parse_args(validate=validate)


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    if argv in (["-h"], ["--help"]):
        return argv, []
    return [], argv


@contextlib.contextmanager
def _temporary_argv(argv: list[str]) -> Iterator[None]:
    old_argv = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
