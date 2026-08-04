# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import contextlib
import re
from collections.abc import Iterator


_NEGATIVE_NUMBER_RE = re.compile(r"^-\d+$|^-\d*\.\d+$")


@contextlib.contextmanager
def capture_registered_options() -> Iterator[set[str]]:
    """Collect option strings from every parser used during one CLI parse."""
    registered: set[str] = set()
    original = argparse.ArgumentParser.parse_known_args

    def parse_known_args(
        parser: argparse.ArgumentParser,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        registered.update(parser._option_string_actions)
        return original(parser, args=args, namespace=namespace)

    argparse.ArgumentParser.parse_known_args = parse_known_args  # type: ignore[method-assign]
    try:
        yield registered
    finally:
        argparse.ArgumentParser.parse_known_args = original  # type: ignore[method-assign]


def find_unknown_options(argv: list[str], registered: set[str]) -> list[str]:
    if not registered:
        return []

    unknown = []
    seen = set()
    registered_short_options = sorted(
        (option for option in registered if option.startswith("-") and not option.startswith("--")),
        key=len,
        reverse=True,
    )
    for item in argv:
        if not item.startswith("-") or item in {"-", "--"}:
            continue
        if item.startswith("--"):
            option = item.split("=", 1)[0]
        elif _NEGATIVE_NUMBER_RE.fullmatch(item):
            continue
        else:
            option = next(
                (candidate for candidate in registered_short_options if item.startswith(candidate)),
                item.split("=", 1)[0],
            )
        if option in registered or option in seen:
            continue
        seen.add(option)
        unknown.append(option)
    return unknown
