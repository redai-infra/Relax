#!/usr/bin/env python3
"""Check changed Python files for duplicate top-level def/class names."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def check_file(path: Path) -> list[str]:
    if not path.exists() or path.suffix != ".py":
        return []

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    seen: dict[tuple[str, str], int] = {}
    errors: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = ("def", node.name)
        elif isinstance(node, ast.ClassDef):
            key = ("class", node.name)
        else:
            continue

        previous = seen.get(key)
        if previous is not None:
            kind, name = key
            errors.append(f"{path}:{node.lineno}: duplicate top-level {kind} {name!r}; first at line {previous}")
        else:
            seen[key] = node.lineno

    return errors


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv if arg.endswith(".py")]
    if not paths:
        print("check_duplicate_defs: no Python files to check")
        return 0

    errors: list[str] = []
    for path in paths:
        errors.extend(check_file(path))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print(f"check_duplicate_defs: checked {len(paths)} Python file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
