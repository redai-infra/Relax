# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
from pathlib import Path


SGLANG_ENGINE_PATH = Path(__file__).resolve().parents[2] / "relax" / "backends" / "sglang" / "sglang_engine.py"


def _method(name: str) -> ast.FunctionDef:
    tree = ast.parse(SGLANG_ENGINE_PATH.read_text(encoding="utf-8"))
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SGLangEngine")
    return next(node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _requests_post(method: ast.FunctionDef) -> ast.Call:
    return next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "requests"
        and node.func.attr == "post"
    )


def test_pause_generation_preserves_mode_and_timeout() -> None:
    method = _method("pause_generation")
    arguments = {argument.arg for argument in method.args.args}
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in _requests_post(method).keywords}

    assert {"self", "mode", "timeout"} <= arguments
    assert keywords["json"] == "{'mode': mode}"
    assert keywords["timeout"] == "timeout"


def test_continue_generation_preserves_cache_control_and_timeout() -> None:
    method = _method("continue_generation")
    arguments = {argument.arg for argument in method.args.args}
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in _requests_post(method).keywords}

    assert {"self", "torch_empty_cache", "timeout"} <= arguments
    assert keywords["json"] == "{'torch_empty_cache': torch_empty_cache}"
    assert keywords["timeout"] == "timeout"
