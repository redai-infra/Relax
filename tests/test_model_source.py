# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType

import pytest

from relax.utils import model_source as m


@pytest.fixture(autouse=True)
def reset_provider_registry(monkeypatch):
    monkeypatch.setattr(m, "_PROVIDERS", {})
    monkeypatch.setattr(m, "_FROZEN", False)


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

    module_name = "relax.utils.arguments"
    original_module = sys.modules.get(module_name)
    module_was_loaded = module_name in sys.modules
    utils_module = sys.modules.get("relax.utils")
    arguments_attr_was_set = utils_module is not None and hasattr(utils_module, "arguments")
    original_arguments_attr = getattr(utils_module, "arguments", None) if arguments_attr_was_set else None

    sys.modules.pop(module_name, None)
    if utils_module is not None and arguments_attr_was_set:
        delattr(utils_module, "arguments")
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if module_was_loaded:
            sys.modules[module_name] = original_module
        if utils_module is not None:
            if arguments_attr_was_set:
                setattr(utils_module, "arguments", original_arguments_attr)
            elif hasattr(utils_module, "arguments"):
                delattr(utils_module, "arguments")


def test_apply_model_source_to_argv_returns_canonical_copy():
    argv = ["train.py", "--hf-checkpoint=old", "--hf-checkpoint", "older", "--foo", "bar"]
    source = m.ModelSource("s3://bucket/model/")

    updated = m.apply_model_source_to_argv(argv, source)

    assert argv == ["train.py", "--hf-checkpoint=old", "--hf-checkpoint", "older", "--foo", "bar"]
    assert updated == ["train.py", "--foo", "bar", "--hf-checkpoint", source.uri]


def test_resolve_model_source_rejects_multiple_matches_and_freezes_registry():
    m.register_model_source_provider("a", lambda argv: m.ModelSource("s3://a/model/"))
    m.register_model_source_provider("b", lambda argv: m.ModelSource("s3://b/model/"))

    with pytest.raises(RuntimeError, match="Multiple model source providers matched: a, b"):
        m.resolve_model_source(["train.py"])
    with pytest.raises(RuntimeError, match="registry is frozen"):
        m.register_model_source_provider("c", lambda argv: None)


def test_resolve_model_source_reports_invalid_provider_result():
    m.register_model_source_provider("broken", lambda argv: "s3://bucket/model/")

    with pytest.raises(RuntimeError, match="provider 'broken' failed: expected ModelSource or None"):
        m.resolve_model_source(["train.py"])


def test_parse_args_temporarily_overlays_provider_source(monkeypatch, arguments_module):
    source = m.ModelSource("s3://bucket/model/", provider_name="test")
    m.register_model_source_provider("test", lambda argv: source)
    original_argv = ["train.py", "--hf-checkpoint", "/models/original", "--foo", "bar"]
    captured = {}

    def fake_parse(add_custom_arguments=None, *, provider_source=None):
        captured["argv"] = list(sys.argv)
        captured["source"] = provider_source
        return "parsed"

    monkeypatch.setattr(sys, "argv", original_argv)
    monkeypatch.setattr(arguments_module, "_parse_args_impl", fake_parse)

    assert arguments_module.parse_args() == "parsed"
    assert captured == {
        "argv": ["train.py", "--foo", "bar", "--hf-checkpoint", source.uri],
        "source": source,
    }
    assert sys.argv is original_argv
