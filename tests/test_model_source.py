# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys

import pytest

from relax.utils import model_source as m


@pytest.fixture(autouse=True)
def reset_provider_registry(monkeypatch):
    monkeypatch.setattr(m, "_PROVIDERS", {})
    monkeypatch.setattr(m, "_FROZEN", False)


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


def test_parse_args_temporarily_overlays_provider_source(monkeypatch):
    from relax.utils import arguments

    source = m.ModelSource("s3://bucket/model/", provider_name="test")
    m.register_model_source_provider("test", lambda argv: source)
    original_argv = ["train.py", "--hf-checkpoint", "/models/original", "--foo", "bar"]
    captured = {}

    def fake_parse(add_custom_arguments=None, *, provider_source=None):
        captured["argv"] = list(sys.argv)
        captured["source"] = provider_source
        return "parsed"

    monkeypatch.setattr(sys, "argv", original_argv)
    monkeypatch.setattr(arguments, "_parse_args_impl", fake_parse)

    assert arguments.parse_args() == "parsed"
    assert captured == {
        "argv": ["train.py", "--foo", "bar", "--hf-checkpoint", source.uri],
        "source": source,
    }
    assert sys.argv is original_argv
