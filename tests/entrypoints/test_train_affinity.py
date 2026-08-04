# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for baseline node-group pinning."""

import os
from argparse import Namespace

import pytest

from relax.core.node_group_affinity import _maybe_pin_baseline_to_stable


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("RELAX_INITIAL_NODE_GROUP", raising=False)
    return monkeypatch


def _write_yaml(tmp_path, enabled: bool):
    """Write a minimal autoscaler config."""
    path = tmp_path / "autoscaler.yaml"
    path.write_text(f"enabled: {'true' if enabled else 'false'}\n")
    return str(path)


def test_pins_stable_for_enabled_elastic_job(clean_env, tmp_path):
    """autoscaler_config present + YAML enabled: true -> env set to stable."""
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    _maybe_pin_baseline_to_stable(args)

    assert os.environ.get("RELAX_INITIAL_NODE_GROUP") == "stable"


def test_disabled_autoscaler_does_not_set_env(clean_env, tmp_path):
    """YAML enabled: false -> not an active elastic job -> env stays unset (so
    a disabled-autoscaler run never risks hanging on missing markers)."""
    cfg = _write_yaml(tmp_path, enabled=False)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    _maybe_pin_baseline_to_stable(args)

    assert not os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip()


def test_enable_affinity_false_no_longer_affects_env(clean_env, tmp_path):
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=False)
    _maybe_pin_baseline_to_stable(args)

    assert os.environ.get("RELAX_INITIAL_NODE_GROUP") == "stable"


def test_non_elastic_job_does_not_set_env(clean_env):
    args = Namespace(autoscaler_config=None, enable_affinity=True)
    _maybe_pin_baseline_to_stable(args)

    assert not os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip()


def test_unreadable_config_does_not_set_env(clean_env, tmp_path):
    """A missing / unreadable config is treated as "not enabled": we log a
    warning and leave the env unset rather than crash startup."""
    args = Namespace(autoscaler_config=str(tmp_path / "does_not_exist.yaml"), enable_affinity=True)
    _maybe_pin_baseline_to_stable(args)

    assert not os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip()


def test_explicit_env_value_is_preserved(clean_env, tmp_path):
    clean_env.setenv("RELAX_INITIAL_NODE_GROUP", "custom")
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    _maybe_pin_baseline_to_stable(args)

    assert os.environ.get("RELAX_INITIAL_NODE_GROUP") == "custom"


def _affinity_parser():
    import argparse

    pytest.importorskip("sglang.srt.server_args")

    try:
        from relax.utils.arguments import get_slime_extra_args_provider
    except (ImportError, AssertionError) as exc:
        pytest.skip(f"relax.utils.arguments unavailable: {exc}")

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    get_slime_extra_args_provider()(parser)
    return parser


def test_enable_affinity_defaults_true():
    parser = _affinity_parser()
    args, _ = parser.parse_known_args([])
    assert args.enable_affinity is True


def test_enable_affinity_flag_sets_true():
    parser = _affinity_parser()
    args, _ = parser.parse_known_args(["--enable-affinity"])
    assert args.enable_affinity is True


def test_no_enable_affinity_flag_sets_false():
    """BooleanOptionalAction registers the paired ``--no-enable-affinity``
    form, now consumed by create_placement_group (node_group_affinity)."""
    parser = _affinity_parser()
    args, _ = parser.parse_known_args(["--no-enable-affinity"])
    assert args.enable_affinity is False
