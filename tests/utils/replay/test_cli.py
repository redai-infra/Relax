# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CLI entry point tests."""

from __future__ import annotations

from relax.tools.trajectory_replay.cli import main
from tests.utils.replay.helpers import build_grpo_bundle


def test_cli_inspect_valid(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["inspect", str(bundle)]) == 0


def test_cli_inspect_missing(tmp_path):
    assert main(["inspect", str(tmp_path / "nope")]) == 1


def test_cli_validate_valid(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["validate", str(bundle)]) == 0


def test_cli_validate_invalid(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    (bundle / "COMPLETE").unlink()
    assert main(["validate", str(bundle)]) == 1


def test_cli_replay_valid(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["replay", str(bundle)]) == 0


def test_cli_replay_divergent(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", pr65_bug=True)
    assert main(["replay", str(bundle)]) == 1


def test_cli_replay_sample_selection(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["replay", str(bundle), "--sample", "s-0"]) == 0


def test_cli_replay_batch_selection(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["replay", str(bundle), "--batch", "mb-0007"]) == 0


def test_cli_replay_step_assert_match(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["replay", str(bundle), "--step", "120:0"]) == 0


def test_cli_replay_step_assert_mismatch(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["replay", str(bundle), "--step", "999:0"]) == 1
