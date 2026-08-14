# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CLI entry point tests.

Happy-path replay, sample/batch selection and PR #65 localization are covered
by the runner tests. This module only checks argv wiring, exit codes and the
--step resolver unique to the CLI.
"""

from __future__ import annotations

from relax.tools.trajectory_replay.cli import main
from relax.utils.replay.capture import build_bundle_from_record
from tests.utils.replay.helpers import build_grpo_bundle, make_rollout_capture_record


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


def test_cli_replay_divergent(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", pr65_bug=True)
    assert main(["replay", str(bundle)]) == 1


def test_cli_replay_step_assert_mismatch(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["replay", str(bundle), "--step", "999:0"]) == 1


def test_cli_replay_step_on_rollout_bundle(tmp_path):
    bundle = build_bundle_from_record(make_rollout_capture_record("b-rollout"), tmp_path)
    assert main(["replay", str(bundle), "--step", "120:0"]) == 0
    assert main(["replay", str(bundle), "--step", "121:0"]) == 1


def test_cli_replay_step_picks_from_capture_dir(tmp_path):
    store = tmp_path / "rank-0"
    build_grpo_bundle(store / "replay-120-0")
    build_bundle_from_record(make_rollout_capture_record("replay-rollout-120"), store)
    assert main(["replay", str(tmp_path), "--step", "120:0"]) == 0
    assert main(["replay", str(tmp_path), "--step", "999:0"]) == 1
