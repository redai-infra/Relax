# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for the Task40 artifact verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[4] / "scripts" / "experiments" / "task40" / "verify_artifacts.py"
SPEC = importlib.util.spec_from_file_location("task40_verify_artifacts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY_ARTIFACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY_ARTIFACTS)


def test_scan_text_artifacts_ignores_escaped_newlines_and_binary_files(tmp_path: Path) -> None:
    """Ordinary escaped output and binary plots must not look like Windows
    paths."""
    (tmp_path / "run.log").write_text(
        r"Final Answer:\nTensorDict(fields:\n  response_lengths: tensor(...))",
        encoding="utf-8",
    )
    (tmp_path / "curve.png").write_bytes(b"\x89PNG\r\n\x1a\n\0binary")

    assert VERIFY_ARTIFACTS.scan_text_artifacts(tmp_path) == []


def test_scan_text_artifacts_rejects_windows_absolute_path(tmp_path: Path) -> None:
    """A real Windows drive path remains a fail-closed verification error."""
    artifact = tmp_path / "manifest.txt"
    artifact.write_text(r"source=C:\Users\alice\private\trace.txt", encoding="utf-8")

    assert VERIFY_ARTIFACTS.scan_text_artifacts(tmp_path) == [f"Windows absolute path found: {artifact}"]


def test_scan_text_artifacts_allows_windows_fixture_only_in_packaged_tests(tmp_path: Path) -> None:
    """Packaged test source may retain the synthetic path used by its own
    rejection test."""
    fixture = tmp_path / "delivery" / "source" / "Relax" / "tests" / "test_windows_fixture.py"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(r'WINDOWS_FIXTURE = "C:\Users\alice\private\trace.txt"', encoding="utf-8")

    assert VERIFY_ARTIFACTS.scan_text_artifacts(tmp_path) == []
