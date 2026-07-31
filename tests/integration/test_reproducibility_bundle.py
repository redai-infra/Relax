# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys

from relax.entrypoints.reproducibility import main
from relax.utils.reproducibility import build_manifest, write_manifest


def test_manifest_reruns_minimal_cpu_task(tmp_path):
    marker = tmp_path / "replayed.txt"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('replayed', encoding='utf-8')",
    ]
    manifest = build_manifest(argv=command, cwd=tmp_path)
    manifest_path = write_manifest(manifest, tmp_path / "experiment-manifest.json")

    exit_code = main(["rerun", str(manifest_path), "--execute"])

    assert exit_code == 0
    assert marker.read_text(encoding="utf-8") == "replayed"
