# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import csv
import importlib.util
import tarfile
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "task22_hybrid_async_text" / "package_evidence.py"
SPEC = importlib.util.spec_from_file_location("task22_evidence_package", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def test_package_evidence_rejects_incomplete_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="raw evidence is incomplete"):
        evidence.package_evidence(tmp_path / "missing", tmp_path / "results")


def test_package_evidence_requires_all_runs_and_redacts_private_values(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    private_ip = ".".join(("10", "15", "1", "9"))
    for variant in evidence.VARIANTS:
        for run_id in evidence.RUN_IDS:
            run_dir = artifact_root / variant / f"run-{run_id}"
            run_dir.mkdir(parents=True)
            for filename in evidence.REQUIRED_FILES:
                (run_dir / filename).write_text(
                    "model=/home/example/model/Qwen3-0.6B\n"
                    "HF_TOKEN=secret-value\n"
                    "url=https://user:password@example.com/api\n"
                    f"ray_address={private_ip}:6379\n"
                    "author=person@example.com\n"
                )

    output_dir = tmp_path / "results"
    archive, index, checksum = evidence.package_evidence(artifact_root, output_dir)

    assert archive.is_file()
    assert index.is_file()
    assert checksum.read_text().endswith("  raw-evidence.tar.gz\n")
    with index.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(evidence.VARIANTS) * len(evidence.RUN_IDS) * len(evidence.REQUIRED_FILES)
    assert all(row["source_sha256"] and row["delivered_sha256"] for row in rows)

    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile("raw-evidence/baseline/run-1/manifest.txt")
        assert member is not None
        delivered = member.read().decode()
    assert "$HOME/model/Qwen3-0.6B" in delivered
    assert "HF_TOKEN=<redacted>" in delivered
    assert "https://<redacted>@example.com/api" in delivered
    assert "ray_address=<private-ip>:6379" in delivered
    assert "author=<redacted-email>" in delivered
    assert "/home/example" not in delivered
    assert private_ip not in delivered
    assert "secret-value" not in delivered
