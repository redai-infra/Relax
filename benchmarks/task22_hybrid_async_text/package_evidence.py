#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Create a redacted, checksummed Task 22 raw-evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import tarfile
import tempfile
from pathlib import Path


VARIANTS = ("baseline", "zero_kl", "optimized")
RUN_IDS = (1, 2, 3)
REQUIRED_FILES = ("manifest.txt", "submit.log", "train.log", "gpu.csv")
HOME_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:/public)?/home/[^/\s'\";,]+")
PRIVATE_IPV4 = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
EMAIL_ADDRESS = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY)[A-Z0-9_]*)=([^\s]+)"
)
URL_CREDENTIAL = re.compile(r"(https?://)[^/@\s]+@")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redact_text(value: str) -> str:
    value = HOME_PATH.sub("$HOME", value)
    value = URL_CREDENTIAL.sub(r"\1<redacted>@", value)
    value = PRIVATE_IPV4.sub("<private-ip>", value)
    value = EMAIL_ADDRESS.sub("<redacted-email>", value)
    return SECRET_ASSIGNMENT.sub(r"\1=<redacted>", value)


def collect_sources(artifact_root: Path) -> list[tuple[str, int, Path]]:
    sources: list[tuple[str, int, Path]] = []
    missing: list[Path] = []
    for variant in VARIANTS:
        for run_id in RUN_IDS:
            run_dir = artifact_root / variant / f"run-{run_id}"
            for filename in REQUIRED_FILES:
                path = run_dir / filename
                if path.is_file():
                    sources.append((variant, run_id, path))
                else:
                    missing.append(path)
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Task 22 raw evidence is incomplete:\n{formatted}")
    return sources


def package_evidence(artifact_root: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    sources = collect_sources(artifact_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "raw-evidence.tar.gz"
    index_path = output_dir / "raw-evidence-index.csv"
    checksum_path = output_dir / "raw-evidence.sha256"

    with tempfile.TemporaryDirectory(prefix="task22-evidence-") as temporary:
        staging_root = Path(temporary) / "raw-evidence"
        rows: list[dict[str, str | int]] = []
        for variant, run_id, source in sources:
            relative = Path(variant) / f"run-{run_id}" / source.name
            destination = staging_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_bytes = source.read_bytes()
            delivered_bytes = redact_text(source_bytes.decode("utf-8", errors="replace")).encode("utf-8")
            destination.write_bytes(delivered_bytes)
            rows.append(
                {
                    "variant": variant,
                    "run_id": run_id,
                    "file": relative.as_posix(),
                    "source_sha256": sha256_bytes(source_bytes),
                    "delivered_sha256": sha256_bytes(delivered_bytes),
                    "delivered_bytes": len(delivered_bytes),
                }
            )

        staged_index = staging_root / "raw-evidence-index.csv"
        with staged_index.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (staging_root / "README.txt").write_text(
            "Task 22 raw evidence bundle.\n"
            "Home paths, private IPs, emails, URL credentials, and secret-like assignments were redacted.\n"
            "source_sha256 identifies the original local file; delivered_sha256 identifies the redacted file.\n"
        )
        index_path.write_bytes(staged_index.read_bytes())
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(staging_root, arcname="raw-evidence")

    checksum_path.write_text(f"{sha256_file(archive_path)}  {archive_path.name}\n")
    return archive_path, index_path, checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive, index, checksum = package_evidence(args.artifact_root.resolve(), args.output_dir.resolve())
    print(f"Wrote {archive}")
    print(f"Wrote {index}")
    print(f"Wrote {checksum}")


if __name__ == "__main__":
    main()
