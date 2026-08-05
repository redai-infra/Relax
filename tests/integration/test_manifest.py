# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Integration checks for manifest collection and non-blocking persistence."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from relax.utils import manifest


def test_manifest_generation_is_fast_small_and_sanitized(tmp_path: Path) -> None:
    args = SimpleNamespace(
        tensorboard_dir=str(tmp_path),
        hf_checkpoint="/home/alice/models/qwen",
        prompt_data="/home/alice/data/train.jsonl",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        runtime_env_json='{"HF_TOKEN":"do-not-write-this"}',
    )

    started = time.perf_counter()
    path = manifest.collect_and_save_manifest(args, {"env_vars": {"API_TOKEN": "also-secret"}})
    duration = time.perf_counter() - started

    assert path is not None
    output = Path(path)
    payload = output.read_text(encoding="utf-8")
    data = json.loads(payload)
    assert data["schema_version"] == "1.0"
    assert {"gpu_count", "gpu_model", "gpu_memory_gb"} <= data["hardware"].keys()
    assert "parallel_topology" in data["runtime"]
    assert {"model", "dataset"} <= data["inputs"].keys()
    assert duration < 5
    assert output.stat().st_size < manifest.MAX_MANIFEST_BYTES
    assert "do-not-write-this" not in payload
    assert "also-secret" not in payload
    assert "/home/alice" not in payload


def test_collection_failure_and_non_primary_rank_do_not_block(tmp_path: Path) -> None:
    args = SimpleNamespace(tensorboard_dir=str(tmp_path))
    with patch.object(manifest, "collect_manifest", side_effect=RuntimeError("boom")):
        assert manifest.collect_and_save_manifest(args) is None
    with patch.dict("os.environ", {"RANK": "1"}, clear=True):
        assert manifest.collect_and_save_manifest(args) is None
    assert list(tmp_path.iterdir()) == []


def test_minimal_cpu_generate_check_and_confirmed_rerun(tmp_path: Path) -> None:
    marker = tmp_path / "rerun-complete"
    manifest_path = tmp_path / "cpu-manifest.json"
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
    recorded = manifest.collect_manifest(argv=command)
    manifest_path.write_text(json.dumps(recorded), encoding="utf-8")

    assert manifest.main(["check", str(manifest_path)]) == 0
    assert manifest.main(["rerun", str(manifest_path), "--dry-run"]) == 0
    assert not marker.exists()
    assert manifest.main(["rerun", str(manifest_path), "--confirm"]) == 0
    assert marker.exists()
