# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed consistency checks for prepared GraphGPO recipe inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.graphgpo.manifest import MANIFEST_VERSION
from examples.graphgpo.prepare_alfworld import PREPARE_SCHEMA_VERSION


MODEL_LOCK_SCHEMA = "task37-huggingface-model-lock-v1"
MODEL_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_FILE_COUNT = 9
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_locked_file(root: Path, relative_path: str, *, label: str) -> Path:
    relative = Path(_non_empty_string(relative_path, label=label))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a safe relative path")
    try:
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is missing or outside its locked root") from exc
    if not candidate.is_file():
        raise ValueError(f"{label} must reference a regular file")
    return candidate


def _verify_locked_file(
    *,
    root: Path,
    relative_path: str,
    metadata: Mapping[str, Any],
    size_key: str,
    label: str,
) -> Path:
    path = _safe_locked_file(root, relative_path, label=label)
    expected_size = _positive_int(metadata.get(size_key), label=f"{label} {size_key}")
    expected_sha = _non_empty_string(metadata.get("sha256"), label=f"{label} sha256")
    if _SHA256_HEX.fullmatch(expected_sha) is None:
        raise ValueError(f"{label} sha256 must be 64 lowercase hexadecimal characters")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size does not match its lock")
    if _sha256(path) != expected_sha:
        raise ValueError(f"{label} SHA256 does not match its lock")
    return path


def verify_model_checkpoint(
    *,
    model_lock_path: Path,
    checkpoint_path: Path,
    model_revision: str,
) -> None:
    """Hash every independently locked file in the local model snapshot."""

    model_revision = _non_empty_string(model_revision, label="model_revision")
    if re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
        raise ValueError("model_revision must be a pinned 40-character commit SHA")
    checkpoint_root = checkpoint_path.resolve(strict=True)
    if not checkpoint_root.is_dir():
        raise ValueError("checkpoint_path must be a directory")
    lock = _read_json(model_lock_path, label="model lock")
    if lock.get("schema") != MODEL_LOCK_SCHEMA:
        raise ValueError("model lock schema mismatch")
    if lock.get("repo_id") != MODEL_REPO_ID:
        raise ValueError("model lock repo_id mismatch")
    if lock.get("revision") != model_revision:
        raise ValueError("model revision does not match the model lock")
    files = lock.get("files")
    if not isinstance(files, Mapping) or len(files) != MODEL_FILE_COUNT:
        raise ValueError(f"model lock must contain exactly {MODEL_FILE_COUNT} files")
    for relative_path, metadata in files.items():
        if not isinstance(relative_path, str) or not isinstance(metadata, Mapping):
            raise ValueError("model lock files must map relative paths to metadata")
        _verify_locked_file(
            root=checkpoint_root,
            relative_path=relative_path,
            metadata=metadata,
            size_key="bytes",
            label=f"model file {relative_path!r}",
        )


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_empty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_batch_arithmetic(
    *,
    task_groups: int,
    group_size: int,
    global_batch_size: int,
) -> None:
    """Require every rollout's trajectory count to form complete train
    batches."""

    task_groups = _positive_int(task_groups, label="task_groups")
    group_size = _positive_int(group_size, label="group_size")
    global_batch_size = _positive_int(global_batch_size, label="global_batch_size")
    rollout_sample_count = task_groups * group_size
    if rollout_sample_count % global_batch_size != 0:
        raise ValueError("task_groups * group_size must be divisible by global_batch_size")


def _prompt_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"prompt data has a blank row at line {line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"prompt row {line_number} must be a JSON object")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prompt JSONL from {path}") from exc
    return rows


def _verify_manifest_assets(
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
    split: str,
) -> None:
    """Rehash every ALFWorld file referenced by one task manifest."""

    data_root = data_root.resolve(strict=True)
    if not data_root.is_dir():
        raise ValueError("ALFWorld data root must be a directory")
    locked_entries: list[tuple[str, Mapping[str, Any], str]] = []
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"{split} manifest tasks must be a list")
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ValueError(f"{split} manifest task {index} must be an object")
        for field in ("game", "trajectory"):
            entry = task.get(field)
            if not isinstance(entry, Mapping):
                raise ValueError(f"{split} manifest task {index} is missing {field} metadata")
            relative_path = _non_empty_string(
                entry.get("relative_path"),
                label=f"{split} manifest task {index} {field}.relative_path",
            )
            locked_entries.append((relative_path, entry, f"{split} task {index} {field}"))

    shared_assets = manifest.get("shared_assets")
    if not isinstance(shared_assets, list) or not shared_assets:
        raise ValueError(f"{split} manifest shared_assets must be a non-empty list")
    for index, asset in enumerate(shared_assets):
        if not isinstance(asset, Mapping) or not isinstance(asset.get("file"), Mapping):
            raise ValueError(f"{split} shared asset {index} is missing file metadata")
        entry = asset["file"]
        relative_path = _non_empty_string(
            entry.get("relative_path"),
            label=f"{split} shared asset {index} relative_path",
        )
        locked_entries.append((relative_path, entry, f"{split} shared asset {index}"))

    relative_paths = [relative_path for relative_path, _, _ in locked_entries]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError(f"{split} manifest references duplicate ALFWorld files")
    for relative_path, metadata, label in locked_entries:
        _verify_locked_file(
            root=data_root,
            relative_path=relative_path,
            metadata=metadata,
            size_key="size_bytes",
            label=label,
        )


def verify_split(
    *,
    split: str,
    prompt_path: Path,
    manifest_path: Path,
    lock_split: Mapping[str, Any],
    max_steps: int,
    data_root: Path,
) -> None:
    expected_manifest_sha = _non_empty_string(lock_split.get("manifest_sha256"), label=f"{split} manifest_sha256")
    expected_prompt_sha = _non_empty_string(lock_split.get("prompt_data_sha256"), label=f"{split} prompt_data_sha256")
    if _sha256(manifest_path) != expected_manifest_sha:
        raise ValueError(f"{split} manifest SHA256 does not match prepare.lock")
    if _sha256(prompt_path) != expected_prompt_sha:
        raise ValueError(f"{split} prompt SHA256 does not match prepare.lock")

    manifest = _read_json(manifest_path, label=f"{split} manifest")
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError(f"{split} manifest schema_version mismatch")
    if manifest.get("split") != split:
        raise ValueError(f"{split} manifest split mismatch")
    _verify_manifest_assets(manifest, data_root=data_root, split=split)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{split} manifest tasks must be a non-empty list")
    expected_count = _positive_int(lock_split.get("task_count"), label=f"{split} task_count")
    if len(tasks) != expected_count:
        raise ValueError(f"{split} manifest task count does not match prepare.lock")

    task_paths: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ValueError(f"{split} manifest task {index} must be an object")
        if task.get("split") != split:
            raise ValueError(f"{split} manifest task {index} has the wrong split")
        game = task.get("game")
        if not isinstance(game, Mapping):
            raise ValueError(f"{split} manifest task {index} is missing game metadata")
        task_paths.append(
            _non_empty_string(
                game.get("relative_path"),
                label=f"{split} manifest task {index} game.relative_path",
            )
        )
    if task_paths != sorted(task_paths) or len(task_paths) != len(set(task_paths)):
        raise ValueError(f"{split} manifest tasks are not in unique stable order")

    rows = _prompt_rows(prompt_path)
    if len(rows) != expected_count:
        raise ValueError(f"{split} prompt row count does not match prepare.lock")
    prompt_task_paths: list[str] = []
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{split} prompt row {index} is missing metadata")
        if metadata.get("alfworld_train_eval") != split:
            raise ValueError(f"{split} prompt row {index} has the wrong split")
        if metadata.get("manifest_index") != index:
            raise ValueError(f"{split} prompt row {index} has the wrong manifest_index")
        if metadata.get("manifest_sha256") != expected_manifest_sha:
            raise ValueError(f"{split} prompt row {index} has the wrong manifest SHA256")
        if metadata.get("max_steps") != max_steps:
            raise ValueError(f"{split} prompt row {index} has the wrong max_steps")
        manifest_task_id = _non_empty_string(
            metadata.get("manifest_task_id"),
            label=f"{split} prompt row {index} manifest_task_id",
        )
        if metadata.get("task_id") != manifest_task_id:
            raise ValueError(f"{split} prompt row {index} task_id mismatch")
        prompt_task_paths.append(manifest_task_id)
    if prompt_task_paths != task_paths:
        raise ValueError(f"{split} prompt task order does not match the manifest")


def verify_prepared_artifacts(
    *,
    prepare_lock_path: Path,
    split_artifacts: Mapping[str, tuple[Path, Path]],
    max_steps: int,
    model_revision: str,
    alfworld_data_root: Path,
) -> None:
    """Verify the exact prepared artifacts selected by a launcher."""

    max_steps = _positive_int(max_steps, label="max_steps")
    model_revision = _non_empty_string(model_revision, label="model_revision")
    if re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
        raise ValueError("model_revision must be a pinned 40-character commit SHA")
    if not split_artifacts:
        raise ValueError("at least one split artifact binding is required")

    lock = _read_json(prepare_lock_path, label="prepare.lock")
    if lock.get("schema_version") != PREPARE_SCHEMA_VERSION:
        raise ValueError("prepare.lock schema_version mismatch")
    if lock.get("max_steps") != max_steps:
        raise ValueError("max_steps does not match prepare.lock")
    if lock.get("model_revision") != model_revision:
        raise ValueError("model_revision does not match prepare.lock")
    splits = lock.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("prepare.lock splits must be an object")

    for split, (prompt_path, manifest_path) in split_artifacts.items():
        lock_split = splits.get(split)
        if not isinstance(lock_split, Mapping):
            raise ValueError(f"prepare.lock does not contain split {split!r}")
        verify_split(
            split=split,
            prompt_path=prompt_path,
            manifest_path=manifest_path,
            lock_split=lock_split,
            max_steps=max_steps,
            data_root=alfworld_data_root,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify prepared GraphGPO recipe inputs.")
    parser.add_argument("--prepare-lock", type=Path, required=True)
    parser.add_argument("--alfworld-data-root", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--task-groups", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument(
        "--split-artifact",
        action="append",
        nargs=3,
        metavar=("SPLIT", "PROMPT_DATA", "MANIFEST"),
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_batch_arithmetic(
        task_groups=args.task_groups,
        group_size=args.group_size,
        global_batch_size=args.global_batch_size,
    )
    verify_model_checkpoint(
        model_lock_path=args.model_lock,
        checkpoint_path=args.checkpoint,
        model_revision=args.model_revision,
    )
    split_artifacts: dict[str, tuple[Path, Path]] = {}
    for split, prompt_data, manifest in args.split_artifact:
        if split in split_artifacts:
            raise ValueError(f"duplicate split artifact binding: {split!r}")
        split_artifacts[split] = (Path(prompt_data), Path(manifest))
    verify_prepared_artifacts(
        prepare_lock_path=args.prepare_lock,
        split_artifacts=split_artifacts,
        max_steps=args.max_steps,
        model_revision=args.model_revision,
        alfworld_data_root=args.alfworld_data_root,
    )
    print("GraphGPO prepared-artifact preflight passed.")


if __name__ == "__main__":
    main()
