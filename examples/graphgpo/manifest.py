# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deterministic, content-addressed ALFWorld task manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


MANIFEST_VERSION = "alfworld-task-manifest-v1"
TASK_FILENAMES = ("game.tw-pddl", "traj_data.json")
SHARED_FILENAMES = ("alfred.pddl", "alfred.twl2")
ALFWORLD_TASK_TYPES = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)


@dataclass(frozen=True)
class FileManifestEntry:
    """One immutable file referenced by the task manifest."""

    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class TaskManifestEntry:
    """One ALFWorld task and both files needed to reproduce it."""

    task_id: str
    split: str
    task_type: str
    game: FileManifestEntry
    trajectory: FileManifestEntry


@dataclass(frozen=True)
class SharedAssetEntry:
    """One shared ALFWorld grammar or PDDL asset."""

    asset_name: str
    file: FileManifestEntry


@dataclass(frozen=True)
class TaskManifest:
    """A fixed-order collection of content-addressed ALFWorld tasks."""

    schema_version: str
    split: str
    tasks: tuple[TaskManifestEntry, ...]
    shared_assets: tuple[SharedAssetEntry, ...]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_task_type(relative_path: str) -> str:
    """Infer the official ALFWorld task family from a relative path."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("relative_path must be a non-empty string")
    for part in Path(relative_path).parts:
        for task_type in ALFWORLD_TASK_TYPES:
            if part == task_type or part.startswith(f"{task_type}-"):
                return task_type
    raise ValueError(f"cannot infer ALFWorld task type from {relative_path!r}")


def discover_task_files(root: Path, *, filename: str = "game.tw-pddl") -> list[Path]:
    """Discover task files in stable relative-path order."""

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("root must be a directory")
    if not isinstance(filename, str) or not filename:
        raise ValueError("filename must be a non-empty string")
    return sorted(
        (path for path in root.rglob(filename) if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def discover_shared_files(root: Path) -> list[Path]:
    """Find exactly one copy of every required shared ALFWorld asset."""

    root = root.resolve(strict=True)
    shared_files: list[Path] = []
    for filename in SHARED_FILENAMES:
        matches = sorted(
            (path for path in root.rglob(filename) if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {filename!r} below root, found {len(matches)}")
        shared_files.append(matches[0])
    return shared_files


def _resolve_task_path(root: Path, path: Path) -> tuple[Path, str]:
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.resolve(strict=True)
    try:
        relative_path = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"task path is outside root: {path}") from exc
    if not candidate.is_file():
        raise ValueError(f"task path is not a regular file: {path}")
    return candidate, relative_path


def _file_entry(path: Path, relative_path: str) -> FileManifestEntry:
    return FileManifestEntry(
        relative_path=relative_path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def build_manifest(
    root: Path,
    *,
    split: str,
    task_files: Sequence[Path] | None = None,
    shared_files: Sequence[Path] | None = None,
) -> TaskManifest:
    """Hash every task and shared asset required for a reproducible run.

    ``task_files`` contains ``game.tw-pddl`` paths.  The sibling
    ``traj_data.json`` is mandatory and is frozen in the same task entry.
    ``shared_files`` must contain one ``alfred.pddl`` and one ``alfred.twl2``;
    when omitted, the builder discovers exactly one of each below ``root``.
    """

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("root must be a directory")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    split = split.strip()

    if task_files is None:
        task_files = discover_task_files(root)
    elif isinstance(task_files, (str, bytes)) or not isinstance(task_files, Sequence):
        raise TypeError("task_files must be a sequence of paths")

    resolved: list[tuple[Path, str]] = [_resolve_task_path(root, Path(path)) for path in task_files]
    resolved.sort(key=lambda item: item[1])
    relative_paths = [relative_path for _, relative_path in resolved]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("task_files contains duplicate paths")

    entries: list[TaskManifestEntry] = []
    for game_path, game_relative_path in resolved:
        if game_path.name != TASK_FILENAMES[0]:
            raise ValueError(f"task file must be named {TASK_FILENAMES[0]!r}: {game_path}")
        trajectory_path = game_path.with_name(TASK_FILENAMES[1])
        trajectory_path, trajectory_relative_path = _resolve_task_path(root, trajectory_path)
        task_parent = Path(game_relative_path).parent.as_posix()
        entries.append(
            TaskManifestEntry(
                task_id=task_parent,
                split=split,
                task_type=infer_task_type(game_relative_path),
                game=_file_entry(game_path, game_relative_path),
                trajectory=_file_entry(
                    trajectory_path,
                    trajectory_relative_path,
                ),
            )
        )

    if shared_files is None:
        shared_files = discover_shared_files(root)
    elif isinstance(shared_files, (str, bytes)) or not isinstance(shared_files, Sequence):
        raise TypeError("shared_files must be a sequence of paths")

    shared_by_name: dict[str, SharedAssetEntry] = {}
    for shared_file in shared_files:
        path, relative_path = _resolve_task_path(root, Path(shared_file))
        if path.name not in SHARED_FILENAMES:
            raise ValueError(f"unexpected shared asset: {path.name!r}")
        if path.name in shared_by_name:
            raise ValueError(f"duplicate shared asset: {path.name!r}")
        shared_by_name[path.name] = SharedAssetEntry(
            asset_name=path.name,
            file=_file_entry(path, relative_path),
        )
    missing_shared = set(SHARED_FILENAMES) - set(shared_by_name)
    if missing_shared:
        raise ValueError(f"missing shared assets: {sorted(missing_shared)!r}")

    return TaskManifest(
        schema_version=MANIFEST_VERSION,
        split=split,
        tasks=tuple(entries),
        shared_assets=tuple(shared_by_name[filename] for filename in SHARED_FILENAMES),
    )


def manifest_bytes(manifest: TaskManifest) -> bytes:
    if not isinstance(manifest, TaskManifest):
        raise TypeError("manifest must be a TaskManifest")
    payload = {
        "schema_version": manifest.schema_version,
        "split": manifest.split,
        "tasks": [asdict(task) for task in manifest.tasks],
        "shared_assets": [asdict(asset) for asset in manifest.shared_assets],
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def manifest_sha256(manifest: TaskManifest) -> str:
    return hashlib.sha256(manifest_bytes(manifest)).hexdigest()


def write_manifest(path: Path, manifest: TaskManifest) -> None:
    path.write_bytes(manifest_bytes(manifest))
