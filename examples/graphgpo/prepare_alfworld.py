# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare deterministic ALFWorld manifests and Relax prompt rows.

This module intentionally does not import ALFWorld.  It operates on an already-
downloaded text-only dataset and can therefore run in a lightweight CPU
environment before the real ``reset``/``step`` smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.graphgpo.manifest import (
    ALFWORLD_TASK_TYPES,
    TaskManifest,
    build_manifest,
    discover_task_files,
    manifest_bytes,
    manifest_sha256,
)


PREPARE_SCHEMA_VERSION = "graphgpo-alfworld-prepare-v1"
PINNED_ALFWORLD_VERSION = "0.4.2"
PINNED_MODEL_REVISION = "775b11afaf83e0dc75bd5abaf90133e47b3ec082"
DEFAULT_SPLIT_PATHS = {
    "train": Path("json_2.1.1/train"),
    "eval_in_distribution": Path("json_2.1.1/valid_seen"),
}
DEFAULT_LIMITS = {
    "train": None,
    "eval_in_distribution": 128,
}
TRAIN_TEMPERATURE = 1.0
EVAL_TEMPERATURE = 0.4
SAMPLING_TOP_P = 1.0
SAMPLING_MAX_TOKENS = 512
_SPLIT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_PLACEHOLDER_MESSAGE = (
    "Start the ALFWorld episode. The managed agent builds every trainable turn prompt from the environment state."
)


@dataclass(frozen=True)
class PreparedSplit:
    """Files and hashes produced for one ALFWorld split."""

    split: str
    manifest_name: str
    prompt_name: str
    task_count: int
    manifest_sha256: str
    prompt_sha256: str
    temperature: float
    top_p: float
    max_tokens: int


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def is_supported_solvable_task(game_path: Path) -> bool:
    """Match the filters used by ALFWorld 0.4.2's text environment."""

    normalized_path = game_path.as_posix().lower()
    if "movable" in normalized_path or "sliced" in normalized_path:
        return False
    trajectory_path = game_path.with_name("traj_data.json")
    trajectory = _read_json_object(trajectory_path, label="trajectory")
    if trajectory.get("task_type") not in ALFWORLD_TASK_TYPES:
        return False
    game = _read_json_object(game_path, label="game")
    return game.get("solvable") is True


def eligible_task_files(split_root: Path) -> list[Path]:
    """Return supported, solvable games in stable relative-path order."""

    split_root = split_root.resolve(strict=True)
    if not split_root.is_dir():
        raise ValueError(f"split root must be a directory: {split_root}")
    return [path for path in discover_task_files(split_root) if is_supported_solvable_task(path)]


def select_task_files(task_files: Sequence[Path], *, limit: int | None) -> list[Path]:
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("split limit must be a positive integer or null")
        if len(task_files) < limit:
            raise ValueError(f"requested {limit} tasks but only {len(task_files)} eligible tasks are available")
        return list(task_files[:limit])
    return list(task_files)


def prompt_rows_bytes(
    manifest: TaskManifest,
    *,
    max_steps: int,
    temperature: float | None = None,
    top_p: float = SAMPLING_TOP_P,
    max_tokens: int = SAMPLING_MAX_TOKENS,
) -> bytes:
    """Serialize one deterministic Relax seed row per manifest task."""

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if temperature is None:
        temperature = TRAIN_TEMPERATURE if manifest.split == "train" else EVAL_TEMPERATURE
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0.0 <= float(temperature)
    ):
        raise ValueError("temperature must be a non-negative real number")
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(float(top_p))
        or not 0.0 < float(top_p) <= 1.0
    ):
        raise ValueError("top_p must be in (0, 1]")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    temperature = float(temperature)
    top_p = float(top_p)
    digest = manifest_sha256(manifest)
    rows: list[str] = []
    for index, task in enumerate(manifest.tasks):
        row = {
            "messages": [{"role": "user", "content": _PLACEHOLDER_MESSAGE}],
            "metadata": {
                "alfworld_train_eval": manifest.split,
                "manifest_index": index,
                "manifest_sha256": digest,
                "manifest_task_id": task.game.relative_path,
                "max_tokens": max_tokens,
                "max_steps": max_steps,
                "task_id": task.game.relative_path,
                "task_type": task.task_type,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        rows.append(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    if not rows:
        raise ValueError(f"split {manifest.split!r} contains no eligible tasks")
    return ("\n".join(rows) + "\n").encode("utf-8")


def _write_idempotent(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file():
            raise FileExistsError(f"output path exists and is not a file: {path}")
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to replace non-identical prepared artifact: {path}")
    path.write_bytes(content)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def prepare_split(
    *,
    data_root: Path,
    output_dir: Path,
    split: str,
    split_root: Path,
    shared_files: Sequence[Path],
    limit: int | None,
    max_steps: int,
) -> PreparedSplit:
    if not isinstance(split, str) or _SPLIT_NAME.fullmatch(split) is None:
        raise ValueError(f"invalid split name: {split!r}")
    selected = select_task_files(
        eligible_task_files(split_root),
        limit=limit,
    )
    manifest = build_manifest(
        data_root,
        split=split,
        task_files=selected,
        shared_files=shared_files,
    )
    manifest_content = manifest_bytes(manifest)
    prompt_content = prompt_rows_bytes(manifest, max_steps=max_steps)
    temperature = TRAIN_TEMPERATURE if manifest.split == "train" else EVAL_TEMPERATURE
    manifest_name = f"{split}.manifest.json"
    prompt_name = f"{split}.prompts.jsonl"
    _write_idempotent(output_dir / manifest_name, manifest_content)
    _write_idempotent(output_dir / prompt_name, prompt_content)
    return PreparedSplit(
        split=split,
        manifest_name=manifest_name,
        prompt_name=prompt_name,
        task_count=len(manifest.tasks),
        manifest_sha256=_sha256_bytes(manifest_content),
        prompt_sha256=_sha256_bytes(prompt_content),
        temperature=temperature,
        top_p=SAMPLING_TOP_P,
        max_tokens=SAMPLING_MAX_TOKENS,
    )


def prepare_artifacts(
    *,
    data_root: Path,
    output_dir: Path,
    split_roots: Mapping[str, Path],
    shared_files: Sequence[Path] | None = None,
    limits: Mapping[str, int | None] | None = None,
    max_steps: int = 50,
) -> dict[str, Any]:
    """Build all split artifacts and a path-free content lock."""

    data_root = data_root.resolve(strict=True)
    if not data_root.is_dir():
        raise ValueError(f"data_root must be a directory: {data_root}")
    if not split_roots:
        raise ValueError("at least one split root is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    if shared_files is None:
        shared_files = (
            data_root / "logic" / "alfred.pddl",
            data_root / "logic" / "alfred.twl2",
        )
    resolved_shared = tuple(Path(path).resolve(strict=True) for path in shared_files)
    resolved_limits = dict(limits or {})

    prepared: list[PreparedSplit] = []
    for split in sorted(split_roots):
        prepared.append(
            prepare_split(
                data_root=data_root,
                output_dir=output_dir,
                split=split,
                split_root=Path(split_roots[split]),
                shared_files=resolved_shared,
                limit=resolved_limits.get(split),
                max_steps=max_steps,
            )
        )

    lock = {
        "schema_version": PREPARE_SCHEMA_VERSION,
        "alfworld_version": PINNED_ALFWORLD_VERSION,
        "model_revision": PINNED_MODEL_REVISION,
        "max_steps": max_steps,
        "splits": {
            item.split: {
                "manifest": item.manifest_name,
                "manifest_sha256": item.manifest_sha256,
                "prompt_data": item.prompt_name,
                "prompt_data_sha256": item.prompt_sha256,
                "sampling": {
                    "max_tokens": item.max_tokens,
                    "temperature": item.temperature,
                    "top_p": item.top_p,
                },
                "task_count": item.task_count,
            }
            for item in prepared
        },
    }
    lock_content = (
        json.dumps(
            lock,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_idempotent(output_dir / "prepare.lock.json", lock_content)
    return lock


def _assignment(value: str, *, option: str) -> tuple[str, str]:
    name, separator, raw_value = value.partition("=")
    if not separator or not name or not raw_value:
        raise argparse.ArgumentTypeError(f"{option} must use NAME=VALUE syntax")
    if _SPLIT_NAME.fullmatch(name) is None:
        raise argparse.ArgumentTypeError(f"{option} has an invalid split name: {name!r}")
    return name, raw_value


def _split_assignments(
    raw_values: Sequence[str] | None,
    *,
    data_root: Path,
) -> dict[str, Path]:
    if not raw_values:
        return {name: data_root / relative_path for name, relative_path in DEFAULT_SPLIT_PATHS.items()}
    result: dict[str, Path] = {}
    for raw_value in raw_values:
        name, value = _assignment(raw_value, option="--split")
        if name in result:
            raise ValueError(f"duplicate split assignment: {name!r}")
        result[name] = Path(value)
    return result


def _limit_assignments(raw_values: Sequence[str] | None) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for raw_value in raw_values or ():
        name, value = _assignment(raw_value, option="--limit")
        if name in result:
            raise ValueError(f"duplicate limit assignment: {name!r}")
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"limit for {name!r} must be an integer") from exc
        if parsed < 0:
            raise ValueError(f"limit for {name!r} must be non-negative")
        result[name] = None if parsed == 0 else parsed
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic ALFWorld manifests and GraphGPO prompt rows.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        help=(
            "Split mapping in NAME=PATH form. May be repeated. Defaults to "
            "train and eval_in_distribution below --data-root."
        ),
    )
    parser.add_argument(
        "--limit",
        action="append",
        help="Per-split task limit in NAME=COUNT form; COUNT=0 means all.",
    )
    parser.add_argument(
        "--shared-file",
        action="append",
        type=Path,
        help=(
            "Explicit alfred.pddl/alfred.twl2 path. Repeat twice. Defaults "
            "to --data-root/logic/{alfred.pddl,alfred.twl2}."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args(argv)
    if args.shared_file is not None and len(args.shared_file) != 2:
        parser.error("--shared-file must be omitted or supplied exactly twice")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = args.data_root.resolve(strict=True)
    split_roots = _split_assignments(args.split, data_root=data_root)
    explicit_limits = _limit_assignments(args.limit)
    unknown_limits = set(explicit_limits) - set(split_roots)
    if unknown_limits:
        raise ValueError(f"limits refer to unknown splits: {sorted(unknown_limits)!r}")
    limits = {split: DEFAULT_LIMITS.get(split) for split in split_roots}
    limits.update(explicit_limits)
    lock = prepare_artifacts(
        data_root=data_root,
        output_dir=args.output_dir,
        split_roots=split_roots,
        shared_files=args.shared_file,
        limits=limits,
        max_steps=args.max_steps,
    )
    print(json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
