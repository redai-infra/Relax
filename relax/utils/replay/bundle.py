# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay bundle writer and reader.

The writer is crash-safe: payloads and metadata land in a sibling temp
directory, then the directory is atomically renamed into place only after the
completion sentinel is written. The reader refuses to open a bundle that is
missing COMPLETE, a payload, a rank shard, or whose checksums do not match the
manifest — and it loads tensors with weights_only=True only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from relax.utils.replay.schema import (
    BundleIndex,
    Manifest,
    PayloadSpec,
    index_from_dict,
    index_to_dict,
    manifest_from_dict,
    manifest_to_dict,
)


_COMPLETE = "COMPLETE"
_PAYLOAD_DIR = "payloads"
_MANIFEST = "manifest.json"
_INDEX = "index.json"
_EXPECTED = "expected.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class LoadedBundle:
    """A fully validated bundle ready for replay."""

    manifest: Manifest
    index: BundleIndex
    expected: dict[str, Any]
    tensors: dict[str, torch.Tensor]

    @property
    def sample_ids(self) -> list[str]:
        return [record.sample_id for record in self.index.samples]

    @property
    def response_lengths(self) -> list[int]:
        return [record.response_length for record in self.index.samples]


class BundleWriter:
    """Writes a single-rank (or coordinator) replay bundle atomically."""

    def __init__(
        self,
        path: str | Path,
        manifest: Manifest,
        index: BundleIndex,
        expected: dict[str, Any],
        *,
        rank: int = 0,
    ) -> None:
        self._final_path = Path(path)
        self._tmp_path = self._final_path.with_name(f".{self._final_path.name}.tmp-{os.getpid()}")
        self._manifest = manifest
        self._index = index
        self._expected = expected
        self._rank = rank
        self._payloads: dict[str, PayloadSpec] = {}

    def _payload_dir(self) -> Path:
        return self._tmp_path / _PAYLOAD_DIR

    def write_payload(self, name: str, tensor: torch.Tensor) -> None:
        """Store one detached CPU tensor payload under name."""
        tensor = tensor.detach().cpu().contiguous()
        path = self._payload_dir() / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
        self._payloads[name] = PayloadSpec(
            name=name,
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )

    def finalize(self, ranks: list[int]) -> None:
        """Flush metadata, write COMPLETE sentinels and atomically publish."""
        self._manifest.payloads = dict(self._payloads)
        _write_json(self._tmp_path / _MANIFEST, manifest_to_dict(self._manifest))
        _write_json(self._tmp_path / _INDEX, index_to_dict(self._index))
        _write_json(self._tmp_path / _EXPECTED, self._expected)

        step = self._index.identity.actor_step_id
        rank_complete = {
            "actor_step_id": ({"rollout_id": step.rollout_id, "step_id": step.step_id} if step is not None else None),
            "rollout_id": self._index.identity.rollout_id,
            "payloads": {name: spec.sha256 for name, spec in self._payloads.items()},
        }
        _write_json(self._tmp_path / f"{_COMPLETE}.{self._rank}", rank_complete)
        _write_json(self._tmp_path / _COMPLETE, {"ranks": ranks})

        if self._final_path.exists():
            raise FileExistsError(f"bundle already exists at {self._final_path}")
        os.replace(self._tmp_path, self._final_path)


def write_rank_shard(
    path: str | Path,
    *,
    rank: int,
    actor_step_id: tuple[int, int],
    payloads: dict[str, torch.Tensor],
) -> None:
    """Write one rank's payload shard and COMPLETE.<rank> into an existing
    bundle dir.

    Used by multi-rank producers; the coordinator calls finalize_bundle once
    every rank has flushed.
    """
    bundle_dir = Path(path)
    payload_dir = bundle_dir / _PAYLOAD_DIR
    payload_dir.mkdir(parents=True, exist_ok=True)

    checksums: dict[str, str] = {}
    for name, tensor in payloads.items():
        tensor = tensor.detach().cpu().contiguous()
        payload_path = payload_dir / f"{name}.pt"
        torch.save(tensor, payload_path)
        checksums[name] = sha256_file(payload_path)

    _write_json(
        bundle_dir / f"{_COMPLETE}.{rank}",
        {
            "actor_step_id": {"rollout_id": actor_step_id[0], "step_id": actor_step_id[1]},
            "payloads": checksums,
        },
    )


def finalize_bundle(path: str | Path, ranks: list[int]) -> None:
    """Write the final COMPLETE sentinel after every rank shard is present."""
    bundle_dir = Path(path)
    for rank in ranks:
        if not (bundle_dir / f"{_COMPLETE}.{rank}").is_file():
            raise FileNotFoundError(f"missing COMPLETE.{rank} in {bundle_dir}")
    _write_json(bundle_dir / _COMPLETE, {"ranks": ranks})


class IncompleteBundleError(ValueError):
    """Raised when a bundle is missing a sentinel, shard or payload."""


class BundleReader:
    """Reads and validates a replay bundle."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self, *, allow_partial_read: bool = False) -> LoadedBundle:
        """Load manifest, index, expected outputs and tensor payloads.

        allow_partial_read is for inspection of incomplete bundles; replay and
        validate always require a complete bundle.
        """
        if not self._path.is_dir():
            raise IncompleteBundleError(f"bundle directory not found: {self._path}")

        complete = self._path / _COMPLETE
        if not complete.is_file():
            raise IncompleteBundleError(f"missing {_COMPLETE} sentinel in {self._path}")

        manifest = manifest_from_dict(_read_json(self._path / _MANIFEST))
        index = index_from_dict(_read_json(self._path / _INDEX))
        expected = _read_json(self._path / _EXPECTED)

        if manifest.bundle_id != index.bundle_id:
            raise IncompleteBundleError(
                f"manifest bundle_id {manifest.bundle_id!r} != index bundle_id {index.bundle_id!r}"
            )

        ranks = self._read_ranks()
        self._validate_rank_shards(manifest, ranks)
        self._validate_payloads(manifest)

        if not allow_partial_read:
            tensors = {name: self._read_payload(spec) for name, spec in manifest.payloads.items()}
        else:
            tensors = {}
        return LoadedBundle(manifest=manifest, index=index, expected=expected, tensors=tensors)

    def _read_ranks(self) -> list[int]:
        complete_raw = _read_json(self._path / _COMPLETE)
        ranks = complete_raw.get("ranks")
        if not isinstance(ranks, list) or not all(isinstance(rank, int) for rank in ranks):
            raise IncompleteBundleError(f"malformed {_COMPLETE}: missing integer 'ranks' list")
        return ranks

    def _validate_rank_shards(self, manifest: Manifest, ranks: list[int]) -> None:
        accounted: dict[str, str] = {}
        for rank in ranks:
            shard_path = self._path / f"{_COMPLETE}.{rank}"
            if not shard_path.is_file():
                raise IncompleteBundleError(f"missing COMPLETE.{rank} in {self._path}")
            shard = _read_json(shard_path)
            for name, checksum in shard.get("payloads", {}).items():
                if name in accounted:
                    raise IncompleteBundleError(f"payload {name!r} written by more than one rank")
                accounted[name] = checksum

        manifest_names = set(manifest.payloads)
        if set(accounted) != manifest_names:
            missing = sorted(manifest_names - set(accounted))
            extra = sorted(set(accounted) - manifest_names)
            raise IncompleteBundleError(f"rank shards do not match manifest (missing={missing}, extra={extra})")

        for name, checksum in accounted.items():
            if checksum != manifest.payloads[name].sha256:
                raise IncompleteBundleError(f"payload {name!r} checksum mismatch between shard and manifest")

    def _validate_payloads(self, manifest: Manifest) -> None:
        for name, spec in manifest.payloads.items():
            path = self._path / _PAYLOAD_DIR / f"{name}.pt"
            if not path.is_file():
                raise IncompleteBundleError(f"missing payload file {path}")
            if path.stat().st_size != spec.bytes:
                raise IncompleteBundleError(f"payload {name!r} byte size mismatch")
            if sha256_file(path) != spec.sha256:
                raise IncompleteBundleError(f"payload {name!r} checksum mismatch")

    def _read_payload(self, spec: PayloadSpec) -> torch.Tensor:
        path = self._path / _PAYLOAD_DIR / f"{spec.name}.pt"
        try:
            tensor = torch.load(path, weights_only=True)
        except Exception as exc:  # noqa: BLE001 — reject any unsafe/unpicklable payload as incomplete
            raise IncompleteBundleError(f"payload {spec.name!r} could not be loaded safely: {exc}") from exc
        if not isinstance(tensor, torch.Tensor):
            raise IncompleteBundleError(f"payload {spec.name!r} is not a tensor (weights_only loader rejected it)")
        if str(tensor.dtype) != spec.dtype:
            raise IncompleteBundleError(f"payload {spec.name!r} dtype {tensor.dtype} != manifest {spec.dtype}")
        if list(tensor.shape) != spec.shape:
            raise IncompleteBundleError(f"payload {spec.name!r} shape {tuple(tensor.shape)} != manifest {spec.shape}")
        return tensor
