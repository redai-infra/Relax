# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Raw-byte payload assertions shared by TransferQueue tests.

These helpers intentionally live under ``tests``: production code does not need
generic payload traversal, while the data-plane contract must distinguish byte
identity from value equality for NaNs, signed zero, and nested tensors.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch


LeafDigest = tuple[str, str, str]


def _tensor_digest(value: torch.Tensor) -> LeafDigest:
    flat = value.detach().cpu().contiguous().reshape(-1)
    raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    shape = "x".join(str(dim) for dim in value.shape)
    return (str(value.dtype), shape, hashlib.sha256(raw).hexdigest())


def _ndarray_digest(value: np.ndarray) -> LeafDigest:
    contiguous = np.ascontiguousarray(value)
    shape = "x".join(str(dim) for dim in value.shape)
    return (f"np.{contiguous.dtype}", shape, hashlib.sha256(contiguous.tobytes()).hexdigest())


def _scalar_digest(value: Any) -> LeafDigest:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = repr(value).encode("utf-8")
    return (f"py.{type(value).__name__}", "", hashlib.sha256(raw).hexdigest())


def _unwrap_non_tensor(value: Any) -> Any:
    if type(value).__name__ == "NonTensorStack":
        return value.tolist()
    if type(value).__name__ == "NonTensorData":
        return value.data
    return value


def leaf_digests(payload: Any, prefix: str = "payload") -> dict[str, LeafDigest]:
    """Map every supported leaf to ``(dtype, shape, raw-byte SHA-256)``."""
    payload = _unwrap_non_tensor(payload)
    digests: dict[str, LeafDigest] = {}
    if isinstance(payload, torch.Tensor):
        if payload.is_nested:
            for row_index, row in enumerate(payload.unbind()):
                digests[f"{prefix}[{row_index}]"] = _tensor_digest(row)
        else:
            digests[prefix] = _tensor_digest(payload)
    elif isinstance(payload, np.ndarray):
        digests[prefix] = _ndarray_digest(payload)
    elif isinstance(payload, dict):
        for key in sorted(payload):
            digests.update(leaf_digests(payload[key], f"{prefix}.{key}"))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            digests.update(leaf_digests(item, f"{prefix}[{index}]"))
    elif isinstance(payload, (str, bytes, bool, int, float)) or payload is None:
        digests[prefix] = _scalar_digest(payload)
    else:
        raise TypeError(f"Unsupported payload leaf at {prefix}: {type(payload).__name__}")
    return digests


def diff_digests(expected: dict[str, LeafDigest], actual: dict[str, LeafDigest]) -> list[str]:
    """Return mismatch descriptions; an empty list means byte-exact."""
    problems: list[str] = []
    for path in sorted(expected.keys() | actual.keys()):
        want, have = expected.get(path), actual.get(path)
        if want is None:
            problems.append(f"{path}: unexpected extra leaf {have}")
        elif have is None:
            problems.append(f"{path}: missing (expected {want})")
        elif want != have:
            for axis, want_part, have_part in zip(("dtype", "shape", "sha256"), want, have, strict=True):
                if want_part != have_part:
                    problems.append(f"{path}: {axis} mismatch (expected {want_part}, got {have_part})")
    return problems
