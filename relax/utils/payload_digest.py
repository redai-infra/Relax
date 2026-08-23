# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Leaf-level byte fingerprints for data-plane payloads.

Byte-exact acceptance for the TransferQueue data plane needs one canonical
fingerprint definition shared by tests, benchmarks, and the multimodal fixture
generator.  ``torch.equal`` is a *value* comparison (``NaN != NaN``, and
``-0.0 == 0.0``), so it cannot prove byte identity; these helpers hash the raw
storage bytes instead.

The unit of comparison is a *leaf*: payloads are traversed recursively
(dict / list / tuple), tensors are normalized to contiguous row-major CPU
storage, and every leaf yields ``(dtype, shape, sha256)``.  Comparing two
payloads reduces to comparing their digest maps, which also produces precise
"which leaf, which axis" mismatch reports for acceptance logs.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np
import torch


LeafDigest = tuple[str, str, str]  # (dtype, shape, sha256 of raw bytes)


def _tensor_digest(value: torch.Tensor) -> LeafDigest:
    """Digest a tensor's storage bytes, normalized to contiguous CPU layout.

    ``view(torch.uint8)`` reinterprets storage without conversion, so bfloat16
    and other numpy-unsupported dtypes hash losslessly.
    """
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
    elif isinstance(value, float):
        # repr(float("nan")) discards the payload bits. Pack the Python double
        # directly so distinct NaNs and signed zero remain byte-distinguishable.
        raw = struct.pack("!d", value)
    else:  # bool / int / None — repr is canonical for these types.
        raw = repr(value).encode("utf-8")
    return (f"py.{type(value).__name__}", "", hashlib.sha256(raw).hexdigest())


def _unwrap_non_tensor(value: Any) -> Any:
    """Unwrap tensordict ``NonTensorData`` / ``NonTensorStack`` wrappers.

    TransferQueue returns non-tensor fields re-wrapped by tensordict; the
    fingerprint must see the underlying Python object so that put-side and get-
    side digests are comparable.
    """
    if type(value).__name__ in ("NonTensorData", "NonTensorStack"):
        return value.tolist() if type(value).__name__ == "NonTensorStack" else value.data
    return value


def _dict_child_path(prefix: str, key: Any) -> str:
    """Render a string dict key without colliding with nested/list paths."""
    if not isinstance(key, str):
        raise TypeError(f"Unsupported payload dict key at {prefix}: {type(key).__name__}")
    if key.isidentifier():
        return f"{prefix}.{key}"
    return f"{prefix}[{key!r}]"


def leaf_digests(payload: Any, prefix: str = "payload") -> dict[str, LeafDigest]:
    """Map every leaf of *payload* to ``(dtype, shape, sha256)``.

    Supported nodes: dict (sorted keys), list/tuple, ``torch.Tensor``
    (including jagged ``NestedTensor``, digested per row so put-side lists and
    get-side NestedTensors compare equal), ``np.ndarray``, and scalar leaves
    (str/bytes/bool/int/float/None).  Unknown node types raise ``TypeError`` so
    no leaf is ever silently skipped.
    """
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
        for key in payload:
            if not isinstance(key, str):
                raise TypeError(f"Unsupported payload dict key at {prefix}: {type(key).__name__}")
        for key in sorted(payload):
            digests.update(leaf_digests(payload[key], _dict_child_path(prefix, key)))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            digests.update(leaf_digests(item, f"{prefix}[{index}]"))
    elif isinstance(payload, (str, bytes, bool, int, float)) or payload is None:
        digests[prefix] = _scalar_digest(payload)
    else:
        raise TypeError(f"Unsupported payload leaf at {prefix}: {type(payload).__name__}")
    return digests


def diff_digests(expected: dict[str, LeafDigest], actual: dict[str, LeafDigest]) -> list[str]:
    """Return human-readable mismatch lines; empty list means byte-exact."""
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


def total_leaf_bytes(payload: Any) -> int:
    """Total payload bytes across all tensor/ndarray leaves (for effective-
    bandwidth accounting).

    Scalar leaves count their encoded byte length; container overhead is
    excluded because acceptance bandwidth is defined over payload bytes.
    """
    payload = _unwrap_non_tensor(payload)
    if isinstance(payload, torch.Tensor):
        if payload.is_nested:
            return sum(row.numel() * row.element_size() for row in payload.unbind())
        return payload.numel() * payload.element_size()
    if isinstance(payload, np.ndarray):
        return payload.nbytes
    if isinstance(payload, dict):
        return sum(total_leaf_bytes(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(total_leaf_bytes(item) for item in payload)
    if isinstance(payload, bytes):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    return 0
