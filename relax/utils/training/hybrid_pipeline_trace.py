# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import atexit
import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, TextIO


_WRITERS: dict[Path, TextIO] = {}
_WRITER_LOCK = threading.Lock()
_SCALAR_TYPES = (str, int, float, bool, type(None))
_FINGERPRINT_BITS = 128
_FINGERPRINT_MODULUS = 1 << _FINGERPRINT_BITS
_RECORD_FIELDS = {
    "event",
    "monotonic_ns",
    "rollout_id",
    "chunk_index",
    "sample_count",
    "total_tokens",
    "response_tokens",
    "multimodal_tensor_bytes",
    "role",
    "hostname",
    "pid",
    "global_rank",
    "cuda_visible_devices",
    "cuda_max_allocated_bytes",
    "cuda_max_reserved_bytes",
    "global_indexes_fingerprint",
}


def _close_writers() -> None:
    with _WRITER_LOCK:
        for writer in _WRITERS.values():
            writer.close()
        _WRITERS.clear()


atexit.register(_close_writers)


def _global_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except (ImportError, RuntimeError):
        pass
    return -1


def _cuda_memory_peaks() -> tuple[int | None, int | None]:
    try:
        import torch

        if torch.cuda.is_available():
            return (
                int(torch.cuda.max_memory_allocated()),
                int(torch.cuda.max_memory_reserved()),
            )
    except (ImportError, RuntimeError):
        pass
    return None, None


def _tensor_bytes(value: Any) -> int:
    if value is None or isinstance(value, (str, bytes)):
        return 0
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)

    nelement = getattr(value, "nelement", None)
    element_size = getattr(value, "element_size", None)
    if callable(nelement) and callable(element_size):
        return int(nelement()) * int(element_size())
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return 0


def _sum_ints(values: Any) -> int | None:
    if values is None:
        return None
    try:
        return sum(int(value) for value in values)
    except (TypeError, ValueError):
        return None


def fingerprint_global_indexes(global_indexes: Any) -> str | None:
    """Return an order-independent, multiplicity-sensitive digest.

    Digests are added modulo 2**128 so an analyzer can combine chunk digests
    without recording sample indexes or depending on producer/fetch grouping.
    """
    if global_indexes is None:
        return None
    normalized = [int(index) for index in global_indexes]
    accumulator = 0
    for index in normalized:
        digest = hashlib.blake2b(str(index).encode("ascii"), digest_size=16).digest()
        accumulator = (accumulator + int.from_bytes(digest, "big")) % _FINGERPRINT_MODULUS
    return f"{accumulator:032x}"


def _trace_path(trace_dir: str, role: str, hostname: str, pid: int, rank: int) -> Path:
    safe_hostname = hostname.replace("/", "_")
    safe_role = role.replace("/", "_")
    return Path(trace_dir) / f"hybrid-pipeline-{safe_role}-{safe_hostname}-pid{pid}-rank{rank}.jsonl"


def emit_hybrid_pipeline_event(
    args: Any,
    event: str,
    *,
    rollout_id: int,
    role: str,
    chunk_index: int | None = None,
    sample_count: int | None = None,
    batch: dict[str, Any] | None = None,
    global_indexes: Any = None,
    monotonic_ns: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Append one content-free Hybrid pipeline event to this process's
    JSONL."""
    trace_dir = getattr(args, "hybrid_pipeline_trace_dir", None)
    if not trace_dir:
        return None
    if details is not None:
        invalid = {key: value for key, value in details.items() if not isinstance(value, _SCALAR_TYPES)}
        if invalid:
            raise TypeError(f"Hybrid pipeline trace details must be scalar, got {invalid}")
        reserved = sorted(_RECORD_FIELDS.intersection(details))
        if reserved:
            raise ValueError(f"Hybrid pipeline trace details cannot replace record fields: {reserved}")

    hostname = socket.gethostname()
    pid = os.getpid()
    rank = _global_rank()
    cuda_allocated, cuda_reserved = _cuda_memory_peaks()
    total_lengths = batch.get("total_lengths") if batch is not None else None
    response_lengths = batch.get("response_lengths") if batch is not None else None
    if sample_count is None and total_lengths is not None:
        sample_count = len(total_lengths)

    record = {
        "event": event,
        "monotonic_ns": time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns),
        "rollout_id": int(rollout_id),
        "chunk_index": chunk_index,
        "sample_count": sample_count,
        "total_tokens": _sum_ints(total_lengths),
        "response_tokens": _sum_ints(response_lengths),
        "multimodal_tensor_bytes": (
            _tensor_bytes(batch.get("multimodal_train_inputs")) if batch is not None else None
        ),
        "role": role,
        "hostname": hostname,
        "pid": pid,
        "global_rank": rank,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_max_allocated_bytes": cuda_allocated,
        "cuda_max_reserved_bytes": cuda_reserved,
        "global_indexes_fingerprint": fingerprint_global_indexes(global_indexes),
    }
    if details:
        record.update(details)

    path = _trace_path(os.fspath(trace_dir), role, hostname, pid, rank)
    with _WRITER_LOCK:
        writer = _WRITERS.get(path)
        if writer is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = path.open("a", encoding="utf-8", buffering=1)
            _WRITERS[path] = writer
        writer.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return record
