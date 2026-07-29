# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate and summarize Task 21 Hybrid pipeline benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


TRACE_REQUIRED_FIELDS = {
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
NVML_COLUMNS = (
    "timestamp",
    "gpu_index",
    "gpu_name",
    "pstate",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "gpu_util_percent",
    "memory_util_percent",
    "memory_used_mib",
    "power_w",
)
DEFAULT_WINDOWS = ((4, 8), (9, 13), (14, 18))
PERFORMANCE_TAGS = (
    "perf/step_token_per_s",
    "perf/step_resp_token_per_s",
    "perf/step_time",
    "perf/hybrid_phase1_time",
)
COMPARISON_PERFORMANCE_TAGS = PERFORMANCE_TAGS + ("perf/wall_clock_samples_per_s",)
CORRECTNESS_GUARDRAIL_TAGS = (
    "rollout/raw_reward",
    "rollout/truncated_ratio",
    "train/loss",
    "train/grad_norm",
    "train/ppo_kl",
    "train/pg_clipfrac",
)
QUALITY_TAG_FRAGMENTS = (
    "raw_reward",
    "reward",
    "loss",
    "grad_norm",
    "response_length",
    "truncated",
    "staleness",
)
RUN_MANIFEST_REQUIRED_FIELDS = {
    "hostname",
    "condition",
    "order",
    "seed",
    "rollout_seed",
    "num_rollout",
    "max_staleness",
    "global_batch_size",
    "rollout_batch_size",
    "n_samples_per_prompt",
    "num_iters_per_train_update",
    "hybrid_pipeline_forward",
    "hybrid_pipeline_trace_dir",
    "hybrid_pipeline_fetch_timeout_s",
    "git_commit",
    "git_branch",
    "git_status_porcelain",
    "image_archive_sha256",
    "image_manifest_digest",
    "image_id",
    "transferqueue_commit",
    "python",
    "entrypoint",
}
COMPARISON_FIXED_MANIFEST_FIELDS = (
    "hostname",
    "num_rollout",
    "max_staleness",
    "global_batch_size",
    "rollout_batch_size",
    "n_samples_per_prompt",
    "num_iters_per_train_update",
    "hybrid_pipeline_fetch_timeout_s",
    "git_commit",
    "git_branch",
    "image_archive_sha256",
    "image_manifest_digest",
    "image_id",
    "transferqueue_commit",
    "python",
    "entrypoint",
)
COMPARISON_WORKLOAD_MANIFEST_FIELDS = (
    "schema_version",
    "baseline_commit",
    "model_variant",
    "model_name",
    "model_dir",
    "model_config_file",
    "data_file",
    "rollout_max_response_len",
    "rollout_max_prompt_len",
    "rollout_max_context_len",
    "actor_max_tokens_per_gpu",
    "resource",
    "rollout_num_gpus_per_engine",
    "physical_gpu_indices",
    "container_cuda_visible_devices",
    "gpu_hardware_fingerprint",
    "checkpoint_save",
    "sglang_deterministic_inference",
    "sglang_mem_fraction_static",
    "load_debug_rollout_data",
    "save_debug_rollout_data",
    "save_debug_train_data",
)
REPRODUCIBILITY_ARTIFACTS = (
    "pip-freeze.txt",
    "inputs.sha256",
    "transferqueue-wheel.sha256",
    "logs/launcher.log",
    "manifests/static-input-verification.log",
)
EXIT_STATUS_ARTIFACTS = (
    "training_exit_status.txt",
    "validation_exit_status.txt",
    "exit_status.txt",
)


class BenchmarkValidationError(RuntimeError):
    """Raised when benchmark artifacts violate a registered invariant."""


@dataclass(frozen=True)
class RunAnalysis:
    run_dir: Path
    manifest: dict[str, Any]
    trace_rows: list[dict[str, Any]]
    actor_rank_rows: list[dict[str, Any]]
    scalar_rows: list[dict[str, Any]]
    nvml_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _fail(message: str) -> None:
    raise BenchmarkValidationError(message)


def _validate_finite(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            _fail(f"{context} contains non-finite numeric value {value!r}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{context}[{index}]")


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        _fail(f"missing run manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse {path}: {exc}")
    if not isinstance(manifest, dict):
        _fail(f"{path} must contain a JSON object")
    missing = sorted(RUN_MANIFEST_REQUIRED_FIELDS - set(manifest))
    if missing:
        _fail(f"{path} is missing required fields {missing}")
    _validate_finite(manifest, str(path))
    return manifest


def _validate_hex_digest(value: Any, *, length: int, context: str, prefix: str = "") -> None:
    if not isinstance(value, str) or not value.startswith(prefix):
        _fail(f"{context} must be a {prefix!r}-prefixed hexadecimal string")
    payload = value[len(prefix) :]
    if len(payload) != length:
        _fail(f"{context} must contain {length} hexadecimal characters, got {len(payload)}")
    try:
        int(payload, 16)
    except ValueError:
        _fail(f"{context} contains non-hexadecimal characters")


def _validate_run_manifest(
    manifest: dict[str, Any],
    *,
    run_dir: Path,
    windows: Sequence[tuple[int, int]],
) -> None:
    for key in (
        "seed",
        "rollout_seed",
        "num_rollout",
        "max_staleness",
        "global_batch_size",
        "rollout_batch_size",
        "n_samples_per_prompt",
        "num_iters_per_train_update",
        "hybrid_pipeline_forward",
    ):
        if type(manifest[key]) is not int:
            _fail(f"{run_dir} manifest field {key!r} must be an integer, got {manifest[key]!r}")
    if "hybrid_pipeline_overlap" in manifest and (
        type(manifest["hybrid_pipeline_overlap"]) is not int or manifest["hybrid_pipeline_overlap"] not in (0, 1)
    ):
        _fail(
            f"{run_dir} manifest field 'hybrid_pipeline_overlap' must be integer 0 or 1, "
            f"got {manifest['hybrid_pipeline_overlap']!r}"
        )
    if manifest["seed"] != manifest["rollout_seed"]:
        _fail(
            f"{run_dir} must use the same paired Megatron/rollout seed, got "
            f"{manifest['seed']} and {manifest['rollout_seed']}"
        )
    if manifest["num_rollout"] <= max(end for _, end in windows):
        _fail(
            f"{run_dir} num_rollout={manifest['num_rollout']} does not cover "
            f"steady window ending at step {max(end for _, end in windows)}"
        )
    for key in (
        "global_batch_size",
        "rollout_batch_size",
        "n_samples_per_prompt",
        "num_iters_per_train_update",
    ):
        if manifest[key] <= 0:
            _fail(f"{run_dir} manifest field {key!r} must be positive, got {manifest[key]!r}")
    if manifest["max_staleness"] < 0:
        _fail(f"{run_dir} max_staleness must be non-negative, got {manifest['max_staleness']!r}")
    produced_samples = manifest["rollout_batch_size"] * manifest["n_samples_per_prompt"]
    if produced_samples != manifest["global_batch_size"]:
        _fail(
            f"{run_dir} must benchmark exactly one optimizer mini per rollout: "
            f"rollout_batch_size * n_samples_per_prompt={produced_samples}, "
            f"global_batch_size={manifest['global_batch_size']}"
        )
    if manifest["git_status_porcelain"] != "":
        _fail(f"{run_dir} was captured from a dirty working tree: {manifest['git_status_porcelain']!r}")

    timeout = manifest["hybrid_pipeline_fetch_timeout_s"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        _fail(f"{run_dir} has invalid hybrid_pipeline_fetch_timeout_s={timeout!r}")

    trace_dir = Path(str(manifest["hybrid_pipeline_trace_dir"])).resolve()
    expected_trace_dir = (run_dir / "timeline").resolve()
    if trace_dir != expected_trace_dir:
        _fail(f"{run_dir} manifest trace directory is {trace_dir}, expected {expected_trace_dir}")

    for key in ("hostname", "condition", "order", "git_branch", "python", "entrypoint"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            _fail(f"{run_dir} manifest field {key!r} must be a non-empty string")

    _validate_hex_digest(manifest["git_commit"], length=40, context=f"{run_dir} git_commit")
    _validate_hex_digest(
        manifest["image_archive_sha256"],
        length=64,
        context=f"{run_dir} image_archive_sha256",
    )
    _validate_hex_digest(
        manifest["image_manifest_digest"],
        length=64,
        prefix="sha256:",
        context=f"{run_dir} image_manifest_digest",
    )
    _validate_hex_digest(
        manifest["image_id"],
        length=64,
        prefix="sha256:",
        context=f"{run_dir} image_id",
    )
    _validate_hex_digest(
        manifest["transferqueue_commit"],
        length=40,
        context=f"{run_dir} transferqueue_commit",
    )
    if "gpu_hardware_fingerprint" in manifest:
        _validate_hex_digest(
            manifest["gpu_hardware_fingerprint"],
            length=64,
            context=f"{run_dir} gpu_hardware_fingerprint",
        )


def _require_reproducibility_artifacts(run_dir: Path) -> None:
    missing = []
    for relative_path in REPRODUCIBILITY_ARTIFACTS:
        path = run_dir / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative_path)
    if missing:
        _fail(f"{run_dir} is missing non-empty reproducibility artifacts {missing}")
    for relative_path in EXIT_STATUS_ARTIFACTS:
        path = run_dir / relative_path
        if not path.is_file():
            _fail(f"{run_dir} is missing exit-status artifact {relative_path}")
        try:
            status = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            _fail(f"cannot read {path}: {exc}")
        if status != "0":
            _fail(f"{run_dir} has non-zero or invalid {relative_path}: {status!r}")


def _load_trace_rows(run_dir: Path) -> list[dict[str, Any]]:
    trace_dir = run_dir / "timeline"
    paths = sorted(trace_dir.glob("*.jsonl"))
    if not paths:
        _fail(f"no Hybrid pipeline JSONL files found under {trace_dir}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        previous_ns = -1
        with path.open(encoding="utf-8") as reader:
            for line_number, line in enumerate(reader, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    _fail(f"{path}:{line_number} is invalid JSON: {exc}")
                if not isinstance(row, dict):
                    _fail(f"{path}:{line_number} must contain a JSON object")
                missing = sorted(TRACE_REQUIRED_FIELDS - set(row))
                if missing:
                    _fail(f"{path}:{line_number} is missing trace fields {missing}")
                _validate_finite(row, f"{path}:{line_number}")
                monotonic_ns = row["monotonic_ns"]
                if type(monotonic_ns) is not int or monotonic_ns < 0:
                    _fail(f"{path}:{line_number} has invalid monotonic_ns={monotonic_ns!r}")
                if monotonic_ns < previous_ns:
                    _fail(f"{path}:{line_number} is not monotonic: previous={previous_ns}, current={monotonic_ns}")
                previous_ns = monotonic_ns
                row["_source_file"] = path.name
                row["_source_line"] = line_number
                rows.append(row)

    if not rows:
        _fail(f"trace files under {trace_dir} contain no events")
    hostnames = {row["hostname"] for row in rows}
    if len(hostnames) != 1:
        _fail(f"trace events span multiple hostnames and cannot share one monotonic clock: {sorted(hostnames)}")
    return sorted(rows, key=lambda row: (row["monotonic_ns"], row["_source_file"], row["_source_line"]))


def _pair_events(
    rows: Sequence[dict[str, Any]],
    start_event: str,
    end_event: str,
    *,
    key: str,
    context: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    starts: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    ends: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["event"] == start_event:
            starts[row.get(key)].append(row)
        elif row["event"] == end_event:
            ends[row.get(key)].append(row)
    if set(starts) != set(ends):
        _fail(
            f"{context} has unmatched {start_event}/{end_event} keys: "
            f"starts={sorted(starts, key=str)}, ends={sorted(ends, key=str)}"
        )

    pairs = []
    for event_key in sorted(starts, key=str):
        if len(starts[event_key]) != 1 or len(ends[event_key]) != 1:
            _fail(
                f"{context} requires one {start_event}/{end_event} pair for {key}={event_key!r}, "
                f"got starts={len(starts[event_key])}, ends={len(ends[event_key])}"
            )
        start, end = starts[event_key][0], ends[event_key][0]
        if start["monotonic_ns"] > end["monotonic_ns"]:
            _fail(
                f"{context} has {start_event} after {end_event} for {key}={event_key!r}: "
                f"{start['monotonic_ns']} > {end['monotonic_ns']}"
            )
        pairs.append((start, end))
    return pairs


def _require_count(rows: Sequence[dict[str, Any]], event: str, count: int, context: str) -> list[dict[str, Any]]:
    matches = [row for row in rows if row["event"] == event]
    if len(matches) != count:
        _fail(f"{context} expected {count} {event!r} events, got {len(matches)}")
    return matches


def _stream_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return row["hostname"], int(row["pid"]), int(row["global_rank"])


def _combine_global_index_fingerprints(
    rows: Sequence[dict[str, Any]],
    *,
    context: str,
) -> str:
    """Combine additive 128-bit chunk digests without exposing sample
    indexes."""
    accumulator = 0
    modulus = 1 << 128
    for row in rows:
        fingerprint = row.get("global_indexes_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 32:
            _fail(f"{context} has invalid global index fingerprint {fingerprint!r}")
        try:
            value = int(fingerprint, 16)
        except ValueError:
            _fail(f"{context} has non-hex global index fingerprint {fingerprint!r}")
        accumulator = (accumulator + value) % modulus
    return f"{accumulator:032x}"


def _analyze_trace(
    rows: Sequence[dict[str, Any]],
    *,
    pipeline_enabled: bool,
    pipeline_overlap_enabled: bool,
    expected_samples: int,
    expected_actor_chunks: int,
    expected_producer_chunks: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    producer_by_rollout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    actor_by_rollout_stream: dict[tuple[int, tuple[str, int, int]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rollout_id = int(row["rollout_id"])
        if row["role"] == "rollout":
            producer_by_rollout[rollout_id].append(row)
        elif row["role"] == "actor":
            actor_by_rollout_stream[(rollout_id, _stream_key(row))].append(row)

    producer_rollouts = set(producer_by_rollout)
    actor_rollouts = {rollout_id for rollout_id, _ in actor_by_rollout_stream}
    if not producer_rollouts or producer_rollouts != actor_rollouts:
        _fail(
            "producer and actor rollout IDs differ: "
            f"producer={sorted(producer_rollouts)}, actor={sorted(actor_rollouts)}"
        )

    expected_stream_chunks = expected_actor_chunks if pipeline_enabled else 1
    rollout_rows: list[dict[str, Any]] = []
    actor_rank_rows: list[dict[str, Any]] = []

    for rollout_id in sorted(producer_rollouts):
        producer_rows = producer_by_rollout[rollout_id]
        put_pairs = _pair_events(
            producer_rows,
            "tq_put_start",
            "tq_put_done",
            key="event_id",
            context=f"rollout_id={rollout_id} producer",
        )
        if not put_pairs:
            _fail(f"rollout_id={rollout_id} producer has no completed puts")
        put_start = [start for start, _ in put_pairs]
        put_done = [end for _, end in put_pairs]
        if any(row["sample_count"] is None for row in put_done):
            _fail(f"rollout_id={rollout_id} producer put is missing sample_count")
        if expected_producer_chunks is not None:
            if expected_samples % expected_producer_chunks != 0:
                _fail(
                    "expected_samples must be divisible by expected_producer_chunks, "
                    f"got {expected_samples=} and {expected_producer_chunks=}"
                )
            expected_producer_chunk_samples = expected_samples // expected_producer_chunks
            producer_chunk_samples = [int(row["sample_count"] or 0) for row in put_done]
            if len(put_pairs) != expected_producer_chunks or any(
                sample_count != expected_producer_chunk_samples for sample_count in producer_chunk_samples
            ):
                _fail(
                    f"rollout_id={rollout_id} expected {expected_producer_chunks} producer "
                    f"chunks of {expected_producer_chunk_samples} samples, got "
                    f"{producer_chunk_samples}"
                )
        producer_samples = sum(int(row["sample_count"]) for row in put_done)
        if producer_samples != expected_samples:
            _fail(
                f"rollout_id={rollout_id} producer sample conservation failed: "
                f"expected={expected_samples}, actual={producer_samples}"
            )
        producer_fingerprint = _combine_global_index_fingerprints(
            put_done,
            context=f"rollout_id={rollout_id} producer",
        )

        streams = {
            stream: stream_rows
            for (stream_rollout_id, stream), stream_rows in actor_by_rollout_stream.items()
            if stream_rollout_id == rollout_id
        }
        if not streams:
            _fail(f"rollout_id={rollout_id} has no actor trace stream")

        for stream, stream_rows in streams.items():
            context = f"rollout_id={rollout_id} actor_stream={stream}"
            fetch_pairs = _pair_events(
                stream_rows,
                "chunk_fetch_start",
                "chunk_fetch_end",
                key="chunk_index",
                context=context,
            )
            forward_pairs = _pair_events(
                stream_rows,
                "actor_forward_start",
                "actor_forward_end",
                key="chunk_index",
                context=context,
            )
            restore_pairs = _pair_events(
                stream_rows,
                "actor_restore_start",
                "actor_restore_end",
                key="chunk_index",
                context=context,
            )
            if len(fetch_pairs) != expected_stream_chunks or len(forward_pairs) != expected_stream_chunks:
                _fail(
                    f"{context} expected {expected_stream_chunks} fetch/forward chunks, "
                    f"got fetch={len(fetch_pairs)}, forward={len(forward_pairs)}"
                )
            if len(restore_pairs) != 1:
                _fail(f"{context} expected exactly one actor restore, got {len(restore_pairs)}")
            _pair_events(
                stream_rows,
                "advantages_start",
                "advantages_end",
                key="chunk_index",
                context=context,
            )
            _pair_events(
                stream_rows,
                "optimizer_start",
                "optimizer_end",
                key="chunk_index",
                context=context,
            )
            _require_count(stream_rows, "advantages_start", 1, context)
            _require_count(stream_rows, "optimizer_start", 1, context)

            fetch_end = [end for _, end in fetch_pairs]
            forward_start = [start for start, _ in forward_pairs]
            forward_end = [end for _, end in forward_pairs]
            for event_rows, event_name in ((fetch_end, "fetch"), (forward_start, "forward")):
                if any(row["sample_count"] is None for row in event_rows):
                    _fail(f"{context} {event_name} event is missing sample_count")
                actual_samples = sum(int(row["sample_count"]) for row in event_rows)
                if actual_samples != expected_samples:
                    _fail(
                        f"{context} {event_name} sample conservation failed: "
                        f"expected={expected_samples}, actual={actual_samples}"
                    )
            if any(row["global_indexes_fingerprint"] is None for row in fetch_end):
                _fail(f"{context} fetch event is missing global index fingerprint")
            fetch_fingerprint = _combine_global_index_fingerprints(
                fetch_end,
                context=f"{context} fetch",
            )
            forward_fingerprint = _combine_global_index_fingerprints(
                forward_start,
                context=f"{context} forward",
            )
            if fetch_fingerprint != producer_fingerprint:
                _fail(
                    f"{context} producer/fetch global index fingerprints differ: "
                    f"producer={producer_fingerprint}, actor={fetch_fingerprint}"
                )
            if forward_fingerprint != fetch_fingerprint:
                _fail(
                    f"{context} fetch/forward global index fingerprints differ: "
                    f"fetch={fetch_fingerprint}, forward={forward_fingerprint}"
                )

            fetch_by_chunk = {end["chunk_index"]: (start, end) for start, end in fetch_pairs}
            for forward_start_row, forward_end_row in forward_pairs:
                chunk_index = forward_start_row["chunk_index"]
                if chunk_index not in fetch_by_chunk:
                    _fail(f"{context} forward chunk {chunk_index!r} has no matching fetch")
                _, fetch_end_row = fetch_by_chunk[chunk_index]
                if fetch_end_row["monotonic_ns"] > forward_start_row["monotonic_ns"]:
                    _fail(f"{context} forward chunk {chunk_index!r} starts before fetch completes")
                if forward_start_row["monotonic_ns"] > forward_end_row["monotonic_ns"]:
                    _fail(f"{context} forward chunk {chunk_index!r} ends before it starts")

            first_phase_ns = min(
                min(start["monotonic_ns"] for start, _ in fetch_pairs),
                restore_pairs[0][0]["monotonic_ns"],
            )
            last_forward_ns = max(row["monotonic_ns"] for row in forward_end)
            first_forward_ns = min(row["monotonic_ns"] for row in forward_start)
            last_fetch_end_ns = max(row["monotonic_ns"] for row in fetch_end)
            last_put_start_ns = max(row["monotonic_ns"] for row in put_start)
            last_put_done_ns = max(row["monotonic_ns"] for row in put_done)
            chunk_schedule_overlapped = first_forward_ns < last_fetch_end_ns
            if pipeline_enabled and chunk_schedule_overlapped != pipeline_overlap_enabled:
                expected_order = (
                    "the first forward before the final fetch completed"
                    if pipeline_overlap_enabled
                    else "all fetches before the first forward"
                )
                _fail(f"{context} does not implement the requested overlap mode: expected {expected_order}")
            ready_rollout_ids = [
                int(row["rollout_id"])
                for row in rows
                if row["role"] == "rollout"
                and row["event"] == "tq_put_done"
                and row["monotonic_ns"] <= first_forward_ns
            ]
            # A completed actor fetch proves the current rollout was ready even
            # if the producer's post-async_put trace write loses the scheduling
            # race with the consumer process.
            ready_rollout_ids.append(rollout_id)
            producer_lead = max(ready_rollout_ids) - rollout_id
            actor_rank_rows.append(
                {
                    "rollout_id": rollout_id,
                    "hostname": stream[0],
                    "pid": stream[1],
                    "global_rank": stream[2],
                    "pipeline_enabled": pipeline_enabled,
                    "pipeline_overlap_enabled": pipeline_overlap_enabled,
                    "producer_samples": producer_samples,
                    "actor_fetch_samples": sum(int(row["sample_count"]) for row in fetch_end),
                    "actor_forward_samples": sum(int(row["sample_count"]) for row in forward_start),
                    "producer_put_count": len(put_pairs),
                    "actor_fetch_count": len(fetch_pairs),
                    "actor_forward_count": len(forward_pairs),
                    "actor_restore_count": len(restore_pairs),
                    "phase1_s": (last_forward_ns - first_phase_ns) / 1e9,
                    # Strict evidence of producer/actor overlap: the actor
                    # starts forwarding before the producer even begins its
                    # final put. A delayed post-put trace write alone cannot
                    # make this condition true.
                    "first_forward_before_last_put_start": first_forward_ns < last_put_start_ns,
                    # Diagnostic only: async_put may have made data visible
                    # before the producer coroutine records tq_put_done.
                    "first_forward_before_last_put_done": first_forward_ns < last_put_done_ns,
                    "first_forward_before_last_fetch_end": chunk_schedule_overlapped,
                    "all_chunks_fetched_before_first_forward": not chunk_schedule_overlapped,
                    "producer_overlap_s": max(0, last_put_start_ns - first_forward_ns) / 1e9,
                    "transfer_overlap_s": max(0, last_put_done_ns - first_forward_ns) / 1e9,
                    "producer_lead_at_first_forward": producer_lead,
                    "actor_multimodal_tensor_bytes": sum(
                        int(row["multimodal_tensor_bytes"] or 0) for row in fetch_end
                    ),
                    "actor_total_tokens": sum(int(row["total_tokens"] or 0) for row in fetch_end),
                    "actor_response_tokens": sum(int(row["response_tokens"] or 0) for row in fetch_end),
                    "cuda_max_allocated_bytes": max(
                        (int(row["cuda_max_allocated_bytes"] or 0) for row in stream_rows),
                        default=0,
                    ),
                    "cuda_max_reserved_bytes": max(
                        (int(row["cuda_max_reserved_bytes"] or 0) for row in stream_rows),
                        default=0,
                    ),
                    "fetch_global_indexes_fingerprint": fetch_fingerprint,
                    "forward_global_indexes_fingerprint": forward_fingerprint,
                }
            )

        primary = min(
            (row for row in actor_rank_rows if row["rollout_id"] == rollout_id),
            key=lambda row: (
                row["global_rank"] < 0,
                row["global_rank"] if row["global_rank"] >= 0 else row["pid"],
            ),
        )
        put_start_times = [row["monotonic_ns"] for row in put_start]
        put_done_times = [row["monotonic_ns"] for row in put_done]
        rollout_rows.append(
            {
                **primary,
                "producer_global_indexes_fingerprint": producer_fingerprint,
                "producer_first_put_start_ns": min(put_start_times),
                "producer_last_put_start_ns": max(put_start_times),
                "producer_first_put_done_ns": min(put_done_times),
                "producer_last_put_done_ns": max(put_done_times),
                "producer_ready_window_s": (max(put_done_times) - min(put_done_times)) / 1e9,
                "producer_multimodal_tensor_bytes": sum(int(row["multimodal_tensor_bytes"] or 0) for row in put_done),
                "producer_total_tokens": sum(int(row["total_tokens"] or 0) for row in put_done),
                "producer_response_tokens": sum(int(row["response_tokens"] or 0) for row in put_done),
            }
        )

    producer_overlap_count = sum(bool(row["first_forward_before_last_put_start"]) for row in rollout_rows)
    transfer_overlap_count = sum(bool(row["first_forward_before_last_put_done"]) for row in rollout_rows)
    chunk_schedule_overlap_count = sum(bool(row["first_forward_before_last_fetch_end"]) for row in rollout_rows)
    summary = {
        "hostname": next(iter({row["hostname"] for row in rows})),
        "pipeline_enabled": pipeline_enabled,
        "pipeline_overlap_enabled": pipeline_overlap_enabled,
        "rollout_count": len(rollout_rows),
        "actor_stream_count": len({_stream_key(row) for row in rows if row["role"] == "actor"}),
        "producer_overlap_rollout_count": producer_overlap_count,
        "producer_overlap_rollout_ratio": producer_overlap_count / len(rollout_rows),
        "transfer_overlap_rollout_count": transfer_overlap_count,
        "transfer_overlap_rollout_ratio": transfer_overlap_count / len(rollout_rows),
        "chunk_schedule_overlap_rollout_count": chunk_schedule_overlap_count,
        "chunk_schedule_overlap_rollout_ratio": chunk_schedule_overlap_count / len(rollout_rows),
        "mean_phase1_s": statistics.fmean(row["phase1_s"] for row in rollout_rows),
        "mean_producer_overlap_s": statistics.fmean(row["producer_overlap_s"] for row in rollout_rows),
        "mean_transfer_overlap_s": statistics.fmean(row["transfer_overlap_s"] for row in rollout_rows),
        "mean_producer_lead_at_first_forward": statistics.fmean(
            row["producer_lead_at_first_forward"] for row in rollout_rows
        ),
        "max_producer_lead_at_first_forward": max(row["producer_lead_at_first_forward"] for row in rollout_rows),
        "mean_producer_ready_window_s": statistics.fmean(row["producer_ready_window_s"] for row in rollout_rows),
        "max_cuda_allocated_bytes": max(row["cuda_max_allocated_bytes"] for row in actor_rank_rows),
        "max_cuda_reserved_bytes": max(row["cuda_max_reserved_bytes"] for row in actor_rank_rows),
    }
    return rollout_rows, actor_rank_rows, summary


def _load_tensorboard_scalars(run_dir: Path) -> list[dict[str, Any]]:
    event_paths = sorted(path for path in run_dir.rglob("events.out.tfevents.*") if path.is_file())
    if not event_paths:
        return []
    # TensorBoard 2.10's generated protobuf bindings need the pure-Python
    # compatibility path when the host has a newer protobuf runtime.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for path in event_paths:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:
            _fail(f"cannot load TensorBoard event file {path}: {exc}")
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                _validate_finite(event.value, f"{path}:{tag}:step={event.step}")
                row = {
                    "tag": tag,
                    "step": int(event.step),
                    "value": float(event.value),
                    "wall_time": float(event.wall_time),
                    "source_file": str(path.relative_to(run_dir)),
                }
                key = tag, int(event.step)
                if key not in by_key or row["wall_time"] >= by_key[key]["wall_time"]:
                    by_key[key] = row
    return sorted(by_key.values(), key=lambda row: (row["tag"], row["step"]))


def _parse_nvml_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "telemetry" / "nvidia-smi.csv"
    if not path.is_file():
        return []

    rows = []
    with path.open(encoding="utf-8", newline="") as reader:
        for line_number, values in enumerate(csv.reader(reader), start=1):
            if not values:
                continue
            if len(values) != len(NVML_COLUMNS):
                _fail(f"{path}:{line_number} expected {len(NVML_COLUMNS)} columns, got {len(values)}")
            row = {key: value.strip() for key, value in zip(NVML_COLUMNS, values, strict=True)}
            for key in (
                "gpu_index",
                "temperature_c",
                "sm_clock_mhz",
                "memory_clock_mhz",
                "gpu_util_percent",
                "memory_util_percent",
                "memory_used_mib",
                "power_w",
            ):
                try:
                    row[key] = float(row[key])
                except ValueError as exc:
                    _fail(f"{path}:{line_number} has non-numeric {key}={row[key]!r}: {exc}")
            try:
                row["wall_time"] = datetime.strptime(
                    row["timestamp"],
                    "%Y/%m/%d %H:%M:%S.%f",
                ).timestamp()
            except ValueError:
                row["wall_time"] = None
            _validate_finite(row, f"{path}:{line_number}")
            rows.append(row)
    return rows


def _parse_windows(value: str) -> tuple[tuple[int, int], ...]:
    windows = []
    for item in value.split(","):
        try:
            start_text, end_text = item.strip().split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
        except ValueError:
            _fail(f"invalid steady window {item!r}; expected START-END")
        if start < 0 or end < start:
            _fail(f"invalid steady window {item!r}; require 0 <= START <= END")
        windows.append((start, end))
    if not windows:
        _fail("at least one steady window is required")
    flattened = [step for start, end in windows for step in range(start, end + 1)]
    if len(flattened) != len(set(flattened)):
        _fail(f"steady windows overlap: {windows}")
    return tuple(windows)


def _stable_steps(windows: Sequence[tuple[int, int]]) -> set[int]:
    return {step for start, end in windows for step in range(start, end + 1)}


def _scalar_map(rows: Sequence[dict[str, Any]], tag: str) -> dict[int, float]:
    return {int(row["step"]): float(row["value"]) for row in rows if row["tag"] == tag}


def _aggregate_throughput(
    scalar_rows: Sequence[dict[str, Any]],
    throughput_tag: str,
    windows: Sequence[tuple[int, int]],
) -> float | None:
    step_time = _scalar_map(scalar_rows, "perf/step_time")
    throughput = _scalar_map(scalar_rows, throughput_tag)
    steps = sorted(_stable_steps(windows).intersection(step_time, throughput))
    if not steps:
        return None
    total_time = sum(step_time[step] for step in steps)
    if total_time <= 0:
        _fail(f"{throughput_tag} has non-positive aggregate steady step time")
    return sum(throughput[step] * step_time[step] for step in steps) / total_time


def _aggregate_samples_per_second(
    scalar_rows: Sequence[dict[str, Any]],
    windows: Sequence[tuple[int, int]],
    samples_per_step: int,
) -> float | None:
    step_time = _scalar_map(scalar_rows, "perf/step_time")
    steps = sorted(_stable_steps(windows).intersection(step_time))
    if not steps:
        return None
    total_time = sum(step_time[step] for step in steps)
    if total_time <= 0:
        _fail("perf/step_time has non-positive aggregate steady time")
    return samples_per_step * len(steps) / total_time


def _metric_summary(
    scalar_rows: Sequence[dict[str, Any]],
    windows: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    steps = _stable_steps(windows)
    tags = sorted({row["tag"] for row in scalar_rows})
    summary: dict[str, Any] = {}
    for tag in tags:
        values = [float(row["value"]) for row in scalar_rows if row["tag"] == tag and row["step"] in steps]
        if not values:
            continue
        ordered = sorted(values)
        summary[tag] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "min": ordered[0],
            "max": ordered[-1],
            "p50": statistics.median(ordered),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        }
    for tag in ("perf/step_token_per_s", "perf/step_resp_token_per_s"):
        aggregate = _aggregate_throughput(scalar_rows, tag, windows)
        if aggregate is not None:
            summary.setdefault(tag, {})["aggregate"] = aggregate
    return summary


def _steady_wall_time_intervals(
    scalar_rows: Sequence[dict[str, Any]],
    windows: Sequence[tuple[int, int]],
) -> list[tuple[float, float]]:
    """Map registered steady steps to wall-clock intervals using the
    TensorBoard step end time and perf/step_time duration."""
    steady_steps = _stable_steps(windows)
    intervals = []
    for row in scalar_rows:
        if row["tag"] != "perf/step_time" or int(row["step"]) not in steady_steps:
            continue
        duration_s = float(row["value"])
        wall_time = row.get("wall_time")
        if duration_s <= 0:
            _fail(f"perf/step_time step={row['step']} must be positive, got {duration_s}")
        if not isinstance(wall_time, (int, float)) or not math.isfinite(wall_time):
            _fail(f"perf/step_time step={row['step']} is missing a finite TensorBoard wall_time")
        intervals.append((float(wall_time) - duration_s, float(wall_time)))
    return sorted(intervals)


def _nvml_summary(
    rows: Sequence[dict[str, Any]],
    *,
    steady_intervals: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    if not rows:
        return {}

    def is_steady(row: dict[str, Any]) -> bool:
        wall_time = row.get("wall_time")
        return isinstance(wall_time, (int, float)) and any(
            start <= float(wall_time) <= end for start, end in steady_intervals
        )

    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gpu[int(row["gpu_index"])].append(row)
    per_gpu = {}
    for gpu_index, gpu_rows in sorted(by_gpu.items()):
        full_utilization = [float(row["gpu_util_percent"]) for row in gpu_rows]
        steady_rows = [row for row in gpu_rows if is_steady(row)]
        steady_utilization = [float(row["gpu_util_percent"]) for row in steady_rows]
        per_gpu[str(gpu_index)] = {
            "full_run_sample_count": len(gpu_rows),
            "full_run_mean_gpu_util_percent": statistics.fmean(full_utilization),
            "steady_sample_count": len(steady_rows),
            "steady_mean_gpu_util_percent": (statistics.fmean(steady_utilization) if steady_utilization else None),
            "steady_idle_ratio_below_10_percent": (
                sum(value < 10 for value in steady_utilization) / len(steady_utilization)
                if steady_utilization
                else None
            ),
            "peak_memory_used_mib": max(float(row["memory_used_mib"]) for row in gpu_rows),
            "mean_power_w": statistics.fmean(float(row["power_w"]) for row in gpu_rows),
        }
    steady_gpu_utilization = [
        item["steady_mean_gpu_util_percent"]
        for item in per_gpu.values()
        if item["steady_mean_gpu_util_percent"] is not None
    ]
    steady_idle_ratios = [
        item["steady_idle_ratio_below_10_percent"]
        for item in per_gpu.values()
        if item["steady_idle_ratio_below_10_percent"] is not None
    ]
    return {
        "gpu_count": len(per_gpu),
        "per_gpu": per_gpu,
        "peak_memory_used_mib": max(item["peak_memory_used_mib"] for item in per_gpu.values()),
        "full_run_mean_gpu_util_percent": statistics.fmean(
            item["full_run_mean_gpu_util_percent"] for item in per_gpu.values()
        ),
        "steady_mean_gpu_util_percent": (statistics.fmean(steady_gpu_utilization) if steady_gpu_utilization else None),
        "steady_idle_ratio_below_10_percent": (statistics.fmean(steady_idle_ratios) if steady_idle_ratios else None),
        "steady_wall_time_intervals": [list(interval) for interval in steady_intervals],
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def analyze_run(
    run_dir: Path,
    *,
    windows: Sequence[tuple[int, int]] = DEFAULT_WINDOWS,
    expected_samples: int = 256,
    expected_actor_chunks: int = 2,
    expected_producer_chunks: int | None = None,
    write_outputs: bool = True,
    require_reproducibility_artifacts: bool = False,
) -> RunAnalysis:
    run_dir = run_dir.resolve()
    manifest = _load_manifest(run_dir)
    _validate_run_manifest(manifest, run_dir=run_dir, windows=windows)
    if require_reproducibility_artifacts:
        _require_reproducibility_artifacts(run_dir)
    trace_rows = _load_trace_rows(run_dir)
    pipeline_flag = manifest["hybrid_pipeline_forward"]
    if type(pipeline_flag) is not int or pipeline_flag not in (0, 1):
        _fail(f"{run_dir} hybrid_pipeline_forward must be integer 0 or 1, got {pipeline_flag!r}")
    pipeline_enabled = bool(pipeline_flag)
    overlap_flag = manifest.get("hybrid_pipeline_overlap", pipeline_flag)
    if type(overlap_flag) is not int or overlap_flag not in (0, 1):
        _fail(f"{run_dir} hybrid_pipeline_overlap must be integer 0 or 1, got {overlap_flag!r}")
    pipeline_overlap_enabled = bool(overlap_flag)
    if manifest["global_batch_size"] != expected_samples:
        _fail(
            f"{run_dir} expected_samples={expected_samples} disagrees with "
            f"manifest global_batch_size={manifest['global_batch_size']}"
        )
    if manifest["num_iters_per_train_update"] != expected_actor_chunks:
        _fail(
            f"{run_dir} expected_actor_chunks={expected_actor_chunks} disagrees with "
            f"manifest num_iters_per_train_update={manifest['num_iters_per_train_update']}"
        )
    condition = str(manifest["condition"])
    if condition not in {"baseline", "experiment"}:
        _fail(f"{run_dir} has unsupported condition {condition!r}; expected 'baseline' or 'experiment'")
    if condition == "baseline" and pipeline_enabled and pipeline_overlap_enabled:
        _fail(f"{run_dir} is labeled baseline but both chunk forwarding and producer overlap are enabled")
    if condition == "experiment" and (not pipeline_enabled or not pipeline_overlap_enabled):
        _fail(f"{run_dir} is labeled experiment but chunk forwarding with producer overlap is not enabled")
    rollout_rows, actor_rank_rows, trace_summary = _analyze_trace(
        trace_rows,
        pipeline_enabled=pipeline_enabled,
        pipeline_overlap_enabled=pipeline_overlap_enabled,
        expected_samples=expected_samples,
        expected_actor_chunks=expected_actor_chunks,
        expected_producer_chunks=expected_producer_chunks,
    )
    scalar_rows = _load_tensorboard_scalars(run_dir)
    nvml_rows = _parse_nvml_rows(run_dir)
    steady_wall_time_intervals = _steady_wall_time_intervals(scalar_rows, windows)
    metrics = _metric_summary(scalar_rows, windows)
    samples_per_second = _aggregate_samples_per_second(scalar_rows, windows, expected_samples)
    if samples_per_second is not None:
        metrics["perf/wall_clock_samples_per_s"] = {"aggregate": samples_per_second}
    summary = {
        "run_dir": str(run_dir),
        "condition": manifest["condition"],
        "seed": manifest["seed"],
        "hybrid_pipeline_forward": pipeline_enabled,
        "hybrid_pipeline_overlap": pipeline_overlap_enabled,
        "steady_windows": [list(window) for window in windows],
        "trace": trace_summary,
        "metrics": metrics,
        "nvml": _nvml_summary(nvml_rows, steady_intervals=steady_wall_time_intervals),
        "validation": "passed",
    }

    if write_outputs:
        output_dir = run_dir / "analysis"
        _write_csv(output_dir / "trace_events.csv", trace_rows)
        _write_csv(output_dir / "rollout_summary.csv", rollout_rows)
        _write_csv(output_dir / "actor_rank_summary.csv", actor_rank_rows)
        _write_csv(output_dir / "tensorboard_scalars.csv", scalar_rows)
        _write_csv(output_dir / "nvml_samples.csv", nvml_rows)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return RunAnalysis(
        run_dir=run_dir,
        manifest=manifest,
        trace_rows=rollout_rows,
        actor_rank_rows=actor_rank_rows,
        scalar_rows=scalar_rows,
        nvml_rows=nvml_rows,
        summary=summary,
    )


def _geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(value <= 0 for value in values):
        _fail(f"geometric mean requires positive values, got {values}")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _distribution_summary(values: Iterable[float]) -> dict[str, float | int]:
    samples = list(values)
    if not samples:
        _fail("distribution summary requires at least one value")
    mean = statistics.fmean(samples)
    stddev = statistics.pstdev(samples)
    return {
        "count": len(samples),
        "mean": mean,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "population_stddev": stddev,
        "coefficient_of_variation": stddev / abs(mean) if mean else 0.0,
    }


def _build_comparison(
    analyses: Sequence[RunAnalysis],
    *,
    windows: Sequence[tuple[int, int]],
    enforce_targets: bool,
    expected_gpu_count: int = 8,
) -> dict[str, Any]:
    if expected_gpu_count <= 0:
        _fail(f"expected_gpu_count must be positive, got {expected_gpu_count}")

    by_condition: dict[str, list[RunAnalysis]] = defaultdict(list)
    for analysis in analyses:
        by_condition[str(analysis.manifest["condition"])].append(analysis)
    if set(by_condition) != {"baseline", "experiment"}:
        _fail(f"comparison requires conditions named 'baseline' and 'experiment', got {sorted(by_condition)}")

    comparison_fields = COMPARISON_FIXED_MANIFEST_FIELDS + COMPARISON_WORKLOAD_MANIFEST_FIELDS
    missing_workload_fields = {
        str(analysis.run_dir): sorted(set(COMPARISON_WORKLOAD_MANIFEST_FIELDS) - set(analysis.manifest))
        for analysis in analyses
        if set(COMPARISON_WORKLOAD_MANIFEST_FIELDS) - set(analysis.manifest)
    }
    if missing_workload_fields:
        _fail(f"comparison manifests are missing workload fields: {missing_workload_fields}")

    for field in comparison_fields:
        values = {json.dumps(analysis.manifest[field], sort_keys=True) for analysis in analyses}
        if len(values) != 1:
            _fail(f"comparison requires identical manifest field {field!r}, got {sorted(values)}")

    by_seed: dict[Any, dict[str, RunAnalysis]] = defaultdict(dict)
    for condition, condition_runs in by_condition.items():
        for analysis in condition_runs:
            seed = analysis.manifest["seed"]
            if condition in by_seed[seed]:
                _fail(f"duplicate {condition} run for seed {seed}")
            by_seed[seed][condition] = analysis
    incomplete = {seed: sorted(pair) for seed, pair in by_seed.items() if set(pair) != {"baseline", "experiment"}}
    if incomplete:
        _fail(f"comparison has unpaired seeds: {incomplete}")

    steady_steps = _stable_steps(windows)
    if enforce_targets:
        missing_overlap_mode = [
            str(analysis.run_dir) for analysis in analyses if "hybrid_pipeline_overlap" not in analysis.manifest
        ]
        if missing_overlap_mode:
            _fail(
                f"performance targets require an explicit hybrid_pipeline_overlap manifest field: {missing_overlap_mode}"
            )
        invalid_modes = {
            str(analysis.run_dir): {
                "condition": analysis.manifest["condition"],
                "hybrid_pipeline_forward": analysis.manifest["hybrid_pipeline_forward"],
                "hybrid_pipeline_overlap": analysis.manifest["hybrid_pipeline_overlap"],
            }
            for analysis in analyses
            if analysis.manifest["hybrid_pipeline_forward"] not in (1, True)
            or bool(analysis.manifest["hybrid_pipeline_overlap"]) != (analysis.manifest["condition"] == "experiment")
        }
        if invalid_modes:
            _fail(
                "performance targets require schedule-matched chunk forwarding with overlap disabled "
                f"for baseline and enabled for experiment: {invalid_modes}"
            )
        required_tags = PERFORMANCE_TAGS + CORRECTNESS_GUARDRAIL_TAGS
        for analysis in analyses:
            context = f"{analysis.manifest['condition']} seed={analysis.manifest['seed']}"
            trace_steps = {int(row["rollout_id"]) for row in analysis.trace_rows}
            missing_trace_steps = sorted(steady_steps - trace_steps)
            if missing_trace_steps:
                _fail(f"{context} is missing steady trace steps {missing_trace_steps}")
            for tag in required_tags:
                metric_steps = {
                    int(row["step"])
                    for row in analysis.scalar_rows
                    if row["tag"] == tag and int(row["step"]) in steady_steps
                }
                missing_metric_steps = sorted(steady_steps - metric_steps)
                if missing_metric_steps:
                    _fail(f"{context} {tag} is missing steady steps {missing_metric_steps}")
            gpu_count = analysis.summary["nvml"].get("gpu_count")
            if gpu_count != expected_gpu_count:
                _fail(f"{context} expected NVML data for {expected_gpu_count} GPUs, got {gpu_count!r}")
            per_gpu_nvml = analysis.summary["nvml"].get("per_gpu", {})
            if len(per_gpu_nvml) != expected_gpu_count:
                _fail(
                    f"{context} expected per-GPU NVML summaries for {expected_gpu_count} GPUs, got {len(per_gpu_nvml)}"
                )
            missing_steady_nvml = sorted(
                gpu_index
                for gpu_index, gpu_summary in per_gpu_nvml.items()
                if int(gpu_summary.get("steady_sample_count", 0)) <= 0
            )
            if missing_steady_nvml:
                _fail(
                    f"{context} has no NVML samples aligned to registered steady steps for GPUs {missing_steady_nvml}"
                )
            steady_trace_rows = [row for row in analysis.trace_rows if int(row["rollout_id"]) in steady_steps]
            observed_lead = max(row["producer_lead_at_first_forward"] for row in steady_trace_rows)
            max_staleness = int(analysis.manifest["max_staleness"])
            if observed_lead > max_staleness:
                _fail(
                    f"{context} observed producer lead exceeds configured "
                    f"max_staleness={max_staleness}: {observed_lead}"
                )

    paired = []
    window_speedups = []
    for seed, pair in sorted(by_seed.items(), key=lambda item: str(item[0])):
        row: dict[str, Any] = {"seed": seed}
        baseline_fingerprints = {
            int(trace_row["rollout_id"]): trace_row["producer_global_indexes_fingerprint"]
            for trace_row in pair["baseline"].trace_rows
            if int(trace_row["rollout_id"]) in steady_steps
        }
        experiment_fingerprints = {
            int(trace_row["rollout_id"]): trace_row["producer_global_indexes_fingerprint"]
            for trace_row in pair["experiment"].trace_rows
            if int(trace_row["rollout_id"]) in steady_steps
        }
        if baseline_fingerprints != experiment_fingerprints:
            _fail(
                f"seed {seed} producer global-index fingerprints differ between "
                f"baseline and experiment: baseline={baseline_fingerprints}, "
                f"experiment={experiment_fingerprints}"
            )
        for tag in COMPARISON_PERFORMANCE_TAGS:
            baseline_metrics = pair["baseline"].summary["metrics"].get(tag, {})
            experiment_metrics = pair["experiment"].summary["metrics"].get(tag, {})
            key = "aggregate" if "per_s" in tag else "mean"
            baseline_value = baseline_metrics.get(key)
            experiment_value = experiment_metrics.get(key)
            if baseline_value is None or experiment_value is None:
                continue
            if baseline_value <= 0 or experiment_value <= 0:
                _fail(f"{tag} must be positive for seed {seed}, got {baseline_value=} and {experiment_value=}")
            row[f"{tag}:baseline"] = baseline_value
            row[f"{tag}:experiment"] = experiment_value
            if "time" in tag:
                row[f"{tag}:improvement"] = 1 - experiment_value / baseline_value
            else:
                row[f"{tag}:improvement"] = experiment_value / baseline_value - 1

        for window_start, window_end in windows:
            window = ((window_start, window_end),)
            baseline_throughput = _aggregate_throughput(
                pair["baseline"].scalar_rows,
                "perf/step_token_per_s",
                window,
            )
            experiment_throughput = _aggregate_throughput(
                pair["experiment"].scalar_rows,
                "perf/step_token_per_s",
                window,
            )
            if baseline_throughput is not None and experiment_throughput is not None:
                if baseline_throughput <= 0 or experiment_throughput <= 0:
                    _fail(f"window {window_start}-{window_end} throughput must be positive for seed {seed}")
                window_speedups.append(
                    {
                        "seed": seed,
                        "window_start": window_start,
                        "window_end": window_end,
                        "baseline": baseline_throughput,
                        "experiment": experiment_throughput,
                        "speedup": experiment_throughput / baseline_throughput - 1,
                    }
                )

        baseline_step_p95 = pair["baseline"].summary["metrics"].get("perf/step_time", {}).get("p95")
        experiment_step_p95 = pair["experiment"].summary["metrics"].get("perf/step_time", {}).get("p95")
        if baseline_step_p95 is not None and experiment_step_p95 is not None:
            if baseline_step_p95 <= 0 or experiment_step_p95 <= 0:
                _fail(
                    f"perf/step_time p95 must be positive for seed {seed}, "
                    f"got {baseline_step_p95=} and {experiment_step_p95=}"
                )
            row["perf/step_time:p95_baseline"] = baseline_step_p95
            row["perf/step_time:p95_experiment"] = experiment_step_p95
            row["perf/step_time:p95_regression"] = experiment_step_p95 / baseline_step_p95 - 1

        for tag in CORRECTNESS_GUARDRAIL_TAGS:
            baseline_value = pair["baseline"].summary["metrics"].get(tag, {}).get("mean")
            experiment_value = pair["experiment"].summary["metrics"].get(tag, {}).get("mean")
            if baseline_value is not None and experiment_value is not None:
                row[f"{tag}:baseline"] = baseline_value
                row[f"{tag}:experiment"] = experiment_value
                row[f"{tag}:delta"] = experiment_value - baseline_value

        for field in (
            "actor_fetch_samples",
            "actor_total_tokens",
            "actor_response_tokens",
            "actor_multimodal_tensor_bytes",
        ):
            baseline_total = sum(
                int(trace_row[field])
                for trace_row in pair["baseline"].trace_rows
                if int(trace_row["rollout_id"]) in steady_steps
            )
            experiment_total = sum(
                int(trace_row[field])
                for trace_row in pair["experiment"].trace_rows
                if int(trace_row["rollout_id"]) in steady_steps
            )
            row[f"{field}:baseline"] = baseline_total
            row[f"{field}:experiment"] = experiment_total
            row[f"{field}:relative_delta"] = (
                experiment_total / baseline_total - 1
                if baseline_total
                else (0.0 if experiment_total == 0 else math.inf)
            )

        baseline_vram = pair["baseline"].summary["nvml"].get("peak_memory_used_mib")
        experiment_vram = pair["experiment"].summary["nvml"].get("peak_memory_used_mib")
        if baseline_vram is not None and experiment_vram is not None:
            row["nvml_peak_memory_mib:baseline"] = baseline_vram
            row["nvml_peak_memory_mib:experiment"] = experiment_vram
            row["nvml_peak_memory_mib:delta"] = experiment_vram - baseline_vram

        baseline_lead = statistics.fmean(
            trace_row["producer_lead_at_first_forward"]
            for trace_row in pair["baseline"].trace_rows
            if int(trace_row["rollout_id"]) in steady_steps
        )
        experiment_lead = statistics.fmean(
            trace_row["producer_lead_at_first_forward"]
            for trace_row in pair["experiment"].trace_rows
            if int(trace_row["rollout_id"]) in steady_steps
        )
        row["producer_lead_at_first_forward:baseline"] = baseline_lead
        row["producer_lead_at_first_forward:experiment"] = experiment_lead
        row["producer_lead_at_first_forward:delta"] = experiment_lead - baseline_lead
        paired.append(row)

    token_ratios = [
        1 + row["perf/step_token_per_s:improvement"] for row in paired if "perf/step_token_per_s:improvement" in row
    ]
    phase_ratios = [
        1 - row["perf/hybrid_phase1_time:improvement"]
        for row in paired
        if "perf/hybrid_phase1_time:improvement" in row
    ]
    experiment_overlap_rows = [
        row
        for analysis in by_condition["experiment"]
        for row in analysis.trace_rows
        if int(row["rollout_id"]) in _stable_steps(windows)
    ]
    experiment_producer_overlap_by_run = {}
    for analysis in by_condition["experiment"]:
        rows = [row for row in analysis.trace_rows if int(row["rollout_id"]) in steady_steps]
        experiment_producer_overlap_by_run[str(analysis.run_dir)] = (
            sum(bool(row["first_forward_before_last_put_start"]) for row in rows) / len(rows) if rows else None
        )
    comparison = {
        "paired_runs": paired,
        "window_speedups": window_speedups,
        "paired_seed_count": len(paired),
        "token_throughput_geomean_speedup": (_geometric_mean(token_ratios) - 1 if token_ratios else None),
        "hybrid_phase1_geomean_reduction": (1 - _geometric_mean(phase_ratios) if phase_ratios else None),
        "experiment_steady_producer_overlap_ratio": (
            sum(bool(row["first_forward_before_last_put_start"]) for row in experiment_overlap_rows)
            / len(experiment_overlap_rows)
            if experiment_overlap_rows
            else None
        ),
        "experiment_steady_producer_overlap_ratio_by_run": experiment_producer_overlap_by_run,
        "distributions": {
            "baseline_step_token_per_s": _distribution_summary(
                row["perf/step_token_per_s:baseline"] for row in paired if "perf/step_token_per_s:baseline" in row
            ),
            "experiment_step_token_per_s": _distribution_summary(
                row["perf/step_token_per_s:experiment"] for row in paired if "perf/step_token_per_s:experiment" in row
            ),
            "paired_step_token_per_s_speedup": _distribution_summary(
                row["perf/step_token_per_s:improvement"]
                for row in paired
                if "perf/step_token_per_s:improvement" in row
            ),
            "paired_hybrid_phase1_reduction": _distribution_summary(
                row["perf/hybrid_phase1_time:improvement"]
                for row in paired
                if "perf/hybrid_phase1_time:improvement" in row
            ),
        },
    }

    if enforce_targets:
        if len(paired) < 2:
            _fail(f"performance targets require at least two paired seeds, got {len(paired)}")
        improvements = [row.get("perf/step_token_per_s:improvement") for row in paired]
        if any(value is None for value in improvements):
            _fail("performance targets require perf/step_token_per_s for every paired run")
        if any("perf/wall_clock_samples_per_s:baseline" not in row for row in paired):
            _fail("performance targets require wall-clock samples/s for every paired run")
        if any(value <= 0 for value in improvements):
            _fail(f"every paired token-throughput speedup must be positive, got {improvements}")
        if comparison["token_throughput_geomean_speedup"] < 0.05:
            _fail(
                "token-throughput geometric-mean speedup is below 5%: "
                f"{comparison['token_throughput_geomean_speedup']:.4%}"
            )
        phase_improvements = [row.get("perf/hybrid_phase1_time:improvement") for row in paired]
        if any(value is None for value in phase_improvements):
            _fail("performance targets require perf/hybrid_phase1_time for every paired run")
        if any(value < 0 for value in phase_improvements):
            _fail(f"every paired Hybrid phase-1 result must be non-regressive, got {phase_improvements}")
        if comparison["hybrid_phase1_geomean_reduction"] < 0.15:
            _fail(
                "Hybrid phase-1 geometric-mean reduction is below 15%: "
                f"{comparison['hybrid_phase1_geomean_reduction']:.4%}"
            )
        failing_overlap = {
            run_dir: ratio
            for run_dir, ratio in experiment_producer_overlap_by_run.items()
            if ratio is None or ratio < 0.8
        }
        if failing_overlap:
            _fail(f"each experiment run requires at least 80% steady producer overlap, got {failing_overlap}")

        step_p95_regressions = [row.get("perf/step_time:p95_regression") for row in paired]
        if any(value is None for value in step_p95_regressions):
            _fail("performance targets require perf/step_time p95 for every paired run")
        if any(value > 0.05 for value in step_p95_regressions):
            _fail(f"paired step-time p95 regression exceeds 5%: {step_p95_regressions}")

        for row in paired:
            seed = row["seed"]
            if row["actor_fetch_samples:baseline"] != row["actor_fetch_samples:experiment"]:
                _fail(f"seed {seed} actor fetch sample count changed")
            deterministic = int(by_seed[seed]["baseline"].manifest["sglang_deterministic_inference"]) == 1
            for field in (
                "actor_total_tokens",
                "actor_response_tokens",
                "actor_multimodal_tensor_bytes",
            ):
                baseline_total = row[f"{field}:baseline"]
                experiment_total = row[f"{field}:experiment"]
                if baseline_total <= 0 or experiment_total <= 0:
                    _fail(f"seed {seed} {field} must be positive, got {baseline_total}, {experiment_total}")
                relative_delta = row[f"{field}:relative_delta"]
                if deterministic and baseline_total != experiment_total:
                    _fail(
                        f"seed {seed} {field} must match exactly under deterministic "
                        f"inference, got {baseline_total} and {experiment_total}"
                    )
                if not deterministic and abs(relative_delta) > 0.01:
                    _fail(f"seed {seed} {field} changed by more than 1%: {relative_delta:.4%}")

            vram_baseline = row.get("nvml_peak_memory_mib:baseline")
            vram_delta = row.get("nvml_peak_memory_mib:delta")
            if vram_baseline is None or vram_delta is None:
                _fail(f"seed {seed} is missing paired NVML peak memory")
            allowed_vram_delta = max(1024.0, 0.03 * vram_baseline)
            if vram_delta > allowed_vram_delta:
                _fail(f"seed {seed} peak VRAM increased by {vram_delta:.1f} MiB; allowed={allowed_vram_delta:.1f} MiB")

        accuracy_deltas = [row.get("rollout/raw_reward:delta") for row in paired]
        if any(value is None for value in accuracy_deltas):
            _fail("correctness targets require rollout/raw_reward for every paired run")
        if any(value < -0.03 for value in accuracy_deltas):
            _fail(f"a paired raw-reward/accuracy drop exceeds 3 percentage points: {accuracy_deltas}")
        if statistics.fmean(accuracy_deltas) < -0.02:
            _fail(f"mean paired raw-reward/accuracy drop exceeds 2 percentage points: {accuracy_deltas}")

        truncation_deltas = [row.get("rollout/truncated_ratio:delta") for row in paired]
        if any(value is None for value in truncation_deltas):
            _fail("correctness targets require rollout/truncated_ratio for every paired run")
        if any(value > 0.02 for value in truncation_deltas):
            _fail(f"a paired truncation-rate increase exceeds 2 percentage points: {truncation_deltas}")

        for analysis in analyses:
            ppo_kl = analysis.summary["metrics"]["train/ppo_kl"]["mean"]
            pg_clipfrac = analysis.summary["metrics"]["train/pg_clipfrac"]["mean"]
            if abs(ppo_kl) > 1e-7:
                _fail(f"{analysis.run_dir} same-weight train/ppo_kl exceeds 1e-7: {ppo_kl}")
            if abs(pg_clipfrac) > 1e-7:
                _fail(f"{analysis.run_dir} same-weight train/pg_clipfrac exceeds 1e-7: {pg_clipfrac}")

        staleness_deltas = [row["producer_lead_at_first_forward:delta"] for row in paired]
        if any(value > 0.25 for value in staleness_deltas):
            _fail(f"a paired average producer-lead increase exceeds 0.25: {staleness_deltas}")

        baseline_throughputs = [row["perf/step_token_per_s:baseline"] for row in paired]
        if len(paired) == 2:
            baseline_cv = statistics.pstdev(baseline_throughputs) / statistics.fmean(baseline_throughputs)
            if baseline_cv > 0.05:
                _fail(
                    f"two baseline runs have CV={baseline_cv:.2%} > 5%; "
                    "the preregistered protocol requires a third paired seed"
                )
    return comparison


def _plot_comparison(
    analyses: Sequence[RunAnalysis],
    comparison: dict[str, Any],
    output_dir: Path,
    windows: Sequence[tuple[int, int]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise BenchmarkValidationError(
            "comparison plot generation requires the optional dependency "
            "'matplotlib'; install it in the benchmark environment"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    def finish(filename: str) -> None:
        path = output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        generated.append(str(path))

    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for axis, tag in zip(
        axes,
        ("perf/step_token_per_s", "perf/step_resp_token_per_s"),
        strict=True,
    ):
        for analysis in analyses:
            label = f"{analysis.manifest['condition']}-seed{analysis.manifest['seed']}"
            values = [row for row in analysis.scalar_rows if row["tag"] == tag]
            if values:
                axis.plot(
                    [row["step"] for row in values],
                    [row["value"] for row in values],
                    marker="o",
                    label=label,
                )
        for start, end in windows:
            axis.axvspan(start, end, color="grey", alpha=0.08)
        axis.set_ylabel(tag)
        axis.legend()
    axes[-1].set_xlabel("rollout / optimizer step")
    figure.suptitle("Task 21 step token throughput")
    finish("task21_step_throughput.png")

    figure, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for analysis in analyses:
        label = f"{analysis.manifest['condition']}-seed{analysis.manifest['seed']}"
        axes[0].plot(
            [row["rollout_id"] for row in analysis.trace_rows],
            [row["producer_overlap_s"] for row in analysis.trace_rows],
            marker="o",
            label=label,
        )
        phase1 = [row for row in analysis.scalar_rows if row["tag"] == "perf/hybrid_phase1_time"]
        if phase1:
            axes[1].plot(
                [row["step"] for row in phase1],
                [row["value"] for row in phase1],
                marker="o",
                label=label,
            )
        axes[2].plot(
            [row["rollout_id"] for row in analysis.trace_rows],
            [row["producer_lead_at_first_forward"] for row in analysis.trace_rows],
            marker="o",
            label=label,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("last put start - first actor forward start (s)")
    axes[1].set_ylabel("perf/hybrid_phase1_time (s)")
    max_staleness = int(analyses[0].manifest["max_staleness"])
    axes[2].axhline(
        max_staleness,
        color="red",
        linestyle="--",
        label=f"max_staleness={max_staleness}",
    )
    axes[2].set_xlabel("rollout / optimizer step")
    axes[2].set_ylabel("producer lead at actor forward (steps)")
    for axis in axes:
        axis.legend()
    figure.suptitle("Task 21 producer / actor overlap and phase-1 time")
    finish("task21_phase1_overlap.png")

    def numeric_or_nan(value: Any) -> float:
        return float(value) if value is not None else float("nan")

    labels = [f"{analysis.manifest['condition']}-s{analysis.manifest['seed']}" for analysis in analyses]
    utilization = [
        numeric_or_nan(analysis.summary["nvml"].get("steady_mean_gpu_util_percent")) for analysis in analyses
    ]
    idle_ratio = [
        numeric_or_nan(analysis.summary["nvml"].get("steady_idle_ratio_below_10_percent")) * 100
        for analysis in analyses
    ]
    vram = [numeric_or_nan(analysis.summary["nvml"].get("peak_memory_used_mib")) / 1024 for analysis in analyses]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for analysis in analyses:
        label = f"{analysis.manifest['condition']}-seed{analysis.manifest['seed']}"
        by_timestamp: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in analysis.nvml_rows:
            if row["wall_time"] is not None:
                by_timestamp[float(row["wall_time"])].append(row)
        if by_timestamp:
            start_time = min(by_timestamp)
            timestamps = sorted(by_timestamp)
            elapsed = [timestamp - start_time for timestamp in timestamps]
            mean_utilization = [
                statistics.fmean(float(row["gpu_util_percent"]) for row in by_timestamp[timestamp])
                for timestamp in timestamps
            ]
            peak_vram = [
                max(float(row["memory_used_mib"]) for row in by_timestamp[timestamp]) / 1024
                for timestamp in timestamps
            ]
            axes[0, 0].plot(elapsed, mean_utilization, label=label)
            axes[1, 0].plot(elapsed, peak_vram, label=label)
            for start, end in analysis.summary["nvml"].get("steady_wall_time_intervals", []):
                axes[0, 0].axvspan(start - start_time, end - start_time, color="grey", alpha=0.025)
                axes[1, 0].axvspan(start - start_time, end - start_time, color="grey", alpha=0.025)
    axes[0, 0].set_ylabel("mean GPU utilization (%)")
    axes[0, 0].set_xlabel("seconds since first NVML sample")
    axes[1, 0].set_ylabel("max per-GPU VRAM (GiB)")
    axes[1, 0].set_xlabel("seconds since first NVML sample")
    for axis in (axes[0, 0], axes[1, 0]):
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    positions = list(range(len(labels)))
    axes[0, 1].bar([position - 0.2 for position in positions], utilization, width=0.4, label="mean util")
    axes[0, 1].bar([position + 0.2 for position in positions], idle_ratio, width=0.4, label="idle <10%")
    axes[0, 1].set_xticks(positions, labels)
    axes[0, 1].set_ylabel("steady-window percent")
    axes[0, 1].legend()
    axes[1, 1].bar(labels, vram)
    axes[1, 1].set_ylabel("full-run sampled peak VRAM (GiB)")
    for axis in (axes[0, 1], axes[1, 1]):
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle("Task 21 GPU utilization and VRAM")
    finish("task21_gpu_util_vram.png")

    quality_tags = sorted(
        {
            row["tag"]
            for analysis in analyses
            for row in analysis.scalar_rows
            if any(fragment in row["tag"].lower() for fragment in QUALITY_TAG_FRAGMENTS)
        }
    )
    plt.figure(figsize=(12, 7))
    for analysis in analyses:
        for tag in quality_tags:
            values = [row for row in analysis.scalar_rows if row["tag"] == tag]
            if values:
                plt.plot(
                    [row["step"] for row in values],
                    [row["value"] for row in values],
                    label=f"{analysis.manifest['condition']}-s{analysis.manifest['seed']}:{tag}",
                )
    plt.xlabel("rollout / optimizer step")
    plt.ylabel("raw metric value")
    if quality_tags:
        plt.legend(fontsize=7, ncol=2)
    plt.title("Task 21 correctness and quality guardrails")
    finish("task21_correctness_quality.png")

    paired = comparison["paired_runs"]
    figure, axes = plt.subplots(2, 1, figsize=(12, 9))
    window_speedups = comparison["window_speedups"]
    window_labels = [f"s{row['seed']}:{row['window_start']}-{row['window_end']}" for row in window_speedups]
    axes[0].scatter(
        window_labels,
        [100 * row["speedup"] for row in window_speedups],
        marker="o",
    )
    axes[0].axhline(5, color="red", linestyle="--", label="5% run-level target")
    axes[0].set_ylabel("window token-throughput speedup (%)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()

    pair_labels = [f"seed {row['seed']}" for row in paired]
    speedups = [100 * row.get("perf/step_token_per_s:improvement", float("nan")) for row in paired]
    pair_labels.append("geometric mean")
    speedups.append(100 * (comparison["token_throughput_geomean_speedup"] or 0))
    axes[1].bar(pair_labels, speedups)
    axes[1].axhline(5, color="red", linestyle="--", label="5% target")
    axes[1].set_ylabel("paired run token-throughput speedup (%)")
    axes[1].legend()
    figure.suptitle("Task 21 preregistered windows and paired run summary")
    finish("task21_window_summary.png")
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help="Benchmark run directory. Repeat for paired baseline/experiment comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Comparison output directory; required only when more than one --run-dir is supplied.",
    )
    parser.add_argument(
        "--steady-windows",
        default="4-8,9-13,14-18",
        help="Inclusive, non-overlapping step windows as START-END comma-separated ranges.",
    )
    parser.add_argument("--expected-samples", type=int, default=256)
    parser.add_argument(
        "--expected-actor-chunks",
        type=int,
        default=2,
        help="Expected actor fetch/forward chunks when the pipeline is enabled.",
    )
    parser.add_argument(
        "--expected-producer-chunks",
        type=int,
        default=None,
        help="Require an exact producer put count and equal sample count per put.",
    )
    parser.add_argument(
        "--expected-gpu-count",
        type=int,
        default=8,
        help="Required distinct GPU indexes in NVML telemetry when enforcing targets.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate each run without requiring paired performance target checks.",
    )
    parser.add_argument(
        "--enforce-targets",
        action="store_true",
        help="Require the preregistered 5%% throughput, 15%% phase-1, and 80%% overlap targets.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate comparison PNG files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        windows = _parse_windows(args.steady_windows)
        if (
            args.expected_samples <= 0
            or args.expected_actor_chunks <= 0
            or args.expected_gpu_count <= 0
            or (args.expected_producer_chunks is not None and args.expected_producer_chunks <= 0)
        ):
            _fail("expected sample, actor chunk, producer chunk, and GPU counts must be positive")
        if len(args.run_dir) > 1 and args.output_dir is None:
            _fail("--output-dir is required when comparing multiple runs")
        if args.validate_only and args.enforce_targets:
            _fail("--validate-only and --enforce-targets are mutually exclusive")

        analyses = [
            analyze_run(
                run_dir,
                windows=windows,
                expected_samples=args.expected_samples,
                expected_actor_chunks=args.expected_actor_chunks,
                expected_producer_chunks=args.expected_producer_chunks,
                require_reproducibility_artifacts=args.enforce_targets,
            )
            for run_dir in args.run_dir
        ]
        result: dict[str, Any] = {
            "runs": [analysis.summary for analysis in analyses],
            "validation": "passed",
        }
        if len(analyses) > 1:
            comparison = _build_comparison(
                analyses,
                windows=windows,
                enforce_targets=args.enforce_targets,
                expected_gpu_count=args.expected_gpu_count,
            )
            result["comparison"] = comparison
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            if not args.no_plots:
                result["plots"] = _plot_comparison(analyses, comparison, output_dir, windows)
            (output_dir / "comparison_summary.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_csv(output_dir / "paired_run_summary.csv", comparison["paired_runs"])
            _write_csv(output_dir / "window_speedup_summary.csv", comparison["window_speedups"])
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except BenchmarkValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
