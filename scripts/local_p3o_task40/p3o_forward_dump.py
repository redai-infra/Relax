#!/usr/bin/env python3

"""Task40 BF16 THD forward-dump hook and manifest finalizer.

The module deliberately keeps PyTorch, Megatron, and Relax imports inside the
runtime entry points so that the command-line manifest finalizer and a plain
local import do not require the cluster environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


FORMAT_VERSION = 1
CAPTURE_PHASE = "log_prob"
P0_ORACLE_SOURCE_SHA256 = "d53fdded5f601e0611231520166850dd49f05aad754e36f8adcef0333d31bd8c"
EXPECTED_FIXTURE_SHA256 = "48538d165386dc94006613d857c022a7ba2e979bdc31bc617374eee2dc3c35b8"


def _json_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_sha256(tokens: Any) -> str:
    cpu_tokens = tokens.detach().to(device="cpu", dtype=_torch().int64).contiguous()
    return hashlib.sha256(cpu_tokens.numpy().tobytes()).hexdigest()


def _torch() -> Any:
    import torch

    return torch


def _first_tensor(value: Any) -> Any | None:
    torch = _torch()
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _argument_tensor(args: tuple[Any, ...], kwargs: dict[str, Any], names: tuple[str, ...], index: int) -> Any | None:
    torch = _torch()
    for name in names:
        candidate = kwargs.get(name)
        if torch.is_tensor(candidate):
            return candidate
    if index < len(args) and torch.is_tensor(args[index]):
        return args[index]
    return None


@dataclass
class BatchRecord:
    phase: str
    phase_micro_batch_index: int
    sample_keys: list[str]
    total_lengths: list[int]
    response_lengths: list[int]
    max_seq_lens: list[int] | None
    full_token_ids: list[Any]
    local_sample_indices: Any
    local_token_indices: Any
    local_chunk_indices: Any
    local_real_mask: Any
    derived_position_ids: Any
    cu_seqlens_q: Any | None
    cu_seqlens_kv: Any | None
    max_seqlen_q: int | None
    max_seqlen_kv: int | None
    qkv_format: str


@dataclass
class Capture:
    record: BatchRecord
    directory: Path
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class DumpState:
    args: Any | None = None
    dump_dir: Path | None = None
    rank: int = 0
    world_size: int = 1
    dp_rank: int = 0
    dp_world_size: int = 1
    cp_rank: int = 0
    cp_world_size: int = 1
    tp_rank: int = 0
    tp_world_size: int = 1
    pp_rank: int = 0
    pp_world_size: int = 1
    phase: str = "unclassified"
    phase_counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pending: deque[BatchRecord] = field(default_factory=deque)
    active: Capture | None = None
    hook_installed_model_ids: set[int] = field(default_factory=set)
    capture_directories: list[str] = field(default_factory=list)
    original_get_batch: Callable[..., Any] | None = None
    original_cp_probe: Callable[..., Any] | None = None
    required_stages: set[str] = field(default_factory=set)


_STATE = DumpState()


def _packed_value(packed: Any, name: str) -> Any | None:
    value = getattr(packed, name, None) if packed is not None else None
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return value


def _build_local_token_map(batch: dict[str, Any], cp_size: int, cp_rank: int) -> tuple[Any, Any, Any, Any, Any]:
    """Reproduce ``slice_with_cp`` ownership without changing the batch."""
    torch = _torch()
    full_tokens = batch["unconcat_tokens"]
    local_sample_indices: list[int] = []
    local_token_indices: list[int] = []
    local_chunk_indices: list[int] = []
    local_real_mask: list[bool] = []

    for sample_index, tokens in enumerate(full_tokens):
        total_length = int(tokens.shape[0])
        if cp_size == 1:
            positions = range(total_length)
            chunks = [0] * total_length
        else:
            chunk_size = math.ceil(total_length / (2 * cp_size))
            chunk_ids = (cp_rank, 2 * cp_size - cp_rank - 1)
            positions = [
                position
                for chunk_id in chunk_ids
                for position in range(chunk_id * chunk_size, (chunk_id + 1) * chunk_size)
            ]
            chunks = [chunk_id for chunk_id in chunk_ids for _ in range(chunk_size)]
        for position, chunk_id in zip(positions, chunks, strict=True):
            local_sample_indices.append(sample_index)
            local_token_indices.append(position)
            local_chunk_indices.append(chunk_id)
            local_real_mask.append(position < total_length)

    local_tokens = batch["tokens"]
    local_length = int(local_tokens.numel())
    trailing_padding = local_length - len(local_token_indices)
    if trailing_padding < 0:
        raise ValueError(
            f"forward dump token map exceeds local tensor length: mapped={len(local_token_indices)}, local={local_length}"
        )
    local_sample_indices.extend([-1] * trailing_padding)
    local_token_indices.extend([-1] * trailing_padding)
    local_chunk_indices.extend([-1] * trailing_padding)
    local_real_mask.extend([False] * trailing_padding)

    sample_tensor = torch.tensor(local_sample_indices, dtype=torch.int32)
    token_tensor = torch.tensor(local_token_indices, dtype=torch.int64)
    chunk_tensor = torch.tensor(local_chunk_indices, dtype=torch.int32)
    real_tensor = torch.tensor(local_real_mask, dtype=torch.bool)
    return sample_tensor, token_tensor, chunk_tensor, real_tensor, token_tensor.clone()


def _record_batch(batch: dict[str, Any]) -> BatchRecord:
    phase = _STATE.phase
    micro_batch_index = _STATE.phase_counters[phase]
    _STATE.phase_counters[phase] += 1
    full_tokens = [tokens.detach().cpu() for tokens in batch["unconcat_tokens"]]
    sample_keys = [_token_sha256(tokens) for tokens in full_tokens]
    sample_indices, token_indices, chunk_indices, real_mask, positions = _build_local_token_map(
        batch, _STATE.cp_world_size, _STATE.cp_rank
    )
    packed = batch.get("packed_seq_params")
    raw_max_seq_lens = batch.get("max_seq_lens")
    max_seq_lens = None if raw_max_seq_lens is None else [int(value) for value in raw_max_seq_lens]
    return BatchRecord(
        phase=phase,
        phase_micro_batch_index=micro_batch_index,
        sample_keys=sample_keys,
        total_lengths=[int(value) for value in batch["total_lengths"]],
        response_lengths=[int(value) for value in batch["response_lengths"]],
        max_seq_lens=max_seq_lens,
        full_token_ids=full_tokens,
        local_sample_indices=sample_indices,
        local_token_indices=token_indices,
        local_chunk_indices=chunk_indices,
        local_real_mask=real_mask,
        derived_position_ids=positions,
        cu_seqlens_q=_packed_value(packed, "cu_seqlens_q"),
        cu_seqlens_kv=_packed_value(packed, "cu_seqlens_kv"),
        max_seqlen_q=getattr(packed, "max_seqlen_q", None) if packed is not None else None,
        max_seqlen_kv=getattr(packed, "max_seqlen_kv", None) if packed is not None else None,
        qkv_format=getattr(packed, "qkv_format", getattr(_STATE.args, "qkv_format", "thd")),
    )


def _observed_get_batch(*args: Any, **kwargs: Any) -> Any:
    if _STATE.original_get_batch is None:
        raise RuntimeError("forward dump get_batch wrapper installed without its original callable")
    batch = _STATE.original_get_batch(*args, **kwargs)
    if _STATE.phase == CAPTURE_PHASE and not batch.get("__is_dummy__", False):
        _STATE.pending.append(_record_batch(batch))
    return batch


def _sanitize_stage(stage: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stage)


def _infer_token_axis(tensor: Any, local_length: int, preferred: int | None) -> int | None:
    if preferred is not None and tensor.dim() > preferred and int(tensor.shape[preferred]) == local_length:
        return preferred
    matches = [axis for axis, size in enumerate(tensor.shape) if int(size) == local_length]
    return matches[0] if len(matches) == 1 else None


def _store_stage(stage: str, value: Any, preferred_token_axis: int | None = 0) -> None:
    capture = _STATE.active
    if capture is None:
        return
    tensor = _first_tensor(value)
    if tensor is None:
        return
    torch = _torch()
    cpu_tensor = tensor.detach().cpu().contiguous()
    local_length = int(capture.record.local_token_indices.numel())
    token_axis = _infer_token_axis(cpu_tensor, local_length, preferred_token_axis)
    filename = _sanitize_stage(stage) + ".pt"
    torch.save(cpu_tensor, capture.directory / filename)
    finite = bool(torch.isfinite(cpu_tensor).all()) if cpu_tensor.is_floating_point() else True
    capture.stages[stage] = {
        "dtype": str(cpu_tensor.dtype),
        "finite": finite,
        "path": filename,
        "shape": list(cpu_tensor.shape),
        "token_axis": token_axis,
    }


def _write_rank_manifest() -> None:
    if _STATE.dump_dir is None:
        return
    _json_write(
        _STATE.dump_dir / f"manifest_rank{_STATE.rank}.json",
        {
            "format_version": FORMAT_VERSION,
            "rank": _STATE.rank,
            "world_size": _STATE.world_size,
            "captures": _STATE.capture_directories,
        },
    )


def _root_pre_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    del module, args
    if _STATE.phase != CAPTURE_PHASE:
        return
    if not _STATE.pending:
        raise RuntimeError("forward dump saw a log-prob model forward without matching get_batch metadata")
    if _STATE.active is not None:
        raise RuntimeError("forward dump does not support nested model forwards")
    record = _STATE.pending.popleft()
    if _STATE.dump_dir is None:
        raise RuntimeError("forward dump directory was not configured")
    directory = _STATE.dump_dir / f"rank{_STATE.rank:05d}" / f"micro{record.phase_micro_batch_index:05d}"
    directory.mkdir(parents=True, exist_ok=False)
    _STATE.active = Capture(record=record, directory=directory)

    position_ids = kwargs.get("position_ids")
    token_metadata = {
        "sample_keys": record.sample_keys,
        "full_token_ids": record.full_token_ids,
        "local_sample_indices": record.local_sample_indices,
        "local_token_indices": record.local_token_indices,
        "local_chunk_indices": record.local_chunk_indices,
        "local_real_mask": record.local_real_mask,
        "derived_position_ids": record.derived_position_ids,
        "position_ids_argument": None if position_ids is None else position_ids.detach().cpu(),
        "cu_seqlens_q": record.cu_seqlens_q,
        "cu_seqlens_kv": record.cu_seqlens_kv,
    }
    _torch().save(token_metadata, directory / "token_metadata.pt")


def _root_post_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
    del module, args, kwargs
    capture = _STATE.active
    if capture is None:
        return
    _store_stage("logits", output, preferred_token_axis=1)
    record = capture.record
    metadata = {
        "format_version": FORMAT_VERSION,
        "complete": True,
        "phase": record.phase,
        "phase_micro_batch_index": record.phase_micro_batch_index,
        "rank": _STATE.rank,
        "world_size": _STATE.world_size,
        "dp_rank": _STATE.dp_rank,
        "dp_world_size": _STATE.dp_world_size,
        "cp_rank": _STATE.cp_rank,
        "cp_world_size": _STATE.cp_world_size,
        "tp_rank": _STATE.tp_rank,
        "tp_world_size": _STATE.tp_world_size,
        "pp_rank": _STATE.pp_rank,
        "pp_world_size": _STATE.pp_world_size,
        "qkv_format": record.qkv_format,
        "micro_batch_size": int(getattr(_STATE.args, "micro_batch_size", 0)),
        "sample_keys": record.sample_keys,
        "total_lengths": record.total_lengths,
        "response_lengths": record.response_lengths,
        "max_seq_lens": record.max_seq_lens,
        "max_seqlen_q": None if record.max_seqlen_q is None else int(record.max_seqlen_q),
        "max_seqlen_kv": None if record.max_seqlen_kv is None else int(record.max_seqlen_kv),
        "position_ids_argument_was_none": True,
        "token_key_contract": "(sample_sha256_of_full_int64_token_ids, global_zero_based_token_index)",
        "token_metadata_path": "token_metadata.pt",
        "required_stages": sorted(_STATE.required_stages, key=_stage_sort_key),
        "stages": capture.stages,
    }
    token_metadata = _torch().load(capture.directory / "token_metadata.pt", map_location="cpu", weights_only=False)
    metadata["position_ids_argument_was_none"] = token_metadata["position_ids_argument"] is None
    _json_write(capture.directory / "metadata.json", metadata)
    relative = str(capture.directory.relative_to(_STATE.dump_dir))
    _STATE.capture_directories.append(relative)
    _STATE.active = None
    _write_rank_manifest()


def _register_pre_post(module: Any, stage_prefix: str, preferred_axis: int | None = 0) -> None:
    def pre_hook(hook_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del hook_module
        tensor = _argument_tensor(args, kwargs, ("hidden_states",), 0)
        _store_stage(f"{stage_prefix}.input", tensor, preferred_axis)

    def post_hook(hook_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        del hook_module, args, kwargs
        _store_stage(f"{stage_prefix}.output", output, preferred_axis)

    module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    module.register_forward_hook(post_hook, with_kwargs=True)


def _register_core_attention(module: Any, layer_prefix: str) -> None:
    def pre_hook(hook_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del hook_module
        names = (
            (("query", "q", "query_layer"), 0, "query"),
            (("key", "k", "key_layer"), 1, "key"),
            (("value", "v", "value_layer"), 2, "value"),
        )
        for aliases, index, label in names:
            _store_stage(
                f"{layer_prefix}.attention_{label}",
                _argument_tensor(args, kwargs, aliases, index),
                preferred_token_axis=0,
            )

    def post_hook(hook_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        del hook_module, args, kwargs
        _store_stage(f"{layer_prefix}.attention_output", output, preferred_token_axis=0)

    module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    module.register_forward_hook(post_hook, with_kwargs=True)


def _layer_number(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)$", name)
    return int(match.group(1)) if match else None


def _stage_sort_key(stage: str) -> tuple[int, str]:
    match = re.match(r"layer_(\d+)\.(.+)", stage)
    return (int(match.group(1)), match.group(2)) if match else (10**9, stage)


def _install_forward_dump(model: Any) -> None:
    model_id = id(model)
    if model_id in _STATE.hook_installed_model_ids:
        return
    _STATE.hook_installed_model_ids.add(model_id)

    layer_names: dict[str, int] = {}
    modules = list(model.named_modules())
    for name, module in modules:
        layer = _layer_number(name)
        if layer is not None:
            layer_names[name] = layer
            _register_pre_post(module, f"layer_{layer:03d}.block", preferred_axis=0)
            prefix = f"layer_{layer:03d}"
            _STATE.required_stages.update(
                {
                    f"{prefix}.block.input",
                    f"{prefix}.block.output",
                    f"{prefix}.self_attention.input",
                    f"{prefix}.self_attention.output",
                    f"{prefix}.qkv_projection.input",
                    f"{prefix}.qkv_projection.output",
                    f"{prefix}.attention_query",
                    f"{prefix}.attention_key",
                    f"{prefix}.attention_value",
                    f"{prefix}.attention_output",
                }
            )

    for name, module in modules:
        owner = next((layer for prefix, layer in layer_names.items() if name.startswith(prefix + ".")), None)
        if owner is None:
            continue
        layer_prefix = f"layer_{owner:03d}"
        if name.endswith(".self_attention.linear_qkv"):
            _register_pre_post(module, f"{layer_prefix}.qkv_projection", preferred_axis=0)
        elif name.endswith(".self_attention.core_attention"):
            _register_core_attention(module, layer_prefix)
        elif name.endswith(".self_attention"):
            _register_pre_post(module, f"{layer_prefix}.self_attention", preferred_axis=0)

    model.register_forward_pre_hook(_root_pre_hook, with_kwargs=True)
    model.register_forward_hook(_root_post_hook, with_kwargs=True)
    _STATE.required_stages.add("logits")


def _runtime_metadata(args: Any) -> dict[str, Any]:
    fixture = Path(str(getattr(args, "load_debug_rollout_data", "")))
    fixture_sha = _sha256(fixture) if fixture.is_file() else None
    return {
        "format_version": FORMAT_VERSION,
        "rank": _STATE.rank,
        "world_size": _STATE.world_size,
        "dp_rank": _STATE.dp_rank,
        "dp_world_size": _STATE.dp_world_size,
        "cp_rank": _STATE.cp_rank,
        "cp_world_size": _STATE.cp_world_size,
        "tp_rank": _STATE.tp_rank,
        "tp_world_size": _STATE.tp_world_size,
        "pp_rank": _STATE.pp_rank,
        "pp_world_size": _STATE.pp_world_size,
        "bf16": bool(getattr(args, "bf16", False)),
        "fp16": bool(getattr(args, "fp16", False)),
        "params_dtype": str(getattr(args, "params_dtype", None)),
        "qkv_format": str(getattr(args, "qkv_format", None)),
        "micro_batch_size": int(getattr(args, "micro_batch_size", 0)),
        "global_batch_size": int(getattr(args, "global_batch_size", 0)),
        "fixture": str(fixture),
        "fixture_sha256": fixture_sha,
        "expected_fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "capture_phase": CAPTURE_PHASE,
        "p0_oracle_source_sha256": P0_ORACLE_SOURCE_SHA256,
        "model_provider_cp_probe": "relax.backends.megatron.model_provider._install_cp_probe",
    }


def configure(args: Any) -> None:
    """Install the dump hooks through ``--custom-megatron-init-path``."""
    import torch.distributed as dist
    from megatron.core import mpu

    from relax.backends.megatron import model as model_backend
    from relax.backends.megatron import model_provider

    if _STATE.args is not None:
        raise RuntimeError("Task40 forward dump configure() was called more than once in one process")
    if str(getattr(args, "qkv_format", "thd")) != "thd":
        raise ValueError("Task40 Batch 3 forward dump supports only THD")
    if int(getattr(args, "micro_batch_size", 0)) != 1:
        raise ValueError("Task40 Batch 3 forward dump supports only micro-batch size 1")
    if not bool(getattr(args, "bf16", False)) or bool(getattr(args, "fp16", False)):
        raise ValueError("Task40 Batch 3 forward dump requires BF16 and forbids FP16/FP32 templates")
    fixture = Path(str(getattr(args, "load_debug_rollout_data", "")))
    if not fixture.is_file():
        raise ValueError(f"Task40 Step-0 fixture does not exist: {fixture}")
    fixture_sha = _sha256(fixture)
    if fixture_sha != EXPECTED_FIXTURE_SHA256:
        raise ValueError(
            f"Task40 Step-0 fixture SHA-256 mismatch: got {fixture_sha}, expected {EXPECTED_FIXTURE_SHA256}"
        )

    _STATE.args = args
    _STATE.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    _STATE.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    _STATE.dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
    _STATE.dp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    _STATE.cp_rank = mpu.get_context_parallel_rank()
    _STATE.cp_world_size = mpu.get_context_parallel_world_size()
    _STATE.tp_rank = mpu.get_tensor_model_parallel_rank()
    _STATE.tp_world_size = mpu.get_tensor_model_parallel_world_size()
    _STATE.pp_rank = mpu.get_pipeline_model_parallel_rank()
    _STATE.pp_world_size = mpu.get_pipeline_model_parallel_world_size()
    _STATE.dump_dir = Path(args.dump_details).parent / "dump"
    _STATE.dump_dir.mkdir(parents=True, exist_ok=True)
    (_STATE.dump_dir / f"rank{_STATE.rank:05d}").mkdir(exist_ok=True)
    _json_write(_STATE.dump_dir / f"runtime_rank{_STATE.rank}.json", _runtime_metadata(args))

    _STATE.original_get_batch = model_backend.get_batch
    model_backend.get_batch = _observed_get_batch

    _STATE.original_cp_probe = model_provider._install_cp_probe

    def install_cp_probe_and_dump(model: Any) -> None:
        if _STATE.original_cp_probe is None:
            raise RuntimeError("forward dump lost the original CP probe")
        _STATE.original_cp_probe(model)
        _install_forward_dump(model)

    model_provider._install_cp_probe = install_cp_probe_and_dump


def before_log_prob(args: Any, model: Any, store_prefix: str) -> None:
    """Select the one production log-prob pass captured by this harness."""
    del args, model
    _STATE.phase = CAPTURE_PHASE if store_prefix == "" else f"ignored_{store_prefix}log_prob"


def before_train_step(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Stop capture before the replayed P3O stats/train forwards."""
    del args, rollout_id, step_id, model, optimizer, opt_param_scheduler
    _STATE.phase = "train"


def finalize_manifest(run_dir: Path) -> dict[str, Any]:
    """Validate rank-local captures and write the run-level manifest."""
    dump_dir = run_dir / "dump"
    runtimes = [json.loads(path.read_text()) for path in sorted(dump_dir.glob("runtime_rank*.json"))]
    rank_manifests = [json.loads(path.read_text()) for path in sorted(dump_dir.glob("manifest_rank*.json"))]
    errors: list[str] = []
    if not runtimes:
        errors.append("no runtime_rank*.json files")
    expected_world_sizes = sorted({int(runtime["world_size"]) for runtime in runtimes})
    if len(expected_world_sizes) != 1:
        errors.append(f"inconsistent world sizes: {expected_world_sizes}")
    expected_ranks = expected_world_sizes[0] if len(expected_world_sizes) == 1 else 0
    runtime_ranks = {int(runtime["rank"]) for runtime in runtimes}
    manifest_ranks = {int(manifest["rank"]) for manifest in rank_manifests}
    if runtime_ranks != set(range(expected_ranks)):
        errors.append(f"runtime ranks={sorted(runtime_ranks)}, expected={list(range(expected_ranks))}")
    if manifest_ranks != set(range(expected_ranks)):
        errors.append(f"manifest ranks={sorted(manifest_ranks)}, expected={list(range(expected_ranks))}")

    runtime_by_rank = {int(runtime["rank"]): runtime for runtime in runtimes}
    for rank_manifest in rank_manifests:
        rank = int(rank_manifest["rank"])
        runtime = runtime_by_rank.get(rank)
        if runtime is None:
            continue
        denominator = int(runtime["dp_world_size"]) * int(runtime["micro_batch_size"])
        global_batch_size = int(runtime["global_batch_size"])
        if denominator <= 0 or global_batch_size % denominator != 0:
            errors.append(
                f"rank {rank} cannot derive micro-batch count from "
                f"global_batch_size={global_batch_size}, dp_world_size={runtime['dp_world_size']}, "
                f"micro_batch_size={runtime['micro_batch_size']}"
            )
            continue
        expected_micro_batches = global_batch_size // denominator
        actual_micro_batches = len(rank_manifest.get("captures", []))
        if actual_micro_batches != expected_micro_batches:
            errors.append(f"rank {rank} capture count={actual_micro_batches}, expected={expected_micro_batches}")

    captures: list[str] = []
    for manifest in rank_manifests:
        captures.extend(str(path) for path in manifest.get("captures", []))
    stage_file_count = 0
    nonfinite_stages: list[str] = []
    for relative in captures:
        metadata_path = dump_dir / relative / "metadata.json"
        if not metadata_path.is_file():
            errors.append(f"missing metadata: {relative}")
            continue
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get("complete", False):
            errors.append(f"incomplete capture: {relative}")
        required_stages = set(metadata.get("required_stages", []))
        actual_stages = set(metadata.get("stages", {}))
        if not required_stages:
            errors.append(f"capture declares no required stages: {relative}")
        if required_stages != actual_stages:
            errors.append(
                f"stage contract mismatch for {relative}: "
                f"missing={sorted(required_stages - actual_stages)}, unexpected={sorted(actual_stages - required_stages)}"
            )
        for stage, stage_meta in metadata.get("stages", {}).items():
            stage_file_count += 1
            if not (dump_dir / relative / stage_meta["path"]).is_file():
                errors.append(f"missing tensor: {relative}/{stage_meta['path']}")
            if not stage_meta.get("finite", False):
                nonfinite_stages.append(f"{relative}:{stage}")
    if nonfinite_stages:
        errors.append(f"non-finite stages: {nonfinite_stages[:8]}")

    fixture_hashes = sorted({str(runtime.get("fixture_sha256")) for runtime in runtimes})
    if fixture_hashes != [EXPECTED_FIXTURE_SHA256]:
        errors.append(f"fixture SHA-256 values={fixture_hashes}, expected={[EXPECTED_FIXTURE_SHA256]}")
    manifest = {
        "format_version": FORMAT_VERSION,
        "complete": not errors,
        "errors": errors,
        "run_dir": str(run_dir.resolve()),
        "dump_dir": "dump",
        "runtime_files": [str(path.relative_to(run_dir)) for path in sorted(dump_dir.glob("runtime_rank*.json"))],
        "rank_manifest_files": [
            str(path.relative_to(run_dir)) for path in sorted(dump_dir.glob("manifest_rank*.json"))
        ],
        "capture_count": len(captures),
        "stage_file_count": stage_file_count,
        "fixture_sha256_values": fixture_hashes,
        "captures": sorted(captures),
    }
    _json_write(run_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize_parser = subparsers.add_parser("finalize-manifest", help="validate rank dumps and write manifest.json")
    finalize_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = finalize_manifest(args.run_dir)
    sys.stdout.write(
        json.dumps({"complete": manifest["complete"], "errors": manifest["errors"]}, sort_keys=True) + "\n"
    )
    if not manifest["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
