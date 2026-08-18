#!/usr/bin/env python3

"""Batch-7 BF16 Step-0 observability hook.

The hook is intentionally local to Task40.  It records the fixed-input P3O
oracle, the checkpoint-loaded parameter identity, and finalized gradients
without changing the production training path.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator


EXPECTED_FIXTURE_SHA256 = "48538d165386dc94006613d857c022a7ba2e979bdc31bc617374eee2dc3c35b8"
FORMAT_VERSION = 1
DEFAULT_PRE_SYNC_GRADIENT_TARGETS = (
    "decoder.final_layernorm.weight",
    "decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
    "decoder.layers.0.self_attention.linear_qkv.bias",
    "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
    "decoder.layers.0.self_attention.linear_qkv.weight",
)
_STATE: dict[str, Any] = {
    "args": None,
    "output_dir": None,
    "rank": 0,
    "local_counter": 0,
    "sync_counter": 0,
    "parameters_captured": False,
    "parameter_capture_error": None,
    "parameter_capture_thread": None,
    "gradients_captured": False,
    "pre_sync_gradients_captured": False,
    "pre_sync_gradient_targets": DEFAULT_PRE_SYNC_GRADIENT_TARGETS,
    "strict_dp_anchor": True,
    "optimizer_ids": set(),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _cpu(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _tensor_bytes(tensor: Any) -> tuple[Any, bytes]:
    import torch

    cpu = tensor.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return cpu, raw


def _named_parameters(model: Any) -> Iterator[tuple[str, Any]]:
    seen: set[int] = set()
    for chunk_index, chunk in enumerate(model):
        for name, parameter in chunk.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield f"chunk{chunk_index:03d}.{name}", parameter


def _capture_initial_parameters(model: Any) -> None:
    if _STATE["parameters_captured"]:
        return
    output_dir: Path = _STATE["output_dir"]
    rank = int(_STATE["rank"])
    aggregate = hashlib.sha256()
    tensors = []
    for name, parameter in _named_parameters(model):
        cpu, raw = _tensor_bytes(parameter)
        header = json.dumps(
            {"dtype": str(cpu.dtype), "name": name, "shape": list(cpu.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        aggregate.update(len(header).to_bytes(8, "little"))
        aggregate.update(header)
        aggregate.update(len(raw).to_bytes(8, "little"))
        aggregate.update(raw)
        tensors.append(
            {
                "name": name,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "numel": cpu.numel(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    _write_json(
        output_dir / f"initial_parameters_rank{rank}.json",
        {
            "format_version": FORMAT_VERSION,
            "rank": rank,
            "capture_point": (
                "hash worker launched after checkpoint load and before the first P3O stats/train forward; "
                "joined before optimizer prepare/update while parameters remain immutable"
            ),
            "parameter_sha256": aggregate.hexdigest(),
            "tensor_count": len(tensors),
            "total_numel": sum(int(record["numel"]) for record in tensors),
            "tensors": tensors,
        },
    )
    _STATE["parameters_captured"] = True


def _start_initial_parameter_capture(model: Any) -> None:
    if _STATE["parameter_capture_thread"] is not None:
        raise RuntimeError("Task40 Batch 7 parameter capture worker was already started")

    def worker() -> None:
        try:
            _capture_initial_parameters(model)
        except BaseException as exc:  # surfaced on the training thread by the join below
            _STATE["parameter_capture_error"] = exc

    thread = threading.Thread(target=worker, name="task40-b7-parameter-sha256", daemon=False)
    _STATE["parameter_capture_thread"] = thread
    thread.start()


def _join_initial_parameter_capture() -> None:
    thread = _STATE["parameter_capture_thread"]
    if thread is None:
        raise RuntimeError("Task40 Batch 7 parameter capture worker was not started")
    thread.join()
    error = _STATE["parameter_capture_error"]
    if error is not None:
        raise RuntimeError("Task40 Batch 7 initial parameter SHA-256 worker failed") from error
    if not _STATE["parameters_captured"]:
        raise RuntimeError("Task40 Batch 7 initial parameter SHA-256 worker produced no manifest")


def _capture_prepared_gradient_shards(model: Any, optimizer: Any) -> None:
    import torch

    if _STATE["gradients_captured"]:
        raise RuntimeError("Task40 Batch 7 attempted to capture gradients more than once")
    output_dir: Path = _STATE["output_dir"]
    rank = int(_STATE["rank"])
    gradient_dir = output_dir / "gradients"
    gradient_dir.mkdir(exist_ok=True)
    vector_dir = gradient_dir / f"rank{rank:05d}"
    vector_dir.mkdir(exist_ok=False)

    aggregate = hashlib.sha256()
    tensors = []
    missing_owned = []
    aggregate_l2_sq = 0.0
    aggregate_numel = 0
    names = {id(parameter): (name, parameter) for name, parameter in _named_parameters(model)}
    inner_optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    shard_index = 0
    for optimizer_index, inner in enumerate(inner_optimizers):
        if not hasattr(inner, "model_param_group_index_map") or not hasattr(inner, "_get_model_param_range_map"):
            raise RuntimeError(
                "Task40 Batch 7 gradient reconstruction requires Megatron DistributedOptimizer shard metadata"
            )
        for model_parameter, (group_index, group_order) in inner.model_param_group_index_map.items():
            if id(model_parameter) not in names:
                raise RuntimeError("Distributed optimizer owns a model parameter that is absent from named_parameters")
            name, parameter = names[id(model_parameter)]
            parameter_range = inner._get_model_param_range_map(model_parameter)["param"]
            optimizer_parameter = inner.optimizer.param_groups[group_index]["params"][group_order]
            gradient = optimizer_parameter.grad
            record: dict[str, Any] = {
                "index": shard_index,
                "optimizer_index": optimizer_index,
                "name": name,
                "parameter_shape": list(parameter.shape),
                "parameter_numel": parameter.numel(),
                "range_start": int(parameter_range.start),
                "range_end": int(parameter_range.end),
                "requires_grad": bool(parameter.requires_grad),
                "source": "prepared_distributed_optimizer_shard",
            }
            shard_index += 1
            if gradient is None:
                record.update({"present": False, "numel": 0, "file": None})
                missing_owned.append(name)
                tensors.append(record)
                continue

            cpu, raw = _tensor_bytes(gradient)
            if cpu.numel() != int(parameter_range.end) - int(parameter_range.start):
                raise RuntimeError(
                    f"prepared gradient shard length mismatch for {name}: gradient={cpu.numel()}, "
                    f"range={parameter_range.start}:{parameter_range.end}"
                )
            values = cpu.to(torch.float64)
            l2_sq = float(torch.sum(values * values))
            finite = bool(torch.isfinite(values).all())
            file_name = f"shard_{record['index']:05d}.pt"
            torch.save(cpu, vector_dir / file_name)
            header = json.dumps(
                {
                    "dtype": str(cpu.dtype),
                    "name": name,
                    "range_end": int(parameter_range.end),
                    "range_start": int(parameter_range.start),
                    "shape": list(cpu.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            aggregate.update(len(header).to_bytes(8, "little"))
            aggregate.update(header)
            aggregate.update(len(raw).to_bytes(8, "little"))
            aggregate.update(raw)
            aggregate_l2_sq += l2_sq
            aggregate_numel += cpu.numel()
            record.update(
                {
                    "present": True,
                    "shape": list(cpu.shape),
                    "dtype": str(cpu.dtype),
                    "numel": cpu.numel(),
                    "finite": finite,
                    "l2_sq": l2_sq,
                    "l2_norm": l2_sq**0.5,
                    "max_abs": float(values.abs().max()) if values.numel() else 0.0,
                    "sum": float(values.sum()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "file": f"rank{rank:05d}/{file_name}",
                }
            )
            tensors.append(record)

    _write_json(
        gradient_dir / f"summary_rank{rank}.json",
        {
            "format_version": FORMAT_VERSION,
            "rank": rank,
            "capture_point": "after DistributedOptimizer.prepare_grads and before clipping/update",
            "gradient_sha256": aggregate.hexdigest(),
            "shard_count": len(tensors),
            "present_shard_count": sum(bool(record["present"]) for record in tensors),
            "aggregate_numel": aggregate_numel,
            "aggregate_l2_sq": aggregate_l2_sq,
            "aggregate_l2_norm": aggregate_l2_sq**0.5,
            "missing_owned_gradients": missing_owned,
            "vector_files_saved": True,
            "tensors": tensors,
        },
    )
    _STATE["gradients_captured"] = True


def _capture_pre_sync_gradients(model: Any, num_tokens: Any) -> None:
    """Capture selected full local gradients before DDP/CP synchronization.

    The normal Batch-7 gradient artifact is intentionally post-finalization.
    This diagnostic snapshot keeps a small, named subset of ``main_grad``
    tensors at the finalizer entry so local CUDA backward/accumulation drift
    can be separated from the following reduce-scatter/all-reduce and token
    normalization.
    """
    import torch

    if _STATE["pre_sync_gradients_captured"]:
        raise RuntimeError("Task40 Batch 7 attempted to capture pre-sync gradients more than once")
    output_dir: Path = _STATE["output_dir"]
    rank = int(_STATE["rank"])
    targets = tuple(str(value) for value in _STATE["pre_sync_gradient_targets"])
    gradient_dir = output_dir / "pre_sync_gradients"
    vector_dir = gradient_dir / f"rank{rank:05d}"
    vector_dir.mkdir(parents=True, exist_ok=False)

    tensors = []
    selected = sorted(
        ((name, parameter) for name, parameter in _named_parameters(model) if any(key in name for key in targets)),
        key=lambda item: item[0],
    )
    if not selected:
        raise RuntimeError(f"Task40 Batch 7 pre-sync gradient targets matched no parameters: {targets}")
    for index, (name, parameter) in enumerate(selected):
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"Task40 Batch 7 pre-sync gradient is missing for {name}")
        cpu, raw = _tensor_bytes(gradient)
        values = cpu.to(torch.float64)
        file_name = f"tensor_{index:03d}.pt"
        torch.save(cpu, vector_dir / file_name)
        l2_sq = float(torch.sum(values * values))
        tensors.append(
            {
                "name": name,
                "parameter_shape": list(parameter.shape),
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "numel": cpu.numel(),
                "finite": bool(torch.isfinite(values).all()),
                "l2_sq": l2_sq,
                "l2_norm": l2_sq**0.5,
                "max_abs": float(values.abs().max()) if values.numel() else 0.0,
                "sum": float(values.sum()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "file": f"rank{rank:05d}/{file_name}",
            }
        )

    local_num_tokens = _cpu(num_tokens)
    if isinstance(local_num_tokens, torch.Tensor):
        local_num_tokens = int(local_num_tokens.reshape(()))
    elif local_num_tokens is not None:
        local_num_tokens = int(local_num_tokens)
    _write_json(
        gradient_dir / f"summary_rank{rank}.json",
        {
            "format_version": FORMAT_VERSION,
            "rank": rank,
            "capture_point": "before DDP/CP gradient synchronization and token normalization",
            "local_num_tokens": local_num_tokens,
            "targets": list(targets),
            "tensor_count": len(tensors),
            "all_finite": all(record["finite"] for record in tensors),
            "tensors": tensors,
        },
    )
    _STATE["pre_sync_gradients_captured"] = True


def configure(args: Any) -> None:
    """Install the Batch-7 BF16-only oracle observers."""
    import torch
    import torch.distributed as dist
    from megatron.core import mpu

    from relax.backends.megatron import p3o_step

    if _STATE["args"] is not None:
        raise RuntimeError("Task40 Batch 7 revalidation configure() was called more than once")
    if not bool(getattr(args, "bf16", False)) or bool(getattr(args, "fp16", False)):
        raise ValueError("Task40 Batch 7 permits BF16 only")
    if str(getattr(args, "qkv_format", "thd")) != "thd":
        raise ValueError("Task40 Batch 7 requires THD")
    if int(getattr(args, "micro_batch_size", 0)) != 1:
        raise ValueError("Task40 Batch 7 requires micro-batch size 1")
    if str(getattr(args, "p3o_ess_scope", "")) != "step":
        raise ValueError("Task40 Batch 7 requires P3O step ESS scope")
    fixture = Path(str(getattr(args, "load_debug_rollout_data", "")))
    if not fixture.is_file():
        raise ValueError(f"Task40 Step-0 fixture does not exist: {fixture}")
    fixture_sha = _sha256(fixture)
    if fixture_sha != EXPECTED_FIXTURE_SHA256:
        raise ValueError(
            f"Task40 Step-0 fixture SHA-256 mismatch: got {fixture_sha}, expected {EXPECTED_FIXTURE_SHA256}"
        )

    output_dir = Path(args.dump_details).parent / "oracle"
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    configured_targets = os.environ.get("P3O_B7_PRE_SYNC_GRADIENT_TARGETS", "").strip()
    pre_sync_gradient_targets = (
        tuple(value.strip() for value in configured_targets.split(",") if value.strip())
        if configured_targets
        else DEFAULT_PRE_SYNC_GRADIENT_TARGETS
    )
    _STATE.update(
        {
            "args": args,
            "output_dir": output_dir,
            "rank": rank,
            "pre_sync_gradient_targets": pre_sync_gradient_targets,
            "strict_dp_anchor": mpu.get_data_parallel_rank(with_context_parallel=False) == 0,
        }
    )
    runtime = {
        "format_version": FORMAT_VERSION,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "dp_rank": mpu.get_data_parallel_rank(with_context_parallel=False),
        "dp_world_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        "cp_rank": mpu.get_context_parallel_rank(),
        "cp_world_size": mpu.get_context_parallel_world_size(),
        "tp_rank": mpu.get_tensor_model_parallel_rank(),
        "tp_world_size": mpu.get_tensor_model_parallel_world_size(),
        "pp_rank": mpu.get_pipeline_model_parallel_rank(),
        "pp_world_size": mpu.get_pipeline_model_parallel_world_size(),
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "params_dtype": str(args.params_dtype),
        "qkv_format": str(args.qkv_format),
        "micro_batch_size": int(args.micro_batch_size),
        "global_batch_size": int(args.global_batch_size),
        "p3o_ess_scope": str(args.p3o_ess_scope),
        "fixture": str(fixture),
        "fixture_sha256": fixture_sha,
        "expected_fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "parameter_capture_point": (
            "worker launched by before_train_step after checkpoint loading and before stats forward; "
            "joined before optimizer prepare/update"
        ),
        "gradient_capture_point": "after DistributedOptimizer.prepare_grads and before clipping/update",
        "pre_sync_gradient_capture_point": "before DDP/CP gradient synchronization and token normalization",
        "pre_sync_gradient_targets": list(pre_sync_gradient_targets),
    }
    _write_json(output_dir / f"runtime_rank{rank}.json", runtime)

    original_local_stats = p3o_step._local_stats_from_batch
    original_synchronize = p3o_step.synchronize_p3o_stats

    def observed_local_stats(local_args: Any, batch: dict[str, Any], log_probs: list[Any]) -> Any:
        result = original_local_stats(local_args, batch, log_probs)
        local_counter = int(_STATE["local_counter"])
        if not batch.get("__is_dummy__", False) and bool(_STATE["strict_dp_anchor"]):
            current = torch.cat(log_probs, dim=0)
            behavior = torch.cat(batch["rollout_log_probs"], dim=0)
            valid_mask = p3o_step.get_cp_local_valid_mask(
                batch["total_lengths"],
                batch["response_lengths"],
                batch["loss_masks"],
                local_args.qkv_format,
                batch.get("max_seq_lens"),
                batch.get("padded_total_lengths"),
                dynamic_cp_size=batch.get("dynamic_cp_size"),
                dynamic_cp_rank=batch.get("dynamic_cp_rank"),
            )
            packed = batch.get("packed_seq_params")
            total_lengths = [int(value) for value in batch["total_lengths"]]
            artifact = {
                "rank": rank,
                "micro_batch_index": local_counter,
                "current_log_probs": _cpu(current),
                "rollout_log_probs": _cpu(behavior),
                "valid_mask": _cpu(valid_mask.bool()),
                "loss_masks": _cpu(batch["loss_masks"]),
                "total_lengths": total_lengths,
                "response_lengths": [int(value) for value in batch["response_lengths"]],
                "token_ids": _cpu(batch.get("unconcat_tokens", batch.get("tokens"))),
                "position_ids": [torch.arange(length, dtype=torch.int64) for length in total_lengths],
                "max_seq_lens": _cpu(batch.get("max_seq_lens")),
                "padded_total_lengths": _cpu(batch.get("padded_total_lengths")),
                "cu_seqlens_q": _cpu(getattr(packed, "cu_seqlens_q", None)),
                "cu_seqlens_kv": _cpu(getattr(packed, "cu_seqlens_kv", None)),
                "max_seqlen_q": getattr(packed, "max_seqlen_q", None),
                "max_seqlen_kv": getattr(packed, "max_seqlen_kv", None),
                "relax_total_lengths": _cpu(getattr(packed, "_relax_total_lengths", None)),
                "relax_attention_pad_multiple": getattr(packed, "_relax_attention_pad_multiple", None),
                "relax_cu_seqlens_cpu": _cpu(getattr(packed, "_relax_cu_seqlens_cpu", None)),
                "local_s1": _cpu(result[0].sum_ratio),
                "local_s2": _cpu(result[0].sum_ratio_sq),
                "local_n": _cpu(result[0].valid_token_count),
                "invalid_flag": _cpu(result[1]),
            }
            torch.save(artifact, output_dir / f"vectors_rank{rank}_micro{local_counter}.pt")
        _STATE["local_counter"] = local_counter + 1
        return result

    def observed_synchronize(*sync_args: Any, **sync_kwargs: Any) -> Any:
        reduced = original_synchronize(*sync_args, **sync_kwargs)
        sync_counter = int(_STATE["sync_counter"])
        torch.save(
            {
                "rank": rank,
                "sync_index": sync_counter,
                "s1": _cpu(reduced.sum_ratio),
                "s2": _cpu(reduced.sum_ratio_sq),
                "n": _cpu(reduced.valid_token_count),
            },
            output_dir / f"global_stats_rank{rank}_sync{sync_counter}.pt",
        )
        _STATE["sync_counter"] = sync_counter + 1
        return reduced

    p3o_step._local_stats_from_batch = observed_local_stats
    p3o_step.synchronize_p3o_stats = observed_synchronize


def before_train_step(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Capture parameters and wrap the one optimizer step for gradients."""
    del args, opt_param_scheduler
    if rollout_id != 0 or step_id != 0:
        raise RuntimeError(f"Task40 Batch 7 permits only rollout=0/step=0, got {rollout_id=}, {step_id=}")
    _start_initial_parameter_capture(model)
    optimizer_id = id(optimizer)
    if optimizer_id in _STATE["optimizer_ids"]:
        raise RuntimeError("Task40 Batch 7 optimizer step observer was already installed")
    _STATE["optimizer_ids"].add(optimizer_id)
    from megatron.core.utils import get_model_config

    config = get_model_config(model[0])
    original_finalize_model_grads = config.finalize_model_grads_func
    if original_finalize_model_grads is None:
        raise RuntimeError("Task40 Batch 7 requires Megatron's gradient finalizer")

    def observed_finalize_model_grads(
        finalizer_model: Any,
        num_tokens: Any = None,
        **finalizer_kwargs: Any,
    ) -> Any:
        _capture_pre_sync_gradients(finalizer_model, num_tokens)
        return original_finalize_model_grads(finalizer_model, num_tokens, **finalizer_kwargs)

    config.finalize_model_grads_func = observed_finalize_model_grads
    original_prepare_grads = optimizer.prepare_grads

    def observed_prepare_grads(*prepare_args: Any, **prepare_kwargs: Any) -> Any:
        _join_initial_parameter_capture()
        found_inf = original_prepare_grads(*prepare_args, **prepare_kwargs)
        if not found_inf:
            _capture_prepared_gradient_shards(model, optimizer)
        return found_inf

    optimizer.prepare_grads = observed_prepare_grads
