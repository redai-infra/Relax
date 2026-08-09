# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Collect and convert Mixture-of-LoRA parameters for rollout sync."""

from argparse import Namespace
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu

from relax.backends.megatron.misc_utils import strip_param_name_prefix
from relax.utils import device as device_utils
from relax.utils.megatron_peft_utils import build_mixture_lora_config, is_mixture_lora_param
from relax.utils.mixture_lora import (
    MixtureLoraStateSpec,
    build_mixture_lora_state_specs,
)

from .common import named_params_and_buffers


@dataclass(frozen=True)
class MixtureLoraParamInfo:
    """One routed tensor and the training rank that owns its PP stage."""

    state: MixtureLoraStateSpec
    local_shape: tuple[int, ...]
    tp_shard_dim: int | None
    src_rank: int
    weight_key: str


def _gather_pipeline_param_infos(
    local_infos: dict[str, MixtureLoraParamInfo],
    *,
    pipeline_group,
    pipeline_world_size: int,
) -> dict[str, MixtureLoraParamInfo]:
    """Collect parameter metadata from every stage in one PP group."""

    gathered_infos = [None] * pipeline_world_size
    dist.all_gather_object(
        gathered_infos,
        (dist.get_rank(), local_infos),
        group=pipeline_group,
    )
    merged_infos: dict[str, MixtureLoraParamInfo] = {}
    for _, stage_infos in gathered_infos:
        for name, info in stage_infos.items():
            previous = merged_infos.get(name)
            if previous is not None and previous != info:
                raise ValueError(f"Conflicting Mixture-of-LoRA metadata for {name}")
            merged_infos[name] = info
    return merged_infos


def _qwen3_attention_dimensions(args: Namespace) -> tuple[int, int, int]:
    head_dim = getattr(args, "kv_channels", None)
    if head_dim is None:
        head_dim = args.hidden_size // args.num_attention_heads
    if args.num_attention_heads % args.num_query_groups != 0:
        raise ValueError("num_attention_heads must be divisible by num_query_groups")
    return head_dim, args.num_attention_heads, args.num_query_groups


def _site_dimensions(args: Namespace, site_id: str) -> tuple[int, int]:
    head_dim, num_attention_heads, num_query_groups = _qwen3_attention_dimensions(args)
    target = site_id.rsplit(".", maxsplit=1)[-1]
    if target == "linear_qkv":
        return args.hidden_size, (num_attention_heads + 2 * num_query_groups) * head_dim
    if target == "linear_proj":
        return num_attention_heads * head_dim, args.hidden_size
    raise ValueError(f"Unsupported Mixture-of-LoRA site: {site_id!r}")


def _parameter_kind(parameter_name: str) -> str:
    for kind in ("experts.lora_A", "experts.lora_B", "router.weight"):
        if parameter_name.endswith(f".mixture_lora.{kind}"):
            return kind
    raise ValueError(f"Unsupported Mixture-of-LoRA parameter name: {parameter_name!r}")


def _tp_shard_dim(site_id: str, parameter_kind: str) -> int | None:
    target = site_id.rsplit(".", maxsplit=1)[-1]
    if target == "linear_qkv":
        return {
            "experts.lora_A": 1,
            "experts.lora_B": 1,
            "router.weight": None,
        }[parameter_kind]
    if target == "linear_proj":
        return {
            "experts.lora_A": 2,
            "experts.lora_B": 1,
            "router.weight": 1,
        }[parameter_kind]
    raise ValueError(f"Unsupported Mixture-of-LoRA site: {site_id!r}")


def _qkv_lora_b_to_sglang(
    tensor: torch.Tensor,
    *,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> torch.Tensor:
    """Convert Megatron's group-interleaved QKV rows to Q/K/V blocks."""

    queries_per_group = num_attention_heads // num_query_groups
    expected_rows = (num_attention_heads + 2 * num_query_groups) * head_dim
    if tensor.shape[1] != expected_rows:
        raise ValueError(f"QKV LoRA B has {tensor.shape[1]} rows, expected {expected_rows}")
    grouped = tensor.reshape(
        tensor.shape[0],
        num_query_groups,
        queries_per_group + 2,
        head_dim,
        tensor.shape[-1],
    )
    query, key, value = torch.split(grouped, [queries_per_group, 1, 1], dim=2)
    return torch.cat(
        [
            query.reshape(tensor.shape[0], -1, tensor.shape[-1]),
            key.reshape(tensor.shape[0], -1, tensor.shape[-1]),
            value.reshape(tensor.shape[0], -1, tensor.shape[-1]),
        ],
        dim=1,
    )


def merge_mixture_lora_tp_shards(
    info: MixtureLoraParamInfo,
    shards: Sequence[torch.Tensor],
    *,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> torch.Tensor:
    """Reconstruct one global tensor and convert backend-specific layout."""

    if not shards:
        raise ValueError(f"No TP shards supplied for {info.state.parameter_name}")
    if any(tuple(shard.shape) != info.local_shape for shard in shards):
        raise ValueError(f"TP shard shape mismatch for {info.state.parameter_name}")
    if info.tp_shard_dim is None:
        if len(shards) != 1:
            raise ValueError(f"Replicated parameter {info.state.parameter_name} expects one tensor")
        merged = shards[0]
    else:
        merged = torch.cat(tuple(shards), dim=info.tp_shard_dim)

    if info.state.site_id.endswith(".linear_qkv") and info.state.parameter_kind == "experts.lora_B":
        merged = _qkv_lora_b_to_sglang(
            merged,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            head_dim=head_dim,
        )
    if tuple(merged.shape) != info.state.global_shape:
        raise ValueError(
            f"Reconstructed {info.state.parameter_name} has shape {tuple(merged.shape)}, "
            f"expected {info.state.global_shape}"
        )
    if merged.dtype != info.state.dtype:
        raise TypeError(
            f"Reconstructed {info.state.parameter_name} has dtype {merged.dtype}, expected {info.state.dtype}"
        )
    return merged.contiguous()


class MixtureLoraSync:
    """Gather every PP/TP shard into SGLang-ready routed tensors."""

    def __init__(self, args: Namespace, model: Sequence[torch.nn.Module]) -> None:
        self.args = args
        self.model = model
        self.config = build_mixture_lora_config(args)
        if self.config is None:
            raise ValueError("MixtureLoraSync requires an enabled Mixture-of-LoRA configuration")
        self.base_sync_done = False
        self.param_infos = self._build_param_infos()
        if not self.param_infos:
            raise ValueError("No Mixture-of-LoRA parameters were found in the training model")
        self._validate_param_infos()

    def _validate_param_infos(self) -> None:
        expected_kinds = {"experts.lora_A", "experts.lora_B", "router.weight"}
        kinds_by_site: dict[str, set[str]] = {}
        for info in self.param_infos:
            kinds_by_site.setdefault(info.state.site_id, set()).add(info.state.parameter_kind)
        incomplete = {
            site_id: sorted(expected_kinds - parameter_kinds)
            for site_id, parameter_kinds in kinds_by_site.items()
            if parameter_kinds != expected_kinds
        }
        if incomplete:
            raise ValueError(f"Incomplete Mixture-of-LoRA parameter schema: {incomplete}")

    def _build_param_infos(self) -> tuple[MixtureLoraParamInfo, ...]:
        rank = dist.get_rank()
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        local_infos: dict[str, MixtureLoraParamInfo] = {}
        vanilla = named_params_and_buffers(self.args, self.model, convert_to_global_name=False)
        global_names = named_params_and_buffers(self.args, self.model, convert_to_global_name=True)
        for (vanilla_name, parameter), (global_name, _) in zip(vanilla, global_names, strict=True):
            parameter_name = strip_param_name_prefix(global_name)
            if not is_mixture_lora_param(parameter_name):
                continue
            kind = _parameter_kind(parameter_name)
            site_id = parameter_name.removesuffix(f".mixture_lora.{kind}")
            input_size, output_size = _site_dimensions(self.args, site_id)
            state = next(
                spec
                for spec in build_mixture_lora_state_specs(
                    self.config,
                    site_id,
                    input_size,
                    output_size,
                    parameter.dtype,
                )
                if spec.parameter_kind == kind
            )
            shard_dim = _tp_shard_dim(site_id, kind)
            expected_local_shape = list(state.global_shape)
            if shard_dim is not None:
                if expected_local_shape[shard_dim] % tp_world_size != 0:
                    raise ValueError(f"{state.parameter_name} cannot be sharded over TP={tp_world_size}")
                expected_local_shape[shard_dim] //= tp_world_size
            if tuple(parameter.shape) != tuple(expected_local_shape):
                raise ValueError(
                    f"Training parameter {state.parameter_name} has local shape {tuple(parameter.shape)}, "
                    f"expected {tuple(expected_local_shape)}"
                )
            weight_key = global_name if self.args.megatron_to_hf_mode == "raw" else vanilla_name
            local_infos[state.parameter_name] = MixtureLoraParamInfo(
                state=state,
                local_shape=tuple(parameter.shape),
                tp_shard_dim=shard_dim,
                src_rank=rank,
                weight_key=weight_key,
            )

        pipeline_world_size = mpu.get_pipeline_model_parallel_world_size()
        if pipeline_world_size > 1:
            local_infos = _gather_pipeline_param_infos(
                local_infos,
                pipeline_group=mpu.get_pipeline_model_parallel_group(),
                pipeline_world_size=pipeline_world_size,
            )
        return tuple(local_infos[name] for name in sorted(local_infos))

    def get_weight_chunks(
        self,
        local_weights: Mapping[str, torch.Tensor],
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Yield full routed tensors in deterministic, size-bounded chunks."""

        rank = dist.get_rank()
        device = device_utils.make_current_torch_device()
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()
        pp_world_size = mpu.get_pipeline_model_parallel_world_size()
        pp_group = mpu.get_pipeline_model_parallel_group() if pp_world_size > 1 else None
        pp_ranks = set(dist.get_process_group_ranks(pp_group)) if pp_group is not None else {rank}
        head_dim, num_attention_heads, num_query_groups = _qwen3_attention_dimensions(self.args)
        chunk: list[tuple[str, torch.Tensor]] = []
        chunk_size = 0

        for info in self.param_infos:
            if rank == info.src_rank:
                if info.weight_key not in local_weights:
                    raise KeyError(f"Missing Mixture-of-LoRA weight {info.weight_key!r}")
                source_tensor = local_weights[info.weight_key]
                if tuple(source_tensor.shape) != info.local_shape:
                    raise ValueError(
                        f"Mixture-of-LoRA weight {info.weight_key!r} has shape {tuple(source_tensor.shape)}, "
                        f"expected {info.local_shape}"
                    )
                if source_tensor.dtype != info.state.dtype:
                    raise TypeError(
                        f"Mixture-of-LoRA weight {info.weight_key!r} has dtype {source_tensor.dtype}, "
                        f"expected {info.state.dtype}"
                    )
                local_tensor = source_tensor.to(device=device)
            else:
                local_tensor = torch.empty(info.local_shape, dtype=info.state.dtype, device=device)
            if pp_group is not None and info.src_rank in pp_ranks:
                dist.broadcast(local_tensor, src=info.src_rank, group=pp_group)

            if info.tp_shard_dim is None or tp_world_size == 1:
                shards = [local_tensor]
            else:
                shards = [torch.empty_like(local_tensor) for _ in range(tp_world_size)]
                dist.all_gather(shards, local_tensor.contiguous(), group=tp_group)
            full_tensor = merge_mixture_lora_tp_shards(
                info,
                shards,
                num_attention_heads=num_attention_heads,
                num_query_groups=num_query_groups,
                head_dim=head_dim,
            )
            tensor_size = full_tensor.numel() * full_tensor.element_size()
            if chunk and chunk_size + tensor_size > self.args.update_weight_buffer_size:
                yield chunk
                chunk = []
                chunk_size = 0
            chunk.append((info.state.parameter_name, full_tensor))
            chunk_size += tensor_size

        if chunk:
            yield chunk
