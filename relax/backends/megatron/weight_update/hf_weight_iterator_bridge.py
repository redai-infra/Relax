# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
from megatron.core import mpu

from relax.utils import device as device_utils
from relax.utils.logging_utils import get_logger
from relax.utils.types import ParamInfo

from .bridge_converter import BridgeConverter
from .common import all_gather_param, named_params_and_buffers
from .hf_weight_iterator_base import HfWeightIteratorBase


logger = get_logger(__name__)

# Weight names that must appear in the same chunk for SGLang's MLA fusion.
_MLA_PAIRED_SUFFIXES = ("q_a_proj.weight", "kv_a_proj_with_mqa.weight")


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bridge_converter = BridgeConverter(
            args=self.args, model=self.model, quantization_config=self.quantization_config
        )
        buckets_result = _build_param_info_buckets(self.args, self.model)
        self._expert_buckets, self._non_expert_buckets, self._vanilla_key_map = buckets_result

    def get_hf_weight_chunks(self, megatron_local_weights):
        yield from _chunk_with_mla_pairing(
            self._iter_hf_params(megatron_local_weights),
            chunk_size=self.args.update_weight_buffer_size,
        )

    def _iter_hf_params(self, megatron_local_weights):
        """Yield individual (name, tensor) pairs for all params.

        Expert weights: load → TP gather + convert (src_rank only) →
        PP+EP broadcast via _broadcast_converted_bucket.

        Non-expert weights: PP/EP broadcast (BF16) → TP all-gather →
        bridge convert.
        """
        param_count = 0
        t_bcast_total = 0.0
        t_gather_total = 0.0
        t_convert_total = 0.0
        t_start = time.monotonic()
        device = device_utils.make_current_torch_device()
        rank = dist.get_rank()
        # Eagerly init bridge converter so all ranks are ready before broadcast.
        self._bridge_converter.init_tasks()
        self._bridge_converter.broadcast_and_apply_configs()

        # --- Expert weights: quantize-before-broadcast path ---
        for bucket_infos in self._expert_buckets:
            t_c0 = time.monotonic()
            params = _load_to_gpu(bucket_infos, megatron_local_weights, self._vanilla_key_map, device, rank)
            all_converted = []
            for info, param in zip(bucket_infos, params, strict=True):
                gathered = all_gather_param(self.args, info.name, param)
                if rank == info.src_rank:
                    all_converted.append(self._bridge_converter.convert(info.name, gathered))
                else:
                    all_converted.append(None)
                del gathered
            del params
            t_convert_total += time.monotonic() - t_c0

            t_b0 = time.monotonic()
            results = _broadcast_converted_bucket(bucket_infos, all_converted, device)
            t_bcast_total += time.monotonic() - t_b0
            param_count += len(results)
            yield from results
            del all_converted, results

        # --- Non-expert weights: original path ---
        for bucket_infos in self._non_expert_buckets:
            t_b0 = time.monotonic()
            params = _load_and_broadcast(bucket_infos, megatron_local_weights, self._vanilla_key_map, device, rank)
            t_b1 = time.monotonic()
            t_bcast_total += t_b1 - t_b0

            for info, param in zip(bucket_infos, params, strict=True):
                t_g0 = time.monotonic()
                gathered = all_gather_param(self.args, info.name, param)
                t_g1 = time.monotonic()
                t_gather_total += t_g1 - t_g0

                converted = self._bridge_converter.convert(info.name, gathered)
                t_convert_total += time.monotonic() - t_g1

                param_count += len(converted)
                yield from converted
                del gathered, converted

            del params

        if rank == 0:
            logger.info(
                "[Bridge Fast] params=%d | bcast=%.1fs | tp_gather=%.1fs | convert=%.1fs | total=%.1fs",
                param_count,
                t_bcast_total,
                t_gather_total,
                t_convert_total,
                time.monotonic() - t_start,
            )


def _build_param_info_buckets(args, model):
    """Build ParamInfo buckets and vanilla-key mapping at init time.

    Exchanges parameter metadata across PP/EP ranks so every rank knows about
    all params.  Also records the vanilla-key (TensorBackuper dict key) for
    each param owned by the current rank.

    Returns:
        expert_buckets: list of ParamInfo lists for expert params
        non_expert_buckets: list of ParamInfo lists for non-expert params
        vanilla_key_map: dict mapping global_name -> vanilla_key (only for
            params owned by this PP rank)
    """
    rank = dist.get_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    vanilla_iter = named_params_and_buffers(args, model, convert_to_global_name=False)
    global_iter = named_params_and_buffers(args, model, convert_to_global_name=True)

    local_infos = {}
    vanilla_key_map = {}
    for (v_name, v_param), (g_name, _g_param) in zip(vanilla_iter, global_iter, strict=True):
        local_infos[g_name] = ParamInfo(
            name=g_name,
            dtype=v_param.dtype,
            shape=v_param.shape,
            attrs={
                "tensor_model_parallel": getattr(v_param, "tensor_model_parallel", False),
                "partition_dim": getattr(v_param, "partition_dim", -1),
                "partition_stride": getattr(v_param, "partition_stride", 1),
                "parallel_mode": getattr(v_param, "parallel_mode", None),
            },
            size=v_param.numel() * v_param.element_size(),
            src_rank=rank,
        )
        vanilla_key_map[g_name] = v_name

    # Exchange across PP so every rank has all PP stages' param infos.
    if pp_size > 1:
        pp_infos_list: list[None | tuple[int, dict]] = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, local_infos),
            object_list=pp_infos_list,
            group=mpu.get_pipeline_model_parallel_group(),
        )
        for src_rank, infos in pp_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in local_infos:
                    if local_infos[name].src_rank > src_rank:
                        local_infos[name] = info
                else:
                    local_infos[name] = info

    # Exchange across EP so every rank has all expert indices.
    # Only expert params need src_rank update — non-expert params are
    # replicated across EP and already have the correct PP-local src_rank.
    if ep_size > 1:
        ep_infos_list: list[None | tuple[int, dict]] = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, local_infos),
            object_list=ep_infos_list,
            group=mpu.get_expert_model_parallel_group(),
        )
        for src_rank, infos in ep_infos_list:
            for name, info in infos.items():
                if name not in local_infos:
                    local_infos[name] = dataclasses.replace(info, src_rank=src_rank)
                elif ".experts." in name and info.src_rank < local_infos[name].src_rank:
                    local_infos[name] = dataclasses.replace(local_infos[name], src_rank=info.src_rank)

    # Sort deterministically and split expert / non-expert.
    all_infos = sorted(local_infos.values(), key=lambda info: info.name)
    expert_infos = [i for i in all_infos if ".experts." in i.name]
    non_expert_infos = [i for i in all_infos if ".experts." not in i.name]

    expert_buckets = _bucket_by_size(expert_infos, args)
    non_expert_buckets = _bucket_by_size(non_expert_infos, args)

    return expert_buckets, non_expert_buckets, vanilla_key_map


def _bucket_by_size(infos, args):
    if not infos:
        return []
    buckets: list[list[ParamInfo]] = [[]]
    bucket_bytes = 0
    for info in infos:
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
        param_size = info.size * tp_size

        if bucket_bytes + param_size > args.update_weight_buffer_size and buckets[-1]:
            buckets.append([])
            bucket_bytes = 0
        buckets[-1].append(info)
        bucket_bytes += param_size
    return buckets


def _load_to_gpu(bucket_infos, megatron_local_weights, vanilla_key_map, device, rank):
    """Load params from CPU dict to GPU.

    No broadcast.
    """
    params = []
    for info in bucket_infos:
        if rank == info.src_rank:
            vanilla_key = vanilla_key_map[info.name]
            cpu_tensor = megatron_local_weights[vanilla_key]
            gpu_tensor = cpu_tensor.to(device=device, non_blocking=True)
            param = torch.nn.Parameter(gpu_tensor, requires_grad=False)
        else:
            param = torch.nn.Parameter(torch.empty(info.shape, dtype=info.dtype, device=device), requires_grad=False)
        for key, value in info.attrs.items():
            setattr(param, key, value)
        params.append(param)
    device_utils.synchronize()
    return params


def _pp_broadcast(bucket_infos, params):
    """PP-broadcast params in-place."""
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    if pp_size <= 1:
        return
    handles = []
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_ranks = dist.get_process_group_ranks(pp_group)
    for info, param in zip(bucket_infos, params, strict=True):
        if info.src_rank in pp_ranks:
            handles.append(dist.broadcast(param, src=info.src_rank, group=pp_group, async_op=True))
    for handle in handles:
        handle.wait()


def _ep_broadcast(bucket_infos, params):
    """EP-broadcast expert params in-place."""
    ep_size = mpu.get_expert_model_parallel_world_size()
    if ep_size <= 1:
        return
    handles = []
    ep_group = mpu.get_expert_model_parallel_group()
    ep_ranks = dist.get_process_group_ranks(ep_group)
    rank = dist.get_rank()
    for info, param in zip(bucket_infos, params, strict=True):
        if ".experts." in info.name:
            src = info.src_rank if info.src_rank in ep_ranks else rank
            handles.append(dist.broadcast(param, src=src, group=ep_group, async_op=True))
    for handle in handles:
        handle.wait()


def _load_and_broadcast(bucket_infos, megatron_local_weights, vanilla_key_map, device, rank):
    """Load params from CPU dict, PP-broadcast, EP-broadcast.

    After this call every rank holds all params from all PP stages and all EP
    shards (still TP-sharded).  Mirrors the broadcast logic in
    ``HfWeightIteratorDirect._get_megatron_full_params``.
    """
    params = _load_to_gpu(bucket_infos, megatron_local_weights, vanilla_key_map, device, rank)
    _pp_broadcast(bucket_infos, params)
    _ep_broadcast(bucket_infos, params)
    return params


def _broadcast_converted_bucket(bucket_infos, all_converted, device):
    """Broadcast converted expert tensors across PP and EP groups.

    ``all_converted[i]`` is ``bridge_converter.convert()`` output for
    ``bucket_infos[i]`` on the owning rank, or ``None`` on non-owners.

    Two-phase NCCL broadcast: PP first, then EP.
    """
    rank = dist.get_rank()

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    if pp_size > 1:
        all_converted = _broadcast_converted_phase(
            bucket_infos,
            all_converted,
            device,
            rank,
            group=mpu.get_pipeline_model_parallel_group(),
        )

    ep_size = mpu.get_expert_model_parallel_world_size()
    if ep_size > 1:
        all_converted = _broadcast_converted_phase(
            bucket_infos,
            all_converted,
            device,
            rank,
            group=mpu.get_expert_model_parallel_group(),
        )

    out: list[tuple[str, torch.Tensor]] = []
    for converted in all_converted:
        if converted is not None:
            out.extend(converted)
    return out


# dtype ↔ int encoding for NCCL metadata tensor
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.int8: 5,
    torch.uint8: 6,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _compute_slot_size(all_converted, bucket_infos):
    """Compute the fixed int count per slot for metadata encoding.

    Every slot (including empty ones) must use the same number of ints so that
    allreduce(SUM) aligns correctly across ranks.
    """
    max_ints = 2  # header: [src+1, n_tensors]
    for converted in all_converted:
        if converted is None:
            continue
        n = 2
        for name, tensor in converted:
            n += 1 + len(name.encode("utf-8")) + 1 + tensor.ndim + 1
        max_ints = max(max_ints, n)
    return max_ints


def _encode_metadata(all_converted, bucket_infos, group_ranks_set, rank, slot_size=0):
    """Encode converted tensor metadata into a fixed-width int64 tensor.

    Each slot occupies exactly ``slot_size`` ints (zero-padded), making the
    total length ``len(bucket_infos) * slot_size``.  This enables correct
    allreduce(SUM) when only one rank has data per slot.

    Format per slot (padded to slot_size):
      [src_rank+1, n_tensors, (name_len, *name_bytes, ndim, *shape, dtype_code) × N, 0...]
    Empty slots: all zeros.
    """
    if slot_size == 0:
        slot_size = _compute_slot_size(all_converted, bucket_infos)
    n_slots = len(bucket_infos)
    buf = [0] * (n_slots * slot_size)
    for i, (info, converted) in enumerate(zip(bucket_infos, all_converted)):
        base = i * slot_size
        if converted is None:
            continue
        src = info.src_rank if info.src_rank in group_ranks_set else rank
        pos = base
        buf[pos] = src + 1
        pos += 1
        buf[pos] = len(converted)
        pos += 1
        for name, tensor in converted:
            name_bytes = name.encode("utf-8")
            buf[pos] = len(name_bytes)
            pos += 1
            for b in name_bytes:
                buf[pos] = b
                pos += 1
            buf[pos] = tensor.ndim
            pos += 1
            for s in tensor.shape:
                buf[pos] = s
                pos += 1
            buf[pos] = _DTYPE_TO_CODE[tensor.dtype]
            pos += 1
    return torch.tensor(buf, dtype=torch.int64, device="cpu")


def _decode_metadata(meta_tensor, slot_size):
    """Decode fixed-width int64 metadata tensor back to per-slot results.

    Each slot occupies ``slot_size`` ints.  src_rank is stored as src_rank+1; 0
    means empty slot.
    """
    data = meta_tensor.tolist()
    n_slots = len(data) // slot_size
    slots = []
    for i in range(n_slots):
        base = i * slot_size
        src_encoded = data[base]
        n_tensors = data[base + 1]
        if src_encoded == 0:
            slots.append(None)
            continue
        src = src_encoded - 1
        pos = base + 2
        tensors_meta = []
        for _ in range(n_tensors):
            name_len = data[pos]
            pos += 1
            name_bytes = bytes(data[pos : pos + name_len])
            pos += name_len
            name = name_bytes.decode("utf-8")
            ndim = data[pos]
            pos += 1
            shape = tuple(data[pos : pos + ndim])
            pos += ndim
            dtype_code = data[pos]
            pos += 1
            tensors_meta.append((name, shape, _CODE_TO_DTYPE[dtype_code]))
        slots.append((src, tensors_meta))
    return slots


def _broadcast_converted_phase(bucket_infos, all_converted, device, rank, group):
    """Single-group broadcast of converted tensors using only NCCL.

    1. Each rank encodes its owned tensors' metadata into an int64 tensor.
    2. Two allreduce calls exchange metadata: one for sizes (MAX), one
       for the content (SUM).  Empty slots are encoded as zeros so the
       SUM correctly merges non-overlapping contributions.
    3. Data tensors are broadcast from their owners.
    """
    group_ranks = dist.get_process_group_ranks(group)
    group_ranks_set = set(group_ranks)

    slot_size = _compute_slot_size(all_converted, bucket_infos)

    # Step 1: allreduce(MAX) to agree on slot_size across the group
    slot_size_t = torch.tensor([slot_size], dtype=torch.int64, device=device)
    dist.all_reduce(slot_size_t, op=dist.ReduceOp.MAX, group=group)
    slot_size = slot_size_t.item()

    local_meta_tensor = _encode_metadata(all_converted, bucket_infos, group_ranks_set, rank, slot_size)

    # Step 2: allreduce(SUM) to merge metadata from all ranks.
    # Each param slot has data from at most one rank; the rest contribute zeros.
    meta_buf = local_meta_tensor.to(device)
    dist.all_reduce(meta_buf, op=dist.ReduceOp.SUM, group=group)

    merged_slots = _decode_metadata(meta_buf.cpu(), slot_size)
    merged: dict[int, tuple[int, list]] = {}
    for i, slot in enumerate(merged_slots):
        if slot is not None:
            merged[i] = slot

    # Group param slots by broadcast source and pack into one buffer per src.
    # This reduces N×M individual broadcasts to one per unique src rank.
    src_to_slots: dict[int, list[tuple[int, list]]] = {}
    for i in range(len(bucket_infos)):
        if i not in merged:
            continue
        src, param_meta = merged[i]
        src_to_slots.setdefault(src, []).append((i, param_meta))

    result = list(all_converted)
    handles = []
    unpack_tasks: list[tuple[int, torch.Tensor, list[tuple[int, list]]]] = []

    for src, slot_list in src_to_slots.items():
        is_owner = rank == src
        # Compute total bytes for this src's tensors
        total_bytes = 0
        for _i, param_meta in slot_list:
            for _name, shape, dtype in param_meta:
                total_bytes += torch.tensor([], dtype=dtype).element_size() * torch.Size(shape).numel()

        if is_owner:
            parts = []
            for i, param_meta in slot_list:
                for j, (_name, _shape, _dtype) in enumerate(param_meta):
                    parts.append(all_converted[i][j][1].contiguous().flatten().view(torch.uint8))
            buf = torch.cat(parts).to(device)
        else:
            buf = torch.empty(total_bytes, dtype=torch.uint8, device=device)

        handles.append(dist.broadcast(buf, src=src, group=group, async_op=True))
        unpack_tasks.append((src, buf, slot_list))

    for h in handles:
        h.wait()

    # Unpack buffers back into named tensors
    for src, buf, slot_list in unpack_tasks:
        is_owner = rank == src
        offset = 0
        for i, param_meta in slot_list:
            tensors: list[tuple[str, torch.Tensor]] = []
            for j, (name, shape, dtype) in enumerate(param_meta):
                n_bytes = torch.tensor([], dtype=dtype).element_size() * torch.Size(shape).numel()
                if is_owner:
                    tensor = all_converted[i][j][1]
                else:
                    tensor = buf[offset : offset + n_bytes].view(dtype).reshape(shape)
                offset += n_bytes
                tensors.append((name, tensor))
            result[i] = tensors

    return result


def _chunk_with_mla_pairing(named_params, chunk_size):
    """Chunk weights by size while keeping MLA weight pairs together.

    SGLang's ``do_load_weights`` fuses ``q_a_proj`` and ``kv_a_proj_with_mqa``
    into ``fused_qkv_a_proj_with_mqa`` using a per-call ``cached_a_proj`` dict.
    Each chunk triggers a separate ``load_weights`` call, so the two weights
    **must** be in the same chunk for the fusion to succeed.

    Strategy: buffer any unpaired MLA weight and flush it together with its
    partner when the partner arrives.  All other weights pass through to the
    normal size-based chunking logic.
    """
    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_size = 0
    pending_mla: OrderedDict[str, tuple[str, torch.Tensor]] = OrderedDict()

    for name, tensor in named_params:
        is_mla = any(name.endswith(suffix) for suffix in _MLA_PAIRED_SUFFIXES)

        if is_mla:
            for suffix in _MLA_PAIRED_SUFFIXES:
                if name.endswith(suffix):
                    layer_key = name[: -len(suffix)]
                    break

            if layer_key in pending_mla:
                partner_name, partner_tensor = pending_mla.pop(layer_key)
                pair = [(partner_name, partner_tensor), (name, tensor)]
                pair_size = partner_tensor.nbytes + tensor.nbytes

                if bucket and (bucket_size + pair_size) >= chunk_size:
                    yield bucket
                    bucket = []
                    bucket_size = 0

                bucket.extend(pair)
                bucket_size += pair_size
            else:
                pending_mla[layer_key] = (name, tensor)
        else:
            obj_size = tensor.nbytes
            if bucket and (bucket_size + obj_size) >= chunk_size:
                yield bucket
                bucket = []
                bucket_size = 0

            bucket.append((name, tensor))
            bucket_size += obj_size

    for layer_key, (name, tensor) in pending_mla.items():
        if dist.get_rank() == 0:
            logger.warning("[Bridge Export] Unpaired MLA weight: %s (layer_key=%s)", name, layer_key)
        obj_size = tensor.nbytes
        if bucket and (bucket_size + obj_size) >= chunk_size:
            yield bucket
            bucket = []
            bucket_size = 0
        bucket.append((name, tensor))
        bucket_size += obj_size

    if bucket:
        yield bucket
