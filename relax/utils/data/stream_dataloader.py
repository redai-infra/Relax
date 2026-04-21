# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging
from argparse import Namespace
from functools import partial
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from tensordict import TensorDict
from transfer_queue.dataloader.streaming_dataloader import StreamingDataLoader
from transfer_queue.dataloader.streaming_dataset import StreamingDataset


logger = logging.getLogger(__name__)


def create_stream_dataloader(
    args: Namespace,
    rollout_id: int,
    task_name: str,
    data_fields: list,
    dp_rank: int,
):
    """Create a streaming dataloader and micro-batch plan for a rollout.

    This function constructs a `StreamingDataset` and wraps it with a
    `StreamingDataLoader`. It then builds a list of dataloader iterators
    (one per virtual pipeline parallel stage) and a list describing the
    number of microbatches to use for each step in the rollout.

    Args:
        args (Namespace): Configuration / runtime arguments. Expected to
            contain `tq_config`, `micro_batch_size`, `n_samples_per_prompt`,
            `rollout_batch_size`, and `global_batch_size` attributes.
        rollout_id (int): Identifier for the current rollout partition.
        task_name (str): Name of the task to fetch from the transfer queue.
        data_fields (list): List of data field names to request from the
            transfer queue.
        dp_rank (int): Data-parallel rank (used by the dataset/queue).

    Returns:
        Tuple[List[StreamingDataLoader], List[int]]: A tuple where the first
        element is a list of `StreamingDataLoader` objects (one per virtual
        pipeline stage) and the second element is a list with the number of
        microbatches for each step in the rollout.
    """

    # Choose the appropriate fetch function based on fully_async mode
    # Use partial to bind the broadcast_pp parameter
    # broadcast_pp is the inverse of fully_async: True for colocate, False for fully async
    fetch_batch_fn = partial(
        get_data_from_transfer_queue, args=args, broadcast_pp=not getattr(args, "fully_async", False)
    )
    dataset = StreamingDataset(
        config=args.tq_config,
        batch_size=args.micro_batch_size * args.n_samples_per_prompt,
        micro_batch_size=args.micro_batch_size,
        data_fields=data_fields,
        partition_id=f"train_{rollout_id}",
        task_name=task_name,
        dp_rank=dp_rank,
        fetch_batch_fn=fetch_batch_fn,
        process_batch_fn=split_dict,
    )

    dataloader = StreamingDataLoader(dataset)

    # Virtual pipeline parallel size may be None when not using vpp.
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1

    # Provide one iterator per virtual pipeline stage. Each element is the
    # same dataloader instance; downstream code uses one per stage.
    data_iterator = [dataloader for _ in range(vpp_size)]

    # Compute how many forward steps (global batch splits) occur per rollout,
    # then compute the number of microbatches for each of those steps.
    num_steps_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size

    num_microbatches = [
        args.global_batch_size
        // mpu.get_data_parallel_world_size(with_context_parallel=False)
        // args.micro_batch_size
        for _ in range(num_steps_per_rollout)
    ]

    return data_iterator, num_microbatches


def split_dict(data_dict: Dict[str, Any], batch_meta, micro_batch_size: int) -> List[Tuple[Dict[str, Any], Any]]:
    """Split a batched dictionary into a list of smaller micro-batch
    dictionaries.

    The function slices each tensor or list in `data_dict` along the batch
    dimension (dimension 0) into chunks of size `micro_batch_size`. The
    corresponding `batch_meta` is also split into matching chunks via
    `batch_meta.chunk(...)` and paired with each data chunk.

    Args:
        data_dict (Dict[str, Any]): Mapping from field name to batched value.
            All values must share the same batch size in dimension 0.
        batch_meta: An auxiliary object describing the batch (must have a
            `.size` attribute and a `.chunk(n)` method that returns a list of
            `n` metadata pieces matching the data chunks).
        micro_batch_size (int): Desired size for each micro-batch. The last
            chunk may be smaller if `batch_meta.size` is not divisible by
            `micro_batch_size`.

    Returns:
        List[Tuple[Dict[str, Any], Any]]: A list of tuples where each tuple
        contains (chunked_data_dict, chunked_batch_meta).

    Raises:
        ValueError: If `micro_batch_size` is not positive.
    """

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    total_size = batch_meta.size
    num_chunks = (total_size + micro_batch_size - 1) // micro_batch_size

    result: List[Tuple[Dict[str, Any], Any]] = []
    batch_meta_list: List = batch_meta.chunk(num_chunks)
    for i in range(num_chunks):
        start = i * micro_batch_size
        end = start + micro_batch_size
        chunk = {key: value[start:end] for key, value in data_dict.items()}
        result.append((chunk, batch_meta_list[i]))

    return result


def _broadcast_routed_experts(
    values: "torch.Tensor | None",
    offsets: "torch.Tensor | None",
    is_src: bool,
    cuda_dev: torch.device,
    broadcast_pp: bool,
    keep_on_gpu: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Broadcast rollout_routed_experts tensors via NCCL dist.broadcast.

    On the source rank (*is_src* = True), *values* and *offsets* are the
    NestedTensor jagged internals.  On non-source ranks they are None and
    will be allocated here.

    Broadcasting mirrors the same pattern used by ``broadcast_object_list``
    in this file: first across the TP group (src = tp_rank 0), then
    optionally across the PP group (src = pp_rank 0).

    Using ``dist.broadcast`` on contiguous GPU tensors is orders of magnitude
    faster than ``broadcast_object_list`` which pickles everything (~14 s for
    377 MB vs sub-second via NCCL).
    """

    def _bcast_tensor(tensor, is_sender, dtype):
        """Broadcast a tensor (any shape) across PP then TP groups.

        Order: PP first, then TP.  This is important because only
        (tp_rank==0, pp_rank==0) has the data.  PP broadcast first
        sends data to (tp_rank==0, pp_rank==1), then TP broadcast in
        each PP stage sends from tp_rank==0 to other tp_ranks.
        """
        # After PP broadcast, every tp_rank==0 has the data.
        # After TP broadcast, every rank has the data.
        is_tp_rank0 = mpu.get_tensor_model_parallel_rank() == 0

        # --- Step 1: PP broadcast (only among tp_rank==0 ranks) ---
        if broadcast_pp and is_tp_rank0:
            pp_group = mpu.get_pipeline_model_parallel_group()
            pp_src_global = dist.get_global_rank(pp_group, 0)

            # Broadcast shape metadata
            if is_sender and tensor is not None:
                ndim_t = torch.tensor([tensor.ndim], dtype=torch.long, device=cuda_dev)
            else:
                ndim_t = torch.tensor([0], dtype=torch.long, device=cuda_dev)
            dist.broadcast(ndim_t, src=pp_src_global, group=pp_group)
            ndim = ndim_t.item()

            if is_sender and tensor is not None:
                shape_t = torch.tensor(list(tensor.shape), dtype=torch.long, device=cuda_dev)
            else:
                shape_t = torch.empty(ndim, dtype=torch.long, device=cuda_dev)
            dist.broadcast(shape_t, src=pp_src_global, group=pp_group)
            shape = torch.Size(shape_t.tolist())

            # Broadcast data
            if is_sender and tensor is not None:
                tensor = tensor.to(dtype=dtype, device=cuda_dev).contiguous()
            else:
                tensor = torch.empty(shape, dtype=dtype, device=cuda_dev)
            dist.broadcast(tensor, src=pp_src_global, group=pp_group)

        # --- Step 2: TP broadcast (tp_rank==0 -> others in each TP group) ---
        tp_group = mpu.get_tensor_model_parallel_group()
        tp_src_global = dist.get_global_rank(tp_group, 0)

        # Now every tp_rank==0 has the tensor (from step 1 or original).
        if is_tp_rank0 and tensor is not None:
            ndim_t = torch.tensor([tensor.ndim], dtype=torch.long, device=cuda_dev)
        else:
            ndim_t = torch.tensor([0], dtype=torch.long, device=cuda_dev)
        dist.broadcast(ndim_t, src=tp_src_global, group=tp_group)
        ndim = ndim_t.item()

        if is_tp_rank0 and tensor is not None:
            shape_t = torch.tensor(list(tensor.shape), dtype=torch.long, device=cuda_dev)
        else:
            shape_t = torch.empty(ndim, dtype=torch.long, device=cuda_dev)
        dist.broadcast(shape_t, src=tp_src_global, group=tp_group)
        shape = torch.Size(shape_t.tolist())

        if is_tp_rank0 and tensor is not None:
            buf = tensor.to(dtype=dtype, device=cuda_dev).contiguous()
        else:
            buf = torch.empty(shape, dtype=dtype, device=cuda_dev)
        dist.broadcast(buf, src=tp_src_global, group=tp_group)

        return buf

    values_gpu = _bcast_tensor(values, is_src, torch.int32)
    offsets_gpu = _bcast_tensor(offsets, is_src, torch.long)

    if keep_on_gpu:
        # When optimize_routing_replay is enabled, keep tensors on GPU to
        # avoid a redundant GPU→CPU→GPU round-trip.  fill_routing_replay's
        # RoutingReplay.record() handles GPU→CPU-pinned copy automatically.
        return values_gpu, offsets_gpu

    # Move back to CPU for downstream consumption (fill_routing_replay etc.)
    return values_gpu.cpu(), offsets_gpu.cpu()


def get_data_from_transfer_queue(
    args,
    tq_client,
    data_fields,
    batch_size,
    partition_id,
    task_name,
    sampling_config,
    batch_index,
    broadcast_pp: bool = True,
):
    """Fetch a batch from the transfer queue and broadcast it across tensor-
    parallel and optionally pipeline-parallel ranks.

    The function queries the transfer queue client (`tq_client`) for
    metadata and data on the appropriate rank(s) based on the broadcast_pp
    parameter. The retrieved pair (data, meta) is then broadcast across
    tensor-parallel ranks and optionally across pipeline-parallel ranks
    using torch.distributed.broadcast_object_list so that every rank has
    the same batch information.

    If the returned `rollout_data` is an instance of `TensorDict`, we
    convert it into a plain Python dictionary. This conversion turns
    tensor-valued entries into lists (so downstream code may index into
    them per-sample) and converts special fields like lengths/reward into
    Python lists as well.

    Args:
        args: Configuration / runtime arguments (used for post-processing).
        tq_client: Transfer-queue client with `get_meta` and `get_data` API.
        data_fields: List of field names to request.
        batch_size: Desired batch size to request.
        partition_id: Partition identifier string for the queue.
        task_name: Task name used by the queue.
        sampling_config: Extra sampling configuration passed to the queue.
        batch_index: Index of the batch to request (used for replay semantics).
        broadcast_pp: Whether to broadcast across pipeline parallel ranks.
            True for colocate mode, False for fully async mode.

    Returns:
        Tuple[Optional[dict], Optional[Any]]: A tuple of (rollout_data, batch_meta).
        If no data is available, both elements are None.
    """

    # Compose request configuration and ask the queue for metadata.
    config = {**sampling_config, "batch_index": batch_index, "partition_id": partition_id}

    # Determine which rank should fetch data based on broadcast_pp
    if broadcast_pp:
        # Colocate mode: only tp_rank==0 AND pp_rank==0 fetches data
        should_fetch = mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == 0
    else:
        # Fully async mode: only tp_rank==0 fetches data (each PP stage independently)
        should_fetch = mpu.get_tensor_model_parallel_rank() == 0

    if should_fetch:
        batch_meta = tq_client.get_meta(
            data_fields=data_fields,
            batch_size=batch_size,
            partition_id=partition_id,
            sampling_config=config,
            task_name=task_name,
        )  # type: ignore

        if batch_meta.size == 0:
            rollout_data = [None, None]
        else:
            rollout_data = [tq_client.get_data(batch_meta), batch_meta]
    else:
        # Non-fetching ranks start with an empty placeholder and
        # will receive the real data via broadcast.
        rollout_data = [None, None]

    # Use an explicit CUDA device so the communication backend (e.g. NCCL)
    # can bind to a known CUDA context.
    cuda_dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    # --- Extract rollout_routed_experts BEFORE broadcast_object_list ---
    # broadcast_object_list uses pickle for the entire payload. When
    # rollout_routed_experts is present (~377 MB for Qwen3-30B-A3B), pickle
    # serialization dominates train_get_data_time (~14s).  We extract it and
    # broadcast the underlying contiguous tensors via dist.broadcast (NCCL
    # zero-copy) instead, reducing the time to sub-second.
    has_routed_experts = "rollout_routed_experts" in data_fields
    routed_experts_values = None
    routed_experts_offsets = None

    if has_routed_experts and should_fetch and rollout_data[0] is not None:
        td = rollout_data[0]
        if isinstance(td, TensorDict) and "rollout_routed_experts" in td.keys():
            nt = td["rollout_routed_experts"]
            # NestedTensor jagged internals: _values (total_tokens, inner_dim), _offsets (batch+1,)
            routed_experts_values = nt._values.contiguous()
            routed_experts_offsets = nt._offsets.contiguous()
            # Remove from TensorDict so broadcast_object_list only pickles ~4 MB
            del td["rollout_routed_experts"]
            rollout_data[0] = td

    # Always broadcast across tensor parallel ranks (now without routed_experts)
    dist.broadcast_object_list(
        rollout_data,
        device=cuda_dev,
        group=mpu.get_tensor_model_parallel_group(),
        group_src=0,
    )

    # Conditionally broadcast across pipeline parallel ranks
    if broadcast_pp:
        dist.broadcast_object_list(
            rollout_data,
            device=cuda_dev,
            group=mpu.get_pipeline_model_parallel_group(),
            group_src=0,
        )

    # Unpack the broadcasted pair.
    rollout_data, batch_meta = rollout_data[0], rollout_data[1]

    if rollout_data is None:
        return None, None

    # --- Broadcast routed_experts tensors via efficient dist.broadcast ---
    if has_routed_experts:
        routed_experts_values, routed_experts_offsets = _broadcast_routed_experts(
            routed_experts_values,
            routed_experts_offsets,
            should_fetch,
            cuda_dev,
            broadcast_pp,
            keep_on_gpu=getattr(args, "optimize_routing_replay", False),
        )

    # If the received object is a Tensordict, convert it into a plain Python
    # dict so downstream code can mix tensors and Python lists freely.
    if isinstance(rollout_data, TensorDict):
        new_rollout_data: Dict[str, Any] = {}
        for k, v in rollout_data.items():
            # Convert length/reward-style fields to Python lists.
            if "lengths" in k or "reward" in k:
                new_rollout_data[k] = v.tolist()
            elif k == "multimodal_train_inputs":
                # multimodal inputs are stored as a list of tensordicts / dicts;
                # some entries may be None for text-only samples in a multimodal
                # batch.  Turn each non-None entry into a plain dict.
                from tensordict.tensorclass import NonTensorData

                new_rollout_data[k] = []
                for item in list(v):
                    # NonTensorStack iteration yields NonTensorData wrappers
                    raw = item.data if isinstance(item, NonTensorData) else item
                    if raw is None:
                        new_rollout_data[k].append(None)
                    elif isinstance(raw, dict):
                        new_rollout_data[k].append(raw)
                    else:
                        # TensorDict or similar — convert to plain dict
                        new_rollout_data[k].append(dict(raw.items()) if hasattr(raw, "items") else dict(raw.data))
            elif k == "rollout_routed_experts":
                # rollout_routed_experts is stored as a NonTensorStack /
                # LinkedList in TensorDict (raw numpy arrays).  Iterating may
                # yield NonTensorData wrappers, so unwrap via `.data` when
                # needed to get the underlying numpy array.
                from tensordict.tensorclass import NonTensorData

                new_rollout_data[k] = [item.data if isinstance(item, NonTensorData) else item for item in v]
            elif isinstance(v, torch.Tensor):
                # Expand a tensor with batch dimension into a Python list of
                # per-sample tensors so downstream code can index them.
                new_rollout_data[k] = [tensor for tensor in v]  # noqa: C416
            else:
                raise TypeError(f"Unsupported rollout_data type for key '{k}': {type(v)}")

        rollout_data = new_rollout_data

    # Re-attach routed_experts as a list of 2D tensors (per-sample)
    if has_routed_experts:
        rollout_data["rollout_routed_experts"] = [
            routed_experts_values[routed_experts_offsets[i] : routed_experts_offsets[i + 1]]
            for i in range(len(routed_experts_offsets) - 1)
        ]

    post_process_rollout_data(args, rollout_data)

    return rollout_data, batch_meta


def post_process_rollout_data(args, rollout_data):
    # move tokens/loss_masks to GPU in-place as a list of tensors (downstream
    # code in this module expects lists of sequence tensors for packing)
    from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

    cuda_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    rollout_data["tokens"] = [torch.tensor(t, dtype=torch.long, device=cuda_dev) for t in rollout_data["tokens"]]
    rollout_data["loss_masks"] = [
        torch.tensor(t, dtype=torch.int, device=cuda_dev) for t in rollout_data["loss_masks"]
    ]
    if "multimodal_train_inputs" in rollout_data:
        # Move multimodal training tensors to GPU in advance.
        # Values may be a single Tensor (e.g. image pixel_values) or a list
        # of Tensors (e.g. video frames), so handle both cases.
        def _to_cuda(v):
            if isinstance(v, torch.Tensor):
                return v.to(device=cuda_dev)
            if isinstance(v, list):
                return [_to_cuda(item) for item in v]
            return v

        rollout_data["multimodal_train_inputs"] = [
            ({key: _to_cuda(val) for key, val in mm_dict.items()} if mm_dict is not None else None)
            for mm_dict in rollout_data["multimodal_train_inputs"]
        ]

    if args.qkv_format == "bshd":
        # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
        max_seq_len = max(rollout_data["total_lengths"])

        # pad to reduce memory fragmentation and maybe make the computation faster
        pad_size = mpu.get_tensor_model_parallel_world_size() * args.data_pad_size_multiplier
        max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

        rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

    for key in ["rollout_log_probs", "teacher_log_probs"]:
        if key not in rollout_data:
            continue
        rollout_data[key] = [
            torch.tensor(
                slice_log_prob_with_cp(
                    log_prob,
                    total_length,
                    response_length,
                    args.qkv_format,
                    rollout_data["max_seq_lens"][i] if args.qkv_format == "bshd" else None,
                ),
                device=cuda_dev,
                dtype=torch.float32,
            )
            for i, (log_prob, total_length, response_length) in enumerate(
                zip(
                    rollout_data[key],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            )
        ]

    if "teacher_topk_token_ids" in rollout_data:
        teacher_topk_k = rollout_data.get("teacher_topk_k", None)
        if isinstance(teacher_topk_k, torch.Tensor):
            teacher_topk_k = teacher_topk_k.tolist()

        topk_tensors = []
        for i, (flat_topk_ids, total_length, response_length) in enumerate(
            zip(
                rollout_data["teacher_topk_token_ids"],
                rollout_data["total_lengths"],
                rollout_data["response_lengths"],
                strict=False,
            )
        ):
            k = int(teacher_topk_k[i]) if teacher_topk_k is not None else 0
            if k <= 0:
                topk_tensors.append(torch.empty((response_length, 0), dtype=torch.long, device=cuda_dev))
                continue

            topk_tensor = torch.tensor(flat_topk_ids, dtype=torch.long, device=cuda_dev)
            expected = response_length * k
            if topk_tensor.numel() < expected:
                topk_tensor = F.pad(topk_tensor, (0, expected - topk_tensor.numel()), value=-1)
            elif topk_tensor.numel() > expected:
                topk_tensor = topk_tensor[:expected]

            topk_tensor = topk_tensor.reshape(response_length, k)
            topk_tensor = slice_log_prob_with_cp(
                topk_tensor,
                total_length,
                response_length,
                args.qkv_format,
                rollout_data["max_seq_lens"][i] if args.qkv_format == "bshd" else None,
            )
            topk_tensors.append(topk_tensor)

        rollout_data["teacher_topk_token_ids"] = topk_tensors

    if "rollout_routed_experts" in rollout_data:
        from tensordict.tensorclass import NonTensorData

        rollout_data["rollout_routed_experts"] = [
            torch.tensor(r.data if isinstance(r, NonTensorData) else r, dtype=torch.long, device=cuda_dev)
            for r in rollout_data["rollout_routed_experts"]
        ]
