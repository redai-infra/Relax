# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import re
from contextlib import contextmanager
from pathlib import Path

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from relax.utils import megatron_bridge_utils
from relax.utils.logging_utils import get_logger


try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = get_logger(__name__)

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    exist = Path(load_path).exists() and _is_dir_nonempty(load_path)

    if exist and _is_megatron_checkpoint(load_path):
        try:
            return _load_checkpoint_megatron(
                ddp_model=ddp_model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            )
        except AssertionError as e:
            if "OptimizerParamScheduler" in str(e):
                raise RuntimeError(_format_opt_param_scheduler_error(args, e)) from e
            raise
    else:
        if not exist:
            load_path = None
            logger.warning(f"{args.load=} does not exist or is an empty directory. use args.hf_checkpoint")
        elif not _is_hf_checkpoint(load_path):
            logger.warning(
                f"{args.load=} exists but is not a valid HF checkpoint (no config.json). "
                "Falling back to args.hf_checkpoint"
            )
            load_path = None
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _format_opt_param_scheduler_error(args, original: AssertionError) -> str:
    # lr_decay_steps = num_rollout * rollout_batch_size * n_samples_per_prompt
    # (see relax/backends/megatron/model.py:get_optimizer_param_scheduler).
    # When any of those args change vs. the saved checkpoint, Megatron's
    # OptimizerParamScheduler refuses to load. Tell the user exactly what to do.
    return (
        f"Resume failed: {original}\n\n"
        f"Megatron's OptimizerParamScheduler rejects mismatched LR/WD schedule values "
        f"between the current run and the loaded checkpoint. This is almost always caused "
        f"by changing one of the args that feed into `lr_decay_steps` / `wd_incr_steps`:\n"
        f"    lr_decay_steps = num_rollout * rollout_batch_size * n_samples_per_prompt\n"
        f"Current values:\n"
        f"    --num-rollout            {getattr(args, 'num_rollout', None)}\n"
        f"    --rollout-batch-size     {getattr(args, 'rollout_batch_size', None)}\n"
        f"    --n-samples-per-prompt   {getattr(args, 'n_samples_per_prompt', None)}\n"
        f"    --global-batch-size      {getattr(args, 'global_batch_size', None)}\n"
        f"    --lr-decay-iters         {getattr(args, 'lr_decay_iters', None)}\n"
        f"    --lr-warmup-iters        {getattr(args, 'lr_warmup_iters', None)}\n"
        f"    --lr-warmup-fraction     {getattr(args, 'lr_warmup_fraction', None)}\n\n"
        f"Pick one:\n"
        f"  (a) Revert the changed arg to match the checkpoint, OR\n"
        f"  (b) Add `--override-opt_param-scheduler` to keep the NEW schedule "
        f"(class values overwrite checkpoint values), OR\n"
        f"  (c) Add `--use-checkpoint-opt_param-scheduler` to keep the OLD schedule "
        f"(checkpoint values overwrite class values)."
    )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _is_hf_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "config.json").is_file()


@contextmanager
def _patch_scatter_dtype_cast():
    """Temporarily patch torch.distributed.scatter to auto-cast scatter_list
    tensors to match output dtype.

    Megatron Bridge's scatter_to_tp_ranks creates `output` with the Megatron
    model dtype (e.g. fp16) and `scatter_list` from HF weights (e.g. bf16). Not
    all mapping types (e.g. GatedMLPMapping) cast HF weights to the target
    dtype before scatter, causing a ValueError from PyTorch's dtype consistency
    check. This patch ensures scatter_list tensors are cast to match output's
    dtype.
    """
    import torch.distributed as dist

    original_scatter = dist.scatter

    def _scatter_with_dtype_cast(output, scatter_list=None, **kwargs):
        if scatter_list is not None and output is not None:
            target_dtype = output.dtype
            scatter_list = [t.to(dtype=target_dtype) if t.dtype != target_dtype else t for t in scatter_list]
        return original_scatter(output, scatter_list=scatter_list, **kwargs)

    dist.scatter = _scatter_with_dtype_cast
    try:
        yield
    finally:
        dist.scatter = original_scatter


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    source_path = load_path or args.hf_checkpoint
    logger.info(
        f"Load checkpoint from HuggingFace model into Megatron (requested_path={load_path}, source_path={source_path})"
    )

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(source_path, trust_remote_code=True)
        with _patch_scatter_dtype_cast():
            bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
