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


def _warm_hf_checkpoint_page_cache(source_path: str) -> None:
    """Pre-fault the HF checkpoint into the Linux page cache, once per node.

    NFS-backed safetensors files are accessed via mmap by the bridge, so the
    first per-page touch incurs a synchronous small-read round-trip (we have
    measured ~20 MB/s effective throughput from ``aten::cat``). Reading the
    files end-to-end once promotes them to the page cache, after which the
    bridge's lazy tensor reads are memory-fast.

    Implementation is pure-Python (``open(...).read()`` in chunks) — no shell
    invocation, so dynamic paths cannot inject commands.

    Coordination — explicit per-node rank-0 pattern:

    - ``LOCAL_RANK == 0`` (the first GPU actor on each host) does the read
      and writes a done-marker under ``/dev/shm`` (tmpfs, so it naturally
      clears on reboot — avoiding stale-marker / cleared-cache mismatches).
    - All other local ranks poll for the marker with a generous timeout.
      They never touch NFS themselves.
    - An advisory ``flock`` still wraps the rank-0 path so that two
      independent Relax jobs sharing a host and the same checkpoint don't
      both warm — only the first acquires the lock, the second sees the
      marker and skips.
    - Errors (missing path, read failure, timeout) are logged and swallowed:
      warmup is a best-effort optimization, never a correctness gate.
    """
    import fcntl
    import glob
    import hashlib
    import time

    if not source_path:
        return
    abs_path = os.path.abspath(source_path)
    if not os.path.isdir(abs_path):
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    digest = hashlib.sha1(abs_path.encode()).hexdigest()[:16]
    marker_dir = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
    lock_path = f"{marker_dir}/relax_hf_warmup_{digest}.lock"
    done_path = f"{marker_dir}/relax_hf_warmup_{digest}.done"

    def _marker_says_warm() -> bool:
        try:
            with open(done_path) as df:
                return df.read().strip() == abs_path
        except OSError:
            return False

    def _stream_files_to_devnull(files: list[str]) -> int:
        """Read each file end-to-end so the kernel pulls every page into the
        page cache.

        Pure-Python — no shell, no command injection surface. Returns total
        bytes read. Shows a per-file tqdm progress bar.
        """
        from tqdm import tqdm

        chunk = 8 * 1024 * 1024  # 8 MiB, large enough that read syscall overhead is negligible
        total = 0
        pbar = tqdm(files, desc="warming HF ckpt", unit="file", dynamic_ncols=True)
        for fp in pbar:
            pbar.set_postfix_str(os.path.basename(fp))
            try:
                with open(fp, "rb") as fh:
                    while True:
                        data = fh.read(chunk)
                        if not data:
                            break
                        total += len(data)
            except OSError as exc:
                logger.warning(f"HF checkpoint warmup: skipping {fp} due to read error: {exc}")
        pbar.close()
        return total

    if local_rank == 0:
        try:
            lf = open(lock_path, "w")
        except OSError as e:
            logger.warning(f"HF checkpoint warmup: cannot open lock file {lock_path}: {e}")
            return
        try:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                if _marker_says_warm():
                    logger.info(f"HF checkpoint page cache already warm on this node: {abs_path}")
                    return
                files = sorted(
                    glob.glob(os.path.join(abs_path, "*.safetensors")) + glob.glob(os.path.join(abs_path, "*.bin"))
                )
                if not files:
                    logger.info(f"HF checkpoint warmup: no *.safetensors / *.bin under {abs_path}, skipping")
                    return
                t0 = time.time()
                logger.info(
                    f"[local_rank=0] Warming HF checkpoint page cache on this node: {abs_path} ({len(files)} files)"
                )
                total_bytes = _stream_files_to_devnull(files)
                elapsed = time.time() - t0
                throughput_mb = total_bytes / max(elapsed, 1e-6) / (1024 * 1024)
                try:
                    with open(done_path, "w") as df:
                        df.write(abs_path)
                except OSError as e:
                    logger.warning(f"HF checkpoint warmup: cannot write marker {done_path}: {e}")
                logger.info(
                    f"[local_rank=0] HF checkpoint page cache warmed in {elapsed:.1f}s "
                    f"({total_bytes / (1024 * 1024):.0f} MiB, {throughput_mb:.0f} MiB/s)"
                )
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        finally:
            lf.close()
    else:
        # Other local ranks just wait for rank 0's marker.
        timeout_s = float(os.environ.get("RELAX_HF_WARMUP_TIMEOUT_S", "1800"))
        poll_interval_s = 1.0
        t0 = time.time()
        logged_waiting = False
        while time.time() - t0 < timeout_s:
            if _marker_says_warm():
                if logged_waiting:
                    logger.info(
                        f"[local_rank={local_rank}] HF checkpoint warmup ready after waiting {time.time() - t0:.1f}s"
                    )
                return
            if not logged_waiting and time.time() - t0 > 5.0:
                logger.info(
                    f"[local_rank={local_rank}] waiting for local_rank=0 to warm HF checkpoint page cache: {abs_path}"
                )
                logged_waiting = True
            time.sleep(poll_interval_s)
        logger.warning(
            f"[local_rank={local_rank}] HF checkpoint warmup wait timed out after {timeout_s:.0f}s; "
            f"proceeding without confirmation (load may still succeed, just slower)"
        )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    # Prefer ref_load (if it's an HF dir) over hf_checkpoint on fallback. INT4 QAT
    # runs set --hf-checkpoint to a compressed-tensors packed dir that the bridge
    # cannot read; --ref-load points at the BF16 HF dir that it can. Mirrors the
    # `args.load = args.ref_load or args.hf_checkpoint` remap in arguments.py.
    if load_path is not None:
        source_path = load_path
    elif args.ref_load and _is_hf_checkpoint(args.ref_load):
        source_path = args.ref_load
    else:
        source_path = args.hf_checkpoint
    logger.info(
        f"Load checkpoint from HuggingFace model into Megatron (requested_path={load_path}, source_path={source_path})"
    )

    if getattr(args, "warm_hf_checkpoint_page_cache", False):
        _warm_hf_checkpoint_page_cache(source_path)

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
