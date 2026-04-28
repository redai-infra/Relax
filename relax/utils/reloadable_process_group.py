# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import socket
from collections.abc import Callable
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger
from relax.utils.memory_utils import available_memory, clear_memory, print_memory


logger = get_logger(__name__)

old_new_group_dict = {}
_wrap_low_level_call_enabled = False


def _safe_call(value_fn: Callable[[], Any]) -> Any:
    try:
        return value_fn()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {type(exc).__name__}: {exc}>"


def _describe_process_group(group: Any) -> dict[str, Any]:
    if isinstance(group, ReloadableProcessGroup):
        inner_group = group.group
        return {
            "type": type(group).__name__,
            "inner_type": type(inner_group).__name__ if inner_group is not None else None,
            "ranks": group.group_info.get("ranks"),
            "rank": _safe_call(lambda: group.rank()),
            "size": _safe_call(lambda: group.size()),
        }

    if isinstance(group, torch.distributed.ProcessGroup):
        return {
            "type": type(group).__name__,
            "rank": _safe_call(lambda: group.rank()),
            "size": _safe_call(lambda: group.size()),
        }

    return {"type": type(group).__name__}


def _collect_distributed_debug_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    env_keys = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "NCCL_SOCKET_IFNAME",
        "RCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "NCCL_DEBUG",
        "TORCH_DISTRIBUTED_DEBUG",
    ]

    process_groups = [
        _describe_process_group(arg)
        for arg in args
        if isinstance(arg, (ReloadableProcessGroup, torch.distributed.ProcessGroup))
    ]
    process_groups.extend(
        _describe_process_group(value)
        for value in kwargs.values()
        if isinstance(value, (ReloadableProcessGroup, torch.distributed.ProcessGroup))
    )

    context: dict[str, Any] = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "env": {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
        "torch_hip": torch.version.hip,
        "process_groups": process_groups,
        "dist_available": _safe_call(dist.is_available),
        "dist_initialized": _safe_call(dist.is_initialized),
    }

    if dist.is_available() and dist.is_initialized():
        context["dist"] = {
            "rank": _safe_call(dist.get_rank),
            "world_size": _safe_call(dist.get_world_size),
            "backend": _safe_call(dist.get_backend),
        }

    if torch.cuda.is_available():
        current_device = _safe_call(torch.cuda.current_device)
        context["cuda"] = {
            "device_count": _safe_call(torch.cuda.device_count),
            "current_device": current_device,
            "current_device_name": _safe_call(lambda: torch.cuda.get_device_name(current_device)),
        }
        if isinstance(current_device, int):
            props = _safe_call(lambda: torch.cuda.get_device_properties(current_device))
            if not isinstance(props, str):
                context["cuda"]["current_device_properties"] = {
                    "name": getattr(props, "name", None),
                    "uuid": getattr(props, "uuid", None),
                    "pci_bus_id": getattr(props, "pci_bus_id", None),
                    "gcn_arch_name": getattr(props, "gcnArchName", None),
                    "total_memory": getattr(props, "total_memory", None),
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                }

    try:
        import ray

        context["ray"] = {
            "is_initialized": ray.is_initialized(),
            "gpu_ids": _safe_call(ray.get_gpu_ids),
            "node_id": _safe_call(lambda: ray.get_runtime_context().get_node_id()),
            "actor_id": _safe_call(lambda: ray.get_runtime_context().get_actor_id()),
        }
    except Exception as exc:  # noqa: BLE001
        context["ray"] = f"<unavailable: {type(exc).__name__}: {exc}>"

    return context


def _log_distributed_exception(func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any], exc: Exception) -> None:
    context = _collect_distributed_debug_context(args, kwargs)
    logger.exception("torch.distributed.%s failed with context: %s", func_name, context)
    if hasattr(exc, "add_note"):
        exc.add_note(f"torch.distributed.{func_name} debug context: {context}")


def monkey_patch_torch_dist(args=None):
    global _wrap_low_level_call_enabled
    _wrap_low_level_call_enabled = getattr(args, "enable_cuda_memory_check", False) if args is not None else False

    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))

        if len(ranks) == 1:
            return group

        group = ReloadableProcessGroup(group, ranks)
        return group

    dist.new_group = new_group

    def get_new_query_function(func):
        """Wrap query functions (get_rank, get_world_size, etc.) without memory
        check."""

        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    def get_new_comm_function(func):
        """Wrap communication functions with memory check."""

        def new_function(*args, **kwargs):
            original_args = args
            original_kwargs = kwargs
            args = tuple(arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args)
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            with _wrap_low_level_call():
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    _log_distributed_exception(func.__name__, original_args, original_kwargs, exc)
                    raise

        return new_function

    dist.get_rank = get_new_query_function(dist.get_rank)
    dist.get_world_size = get_new_query_function(dist.get_world_size)
    dist.get_backend = get_new_query_function(dist.get_backend)
    dist.get_global_rank = get_new_query_function(dist.get_global_rank)
    dist.get_group_rank = get_new_query_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_query_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_comm_function(dist.all_reduce)
    dist.all_gather = get_new_comm_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_comm_function(dist.all_gather_into_tensor)
    dist.all_gather_object = get_new_comm_function(dist.all_gather_object)
    dist.all_to_all = get_new_comm_function(dist.all_to_all)
    dist.all_to_all_single = get_new_comm_function(dist.all_to_all_single)
    dist.broadcast = get_new_comm_function(dist.broadcast)
    dist.reduce = get_new_comm_function(dist.reduce)
    dist.reduce_scatter = get_new_comm_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_comm_function(dist.reduce_scatter_tensor)
    dist.scatter = get_new_comm_function(dist.scatter)
    dist.gather = get_new_comm_function(dist.gather)
    dist.barrier = get_new_comm_function(dist.barrier)
    dist.send = get_new_comm_function(dist.send)
    dist.recv = get_new_comm_function(dist.recv)
    dist._coalescing_manager = get_new_comm_function(dist._coalescing_manager)
    dist.broadcast_object_list = get_new_comm_function(dist.broadcast_object_list)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_comm_function(dist.isend)
    dist.irecv = get_new_comm_function(dist.irecv)

    def get_new_p2pop_function(func):
        def new_function(*args, **kwargs):
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg

            args = (convert(arg) for arg in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    GROUPS = {}

    def __init__(self, group, ranks):
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.group_info = {
            "ranks": ranks,
        }
        pid = os.getpid()
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def __getattr__(self, name):
        return getattr(self.group, name)

    @staticmethod
    def destroy_process_groups():
        pid = os.getpid()
        for reloadable_group in ReloadableProcessGroup.GROUPS.get(pid, []):
            if reloadable_group.group is None:
                continue
            try:
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            del reloadable_group.group
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups(timeout_minutes: int = 30):
        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
        old_new_group = old_new_group_dict.get(pid)
        for reloadable_group in reloadable_groups:
            if reloadable_group.group is not None:
                continue
            group = old_new_group(
                ranks=reloadable_group.group_info["ranks"],
                backend="nccl",
                timeout=timedelta(minutes=timeout_minutes),
            )
            reloadable_group.group = group

    def rank(self) -> int:
        return self.group.rank()

    def size(self) -> int:
        return self.group.size()

    def name(self) -> str:
        return self.group.name()

    def shutdown(self) -> None:
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call():
            try:
                return getattr(inner, method)(*args, **kwargs)
            except Exception as exc:
                _log_distributed_exception(method, (self, *args), kwargs, exc)
                raise

    def _fwd_query(self, method, *args, **kwargs):
        """Forward non-communication calls without memory check."""
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        return getattr(inner, method)(*args, **kwargs)

    def barrier(self, *a, **kw):
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, *a, **kw):
        return self._fwd("allgather_into_tensor_coalesced", *a, **kw)

    def gather(self, *a, **kw):
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, *a, **kw):
        return self._fwd("reduce_scatter_tensor_coalesced", *a, **kw)

    def alltoall_base(self, *a, **kw):
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        return self._fwd("recv_anysource", *a, **kw)

    def _start_coalescing(self, *a, **kw):
        return self._fwd_query("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd_query("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd_query("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd_query("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


def destroy_process_groups():
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups(timeout_minutes: int = 30):
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups(timeout_minutes=timeout_minutes)


@contextmanager
def _wrap_low_level_call():
    if not _wrap_low_level_call_enabled:
        yield
        return
    try:
        mem_info = available_memory()
        if mem_info["free_GB"] < 5:
            clear_memory()
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        e.add_note(f"{mem_info=}")
        raise
