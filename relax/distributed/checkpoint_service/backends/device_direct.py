# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""
DeviceDirectBackend - Communication backend using PyTorch distributed (NCCL/GLOO).

Supports:
- NCCL: For GPU-to-GPU communication, high efficiency
- GLOO: For CPU-based communication, fully async with device computation

Features:
- Process group management
- Broadcast, send/recv operations
- Async operations with CUDA streams
"""

import asyncio
import logging
import socket
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Dict, List, Optional

import httpx
import ray
import requests
import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm
from urllib3.exceptions import NewConnectionError

from relax.backends.megatron.weight_conversion import convert_to_hf
from relax.backends.megatron.weight_update.common import all_gather_param, named_params_and_buffers
from relax.distributed.checkpoint_service.backends.base import CommBackend, TensorFusion
from relax.distributed.checkpoint_service.config import BackendType, RoleInfo
from relax.distributed.checkpoint_service.utils import load_weight
from relax.utils import device as device_utils
from relax.utils.distributed_utils import get_gloo_group, init_process_group
from relax.utils.logging_utils import get_logger


logging.getLogger("httpx").setLevel(logging.WARNING)

logger = get_logger(__name__)


class DeviceDirectBackend(CommBackend):
    """PyTorch distributed communication backend using NCCL (GPU) or GLOO
    (CPU).

    Example:
        backend = DeviceDirectBackend(
            backend_type=BackendType.GLOO,
            role_info=RoleInfo(...)
        )
        backend.init_process_group()
        backend.send({"weight": tensor}, dst=1)
        tensors = backend.recv(src=0)
    """

    def __init__(
        self,
        args,
        backend_type: BackendType,
        role_info: Optional[RoleInfo],
        model: Sequence[torch.nn.Module],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        coordinator_url=None,
        lock: Any = None,
        timeout_seconds: int = 300,
    ) -> None:
        """Initialize DeviceDirectBackend.

        Args:
            args: Backend arguments
            backend_type: GLOO or NCCL
            role_info: Current node information
            model: Model instance(s)
            model_name: Model identifier
            quantization_config: Optional quantization settings
            coordinator_url: URL of the coordinator service
            lock: Remote lock for coordinating weight updates
            timeout_seconds: Operation timeout (default 300)
        """
        super().__init__(backend_type, role_info)
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.http_client = httpx.Client(timeout=30.0)
        self.coordinator_url = coordinator_url
        self.lock = lock
        self.timeout_seconds = timeout_seconds
        self.device = next(model[0].parameters()).device if model else device_utils.current_device()

        self._comm_stream: Optional[Any] = None  # CUDA stream
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._tensor_fusion = TensorFusion()

        # For recv, we need to know tensor shapes in advance or use a metadata channel
        self._pending_recvs: Dict[str, asyncio.Future] = {}
        self._model_update_groups = None
        self._model_update_groups_for_actor_fwd_ref = None

        # World size for process group initialization
        self.world_size: Optional[int] = None

        # Ray actors for rollout communication
        self.rollout_engines: Dict[int, Any] = {}  # rank -> Ray actor handle
        device_utils.set_device(self.device)

        # Bridge-based HF weight converter (lazy-initialized on first use)
        self._use_bridge = getattr(args, "megatron_to_hf_mode", None) == "bridge"
        if self._use_bridge:
            from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter

            self._bridge_converter = BridgeConverter(args=args, model=model, quantization_config=quantization_config)

    def _create_rollout_engines(self, rollout_topology: Dict[int, Dict[str, Any]]) -> None:
        """Create Ray actors for each rollout node.

        Args:
            rollout_topology: Mapping of rank -> node_info (contains 'ip' and 'port').
        """
        logger.info(f"Creating {len(rollout_topology)} RolloutEngine actors...")
        for rank, node_info in rollout_topology.items():
            actor = RolloutEngine.remote(int(rank), node_info)
            self.rollout_engines[int(rank)] = actor
            logger.info(f"Created RolloutEngine actor for rank {rank}")

    def _batch_request(self, endpoint: str, payload: Optional[Dict] = None, get_rank: bool = False) -> List[Any]:
        """Send HTTP requests to all rollout engines and collect futures.

        Args:
            endpoint: Endpoint path (e.g. '/init_weights_update_group').
            payload: Optional JSON payload to send.
            get_rank: If True, payload is expected to be a dict keyed by rank.

        Returns:
            List of Ray futures for the remote calls.
        """
        if not self.rollout_engines:
            logger.warning("No rollout engines available for batch request")
            return []

        futures = []
        for rank, engine in self.rollout_engines.items():
            if get_rank:
                payload_cur = payload.get(int(rank), {}) if payload else None
            else:
                payload_cur = payload
            future = engine.make_request.remote(endpoint, payload_cur)
            futures.append(future)
        return futures

    def _healthcheck_rollout_engines(self, timeout_seconds: int = 5) -> set[int]:
        failed_ranks = set()
        futures_to_rank = {}

        for rank, engine in list(self.rollout_engines.items()):
            try:
                future = engine.health.remote(timeout=float(timeout_seconds))
                futures_to_rank[future] = rank
            except Exception as e:
                logger.warning(f"RolloutEngine #{rank} failed to schedule healthcheck: {e}")
                failed_ranks.add(rank)

        if not futures_to_rank:
            return failed_ranks

        ready_futures, _ = ray.wait(
            list(futures_to_rank.keys()), timeout=timeout_seconds, num_returns=len(futures_to_rank)
        )

        for future in ready_futures:
            rank = futures_to_rank[future]
            try:
                ray.get(future)
            except Exception as e:
                logger.warning(f"RolloutEngine #{rank} healthcheck failed: {e}")
                failed_ranks.add(rank)

        for future in futures_to_rank:
            if future not in ready_futures:
                logger.warning(f"RolloutEngine #{futures_to_rank[future]} healthcheck timed out")
                failed_ranks.add(futures_to_rank[future])

        return failed_ranks

    def _remove_failed_engines(self, failed_ranks: set[int]) -> None:
        for rank in failed_ranks:
            if rank in self.rollout_engines:
                try:
                    ray.kill(self.rollout_engines[rank])
                except Exception as e:
                    logger.warning(f"Error killing failed RolloutEngine #{rank}: {e}")
                del self.rollout_engines[rank]

            for key in [str(rank), rank]:
                if key in self.rollout_topology:
                    del self.rollout_topology[key]

        if failed_ranks:
            logger.info(f"Removed {len(failed_ranks)} failed engines: {failed_ranks}")

    def _cleanup_rollout_engines(self) -> None:
        """Cleanup Ray actors for rollout communication."""
        for rank, actor in self.rollout_engines.items():
            try:
                ray.kill(actor)
                logger.debug(f"Killed RolloutEngine actor for rank {rank}")
            except Exception as e:
                logger.warning(f"Error killing RolloutEngine #{rank}: {e}")
        self.rollout_engines.clear()

    def _update_rollout_engines(self, max_retries: int = 30, retry_interval: float = 10.0):
        """Wait for rollout engines to be ready with retries.

        In fully-async mode, Rollout engines may still be initializing (loading model)
        when Actor attempts to sync weights. This method retries health checks until
        engines are ready, without modifying topology during retries.

        Args:
            max_retries: Maximum number of retry attempts (default: 30).
            retry_interval: Seconds to wait between retries (default: 2.0).

        Raises:
            RuntimeError: If no healthy engines are available after all retries.
        """
        if not self.rollout_topology:
            raise RuntimeError("No rollout engines configured")

        for attempt in range(max_retries):
            failed_ranks = self._healthcheck_rollout_engines()

            if not failed_ranks:
                if attempt > 0:
                    logger.info(f"Rollout engines ready after {attempt + 1} attempts")
                return

            logger.warning(
                f"Healthcheck failed for engines: {failed_ranks}, "
                f"retrying in {retry_interval}s (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(retry_interval)

        # All retries exhausted, remove failed engines and report error
        logger.error(f"Removing failed engines after {max_retries} retries: {failed_ranks}")
        self._remove_failed_engines(failed_ranks)

        raise RuntimeError(f"No healthy rollout engines available after {max_retries} retries")

    _MASTER_PORT_MIN = 11000
    _MASTER_PORT_MAX = 11999

    @staticmethod
    def _find_free_port_in_range(port_min: int, port_max: int) -> int:
        """Find a free port within [port_min, port_max] by attempting to bind.

        Raises RuntimeError if no free port is found in the range.
        """
        import random

        ports = list(range(port_min, port_max + 1))
        random.shuffle(ports)
        for port in ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No free port available in range [{port_min}, {port_max}]")

    def init_process_group_for_rollout(self, topology_data: Optional[Dict] = None) -> None:
        """Initialize PyTorch distributed process group for rollout
        communication."""

        if self.role_info is None:
            raise RuntimeError("Role info not set. Cannot initialize process group.")
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        if self._is_pp_src_rank:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            master_address = ray._private.services.get_node_ip_address()
            self._group_name = f"slime-pp_{pp_rank}"

            if topology_data is None:
                raise RuntimeError("topology_data is required for init_process_group_for_rollout")

            self.rollout_topology = topology_data.get("nodes", {}).get("rollout", {})

            self._create_rollout_engines(self.rollout_topology)
            self._update_rollout_engines()

            if self._model_update_groups is not None:
                try:
                    logger.info("Destroying old process group...")
                    destroy_payload = {"group_name": self._group_name}
                    futures = self._batch_request("/destroy_weights_update_group", destroy_payload)
                    dist.destroy_process_group(self._model_update_groups)
                    ray.get(futures)
                    self._model_update_groups = None
                    # Wait for NCCL socket ports to be released by the OS
                    time.sleep(2.0)
                except Exception as e:
                    logger.warning(f"Error destroying old process group: {e}")
                    self._model_update_groups = None

            default_gpus = self.args.rollout_num_gpus_per_engine
            cumulative_offset = 1
            rank_offsets: dict[int, int] = {}
            for rank, role_info in sorted(self.rollout_topology.items(), key=lambda kv: int(kv[0])):
                metadata = role_info.get("metadata") if isinstance(role_info, dict) else {}
                gpus_for_node = (metadata or {}).get("num_gpus_per_engine", default_gpus)
                rank_offsets[int(rank)] = cumulative_offset
                cumulative_offset += gpus_for_node
            world_size = cumulative_offset

            max_retries = 3
            last_error = None
            for attempt in range(1, max_retries + 1):
                master_port = self._find_free_port_in_range(self._MASTER_PORT_MIN, self._MASTER_PORT_MAX)

                init_payloads = {}
                for rank, role_info in self.rollout_topology.items():
                    init_payloads[int(rank)] = {
                        "master_address": master_address,
                        "master_port": master_port,
                        "rank_offset": rank_offsets[int(rank)],
                        "world_size": world_size,
                        "group_name": self._group_name,
                        "backend": self.backend_type,
                    }

                logger.info(
                    f"Sending init_weights_update_group to {len(self.rollout_topology)} rollout nodes "
                    f"(attempt {attempt}/{max_retries}, port={master_port})..."
                )
                futures = self._batch_request("/init_weights_update_group", init_payloads, get_rank=True)

                try:
                    self._model_update_groups = init_process_group(
                        backend=self.backend_type,
                        init_method=f"tcp://{master_address}:{master_port}",
                        world_size=world_size,
                        rank=0,
                        group_name=self._group_name,
                        timeout=timedelta(seconds=180),
                    )
                    ray.get(futures)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Failed to init process group for rollout (attempt {attempt}/{max_retries}, "
                        f"port={master_port}): {e}",
                        exc_info=(attempt == max_retries),
                    )
                    self._model_update_groups = None
                    try:
                        ray.get(futures, timeout=5)
                    except Exception:
                        pass
                    if attempt < max_retries:
                        time.sleep(5.0 * attempt)

            if last_error is not None:
                raise RuntimeError(
                    f"Failed to init process group for rollout after {max_retries} attempts"
                ) from last_error

    def init_process_groups_for_actor_fwd_ref(self, topology_data) -> None:
        """Initialize process groups used for actor -> actor_fwd weight sync.

        This sets up deterministic groups so actor (source) ranks broadcast
        updated weights and actor_fwd ranks receive them.
        """
        if self.role_info is None:
            raise RuntimeError("Role info not set. Cannot initialize process group.")

        # Determine if this rank is the PP source rank (for weight gathering)
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        global_rank = topology_data.get("global_rank")
        pp_groups = topology_data.get("pp_groups")
        world_size = topology_data.get("world_size", 1)

        if self.role_info.role_name == "actor":
            # Actor side: PP source rank (rank 0) manages connection for this PP stage

            if self._is_pp_src_rank:
                if self._model_update_groups_for_actor_fwd_ref is not None:
                    # TODO: This case should not happen in current design since we only init once, but if we want to support dynamic re-init in the future we need to destroy old groups before creating new ones
                    return

                    # dist.destroy_process_group(self._model_update_groups_for_actor_fwd_ref)
                    # time.sleep(1)  # Ensure all ranks have destroyed before creating new group
                # This actor's PP rank 0 establishes the master address/port for this PP group
                group_name = f"update_actor_pp_{pp_rank}"
                init_method = pp_groups.get(group_name)
                # Create the process group for this actor PP stage
                # Rank 0 is always the actor PP source rank
                self._model_update_groups_for_actor_fwd_ref = init_process_group(
                    backend=self.backend_type,
                    init_method=init_method,
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                    timeout=timedelta(seconds=180),
                )
                logger.info(
                    f"Actor PP{pp_rank} initialized process group {group_name} "
                    f"(world_size={world_size}) at {init_method}"
                )

        else:  # actor_fwd side
            # Actor_fwd side: Each rank joins all groups corresponding to actor PP stages
            # to receive weights for inference reference model updates
            if self._model_update_groups_for_actor_fwd_ref is not None:
                return
                # for group_name, group in self._model_update_groups_for_actor_fwd_ref.items():
                #     dist.destroy_process_group(group)
                # time.sleep(1)  # Ensure all ranks have destroyed before creating new group

            self._model_update_groups_for_actor_fwd_ref = {}

            # Calculate this actor_fwd rank's position in the cluster

            # For each actor PP stage, join the corresponding group in deterministic order
            # This ensures all ranks call init_process_group in the same order
            for group_name, init_method in pp_groups.items():
                # Actor_fwd ranks join with rank = actor_fwd_rank + 1 (rank 0 is actor)
                group = init_process_group(
                    backend=self.backend_type,
                    init_method=init_method,
                    world_size=world_size,
                    rank=global_rank,
                    group_name=group_name,
                    timeout=timedelta(seconds=180),
                )
                self._model_update_groups_for_actor_fwd_ref[group_name] = group
                logger.info(
                    f"Actor_fwd PP{pp_rank} joined group {group_name} as rank {global_rank} (world_size={world_size})"
                )

    @torch.no_grad()
    def update_weights_for_rollout(self, rollout_only=False, actor_fwd_only=False) -> None:
        """Update weights used by rollout nodes.

        Sequence: pause rollout generation, flush caches, gather and broadcast
        model parameters (non-expert then expert), then resume generation.
        """
        self.weight_version += 1

        if not actor_fwd_only:
            if dist.get_rank() == 0:
                # Pause generation on all rollout nodes
                logger.info("Pausing generation on all rollout nodes...")
                ray.get(self._batch_request("/pause_generation"))

                # Flush cache on all rollout nodes
                logger.info("Flushing cache on all rollout nodes...")
                for rank, engine in self.rollout_engines.items():
                    ray.get(engine.flush_cache.remote())

            dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        origin_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights") if self._is_pp_src_rank else None

        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name,
                param,
                converted_named_tensors,
                origin_named_tensors,
                buffer_size,
                rollout_only,
                actor_fwd_only,
                pbar=pbar,
            )

        if converted_named_tensors or origin_named_tensors:
            if not rollout_only:
                self._update_bucket_weights_from_distributed_for_actor_fwd_ref(origin_named_tensors)
            if not actor_fwd_only:
                self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
                converted_named_tensors.clear()
            origin_named_tensors.clear()
        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, rollout_only, actor_fwd_only, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(
                named_tensors, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only, pbar=pbar
            )
        dist.barrier(group=get_gloo_group())
        if not rollout_only:
            if dist.get_rank() == 0:
                payload = {
                    "names": [
                        "weight_updated_stop",
                    ],
                    "dtypes": [],
                    "shapes": [],
                    "group_name": "end",
                }
                logger.info("start post end send_weight_meta to actor fwd nodes...")
                response = self.http_client.post(
                    f"{self.coordinator_url}/send_weight_meta",
                    json=payload,
                )
                response.raise_for_status()
            dist.barrier(group=get_gloo_group())

        if not actor_fwd_only:
            if dist.get_rank() == 0:
                # Continue generation on all rollout nodes
                logger.info("Resuming generation on all rollout nodes...")
                self._batch_request("/continue_generation")
            dist.barrier(group=get_gloo_group())
            self._cleanup_rollout_engines()

        # Release fragmented CUDA reserved memory left behind by the
        # all_gather + HF-convert buffers that were allocated and freed
        # during the weight update loop.  Without this, the caching
        # allocator keeps large reserved blocks that are internally
        # fragmented, which can cause OOM when the optimizer later tries
        # to allocate contiguous Adam state buffers.
        device_utils.empty_cache()

    def _update_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        origin_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        rollout_only=False,
        actor_fwd_only=False,
        pbar: tqdm | None = None,
    ) -> int | None:
        """Gather parameter across TP, convert to HF format and buffer it.

        Returns updated buffer size on the source rank, otherwise None.
        """
        param = all_gather_param(self.args, name, param)
        if not self._is_pp_src_rank:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            if converted_named_tensors or origin_named_tensors:
                if not rollout_only:
                    self._update_bucket_weights_from_distributed_for_actor_fwd_ref(origin_named_tensors)
                if not actor_fwd_only:
                    self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
                    converted_named_tensors.clear()
                origin_named_tensors.clear()
                buffer_size = 0
        origin_named_tensors += [(name, param)]
        if not actor_fwd_only:
            if self._use_bridge:
                converted_named_tensors += self._bridge_converter.convert(name, param)
            else:
                converted_named_tensors += convert_to_hf(
                    self.args, self.model_name, name, param, self.quantization_config
                )
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        rollout_only: bool = False,
        actor_fwd_only: bool = False,
        pbar: tqdm | None = None,
    ) -> int:
        """Gather expert parameter across expert-parallel group and buffer it.

        HF conversion is deferred until bucket flush.
        """
        param = all_gather_param(self.args, name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            if named_tensors:
                self._update_expert_bucket_weights_from_distributed(
                    named_tensors, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only, pbar=pbar
                )
                buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        rollout_only: bool = False,
        actor_fwd_only: bool = False,
        pbar: tqdm | None = None,
    ) -> None:
        """Gather expert partitions, convert to HF format, and broadcast.

        Clears the input buffer when complete.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()

        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=self.device)
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return

        all_gathered_params = sum(all_gathered_params, [])
        if not rollout_only:
            self._update_bucket_weights_from_distributed_for_actor_fwd_ref(all_gathered_params)
        if not actor_fwd_only:
            converted_hf_tensors = []
            for name, param in all_gathered_params:
                if self._use_bridge:
                    converted_hf_tensors += self._bridge_converter.convert(name, param)
                else:
                    converted_hf_tensors += convert_to_hf(
                        self.args, self.model_name, name, param, self.quantization_config
                    )
            self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)
            converted_hf_tensors.clear()
        all_gathered_params.clear()

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Broadcast a bucket of converted tensors to rollout nodes.

        A remote lock is acquired to avoid NCCL deadlocks during concurrent
        broadcasts. This function blocks until all broadcasts and remote
        updates complete.
        """

        while not ray.get(self.lock.acquire.remote()):
            time.sleep(0.1)
        # Prepare payload for weight update
        weight_payload = {
            "names": [name for name, _ in converted_named_tensors],
            "dtypes": [str(param.dtype).replace("torch.", "") for _, param in converted_named_tensors],
            "shapes": [param.shape for _, param in converted_named_tensors],
            "group_name": self._group_name,
            "weight_version": str(self.weight_version),
            "flush_cache": False,
        }
        # Send weight update to all rollout nodes via Ray actors
        futures = self._batch_request("/update_weights_from_distributed", weight_payload)

        # Broadcast weights via PyTorch distributed
        handles = []
        for _, param in converted_named_tensors:
            handles.append(dist.broadcast(param.data, 0, group=self._model_update_groups, async_op=True))
        for handle in handles:
            handle.wait()
        ray.get(futures)  # Ensure remote update completes

        ray.get(self.lock.release.remote())
        if pbar is not None:
            pbar.update(1)

    def _update_bucket_weights_from_distributed_for_actor_fwd_ref(
        self, named_tensors: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Broadcast weights to actor_fwd reference models using dist.

        Metadata describing names/shapes/dtypes is sent to a coordinator so
        receiving nodes can allocate buffers, then weights are broadcast using
        the process group set up for actor_fwd reception.
        """
        # Prepare metadata for weight transfer
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        group_name = f"update_actor_pp_{pp_rank}"
        payload = {
            "names": [name for name, _ in named_tensors],
            "dtypes": [str(param.dtype).replace("torch.", "") for _, param in named_tensors],
            "shapes": [list(param.shape) for _, param in named_tensors],
            "group_name": group_name,
        }
        response = self.http_client.post(
            f"{self.coordinator_url}/send_weight_meta",
            json=payload,
        )
        response.raise_for_status()
        handles = []
        for _, param in named_tensors:
            handles.append(
                dist.broadcast(param.data, 0, group=self._model_update_groups_for_actor_fwd_ref, async_op=True)
            )
        for handle in handles:
            handle.wait()

    def recv_weight(self):
        """Poll coordinator for weight metadata and receive broadcasts.

        This method is intended for actor_fwd processes: it queries the
        coordinator for pending weight metadata, allocates receives, and then
        performs dist.broadcast to get actual tensors into local models. The
        loop ends when a special 'weight_updated_stop' marker is seen.
        """
        index = 0
        long_poll_wait_s = float(getattr(self.args, "dcs_recv_weight_meta_wait_timeout_s", 20.0))
        # Ensure read timeout is longer than long-poll wait duration.
        recv_timeout = httpx.Timeout(connect=5.0, read=max(long_poll_wait_s + 5.0, 10.0), write=30.0, pool=30.0)
        while True:
            try:
                response = self.http_client.get(
                    f"{self.coordinator_url}/recv_weight_meta",
                    params={"index": index, "wait_timeout_s": long_poll_wait_s},
                    timeout=recv_timeout,
                )
            except httpx.ReadTimeout:
                # Long-poll timed out without new metadata; continue waiting.
                continue
            response.raise_for_status()
            data = response.json()
            if not data:
                continue
            for metadata in data:
                index += 1
                names = metadata.get("names")
                # termination marker
                if names and names[0] == "weight_updated_stop":
                    dist.barrier(get_gloo_group())
                    if dist.get_rank() == 0:
                        response = self.http_client.get(f"{self.coordinator_url}/clear_weight_meta")
                        response.raise_for_status()

                    logger.info("Received final weight update marker for actor_fwd nodes")
                    return

                dtypes = metadata.get("dtypes")
                shapes = metadata.get("shapes")
                group_name = metadata.get("group_name")
                weights: list[tuple[str, torch.Tensor]] = []
                handles = []
                for name, dtype, shape in zip(names, dtypes, shapes):
                    target_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                    weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                    handles.append(
                        torch.distributed.broadcast(
                            weight,
                            src=0,
                            group=self._model_update_groups_for_actor_fwd_ref[group_name],
                            async_op=True,
                        )
                    )
                    weights.append((name, weight))
                for handle in handles:
                    handle.wait()

                load_weight(self.args, self.model, weights)


@ray.remote
class RolloutEngine:
    """Ray Actor for handling HTTP requests to rollout nodes.

    Encapsulates HTTP communication with a specific rollout endpoint.
    """

    def __init__(self, rank: int, node_info: Dict[str, Any]):
        """Initialize RolloutEngine actor.

        Args:
            rank: Rank/index of this rollout node
            node_info: Dict with 'ip' and 'port' keys
        """
        self.rank = rank
        self.node_info = node_info
        self.base_url = f"http://{node_info['ip']}:{node_info['port']}"
        logger.info(f"RolloutEngine #{self.rank} initialized for {self.base_url}")

    def health(self, timeout: float = 5.0) -> bool:
        response = requests.get(f"{self.base_url}/health_generate", timeout=timeout)
        response.raise_for_status()
        return True

    def make_request(self, endpoint: str, payload: Optional[Dict] = None) -> Any:
        """Send a synchronous HTTP POST to the rollout node and return JSON.

        Args:
            endpoint: Path on the node (e.g. '/init_weights_update_group').
            payload: Optional JSON payload.

        Returns:
            Parsed JSON response from the remote node.
        """
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def flush_cache(self) -> None:
        """Poll the remote server until its cache is flushed or timeout.

        Retries for a short period and raises on timeout.
        """
        # flush_cache may return non-200 while there are pending requests
        url = f"{self.base_url}/flush_cache"
        for _ in range(60):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    break
            except NewConnectionError:
                raise
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")
