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
import re
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

from relax.backends.megatron.misc_utils import strip_param_name_prefix
from relax.backends.megatron.weight_conversion import convert_to_hf
from relax.backends.megatron.weight_conversion.processors import quantize_params, remove_padding
from relax.backends.megatron.weight_update.common import all_gather_param, named_params_and_buffers
from relax.distributed.checkpoint_service.backends.base import CommBackend, TensorFusion
from relax.distributed.checkpoint_service.config import BackendType, RoleInfo
from relax.distributed.checkpoint_service.utils import load_weight
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
        self.device = next(model[0].parameters()).device if model else torch.cuda.current_device()

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
        torch.cuda.set_device(self.device)

        # Bridge-based HF weight converter (lazy-initialized on first use)
        self._use_bridge = getattr(args, "megatron_to_hf_mode", None) == "bridge"
        self._bridge_task_map: Optional[Dict[str, Any]] = None  # global_param_name -> WeightConversionTask
        self._bridge_mapping_registry = None  # MegatronMappingRegistry for dynamic lookups

    def _init_bridge_tasks(self) -> None:
        """Lazily initialize Bridge conversion tasks and build a lookup table.

        Builds a mapping from global_param_name (unwrapped, e.g.
        ``decoder.layers.0.self_attention.linear_qkv.weight``) to the
        corresponding ``WeightConversionTask``.  Only tasks that belong to the
        current PP rank (i.e. ``task.param_weight is not None``) are indexed.

        After building the task map, eagerly initializes any lazily-created
        inner mappings (e.g. ``AutoMapping._mapping``) so that
        ``_collect_all_mappings`` can discover and patch them later.

        When embeddings are tied, Bridge's ``build_conversion_tasks`` filters
        out ``output_layer`` from its task list.  However,
        ``named_params_and_buffers`` still yields ``output_layer.weight`` on
        the last PP stage.  We detect such missing parameters and supplement
        the task map using the mapping registry so that every local parameter
        has a corresponding Bridge task.
        """
        if self._bridge_task_map is not None:
            return

        from megatron.bridge import AutoBridge
        from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        from relax.utils.megatron_bridge_utils import patch_megatron_model

        bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        with patch_megatron_model(self.model):
            tasks = bridge.get_conversion_tasks(self.model)

        self._bridge_task_map = {}
        for task in tasks:
            if task.param_weight is not None:
                self._bridge_task_map[task.global_param_name] = task

        # Supplement tasks for local parameters that Bridge filtered out
        # (e.g. ``output_layer`` when embeddings are tied).  Walk the local
        # model parameters and, for any that are missing from the task map,
        # look up the mapping from the registry and create a synthetic task.
        self._bridge_mapping_registry = bridge._model_bridge.mapping_registry()
        mapping_registry = self._bridge_mapping_registry
        for name, param in named_params_and_buffers(self.args, self.model):
            global_name = strip_param_name_prefix(name)
            if global_name not in self._bridge_task_map:
                mapping = mapping_registry.megatron_to_hf_lookup(global_name)
                if mapping is not None:
                    self._bridge_task_map[global_name] = WeightConversionTask(
                        param_name=global_name,
                        global_param_name=global_name,
                        mapping=mapping,
                        megatron_module=None,
                        param_weight=param,
                    )

        # Eagerly initialize inner mappings of AutoMapping instances.
        # AutoMapping lazily creates a delegate ``_mapping`` (ColumnParallel /
        # RowParallel / Replicated) on first use.  That delegate has its own
        # process groups obtained from ``mpu`` at construction time.  We must
        # trigger this initialization now so that ``_collect_all_mappings`` can
        # find and patch them before ``megatron_to_hf`` is called.
        for task in self._bridge_task_map.values():
            mapping = task.mapping
            if isinstance(mapping, AutoMapping) and mapping._mapping is None:
                if task.megatron_module is not None:
                    mapping._detected_type = mapping._detect_parallelism_type(task.megatron_module)
                    mapping._mapping = mapping._get_or_create_mapping(mapping._detected_type)
                else:
                    # Supplementary tasks (e.g. tied ``output_layer``) have no
                    # ``megatron_module``, so we cannot detect parallelism type.
                    # These parameters are always replicated (that's why Bridge
                    # filtered them out in the first place).
                    mapping._detected_type = "replicated"
                    mapping._mapping = mapping._get_or_create_mapping("replicated")
            # Also handle AutoMapping nested inside _tp_mapping (e.g. QKVMapping)
            inner_tp = getattr(mapping, "_tp_mapping", None)
            if isinstance(inner_tp, AutoMapping) and inner_tp._mapping is None:
                if task.megatron_module is not None:
                    inner_tp._detected_type = inner_tp._detect_parallelism_type(task.megatron_module)
                    inner_tp._mapping = inner_tp._get_or_create_mapping(inner_tp._detected_type)

        logger.info(f"Bridge task map initialized with {len(self._bridge_task_map)} local tasks")

    @staticmethod
    def _collect_all_mappings(mapping) -> list:
        """Recursively collect a mapping and all its inner sub-mappings.

        Bridge mapping objects may contain inner attributes that are themselves
        ``MegatronParamMapping`` instances with their own process groups.
        Known examples:
        - ``AutoMapping._mapping`` (lazily-created delegate)
        - ``QKVMapping._tp_mapping`` / ``MambaInProjMapping._tp_mapping``
        - ``Qwen3VLMoEGateUpProjMapping._gated_mapping``

        Rather than hard-coding attribute names, we scan all instance
        attributes of each mapping to discover sub-mappings generically.
        This ensures new model-specific wrappers are handled automatically.
        """
        from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

        result: list = []
        visited: set = set()
        stack = [mapping]
        while stack:
            m = stack.pop()
            if id(m) in visited:
                continue
            visited.add(id(m))
            if isinstance(m, MegatronParamMapping):
                result.append(m)
                # Scan all instance attributes for nested MegatronParamMapping
                for attr_val in vars(m).values():
                    if isinstance(attr_val, MegatronParamMapping):
                        stack.append(attr_val)
        return result

    def _convert_to_hf_bridge(self, name: str, param: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Convert a single TP-gathered parameter to HF format using Bridge.

        This is a drop-in replacement for ``convert_to_hf()`` that uses
        megatron-bridge's mapping logic instead of hand-written per-model
        converters.  All collective communication (PP broadcast, TP gather,
        EP gather) is disabled by temporarily setting the process groups to
        ``None``, because the caller has already performed TP gather via
        ``all_gather_param`` and this method runs only on ``_is_pp_src_rank``.

        Args:
            name: Global parameter name with ``module.module.`` prefix
                  (as yielded by ``named_params_and_buffers``).
            param: The TP-gathered parameter tensor.

        Returns:
            List of ``(hf_name, hf_tensor)`` tuples, same interface as
            ``convert_to_hf``.
        """
        self._init_bridge_tasks()

        # Strip the ``module.module.`` prefix to get Bridge's global_param_name
        global_name = strip_param_name_prefix(name)
        # named_params_and_buffers yields names like "vp_stages.0.decoder.layers.0...."
        # Bridge's global_param_name is "decoder.layers.0...."
        # Remove the "vp_stages.{N}." prefix if present
        if global_name.startswith("vp_stages."):
            # "vp_stages.0.decoder..." -> "decoder..."
            parts = global_name.split(".", 2)
            if len(parts) >= 3:
                global_name = parts[2]

        task = self._bridge_task_map.get(global_name)

        # When EP > 1, ``_update_expert_bucket_weights_from_distributed``
        # gathers expert params from ALL EP ranks.  The task map only contains
        # the current EP rank's experts, so params from other EP ranks will be
        # missing.  Dynamically look them up via the mapping registry, create a
        # synthetic task, eagerly initialize its inner mapping, and cache it.
        if task is None:
            from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
            from megatron.bridge.models.conversion.param_mapping import AutoMapping

            mapping = self._bridge_mapping_registry.megatron_to_hf_lookup(global_name)
            assert mapping is not None, (
                f"Bridge mapping registry has no entry for '{global_name}'. "
                f"Available task map keys: {list(self._bridge_task_map.keys())[:10]}..."
            )
            # Do NOT pass param_weight=param here — the param_weight field is
            # only used for HF→Megatron (load) direction, not Megatron→HF
            # (export).  Storing the EP-gathered tensor in the cached task
            # would prevent ~20 GB from being freed on _is_pp_src_rank for
            # MoE models with many experts.
            task = WeightConversionTask(
                param_name=global_name,
                global_param_name=global_name,
                mapping=mapping,
                megatron_module=None,
                param_weight=None,
            )
            # Eagerly initialize AutoMapping inner delegate (same logic as
            # ``_init_bridge_tasks``).  Since ``megatron_module`` is None and
            # all groups will be patched to None anyway, default to replicated.
            if isinstance(mapping, AutoMapping) and mapping._mapping is None:
                mapping._detected_type = "replicated"
                mapping._mapping = mapping._get_or_create_mapping("replicated")
            inner_tp = getattr(mapping, "_tp_mapping", None)
            if isinstance(inner_tp, AutoMapping) and inner_tp._mapping is None:
                inner_tp._detected_type = "replicated"
                inner_tp._mapping = inner_tp._get_or_create_mapping("replicated")
            # Cache for future iterations (task has no tensor references)
            self._bridge_task_map[global_name] = task

        mapping = task.mapping

        # Collect the top-level mapping **and** any inner sub-mappings
        # (e.g. AutoMapping._mapping, QKVMapping._tp_mapping) so that we
        # disable collective ops on every level of the delegation chain.
        all_mappings = self._collect_all_mappings(mapping)

        # Save original process groups for every mapping
        saved_groups: list[tuple] = []
        for m in all_mappings:
            saved_groups.append((m.pp_group, m._tp_group, m._etp_group, m.ep_group))

        # For expert parameters, ``megatron_to_hf`` calls
        # ``gather_from_ep_ranks`` when ``is_expert`` is True.  That method
        # needs ``megatron_module`` to compute ``num_experts_per_rank``, but
        # our synthetic tasks have ``megatron_module = None``.  Since we have
        # already performed EP gather externally and set ``ep_group = None``
        # (``ep_size == 1``), the EP gather inside Bridge is redundant.
        #
        # We must monkey-patch ``gather_from_ep_ranks`` on the concrete class
        # of **every** mapping in the delegation chain, not just the top-level
        # one.  For example, ``AutoMapping.megatron_to_hf`` delegates to
        # ``self._mapping.megatron_to_hf`` (a ``RowParallelMapping``), which
        # calls ``self.gather_from_ep_ranks`` on the *inner* mapping instance.
        # If we only patch the outer ``AutoMapping`` class, the inner
        # ``RowParallelMapping`` class still has the original method.
        #
        # ``gather_from_ep_ranks`` is only defined on the base
        # ``MegatronParamMapping`` class and no subclass overrides it, so
        # deleting the monkey-patch in ``finally`` restores the inherited
        # version via MRO.
        patched_classes: set[type] = set()

        def _noop_gather_from_ep_ranks(self_m, megatron_weights, megatron_module, hf_param_name):
            return {str(hf_param_name): megatron_weights}

        try:
            # Disable all collective ops on every mapping: set groups to None
            # so that pp_size/tp_size/ep_size all return 1 (via get_pg_size(None) == 1)
            for m in all_mappings:
                m.pp_group = None
                m._tp_group = None
                m._etp_group = None
                m.ep_group = None

            # Patch gather_from_ep_ranks on every unique mapping class in the
            # delegation chain so that inner delegates also get the no-op.
            for m in all_mappings:
                cls = type(m)
                if cls not in patched_classes:
                    cls.gather_from_ep_ranks = _noop_gather_from_ep_ranks
                    patched_classes.add(cls)

            # Apply remove_padding before conversion (same as convert_to_hf)
            param = remove_padding(name, param, self.args.vocab_size)

            # Call Bridge's megatron_to_hf — now a pure local format conversion.
            # With all groups set to None, tp_size/pp_size/ep_size are all 1,
            # so no collective communication occurs and the tensor is treated
            # as already gathered.
            converted_dict = mapping.megatron_to_hf(param, task.megatron_module)
        finally:
            # Restore original process groups for every mapping
            for m, (pp, tp, etp, ep) in zip(all_mappings, saved_groups):
                m.pp_group = pp
                m._tp_group = tp
                m._etp_group = etp
                m.ep_group = ep
            # Remove the monkey-patch from every patched class; the inherited
            # base-class method is automatically restored via MRO.
            for cls in patched_classes:
                if "gather_from_ep_ranks" in cls.__dict__:
                    del cls.gather_from_ep_ranks

        # Convert Dict[str, Tensor] -> List[Tuple[str, Tensor]]
        converted_named_tensors = list(converted_dict.items())

        # ── Post-process expert weights ──────────────────────────────────
        # Bridge's ExpertMLPGateUpProjMapping and ExpertMLPDownProjMapping
        # (used by Qwen3-VL MoE) apply an extra ``.transpose(-1, -2)`` in
        # their ``megatron_to_hf`` methods, assuming Megatron stores expert
        # weights in column-major order.  However, the raw ``convert_to_hf``
        # does NOT transpose expert weights — Megatron's expert weights are
        # already in the same layout as HF.  We must undo Bridge's transpose
        # to match the format that SGLang / ``convert_to_hf`` expects.
        #
        # Additionally, Bridge outputs fused names without expert_id:
        #   - ``...experts.gate_up_proj`` with shape [2, D_out, D_in]
        #   - ``...experts.down_proj`` with shape [D_in, D_out]
        # We split into per-expert format with correct names and shapes:
        #   - ``...experts.{E}.gate_proj.weight`` [H, D]
        #   - ``...experts.{E}.up_proj.weight``   [H, D]
        #   - ``...experts.{E}.down_proj.weight``  [D, H]
        expert_id_match = re.search(r"weight(\d+)", global_name)
        if expert_id_match is not None:
            expert_id = expert_id_match.group(1)
            postprocessed: list[tuple[str, torch.Tensor]] = []
            for hf_name, tensor in converted_named_tensors:
                if hf_name.endswith(".experts.gate_up_proj"):
                    # Bridge output: [2, D_out, D_in] (transposed by Bridge)
                    # Undo transpose on each slice: [D_out, D_in] -> [D_in, D_out]
                    gate_tensor = tensor[0].transpose(-1, -2).contiguous()
                    up_tensor = tensor[1].transpose(-1, -2).contiguous()
                    base = hf_name[: -len(".gate_up_proj")]
                    postprocessed.append((f"{base}.{expert_id}.gate_proj.weight", gate_tensor))
                    postprocessed.append((f"{base}.{expert_id}.up_proj.weight", up_tensor))
                elif hf_name.endswith(".experts.down_proj"):
                    # Bridge output: transposed — undo to match raw convert_to_hf
                    base = hf_name[: -len(".down_proj")]
                    postprocessed.append(
                        (f"{base}.{expert_id}.down_proj.weight", tensor.transpose(-1, -2).contiguous())
                    )
                else:
                    postprocessed.append((hf_name, tensor))
            converted_named_tensors = postprocessed

        # Apply quantization (same as convert_to_hf)
        return quantize_params(self.args, name, converted_named_tensors, self.quantization_config)

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

    def _update_rollout_engines(self):
        failed_ranks = self._healthcheck_rollout_engines()
        if failed_ranks:
            logger.warning(f"Healthcheck failed for engines: {failed_ranks}, removing and recreating...")
            self._remove_failed_engines(failed_ranks)
            self._create_rollout_engines(self.rollout_topology)

        if not self.rollout_topology:
            raise RuntimeError("No healthy rollout engines available after healthcheck")

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

            master_port = self._find_free_port_in_range(self._MASTER_PORT_MIN, self._MASTER_PORT_MAX)

            # Prepare init payloads for each rollout node
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

            logger.info(f"Sending init_weights_update_group to {len(self.rollout_topology)} rollout nodes...")
            futures = self._batch_request("/init_weights_update_group", init_payloads, get_rank=True)

            self._model_update_groups = init_process_group(
                backend=self.backend_type,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
                timeout=timedelta(seconds=180),
            )
            ray.get(futures)

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
        torch.cuda.empty_cache()

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
                converted_named_tensors += self._convert_to_hf_bridge(name, param)
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
                    converted_hf_tensors += self._convert_to_hf_bridge(name, param)
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
