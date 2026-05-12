# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Manager for Generative Reward Model Service.

This module implements a simplified manager for genRM engines, similar to
RolloutManager but focused on reward evaluation only.
"""

import logging

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.backends.sglang.sglang_engine import GenRMEngine
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock
from relax.utils.http_utils import init_http_client
from relax.utils.logging_utils import get_logger


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = get_logger(__name__)


@ray.remote
class GenRMManager:
    """Manager for GenRM engines.

    This is a simplified version of RolloutManager focused on:
    - Initializing genRM engines
    - Health checking
    - Onload/offload operations
    """

    def __init__(self, args, pg):
        self.args = args
        self.pg = pg
        init_http_client(args)

        if self.args.debug_train_only:
            self.all_genrm_engines = []
        else:
            num_gpu_per_engine = min(args.genrm_num_gpus_per_engine, args.num_gpus_per_node)
            num_engines = args.genrm_num_gpus // num_gpu_per_engine
            self.all_genrm_engines = [None] * num_engines

        self._engine_addr_and_ports = {}  # rank -> {host, port, ...}
        self.num_new_engines = init_genrm_engines(args, pg, self.all_genrm_engines, self._engine_addr_and_ports)
        self.nodes_per_engine = max(1, args.genrm_num_gpus_per_engine // args.num_gpus_per_node)
        self.genrm_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

    @property
    def genrm_engines(self):
        """Return the head engine of each multi-node engine."""
        return self.all_genrm_engines[:: self.nodes_per_engine]

    def get_genrm_engines_and_lock(self):
        return self.genrm_engines, self.genrm_engine_lock, self.num_new_engines

    def health_check(self):
        """Perform health check on all engines."""
        health_results = []
        for engine in self.genrm_engines:
            if engine is not None:
                try:
                    result = ray.get(engine.health_generate.remote(), timeout=5.0)
                    health_results.append(result)
                except Exception as e:
                    logger.warning(f"GenRM engine health check failed: {e}")
                    health_results.append(False)
            else:
                health_results.append(False)
        return all(health_results)

    def onload(self, tags: list[str] | None = None):
        """Load genRM model weights to GPU.

        Args:
            tags: Optional list of tags to specify which resources to load.
                  Available tags: GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE,
                                 GPU_MEMORY_TYPE_CUDA_GRAPH
        """
        logger.info(f"GenRM engines onload started with tags={tags}")
        onload_handles = [
            engine.resume_memory_occupation.remote(tags=tags) for engine in self.genrm_engines if engine is not None
        ]
        if onload_handles:
            ray.get(onload_handles)
        logger.info("GenRM engines onload completed")

    def offload(self):
        """Offload genRM model weights from GPU to free memory."""
        logger.info("GenRM engines offload started")
        offload_handles = [
            engine.release_memory_occupation.remote() for engine in self.genrm_engines if engine is not None
        ]
        if offload_handles:
            ray.get(offload_handles)
        logger.info("GenRM engines offload completed")

    def get_engine_hosts_ports(self):
        """Return a list of (host, port) tuples for each live genRM engine.

        This is used by the GenRM service to send HTTP generation requests
        directly to the underlying SGLang servers.

        The host/port information is captured during engine initialization
        from the addr_and_ports dict passed to ``init_genrm_engines``.
        """
        results = []
        for rank in range(len(self.all_genrm_engines)):
            engine = self.all_genrm_engines[rank]
            if engine is not None and rank in self._engine_addr_and_ports:
                info = self._engine_addr_and_ports[rank]
                results.append((info["host"], info["port"]))
        return results


def init_genrm_engines(args, pg, all_genrm_engines, engine_addr_and_ports=None):
    """Initialize genRM engines on the placement group.

    Similar to init_rollout_engines but for genRM.

    Args:
        args: Argument namespace containing genRM configuration
        pg: Placement group tuple (pg, bundle_indices, gpu_ids)
        all_genrm_engines: List to store initialized engines

    Returns:
        Number of newly initialized engines
    """
    num_gpu_per_engine = min(args.genrm_num_gpus_per_engine, args.num_gpus_per_node)
    num_engines = args.genrm_num_gpus // num_gpu_per_engine
    assert len(all_genrm_engines) == num_engines

    pg, reordered_bundle_indices, reordered_gpu_ids = pg

    GenRMRayActor = ray.remote(GenRMEngine)

    genrm_engines = []
    for i in range(num_engines):
        if all_genrm_engines[i] is not None:
            continue

        # Lower default fractional-GPU footprint when sharing bundles with rollout
        # (rollout uses 0.2 per actor; 0.2 + 0.2 risks Ray scheduler rejection).
        shared_with_rollout = getattr(args, "_genrm_colocate_with_rollout", False)
        default_ray_num_gpus = 0.1 if shared_with_rollout else 0.2
        num_gpus = getattr(args, "genrm_ray_num_gpus", default_ray_num_gpus)
        num_cpus = num_gpus

        gpu_idx = i * num_gpu_per_engine

        if not args.fully_async and not shared_with_rollout:
            gpu_idx += args.rollout_num_gpus

        # Get the base GPU ID from placement group
        base_gpu_id = int(reordered_gpu_ids[gpu_idx])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[gpu_idx],
        )

        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
            "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
        }
        if getattr(args, "fp16", False):
            env_vars["SGLANG_MAMBA_CONV_DTYPE"] = "float16"

        genrm_engine = GenRMRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={
                "env_vars": env_vars,
            },
        ).remote(args, rank=i, worker_type="regular", base_gpu_id=base_gpu_id)

        genrm_engines.append((i, genrm_engine))
        all_genrm_engines[i] = genrm_engine

    num_new_engines = len(genrm_engines)

    if num_new_engines == 0:
        return num_new_engines

    # Allocate addresses and ports for genRM engines
    addr_and_ports = _allocate_genrm_engine_addr_and_ports(
        args=args, num_engines=num_engines, genrm_engines=genrm_engines
    )

    # Store addr/port info so GenRMManager can expose them
    if engine_addr_and_ports is not None:
        for rank, _ in genrm_engines:
            engine_addr_and_ports[rank] = addr_and_ports[rank]

    # Initialize engines
    init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in genrm_engines]
    ray.get(init_handles)

    return num_new_engines


def _allocate_genrm_engine_addr_and_ports(*, args, num_engines, genrm_engines):
    """Allocate network addresses and ports for genRM engines.

    Similar to _allocate_rollout_engine_addr_and_ports_normal but for genRM.
    """
    num_engines_per_node = max(1, min(args.num_gpus_per_node, args.genrm_num_gpus) // args.genrm_num_gpus_per_engine)
    addr_and_ports = [{} for _ in range(num_engines)]

    visited_nodes = set()
    for rank, engine in genrm_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            start_port = 16000  # Use different port range than rollout (15000)

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

        if args.genrm_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.genrm_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # First node in the engine, allocate dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in genrm_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"GenRM engine {i} {key} is not set."
        logger.info(f"Ports for genRM engine {i}: {addr_and_ports[i]}")

    return addr_and_ports
