# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Manager for Generative Reward Model Service.

This module implements a simplified manager for genRM engines, similar to
RolloutManager but focused on reward evaluation only.
"""

import logging

import ray
import requests
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.backends.sglang.sglang_engine import GenRMEngine
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock
from relax.utils.http_utils import init_http_client
from relax.utils.logging_utils import get_logger


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = get_logger(__name__)

# A GenRM engine process can die on its own (SGLang's scheduler watchdog
# SIGQUITs the server after a CUDA-level hang). The next call into it then
# raises one of these. Everything else is a real bug and must propagate.
#   - ConnectionError / TimeoutError: raised by GenRMEngine.release_memory_occupation
#     when the drain loop hits its dead-server fast-fail or its deadline.
#   - requests.exceptions.{ConnectionError,Timeout}: raised by _make_request, i.e.
#     the resume_memory_occupation path. These are OSError subclasses but NOT
#     builtin ConnectionError/TimeoutError, so they must be listed explicitly --
#     otherwise an engine that died during the offloaded window (only observable
#     at onload) escalates to the global restart this whole mechanism avoids.
#   - RayActorError: the Ray actor itself is gone.
# ray.get re-raises as a class inheriting from BOTH RayTaskError and the
# original cause (ray/exceptions.py::as_instanceof_cause), so isinstance works.
_ENGINE_DEAD_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ray.exceptions.RayActorError,
)

# Rebuilding one engine is ~1.5 min (weight load + cuda graph capture). Bound it
# so a dead *node* -- whose placement-group bundle can never be filled -- degrades
# to "run with N-1 engines" instead of hanging the training step forever.
_ENGINE_REBUILD_TIMEOUT_S = 900.0
_ENGINE_SHUTDOWN_TIMEOUT_S = 60.0


def _is_engine_dead(exc: BaseException) -> bool:
    return isinstance(exc, _ENGINE_DEAD_EXCEPTIONS)


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
        # Track memory-occupation state so repeated onload/offload calls become
        # safe no-ops. Engines start onloaded; the caller (placement_group.py)
        # may immediately offload if offload_rollout is set.
        self._onloaded = True

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

        Also the recovery point for engines that died since the last step: a
        freshly built engine comes up onloaded, which is exactly the state this
        phase wants, and unlike offload() this call is not followed by a gloo
        barrier across the actor ranks (megatron/actor.py:1696 vs :619).

        Args:
            tags: Optional list of tags to specify which resources to load.
                  Available tags: GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE,
                                 GPU_MEMORY_TYPE_CUDA_GRAPH
        """
        rebuilt = self.recover()
        if self._onloaded and tags is None:
            logger.info("GenRM engines already onloaded; skipping")
            return
        logger.info(f"GenRM engines onload started with tags={tags}")
        # Engines rebuilt just above are already onloaded -- resuming them again
        # would be a double-resume, so only touch the ones that survived.
        dead = self._fanout("resume_memory_occupation", skip_ranks=rebuilt, tags=tags)
        if dead:
            # An engine that died while we were offloaded is only discovered
            # here (offload() short-circuits when already offloaded), so it
            # missed the recover() above. Rebuild now rather than leaving the
            # judge a man down for the whole next rollout phase.
            self._retire_engines(dead)
            self.recover()
        self._onloaded = True
        logger.info("GenRM engines onload completed")

    def offload(self):
        """Offload genRM model weights from GPU to free memory.

        Dead engines are retired here but NOT rebuilt: this runs on actor rank
        0 with the other ranks waiting on a gloo barrier, so we keep it short
        and leave the ~1.5 min rebuild to the next onload().
        """
        if not self._onloaded:
            logger.info("GenRM engines already offloaded; skipping")
            return
        logger.info("GenRM engines offload started")
        dead = self._fanout("release_memory_occupation")
        self._retire_engines(dead)
        # Unconditional: the surviving engines did release, so the manager must
        # not claim to still be onloaded just because one engine died.
        self._onloaded = False
        logger.info(f"GenRM engines offload completed (retired {len(dead)} dead)")

    def _fanout(self, method: str, *, skip_ranks: set[int] | None = None, **kwargs) -> list[int]:
        """Call ``method`` on every live engine; return the ranks that are
        dead.

        Per-handle ray.get rather than one ray.get over the list: the batched
        form aborts on the first failure and loses which engine raised.
        """
        skip = skip_ranks or set()
        handles = {}
        for rank in range(0, len(self.all_genrm_engines), self.nodes_per_engine):
            engine = self.all_genrm_engines[rank]
            if engine is None or rank in skip:
                continue
            handles[rank] = getattr(engine, method).remote(**kwargs)

        dead = []
        for rank, handle in handles.items():
            try:
                ray.get(handle)
            except Exception as exc:
                if not _is_engine_dead(exc):
                    raise
                logger.warning(f"GenRM engine rank={rank} died during {method}: {exc}")
                dead.append(rank)
        return dead

    def _retire_engines(self, ranks: list[int]) -> None:
        """Tear down dead engines and null their slots so recover() rebuilds
        them."""
        for rank in ranks:
            for i in range(rank, rank + self.nodes_per_engine):
                engine = self.all_genrm_engines[i]
                if engine is None:
                    continue
                try:
                    # shutdown() kill_process_tree's the SGLang server. Must run
                    # before ray.kill or the scheduler subprocesses are orphaned
                    # and keep holding GPU memory, so the rebuild can't fit.
                    ray.get(engine.shutdown.remote(), timeout=_ENGINE_SHUTDOWN_TIMEOUT_S)
                except Exception as exc:
                    logger.warning(f"GenRM engine rank={i} shutdown failed (killing anyway): {exc}")
                try:
                    ray.kill(engine)
                except Exception as exc:
                    logger.warning(f"GenRM engine rank={i} ray.kill failed: {exc}")
                self.all_genrm_engines[i] = None
                logger.info(f"GenRM engine rank={i} retired")

    def recover(self) -> set[int]:
        """Rebuild engines whose slot is None. Returns the ranks rebuilt.

        Mirrors RolloutServer.recover. ``init_genrm_engines`` already skips
        non-None slots, so it rebuilds exactly the holes, reusing the same
        placement-group bundles and probing fresh ports (surviving engines'
        ports are bound, so they're skipped).
        """
        dead = [i for i, engine in enumerate(self.all_genrm_engines) if engine is None]
        if not dead:
            return set()

        logger.info(f"GenRM recovering {len(dead)} engine(s): ranks={dead}")
        try:
            init_genrm_engines(self.args, self.pg, self.all_genrm_engines, self._engine_addr_and_ports)
        except Exception as exc:
            logger.exception(f"GenRM engine rebuild failed for ranks={dead}: {exc}")

        rebuilt = {i for i in dead if self.all_genrm_engines[i] is not None}
        still_dead = [i for i in dead if i not in rebuilt]
        if still_dead:
            # Degrade rather than escalate: judging on N-1 engines beats a
            # global restart. Only a total wipeout is unrecoverable here.
            if all(engine is None for engine in self.all_genrm_engines):
                raise RuntimeError(f"All GenRM engines are dead and could not be rebuilt (ranks={still_dead})")
            logger.error(f"GenRM engines still dead after recovery, continuing degraded: ranks={still_dead}")
        if rebuilt:
            logger.info(f"GenRM recovered engine ranks={sorted(rebuilt)}")
        return rebuilt

    def is_onloaded(self):
        return self._onloaded

    def get_engine_hosts_ports(self):
        """Return a list of (host, port) tuples for each live genRM engine.

        This is used by the GenRM service to send HTTP generation requests
        directly to the underlying SGLang servers.

        The host/port information is captured during engine initialization
        from the addr_and_ports dict passed to ``init_genrm_engines``.

        ``all_genrm_engines`` holds one entry per *node*, so a multi-node engine
        (genrm_num_gpus_per_engine > num_gpus_per_node) occupies
        ``nodes_per_engine`` consecutive ranks. Only node_rank 0 of each group
        runs the SGLang HTTP server -- the followers are compute-only workers
        and answer /generate with 404 -- so stride the same way the
        ``genrm_engines`` property does and return head nodes only.

        The list is also compacted over dead engines, so callers must swap it
        and any derived round-robin state together.
        """
        results = []
        for rank in range(0, len(self.all_genrm_engines), self.nodes_per_engine):
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
            # See rollout.py: recent SGLang reads SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK
            # (default True) and the deprecation shim value-copies SGL_DISABLE_* into it,
            # so the old DISABLE vars re-enable the check. Set ENABLE=false directly.
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            # NOTE: disable custom all-reduce-v2, same as rollout.py — avoids
            # custom_all_reduce.cuh:37: CUDA error: invalid argument during CUDA graph capture.
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "0",
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
    try:
        # Bounded: a bundle on a dead node never schedules, and an unbounded
        # ray.get would block the training step forever.
        ray.get(init_handles, timeout=_ENGINE_REBUILD_TIMEOUT_S)
    except Exception:
        # The slots were filled with actor handles before init ran, so on any
        # failure they hold engines whose SGLang server never came up. Null them
        # so the caller sees them as still-dead instead of silently healthy.
        for rank, engine in genrm_engines:
            try:
                ray.kill(engine)
            except Exception:
                pass
            all_genrm_engines[rank] = None
        raise

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
