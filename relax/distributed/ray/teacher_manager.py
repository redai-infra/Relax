# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os

import ray
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.backends.sglang.sglang_engine import SGLangEngine
from relax.core.service import create_placement_group
from relax.distributed.ray.rollout import _allocate_rollout_engine_addr_and_ports_normal
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from relax.utils.http_utils import find_available_port
from relax.utils.logging_utils import get_logger
from relax.utils.opd.opd_utils import build_teacher_engine_args, build_teacher_overrides


logger = get_logger(__name__)


def _resolve_teacher_gpu_index(*, args, replica: int, gpus_per_replica: int, shared_pg: bool) -> int:
    if not shared_pg:
        return replica * gpus_per_replica
    return int(args.rollout_num_gpus) + replica * gpus_per_replica


def _build_teacher_engine_env(args) -> dict[str, str]:
    env_vars = dict.fromkeys(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, "1") | {
        # OPD patches default off; enabled only when the corresponding env flag is
        # passed through from the driver. RELAX_OPD_PREEXPANDED_PATCH affects the
        # teacher engine only; RELAX_OPD_PER_POS_TOKEN_IDS affects teacher + student.
        "RELAX_OPD_PREEXPANDED_PATCH": os.environ.get("RELAX_OPD_PREEXPANDED_PATCH", "0"),
        "RELAX_OPD_PER_POS_TOKEN_IDS": os.environ.get("RELAX_OPD_PER_POS_TOKEN_IDS", "0"),
        "RELAX_OPD_TOKEN_IDS_LOGPROB_K": os.environ.get("RELAX_OPD_TOKEN_IDS_LOGPROB_K", "0"),
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
    return env_vars


@ray.remote
class TeacherManager:
    """Launch and own Relax-managed OPD teacher SGLang engine(s)."""

    def __init__(
        self,
        args,
        num_replicas: int,
        gpus_per_replica: int,
        pg: tuple | None = None,
        shared_pg: bool = False,
    ) -> None:
        assert num_replicas >= 1, f"num_replicas must be >= 1, got {num_replicas}."
        assert gpus_per_replica > 0, f"gpus_per_replica must be > 0, got {gpus_per_replica}."
        if shared_pg:
            assert pg is not None, "shared_pg=True requires the full actor/rollout placement group."
            _pg, bundle_indices, gpu_ids = pg
            required = int(args.rollout_num_gpus) + gpus_per_replica * num_replicas
            assert len(bundle_indices) >= required and len(gpu_ids) >= required, (
                f"shared teacher PG too small: bundles={len(bundle_indices)}, "
                f"gpu_ids={len(gpu_ids)}, required={required} (rollout_num_gpus={args.rollout_num_gpus} + "
                f"gpus_per_replica={gpus_per_replica} * num_replicas={num_replicas})."
            )

        self.args = args
        self.num_replicas = num_replicas
        self.gpus_per_replica = gpus_per_replica
        self._pg = pg
        self._shared_pg = shared_pg
        self._engines: list[tuple] = []
        self._urls = self._start()

    def get_urls(self) -> list[str]:
        return list(self._urls)

    def _start(self) -> list[str]:
        """Create the teacher PG(s) + engine actor(s), wait until healthy, and
        return the list of teacher ``/generate`` URLs.

        Multi-replica engines load in parallel: every replica's
        ``engine.init.remote(...)`` is fired first (non-blocking), then all
        handles are awaited in a single ``ray.get`` so a large teacher (e.g.
        122B) doesn't pay N x cold-load latency.
        """
        overrides = build_teacher_overrides(self.args, colocate_sync=self._shared_pg)
        teacher_args = build_teacher_engine_args(self.args, overrides)
        logger.info(
            f"[OPD teacher] launching {self.num_replicas} replica(s), "
            f"TP={self.gpus_per_replica}, model={overrides['model_path']}, "
            f"shared_pg={self._shared_pg}, mem_fraction_static={overrides.get('mem_fraction_static')}"
        )

        # Phase 1: create actors + fire init.remote() for every replica without
        # blocking. Each prep call returns the engine handle and the init future.
        preps = [self._prepare_one(replica, teacher_args, overrides) for replica in range(self.num_replicas)]

        # Phase 2: await every init in parallel (single ray.get on the full list).
        init_handles = [p["init_handle"] for p in preps]
        if init_handles:
            ray.get(init_handles)

        # Phase 3: finalize — pull URLs, register engines, log readiness.
        urls: list[str] = []
        for p in preps:
            base_url = ray.get(p["engine"].get_url.remote())
            url = f"{base_url}/generate"
            self._engines.append((p["pg"], p["engine"], p["owns_pg"]))
            logger.info(f"[OPD teacher] replica {p['replica']} ready at {url}")
            urls.append(url)
        return urls

    def _prepare_one(self, replica: int, teacher_args: object, overrides: dict[str, object]) -> dict:
        """Build the engine actor for a single replica and fire its
        ``init.remote()``, returning the handle + future so the caller can
        await all replicas in parallel."""
        if self._shared_pg:
            assert self._pg is not None
            pg, reordered_bundle_indices, reordered_gpu_ids = self._pg
            owns_pg = False
            gpu_index = _resolve_teacher_gpu_index(
                args=self.args,
                replica=replica,
                gpus_per_replica=self.gpus_per_replica,
                shared_pg=True,
            )
        else:
            pg, reordered_bundle_indices, reordered_gpu_ids = create_placement_group(num_gpus=self.gpus_per_replica)
            owns_pg = True
            gpu_index = _resolve_teacher_gpu_index(
                args=self.args,
                replica=replica,
                gpus_per_replica=self.gpus_per_replica,
                shared_pg=False,
            )

        base_gpu_id = int(reordered_gpu_ids[gpu_index])
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[gpu_index],
        )
        logger.info(
            f"[OPD teacher] replica={replica} gpu_index={gpu_index} "
            f"bundle_index={reordered_bundle_indices[gpu_index]} base_gpu_id={base_gpu_id}"
        )

        engine = (
            ray.remote(SGLangEngine)
            .options(
                num_cpus=0.2,
                num_gpus=0.2,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": _build_teacher_engine_env(self.args)},
            )
            .remote(
                teacher_args,
                rank=0,
                worker_type="regular",
                base_gpu_id=base_gpu_id,
                sglang_overrides=overrides,
                num_gpus_per_engine=self.gpus_per_replica,
                register_sigterm_handler=False,
            )
        )

        base_port = find_available_port(15000)
        addr_and_ports, _ = _allocate_rollout_engine_addr_and_ports_normal(
            args=teacher_args,
            rollout_engines=[(0, engine)],
            worker_type="regular",
            num_gpus_per_engine=self.gpus_per_replica,
            rank_offset=0,
            base_port=base_port,
        )

        # Fire init.remote() WITHOUT awaiting — the caller batches the ray.get
        # across replicas. The teacher is standalone: do NOT register to the
        # rollout router and do NOT register to DCS (it receives no weight sync).
        init_handle = engine.init.remote(
            **addr_and_ports[0],
            router_ip=None,
            router_port=None,
            skip_dcs_registration=True,
            skip_router_registration=True,
        )
        return {
            "replica": replica,
            "pg": pg,
            "engine": engine,
            "owns_pg": owns_pg,
            "init_handle": init_handle,
        }

    def shutdown(self) -> None:
        for pg, engine, owns_pg in self._engines:
            try:
                ray.get(engine.shutdown.remote(), timeout=30)
            except Exception as e:
                logger.warning(f"[OPD teacher] engine shutdown failed: {e}")
            try:
                ray.kill(engine)
            except Exception as e:
                logger.warning(f"[OPD teacher] engine kill failed: {e}")
            if owns_pg:
                try:
                    remove_placement_group(pg)
                except Exception as e:
                    logger.warning(f"[OPD teacher] remove placement group failed: {e}")
        self._engines = []
        logger.info("[OPD teacher] shutdown complete.")

    def offload(self) -> None:
        if not self._engines:
            return
        logger.info("[OPD teacher] offload requested")
        handles = [
            engine.release_memory_occupation.remote() for _pg, engine, _owns_pg in self._engines if engine is not None
        ]
        if handles:
            ray.get(handles)

    def onload(self, tags: list[str] | None = None) -> None:
        if not self._engines:
            return
        logger.info(f"[OPD teacher] onload requested with tags={tags}")
        handles = [
            engine.resume_memory_occupation.remote(tags=tags)
            for _pg, engine, _owns_pg in self._engines
            if engine is not None
        ]
        if handles:
            ray.get(handles)
