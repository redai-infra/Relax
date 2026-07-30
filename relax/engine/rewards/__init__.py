# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import random

import aiohttp
import ray

from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.types import Sample

from .dapo_genrm import async_compute_score_genrm
from .math_utils import extract_answer as extract_boxed_answer
from .registry import get_reward_spec, list_reward_types, register_reward, resolve_rm_type


logger = get_logger(__name__)
_shared_session: aiohttp.ClientSession | None = None


def _get_shared_session() -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(
            limit=64,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=120)
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _shared_session


# ---------------------------------------------------------------------------
# RewardWorker: Ray Actor for process-isolated reward computation
# ---------------------------------------------------------------------------
# Solves three problems at once:
# 1. CPU-intensive reward functions no longer block the async event loop.
# 2. Thread-unsafe libraries (e.g. math_verify) are safely isolated inside
#    their own process – each Actor is single-threaded by default.
# 3. Global concurrency is bounded by the number of workers in the pool
#    combined with the asyncio.Semaphore in batched_async_rm.
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=0.25)
class RewardWorker:
    """Stateless worker that executes synchronous reward functions in a
    dedicated process.

    Each call receives the rm_type and the necessary arguments so the worker
    does not need to hold any state.
    """

    def compute(self, rm_type: str, response: str, label, metadata: dict | None = None):
        """Dispatch to the registered synchronous reward function.

        Returns the same value the original function would return.
        """
        spec = get_reward_spec(rm_type)
        if spec is None or spec.mode != "sync":
            raise NotImplementedError(
                f"RewardWorker: unknown rm_type={rm_type!r}. Available: {list_reward_types('sync')}"
            )
        fn = spec.resolve()
        if spec.pass_metadata:
            return fn(response, label, metadata=metadata)
        return fn(response, label)


# ---------------------------------------------------------------------------
# RewardExecutor: manages the worker pool and global concurrency
# ---------------------------------------------------------------------------


class RewardExecutor:
    """Singleton that manages a pool of RewardWorker actors and an
    asyncio.Semaphore for global concurrency control."""

    _instance: "RewardExecutor | None" = None

    def __init__(self, max_concurrency: int = 64, num_workers: int = 16):
        self._max_concurrency = max_concurrency
        self._num_workers = num_workers
        self._semaphore: asyncio.Semaphore | None = None
        self._workers: list = []
        self._worker_index = 0

    # -- singleton access -----------------------------------------------------

    @classmethod
    def get_or_create(cls, max_concurrency: int = 64, num_workers: int = 16) -> "RewardExecutor":
        if cls._instance is None:
            cls._instance = cls(max_concurrency=max_concurrency, num_workers=num_workers)
        return cls._instance

    # -- lazy init (must happen inside an event loop) -------------------------

    def _ensure_semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)

    def _ensure_workers(self):
        if not self._workers:
            self._workers = [
                RewardWorker.options(
                    name=f"reward_worker_{i}",
                    get_if_exists=True,
                ).remote()
                for i in range(self._num_workers)
            ]
            logger.info(
                "RewardExecutor: created %d RewardWorker actors (max_concurrency=%d)",
                self._num_workers,
                self._max_concurrency,
            )

    def _next_worker(self):
        worker = self._workers[self._worker_index % self._num_workers]
        self._worker_index += 1
        return worker

    # -- public API -----------------------------------------------------------

    async def execute(self, args, sample: Sample, **kwargs):
        """Execute a single reward computation with concurrency control.

        - Async rm_types (remote_rm, dapo-genrm) run in the event loop.
        - Sync rm_types are dispatched to the Ray worker pool.
        - Custom rm paths are called directly (user is responsible for
          async safety).
        """
        self._ensure_semaphore()

        async with self._semaphore:
            # --- custom rm path: delegate to user function directly ---
            if args.custom_rm_path is not None and not kwargs.get("ignore_custom", False):
                rm_function = load_function(args.custom_rm_path)
                return await rm_function(args, sample, **kwargs)

            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            route = resolve_rm_type(
                metadata_rm_type=metadata.get("rm_type"),
                args_rm_type=args.rm_type,
                label=sample.label,
                infer=getattr(args, "rm_type_infer", False),
                fallback=getattr(args, "rm_type_fallback", None),
                sample_index=getattr(sample, "index", None),
            )
            if route.zero_reward:
                # Fallback "zero": no dispatch. Reuse the dummy reward so the
                # 0.0 is reward-key aware and cannot mix dict/scalar shapes.
                return await _dummy_reward(args)

            rm_type = route.rm_type
            response = sample.response
            label = sample.label
            if route.boxed:
                response = extract_boxed_answer(response) or ""

            # --- async rm types: run in event loop -----------------------
            spec = get_reward_spec(rm_type) if rm_type else None
            if spec is not None and spec.mode == "async":
                return await spec.resolve()(args, sample)

            # --- sync rm types: dispatch to worker pool ------------------
            # Default to sync path for any non-empty rm_type not registered
            # as async; unknown types keep failing inside the worker.
            if rm_type:
                self._ensure_workers()
                worker = self._next_worker()
                ref = worker.compute.remote(rm_type, response, label, metadata=metadata)
                return await ref

            # --- no rm_type specified -----------------------------------------
            raise NotImplementedError("Rule-based RM type is not specified.")


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------


async def _dummy_reward(args):
    # No-op reward. Paired with --custom-reward-post-process-path when real
    # scoring is deferred to a post-rollout batch pass. Returns a dict when
    # reward_key is set so downstream sample.reward[reward_key] access does
    # not KeyError before the post-process step overwrites everything.
    reward_key = getattr(args, "reward_key", None)
    if reward_key:
        return {reward_key: 0.0}
    return 0.0


async def remote_rm(args, sample: Sample, max_retries: int = 10):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session = _get_shared_session()
    for attempt in range(max_retries):
        try:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            if attempt + 1 >= max_retries:
                logger.warning(f"remote_rm failed after {attempt + 1} attempts: {e}")
                raise
            backoff = min(2**attempt, 30) + random.random()
            logger.info(f"remote_rm: {type(e).__name__}, retrying in {backoff:.1f}s ({attempt + 1}/{max_retries})")
            await asyncio.sleep(backoff)


# Async rm_types run in the event loop (not dispatched to the worker pool).
register_reward("remote_rm", remote_rm, mode="async")
register_reward("dapo-genrm", async_compute_score_genrm, mode="async")
# `dummy` returns 0 without any computation. Use it when the real reward is
# produced elsewhere (e.g., --custom-reward-post-process-path does batched
# GenRM scoring after all rollout finishes).
register_reward("dummy", lambda args, sample: _dummy_reward(args), mode="async")


async def async_rm(args, sample: Sample, **kwargs):
    """Single-sample reward computation.

    Delegates to RewardExecutor which handles concurrency control and process
    isolation for CPU-bound / thread-unsafe reward functions.
    """
    max_concurrency = getattr(args, "reward_max_concurrency", 64)
    num_workers = getattr(args, "reward_num_workers", 16)
    executor = RewardExecutor.get_or_create(
        max_concurrency=max_concurrency,
        num_workers=num_workers,
    )
    return await executor.execute(args, sample, **kwargs)


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
