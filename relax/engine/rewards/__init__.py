# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import random

import aiohttp
import ray

from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl
from .multiple_choice import get_multiple_choice_reward
from .openr1mm import get_openr1mm_rule_based_reward
from .reward_router import (
    RewardSpec,
    build_reward_registry,
    match_math_label,
    match_multiple_choice_label,
    resolve_reward_route,
)


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


def _score_deepscaler(response, label, metadata):
    del metadata
    return get_deepscaler_rule_based_reward(response, label)


def _score_geo3k(response, label, metadata):
    del metadata
    from .geo3k import get_geo3k_reward

    return get_geo3k_reward(response, label)


def _score_openr1mm(response, label, metadata):
    del metadata
    return get_openr1mm_rule_based_reward(response, label)


def _score_multiple_choice(response, label, metadata):
    del metadata
    return get_multiple_choice_reward(response, label)


def _score_dapo(response, label, metadata):
    del metadata
    return compute_score_dapo(response, label)


def _score_math(response, label, metadata):
    del metadata
    return 1 if grade_answer_verl(response, label) else 0


def _score_mopd(response, label, metadata):
    from .mopd import get_mopd_reward

    return get_mopd_reward(response, label, metadata)


def _score_f1(response, label, metadata):
    del metadata
    return f1_score(response, label)[0]


def _score_gpqa(response, label, metadata):
    return compute_gpqa_reward(response, label, metadata=metadata)


def _score_ifbench(response, label, metadata):
    from .ifbench import compute_ifbench_reward

    return compute_ifbench_reward(response, label, metadata=metadata)


def _score_random(response, label, metadata):
    del response, label, metadata
    return random.randint(0, 1)


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
        """Dispatch to the appropriate synchronous reward function.

        Returns the same value the original function would return.
        """
        spec = REWARD_REGISTRY.get(rm_type)
        if spec is None or spec.mode != "sync":
            raise NotImplementedError(f"RewardWorker: unknown rm_type={rm_type!r}")
        return spec.handler(response, label, metadata or {})


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
            route = resolve_reward_route(metadata, sample.label, getattr(args, "rm_type", None), REWARD_REGISTRY)
            if route.reason is not None:
                logger.warning(
                    "Reward route %s for sample index=%r group_index=%r candidates=%s; selected=%r source=%s",
                    route.reason,
                    getattr(sample, "index", None),
                    getattr(sample, "group_index", None),
                    route.candidates,
                    route.reward_type,
                    route.source,
                )
            if route.reward_type is None:
                return _zero_reward(args)

            spec = REWARD_REGISTRY[route.reward_type]
            response = sample.response
            label = sample.label
            if route.boxed:
                response = extract_boxed_answer(response) or ""

            # --- async rm types: run in event loop -----------------------
            if spec.mode == "async":
                return await spec.handler(args, sample)

            # --- sync rm types: dispatch to worker pool ------------------
            self._ensure_workers()
            worker = self._next_worker()
            ref = worker.compute.remote(route.reward_type, response, label, metadata=metadata)
            return await ref


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------


def _zero_reward(args):
    # No-op reward. Paired with --custom-reward-post-process-path when real
    # scoring is deferred to a post-rollout batch pass. Returns a dict when
    # reward_key is set so downstream sample.reward[reward_key] access does
    # not KeyError before the post-process step overwrites everything.
    reward_key = getattr(args, "reward_key", None)
    if reward_key:
        return {reward_key: 0.0}
    return 0.0


async def _dummy_reward(args):
    return _zero_reward(args)


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


async def _run_remote_rm(args, sample):
    return await remote_rm(args, sample)


async def _run_dapo_genrm(args, sample):
    from .dapo_genrm import async_compute_score_genrm

    return await async_compute_score_genrm(args, sample)


async def _run_dummy(args, sample):
    del sample
    return await _dummy_reward(args)


REWARD_REGISTRY = build_reward_registry(
    (
        ("deepscaler", RewardSpec("sync", _score_deepscaler)),
        ("geo3k", RewardSpec("sync", _score_geo3k)),
        ("openr1mm", RewardSpec("sync", _score_openr1mm)),
        ("multiple_choice", RewardSpec("sync", _score_multiple_choice, match_multiple_choice_label)),
        ("dapo", RewardSpec("sync", _score_dapo)),
        ("math", RewardSpec("sync", _score_math, match_math_label)),
        ("mopd", RewardSpec("sync", _score_mopd)),
        ("f1", RewardSpec("sync", _score_f1)),
        ("gpqa", RewardSpec("sync", _score_gpqa)),
        ("ifbench", RewardSpec("sync", _score_ifbench)),
        ("random", RewardSpec("sync", _score_random)),
        ("remote_rm", RewardSpec("async", _run_remote_rm)),
        ("dapo-genrm", RewardSpec("async", _run_dapo_genrm)),
        ("dummy", RewardSpec("async", _run_dummy)),
    )
)


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
