# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import random
from typing import Any

import aiohttp
import ray

from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.types import Sample

from .dapo_genrm import async_compute_score_genrm
from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl
from .multiple_choice import get_multiple_choice_reward
from .openr1mm import get_openr1mm_rule_based_reward
from .registry import (
    RewardRegistry,
    RewardSpec,
    RouteDecision,
    match_math_label,
    match_multiple_choice_label,
    resolve_reward_route,
)


logger = get_logger(__name__)
_shared_session: aiohttp.ClientSession | None = None
REWARD_REGISTRY = RewardRegistry()


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


def _reward_deepscaler(response: str, label: Any, metadata: dict[str, Any]) -> Any:
    del metadata
    return get_deepscaler_rule_based_reward(response, label)


def _reward_geo3k(response: str, label: Any, metadata: dict[str, Any]) -> Any:
    del metadata
    from .geo3k import get_geo3k_reward

    return get_geo3k_reward(response, label)


def _reward_openr1mm(response: str, label: Any, metadata: dict[str, Any]) -> Any:
    del metadata
    return get_openr1mm_rule_based_reward(response, label)


def _reward_multiple_choice(response: str, label: Any, metadata: dict[str, Any]) -> Any:
    del metadata
    return get_multiple_choice_reward(response, label)


def _reward_dapo(response: str, label: Any, metadata: dict[str, Any]) -> Any:
    del metadata
    return compute_score_dapo(response, label)


def _reward_math(response: str, label: Any, metadata: dict[str, Any]) -> int:
    del metadata
    return 1 if grade_answer_verl(response, label) else 0


def _reward_mopd(response: str, label: Any, metadata: dict[str, Any]) -> Any:
    from .mopd import get_mopd_reward

    return get_mopd_reward(response, label, metadata)


def _reward_f1(response: str, label: Any, metadata: dict[str, Any]) -> Any:
    del metadata
    return f1_score(response, label)[0]


def _reward_gpqa(response: str, label: Any, metadata: dict[str, Any]) -> float:
    return compute_gpqa_reward(response, label, metadata=metadata)


def _reward_ifbench(response: str, label: Any, metadata: dict[str, Any]) -> Any:
    from .ifbench import compute_ifbench_reward

    return compute_ifbench_reward(response, label, metadata=metadata)


def _reward_random(response: str, label: Any, metadata: dict[str, Any]) -> int:
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

    def compute(self, rm_type: str, response: str, label: Any, metadata: dict | None = None) -> Any:
        """Dispatch to the appropriate synchronous reward function.

        Returns the same value the original function would return.
        """
        spec = REWARD_REGISTRY.get(rm_type)
        if spec is None:
            raise NotImplementedError(f"RewardWorker: unknown rm_type={rm_type!r}")
        if spec.mode != "sync":
            raise TypeError(f"RewardWorker cannot execute asynchronous reward type {rm_type!r}.")
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
        self._warned_route_failures: set[tuple[str | None, str | None, str, str | None]] = set()

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

    def _warn_route_once(self, decision: RouteDecision, sample: Sample) -> None:
        key = (decision.reason, decision.requested_type, decision.action, decision.reward_type)
        if key in self._warned_route_failures:
            return
        self._warned_route_failures.add(key)
        logger.warning(
            "Reward routing fallback: reason=%s source=%s requested_type=%r action=%s "
            "selected_type=%r group_index=%r sample_index=%r",
            decision.reason,
            decision.source,
            decision.requested_type,
            decision.action,
            decision.reward_type,
            sample.group_index,
            sample.index,
        )

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
            decision = resolve_reward_route(
                registry=REWARD_REGISTRY,
                metadata=metadata,
                label=sample.label,
                explicit_rm_type=getattr(args, "rm_type", None),
                fallback_type=getattr(args, "rm_type_fallback", "zero"),
            )
            if decision.reason is not None:
                self._warn_route_once(decision, sample)
            if decision.action == "zero":
                return await _dummy_reward(args)
            if decision.action == "error":
                raise ValueError(
                    f"Unable to route reward: reason={decision.reason}, source={decision.source}, "
                    f"requested_type={decision.requested_type!r}."
                )

            spec = REWARD_REGISTRY.get(decision.reward_type)
            if spec is None:
                raise RuntimeError(f"Resolved reward type {decision.reward_type!r} is not registered.")

            response = sample.response
            label = sample.label
            if decision.boxed:
                response = extract_boxed_answer(response) or ""

            if spec.mode == "async":
                return await spec.handler(args, sample)

            if spec.mode == "sync":
                self._ensure_workers()
                worker = self._next_worker()
                ref = worker.compute.remote(spec.name, response, label, metadata=metadata)
                return await ref

            raise RuntimeError(f"Registered reward type {spec.name!r} has invalid mode {spec.mode!r}.")


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


async def _dummy_reward_handler(args, sample: Sample) -> float | dict[str, float]:
    del sample
    return await _dummy_reward(args)


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


# Adding a built-in reward requires a handler adapter and one registration line;
# RewardWorker and RewardExecutor do not need reward-specific branches.
REWARD_REGISTRY.register(RewardSpec("deepscaler", "sync", _reward_deepscaler))
REWARD_REGISTRY.register(RewardSpec("geo3k", "sync", _reward_geo3k))
REWARD_REGISTRY.register(RewardSpec("openr1mm", "sync", _reward_openr1mm))
REWARD_REGISTRY.register(RewardSpec("multiple_choice", "sync", _reward_multiple_choice, match_multiple_choice_label))
REWARD_REGISTRY.register(RewardSpec("dapo", "sync", _reward_dapo))
REWARD_REGISTRY.register(RewardSpec("math", "sync", _reward_math, match_math_label))
REWARD_REGISTRY.register(RewardSpec("mopd", "sync", _reward_mopd))
REWARD_REGISTRY.register(RewardSpec("f1", "sync", _reward_f1))
REWARD_REGISTRY.register(RewardSpec("gpqa", "sync", _reward_gpqa))
REWARD_REGISTRY.register(RewardSpec("ifbench", "sync", _reward_ifbench))
REWARD_REGISTRY.register(RewardSpec("random", "sync", _reward_random))
REWARD_REGISTRY.register(RewardSpec("remote_rm", "async", remote_rm))
REWARD_REGISTRY.register(RewardSpec("dapo-genrm", "async", async_compute_score_genrm))
REWARD_REGISTRY.register(RewardSpec("dummy", "async", _dummy_reward_handler))
