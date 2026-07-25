# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import inspect
import random

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


logger = get_logger(__name__)
_shared_session: aiohttp.ClientSession | None = None


def _sample_context(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    metadata_keys = sorted(str(key) for key in metadata.keys())
    return (
        f"index={getattr(sample, 'index', None)!r}, "
        f"group_index={getattr(sample, 'group_index', None)!r}, "
        f"metadata_keys={metadata_keys!r}"
    )


def _sample_group_context(samples: list[Sample]) -> str:
    return "[" + ", ".join(_sample_context(sample) for sample in samples) + "]"


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

    def __init__(self):
        self._custom_rm_functions: dict[str, object] = {}

    def _load_custom_rm_function(self, custom_rm_path: str):
        rm_function = self._custom_rm_functions.get(custom_rm_path)
        if rm_function is None:
            rm_function = load_function(custom_rm_path)
            self._custom_rm_functions[custom_rm_path] = rm_function
        return rm_function

    def clear_custom_rm_cache(self, custom_rm_path: str | None = None):
        if custom_rm_path is None:
            self._custom_rm_functions.clear()
            return
        self._custom_rm_functions.pop(custom_rm_path, None)

    def compute(self, rm_type: str, response: str, label, metadata: dict | None = None):
        """Dispatch to the appropriate synchronous reward function.

        Returns the same value the original function would return.
        """
        if rm_type == "deepscaler":
            return get_deepscaler_rule_based_reward(response, label)
        elif rm_type == "geo3k":
            from .geo3k import get_geo3k_reward

            return get_geo3k_reward(response, label)
        elif rm_type == "openr1mm":
            return get_openr1mm_rule_based_reward(response, label)
        elif rm_type == "multiple_choice":
            return get_multiple_choice_reward(response, label)
        elif rm_type == "dapo":
            return compute_score_dapo(response, label)
        elif rm_type == "math":
            return 1 if grade_answer_verl(response, label) else 0
        elif rm_type == "mopd":
            from .mopd import get_mopd_reward

            return get_mopd_reward(response, label, metadata)
        elif rm_type == "f1":
            return f1_score(response, label)[0]
        elif rm_type == "gpqa":
            return compute_gpqa_reward(response, label, metadata=metadata)
        elif rm_type == "ifbench":
            from .ifbench import compute_ifbench_reward

            return compute_ifbench_reward(response, label, metadata=metadata)
        elif rm_type == "random":
            return random.randint(0, 1)
        else:
            raise NotImplementedError(f"RewardWorker: unknown rm_type={rm_type!r}")

    def compute_custom(self, custom_rm_path: str, args, sample: Sample, kwargs: dict | None = None):
        rm_function = self._load_custom_rm_function(custom_rm_path)
        # Worker actors only execute synchronous custom rewards; async rewards
        # must stay on the rollout event loop where they can be awaited safely.
        if inspect.iscoroutinefunction(rm_function):
            raise TypeError(f"Custom reward {custom_rm_path!r} is async and cannot run in RewardWorker.")

        kwargs = kwargs or {}
        try:
            result = rm_function(args, sample, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Custom reward failed for sample ({_sample_context(sample)}).") from exc

        if inspect.isawaitable(result):
            raise TypeError(
                f"Custom reward {custom_rm_path!r} returned an awaitable from a synchronous function. "
                "Define it with 'async def' to run it in the event loop."
            )
        return result

    def compute_custom_batch(self, custom_rm_path: str, args, samples: list[Sample], kwargs: dict | None = None):
        rm_function = self._load_custom_rm_function(custom_rm_path)
        # Keep group reward semantics intact by passing the whole sample group
        # to a synchronous custom scorer inside one worker process.
        if inspect.iscoroutinefunction(rm_function):
            raise TypeError(f"Custom reward {custom_rm_path!r} is async and cannot run in RewardWorker.")

        kwargs = kwargs or {}
        try:
            result = rm_function(args, samples, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Custom reward failed for sample group {_sample_group_context(samples)}.") from exc

        if inspect.isawaitable(result):
            raise TypeError(
                f"Custom reward {custom_rm_path!r} returned an awaitable from a synchronous function. "
                "Define it with 'async def' to run it in the event loop."
            )
        return result


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
        self._custom_rm_functions: dict[str, object] = {}

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

    def _load_custom_rm_function(self, custom_rm_path: str):
        rm_function = self._custom_rm_functions.get(custom_rm_path)
        if rm_function is None:
            rm_function = load_function(custom_rm_path)
            self._custom_rm_functions[custom_rm_path] = rm_function
        return rm_function

    def _clear_custom_rm_cache(self, custom_rm_path: str | None = None):
        # Explicit hot reload refreshes sys.modules in the rollout process; drop
        # cached function objects here and in already-created worker actors.
        if custom_rm_path is None:
            self._custom_rm_functions.clear()
        else:
            self._custom_rm_functions.pop(custom_rm_path, None)

        refs = []
        for worker in self._workers:
            try:
                refs.append(worker.clear_custom_rm_cache.remote(custom_rm_path))
            except Exception as exc:
                logger.warning("RewardExecutor: failed to clear custom reward cache on worker: %s", exc)
        if refs:
            try:
                ray.get(refs)
            except Exception as exc:
                logger.warning("RewardExecutor: failed to wait for custom reward cache clear: %s", exc)

    @classmethod
    def clear_custom_rm_cache(cls, custom_rm_path: str | None = None):
        if cls._instance is not None:
            cls._instance._clear_custom_rm_cache(custom_rm_path)

    # -- public API -----------------------------------------------------------

    # Async rm_types run in the event loop (not dispatched to worker pool).
    _ASYNC_RM_DISPATCH = {
        "remote_rm": lambda args, sample: remote_rm(args, sample),
        "dapo-genrm": lambda args, sample: async_compute_score_genrm(args, sample),
        # `dummy` returns 0 without any computation. Use it when the real
        # reward is produced elsewhere (e.g., --custom-reward-post-process-path
        # does batched GenRM scoring after all rollout finishes).
        "dummy": lambda args, sample: _dummy_reward(args),
    }

    # CPU-bound / thread-unsafe rm_types dispatched to the Ray worker pool.
    _SYNC_RM_TYPES = frozenset(
        {
            "deepscaler",
            "geo3k",
            "openr1mm",
            "multiple_choice",
            "dapo",
            "math",
            "mopd",
            "f1",
            "gpqa",
            "ifbench",
            "random",
        }
    )

    async def execute(self, args, sample: Sample, **kwargs):
        """Execute a single reward computation with concurrency control.

        - Async rm_types (remote_rm, dapo-genrm) run in the event loop.
        - Sync rm_types are dispatched to the Ray worker pool.
        - Async custom rm paths run in the event loop.
        - Sync custom rm paths are dispatched to the Ray worker pool.
        """
        self._ensure_semaphore()

        async with self._semaphore:
            if args.custom_rm_path is not None and not kwargs.get("ignore_custom", False):
                return await self._execute_custom(args, sample, kwargs)

            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
            response = sample.response
            label = sample.label
            if rm_type.startswith("boxed_"):
                response = extract_boxed_answer(response) or ""
                rm_type = rm_type[len("boxed_") :]

            # --- async rm types: run in event loop -----------------------
            async_handler = self._ASYNC_RM_DISPATCH.get(rm_type)
            if async_handler is not None:
                return await async_handler(args, sample)

            # --- sync rm types: dispatch to worker pool ------------------
            # Default to sync path for any non-empty rm_type not in async dispatch
            if rm_type:
                self._ensure_workers()
                worker = self._next_worker()
                ref = worker.compute.remote(rm_type, response, label, metadata=metadata)
                return await ref

            # --- no rm_type specified -----------------------------------------
            raise NotImplementedError("Rule-based RM type is not specified.")

    async def _execute_custom(self, args, sample: Sample, kwargs: dict):
        custom_rm_path = args.custom_rm_path
        rm_function = self._load_custom_rm_function(custom_rm_path)
        # Classify before calling the function. Calling first and checking the
        # return value would already block the event loop for sync scorers.
        if inspect.iscoroutinefunction(rm_function):
            try:
                return await rm_function(args, sample, **kwargs)
            except Exception as exc:
                raise RuntimeError(f"Custom reward failed for sample ({_sample_context(sample)}).") from exc

        self._ensure_workers()
        worker = self._next_worker()
        ref = worker.compute_custom.remote(custom_rm_path, args, sample, kwargs)
        return await ref

    async def execute_custom_batch(self, args, samples: list[Sample], **kwargs):
        self._ensure_semaphore()

        async with self._semaphore:
            custom_rm_path = args.custom_rm_path
            rm_function = self._load_custom_rm_function(custom_rm_path)
            if inspect.iscoroutinefunction(rm_function):
                try:
                    return await rm_function(args, samples, **kwargs)
                except Exception as exc:
                    raise RuntimeError(
                        f"Custom reward failed for sample group {_sample_group_context(samples)}."
                    ) from exc

            self._ensure_workers()
            worker = self._next_worker()
            ref = worker.compute_custom_batch.remote(custom_rm_path, args, samples, kwargs)
            return await ref


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
    if not samples:
        return []
    # group_rm custom rewards are documented as whole-group scorers, so keep
    # their batch input shape and offload sync implementations as one worker job.
    if args.custom_rm_path is not None and not kwargs.get("ignore_custom", False) and getattr(args, "group_rm", False):
        max_concurrency = getattr(args, "reward_max_concurrency", 64)
        num_workers = getattr(args, "reward_num_workers", 16)
        executor = RewardExecutor.get_or_create(
            max_concurrency=max_concurrency,
            num_workers=num_workers,
        )
        return await executor.execute_custom_batch(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
