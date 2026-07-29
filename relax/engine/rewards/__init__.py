# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RewardHandler:
    function: Callable
    is_async: bool = False


REWARD_REGISTRY: dict[str, RewardHandler] = {}


def register_reward(name: str, *, is_async: bool = False):
    """Register a reward implementation without changing router branches."""

    def decorator(function: Callable) -> Callable:
        if name in REWARD_REGISTRY:
            raise ValueError(f"Reward type {name!r} is already registered")
        REWARD_REGISTRY[name] = RewardHandler(function=function, is_async=is_async)
        return function

    return decorator


@register_reward("deepscaler")
def _deepscaler_reward(response, label, metadata):
    return get_deepscaler_rule_based_reward(response, label)


@register_reward("geo3k")
def _geo3k_reward(response, label, metadata):
    from .geo3k import get_geo3k_reward

    return get_geo3k_reward(response, label)


@register_reward("openr1mm")
def _openr1mm_reward(response, label, metadata):
    return get_openr1mm_rule_based_reward(response, label)


@register_reward("multiple_choice")
def _multiple_choice_reward(response, label, metadata):
    return get_multiple_choice_reward(response, label)


@register_reward("dapo")
def _dapo_reward(response, label, metadata):
    return compute_score_dapo(response, label)


@register_reward("math")
def _math_reward(response, label, metadata):
    return 1 if grade_answer_verl(response, label) else 0


@register_reward("mopd")
def _mopd_reward(response, label, metadata):
    from .mopd import get_mopd_reward

    return get_mopd_reward(response, label, metadata)


@register_reward("f1")
def _f1_reward(response, label, metadata):
    return f1_score(response, label)[0]


@register_reward("gpqa")
def _gpqa_reward(response, label, metadata):
    return compute_gpqa_reward(response, label, metadata=metadata)


@register_reward("ifbench")
def _ifbench_reward(response, label, metadata):
    from .ifbench import compute_ifbench_reward

    return compute_ifbench_reward(response, label, metadata=metadata)


@register_reward("random")
def _random_reward(response, label, metadata):
    return random.randint(0, 1)


@register_reward("remote_rm", is_async=True)
async def _remote_reward(args, sample):
    return await remote_rm(args, sample)


@register_reward("dapo-genrm", is_async=True)
async def _dapo_genrm_reward(args, sample):
    return await async_compute_score_genrm(args, sample)


@register_reward("dummy", is_async=True)
async def _registered_dummy_reward(args, sample):
    return await _dummy_reward(args)


_ROUTE_KEYS = ("rm_type", "reward_type", "task_type", "format")
_ROUTE_ALIASES = {
    "multiple-choice": "multiple_choice",
    "multiple choice": "multiple_choice",
    "mcq": "multiple_choice",
}
_MULTIPLE_CHOICE_LABEL = re.compile(r"^\s*(?:<answer>\s*)?([A-Za-z])(?:\s*</answer>)?\s*$", re.DOTALL)
_NUMERIC_LABEL = re.compile(r"^\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*$")


def _normalize_rm_type(value) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().lower()
    return _ROUTE_ALIASES.get(value, value)


def _routing_hints(value) -> set[str]:
    if not isinstance(value, Mapping):
        return set()
    return {_normalize_rm_type(value.get(key)) for key in _ROUTE_KEYS if _normalize_rm_type(value.get(key))}


def _infer_rm_type_from_label(label) -> str:
    if isinstance(label, Mapping):
        label = label.get("answer", label.get("label"))
    if not isinstance(label, str):
        return ""
    if _MULTIPLE_CHOICE_LABEL.fullmatch(label):
        return "multiple_choice"
    if "\\boxed{" in label or "\\frac{" in label or "\\sqrt{" in label or _NUMERIC_LABEL.fullmatch(label):
        return "math"
    return ""


def resolve_rm_type(args, sample: Sample) -> str | None:
    """Resolve one sample's reward type, using ``args.rm_type`` as fallback."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    hints = _routing_hints(metadata) | _routing_hints(sample.label)
    fallback = _normalize_rm_type(getattr(args, "rm_type", None))

    if len(hints) > 1:
        logger.warning("Conflicting reward types %s; using fallback %r", sorted(hints), fallback or None)
    elif len(hints) == 1:
        candidate = next(iter(hints))
        if candidate in REWARD_REGISTRY or candidate.startswith("boxed_"):
            return candidate
        logger.warning("Unknown sample reward type %r; using fallback %r", candidate, fallback or None)
    elif not fallback:
        inferred = _infer_rm_type_from_label(sample.label)
        if inferred:
            return inferred

    if fallback in REWARD_REGISTRY or fallback.startswith("boxed_"):
        return fallback
    if fallback:
        logger.warning("Unknown fallback reward type %r; returning zero reward", fallback)
    else:
        logger.warning("Reward type is missing and label format is unknown; returning zero reward")
    return None


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
        """Dispatch to the appropriate synchronous reward function.

        Returns the same value the original function would return.
        """
        handler = REWARD_REGISTRY.get(rm_type)
        if handler is None or handler.is_async:
            raise NotImplementedError(f"RewardWorker: unknown rm_type={rm_type!r}")
        return handler.function(response, label, metadata)


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
            rm_type = resolve_rm_type(args, sample)
            if rm_type is None:
                return 0.0
            response = sample.response
            label = sample.label
            if rm_type.startswith("boxed_"):
                response = extract_boxed_answer(response) or ""
                rm_type = rm_type[len("boxed_") :]

            # --- async rm types: run in event loop -----------------------
            handler = REWARD_REGISTRY.get(rm_type)
            if handler is not None and handler.is_async:
                return await handler.function(args, sample)

            # --- sync rm types: dispatch to worker pool ------------------
            # Default to sync path for any non-empty rm_type not in async dispatch
            if handler is not None:
                self._ensure_workers()
                worker = self._next_worker()
                ref = worker.compute.remote(rm_type, response, label, metadata=metadata)
                return await ref

            logger.warning("Unknown resolved reward type %r; returning zero reward", rm_type)
            return 0.0


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
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
