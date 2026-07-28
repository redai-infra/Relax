# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import inspect
import random
from typing import Any, Callable

import aiohttp
import ray

from relax.engine.rewards.custom_reward import (
    IMPLEMENTATION_VERSION,
    CustomRewardError,
    CustomRewardResolver,
    RewardWorkerConfig,
    WorkerConfigFingerprint,
    build_reward_worker_config,
    wrap_custom_reward_error,
)
from relax.utils.logging_utils import get_logger
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

_CUSTOM_KWARG_KEYS = frozenset({"custom_options", "ignore_custom"})


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


class _StaticGenerationProvider:
    def __init__(self, generation: int = 0):
        self._generation = generation

    def __call__(self, module_name: str = "custom_rm", module_path: str | None = None) -> int:
        return self._generation


def _is_same_generation_provider(left: Callable[..., int], right: Callable[..., int]) -> bool:
    if left is right:
        return True
    left_self = getattr(left, "__self__", None)
    right_self = getattr(right, "__self__", None)
    left_func = getattr(left, "__func__", None)
    right_func = getattr(right, "__func__", None)
    return left_self is not None and left_self is right_self and left_func is not None and left_func is right_func


def get_reward_executor(args) -> "RewardExecutor":
    """Return the process-wide RewardExecutor for the given training args."""
    max_concurrency = getattr(args, "reward_max_concurrency", None)
    if max_concurrency is None or max_concurrency <= 0:
        max_concurrency = 64
    num_workers = getattr(args, "reward_num_workers", None)
    if num_workers is None or num_workers <= 0:
        num_workers = 16
    return RewardExecutor.get_or_create(
        max_concurrency=max_concurrency,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# RewardWorker: Ray Actor for process-isolated reward computation
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=0.25)
class RewardWorker:
    """Worker that executes synchronous reward functions in a dedicated
    process."""

    def __init__(self):
        self._fingerprint: dict[str, Any] | None = None
        self._custom_resolver = CustomRewardResolver()
        self._loaded_generation_by_path: dict[str, int] = {}

    def ensure_fingerprint(self, fingerprint: dict[str, Any]) -> dict[str, Any]:
        if self._fingerprint is None:
            self._fingerprint = dict(fingerprint)
            return self._fingerprint
        if self._fingerprint != fingerprint:
            raise RuntimeError(
                f"RewardWorker fingerprint mismatch: existing={self._fingerprint}, requested={fingerprint}"
            )
        return self._fingerprint

    def compute(self, rm_type: str, response: str, label, metadata: dict | None = None):
        """Dispatch to the appropriate synchronous built-in reward function."""
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

    def compute_custom(
        self,
        *,
        path: str,
        generation: int,
        worker_config: RewardWorkerConfig,
        payload: Sample | list[Sample],
        kwargs: dict,
    ):
        prev_gen = self._loaded_generation_by_path.get(path)
        refresh = prev_gen is None or generation > prev_gen
        loaded = self._custom_resolver.ensure_loaded(path, generation, refresh_modules=refresh)
        if prev_gen is None or generation > prev_gen:
            self._loaded_generation_by_path[path] = generation
        if loaded.is_async:
            raise TypeError("Async custom reward must run in the rollout event loop")
        result = loaded.function(worker_config.as_namespace(), payload, **kwargs)
        if inspect.isawaitable(result):
            raise TypeError(
                f"Sync custom reward at {path!r} returned awaitable; "
                "use async def for async rewards or return a concrete value"
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
        self._custom_resolver = CustomRewardResolver()
        self._generation_provider: Callable[..., int] = _StaticGenerationProvider(0)
        self._generation_provider_bound = False
        self._worker_fingerprint: WorkerConfigFingerprint | None = None

    @classmethod
    def get_or_create(cls, max_concurrency: int = 64, num_workers: int = 16) -> "RewardExecutor":
        if cls._instance is None:
            cls._instance = cls(max_concurrency=max_concurrency, num_workers=num_workers)
        return cls._instance

    def bind_generation_provider(self, provider: Callable[..., int]) -> None:
        if self._generation_provider_bound:
            if _is_same_generation_provider(provider, self._generation_provider):
                return
            raise RuntimeError(
                "RewardExecutor generation provider already bound; call RewardExecutor.reset() before rebinding"
            )
        self._generation_provider = provider
        self._generation_provider_bound = True
        self._custom_resolver.invalidate_all()

    def current_custom_generation(self, path: str | None = None) -> int:
        return int(self._generation_provider("custom_rm", path))

    def _ensure_semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)

    def _expected_fingerprint(self) -> WorkerConfigFingerprint:
        return WorkerConfigFingerprint(
            num_workers=self._num_workers,
            implementation_version=IMPLEMENTATION_VERSION,
        )

    def _ensure_workers(self):
        expected = self._expected_fingerprint()
        if self._workers and self._worker_fingerprint is not None and self._worker_fingerprint != expected:
            raise RuntimeError(
                f"RewardWorker pool fingerprint mismatch: existing={self._worker_fingerprint.as_dict()}, "
                f"requested={expected.as_dict()}"
            )
        if not self._workers:
            fingerprint = expected.as_dict()
            workers = [
                RewardWorker.options(
                    name=f"reward_worker_{i}",
                    get_if_exists=True,
                ).remote()
                for i in range(self._num_workers)
            ]
            # One batched wait instead of N sequential ray.get calls.
            ray.get([worker.ensure_fingerprint.remote(fingerprint) for worker in workers])
            self._workers = workers
            self._worker_fingerprint = expected
            logger.info(
                "RewardExecutor: created %d RewardWorker actors (max_concurrency=%d)",
                self._num_workers,
                self._max_concurrency,
            )

    def _next_worker(self):
        worker = self._workers[self._worker_index % self._num_workers]
        self._worker_index += 1
        return worker

    _ASYNC_RM_DISPATCH = {
        "remote_rm": lambda args, sample: remote_rm(args, sample),
        "dapo-genrm": lambda args, sample: async_compute_score_genrm(args, sample),
        "dummy": lambda args, sample: _dummy_reward(args),
    }

    async def execute(self, args, sample: Sample, **kwargs):
        """Execute a single reward computation with concurrency control."""
        self._ensure_semaphore()
        async with self._semaphore:
            if args.custom_rm_path is not None and not kwargs.get("ignore_custom", False):
                return await self._execute_custom_sample_unlocked(args, sample, **kwargs)
            return await self._execute_builtin_unlocked(args, sample, **kwargs)

    async def execute_custom_sample(self, args, sample: Sample, **kwargs):
        self._ensure_semaphore()
        async with self._semaphore:
            return await self._execute_custom_sample_unlocked(args, sample, **kwargs)

    async def execute_custom_group(self, args, samples: list[Sample], **kwargs):
        self._ensure_semaphore()
        async with self._semaphore:
            return await self._execute_custom_group_unlocked(args, samples, **kwargs)

    async def dispatch_custom_sample(self, args, sample: Sample, **kwargs):
        """Agentic path: caller already holds RewardDomain semaphore."""
        return await self._execute_custom_sample_unlocked(args, sample, **kwargs)

    async def dispatch_custom_group(self, args, samples: list[Sample], **kwargs):
        """Agentic path: caller already holds RewardDomain semaphore."""
        return await self._execute_custom_group_unlocked(args, samples, **kwargs)

    async def _execute_builtin_unlocked(self, args, sample: Sample, **kwargs):
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
        response = sample.response
        label = sample.label
        if rm_type.startswith("boxed_"):
            response = extract_boxed_answer(response) or ""
            rm_type = rm_type[len("boxed_") :]

        async_handler = self._ASYNC_RM_DISPATCH.get(rm_type)
        if async_handler is not None:
            return await async_handler(args, sample)

        if rm_type:
            self._ensure_workers()
            worker = self._next_worker()
            ref = worker.compute.remote(rm_type, response, label, metadata=metadata)
            return await ref

        raise NotImplementedError("Rule-based RM type is not specified.")

    async def _execute_custom_sample_unlocked(self, args, sample: Sample, **kwargs):
        return await self._invoke_custom(args, payload=sample, group=False, **kwargs)

    async def _execute_custom_group_unlocked(self, args, samples: list[Sample], **kwargs):
        result = await self._invoke_custom(args, payload=samples, group=True, **kwargs)
        if not isinstance(result, list):
            raise CustomRewardError(
                f"Custom group reward at {args.custom_rm_path!r} must return list, got {type(result).__name__}"
            )
        if len(result) != len(samples):
            raise CustomRewardError(
                f"Custom group reward at {args.custom_rm_path!r} returned {len(result)} values "
                f"for {len(samples)} samples"
            )
        return result

    async def _invoke_custom(self, args, *, payload: Sample | list[Sample], group: bool, **kwargs):
        path = args.custom_rm_path
        if not path:
            raise CustomRewardError("custom_rm_path is not set")

        custom_options = kwargs.get("custom_options")
        if kwargs.get("ignore_custom", False):
            raise CustomRewardError("ignore_custom=True cannot invoke custom reward")
        call_kwargs = {key: value for key, value in kwargs.items() if key not in _CUSTOM_KWARG_KEYS}

        generation = self.current_custom_generation(path)
        loaded = self._custom_resolver.ensure_loaded(path, generation)

        sample = None if group else payload
        samples = payload if group else None
        try:
            if loaded.is_async:
                result = loaded.function(args, payload, **call_kwargs)
                if not inspect.isawaitable(result):
                    raise TypeError(f"Async custom reward at {path!r} did not return awaitable")
                return await result

            worker_config = build_reward_worker_config(
                args,
                generation=generation,
                custom_options=custom_options,
            )
            self._ensure_workers()
            worker = self._next_worker()
            ref = worker.compute_custom.remote(
                path=path,
                generation=generation,
                worker_config=worker_config,
                payload=payload,
                kwargs=call_kwargs,
            )
            return await ref
        except CustomRewardError:
            raise
        except Exception as error:
            raise wrap_custom_reward_error(error, path=path, sample=sample, samples=samples) from error

    async def close(self) -> None:
        for worker in self._workers:
            try:
                ray.kill(worker)
            except Exception:
                pass
        self._workers = []
        self._worker_fingerprint = None
        self._semaphore = None
        self._custom_resolver.invalidate_all()
        self._generation_provider = _StaticGenerationProvider(0)
        self._generation_provider_bound = False

    @classmethod
    async def reset(cls) -> None:
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------


async def _dummy_reward(args):
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
    """Single-sample reward computation."""
    return await get_reward_executor(args).execute(args, sample, **kwargs)


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None and not kwargs.get("ignore_custom", False):
        # Legacy compatibility: preserve one list[Sample] call even when group_rm=False.
        return await get_reward_executor(args).execute_custom_group(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards


__all__ = [
    "CustomRewardError",
    "RewardExecutor",
    "RewardWorker",
    "RewardWorkerConfig",
    "async_rm",
    "batched_async_rm",
    "get_reward_executor",
    "remote_rm",
]
