# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Helpers for custom reward load/classify/cache and Worker config."""

from __future__ import annotations

import inspect
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from relax.utils.misc import load_function
from relax.utils.types import Sample


IMPLEMENTATION_VERSION = 1

_CONFIG_FIELD_KEYS = frozenset(
    {
        "custom_rm_path",
        "group_rm",
        "reward_key",
        "rm_type",
        "rm_url",
        "implementation_version",
        "generation",
        "custom_options",
    }
)

# Known heavy / non-serializable training attrs never auto-copied into Workers.
_AUTO_PICK_DENYLIST = frozenset(
    {
        "model",
        "tokenizer",
        "processor",
        "hf_tokenizer",
        "hf_processor",
        "train_data",
        "eval_data",
        "rollout_engine",
        "actor",
        "critic",
    }
)


class CustomRewardError(RuntimeError):
    """Driver-boundary error that attaches custom path and sample identity."""


@dataclass(frozen=True)
class RewardWorkerConfig:
    """Serializable whitelist view of args for sync custom rewards."""

    custom_rm_path: str | None = None
    group_rm: bool = False
    reward_key: str | None = None
    rm_type: str | None = None
    rm_url: str | None = None
    implementation_version: int = IMPLEMENTATION_VERSION
    generation: int = 0
    custom_options: dict[str, Any] = field(default_factory=dict)

    def as_namespace(self) -> Namespace:
        data = asdict(self)
        options = data.pop("custom_options") or {}
        ns = Namespace(**data)
        for key, value in options.items():
            setattr(ns, key, value)
        return ns


def _is_jsonish(value: Any, *, depth: int = 0) -> bool:
    if depth > 2:
        return False
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_jsonish(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_jsonish(item, depth=depth + 1) for key, item in value.items())
    return False


def pick_primitive_arg_attrs(args: Any) -> dict[str, Any]:
    """Copy JSON-ish scalar attrs from training args for sync Worker compatibility."""
    source = vars(args) if hasattr(args, "__dict__") else {}
    picked: dict[str, Any] = {}
    for key, value in source.items():
        if key.startswith("_") or key in _CONFIG_FIELD_KEYS or key in _AUTO_PICK_DENYLIST:
            continue
        if _is_jsonish(value):
            picked[key] = value
    return picked


def build_reward_worker_config(
    args: Any,
    *,
    generation: int = 0,
    custom_options: dict[str, Any] | None = None,
) -> RewardWorkerConfig:
    # auto-pick first; explicit custom_options win on key conflicts.
    options = pick_primitive_arg_attrs(args)
    options.update(dict(custom_options or {}))
    return RewardWorkerConfig(
        custom_rm_path=getattr(args, "custom_rm_path", None),
        group_rm=bool(getattr(args, "group_rm", False)),
        reward_key=getattr(args, "reward_key", None),
        rm_type=getattr(args, "rm_type", None),
        rm_url=getattr(args, "rm_url", None),
        implementation_version=IMPLEMENTATION_VERSION,
        generation=generation,
        custom_options=options,
    )


@dataclass(frozen=True)
class WorkerConfigFingerprint:
    num_workers: int
    implementation_version: int = IMPLEMENTATION_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedCustomReward:
    function: Callable
    is_async: bool
    generation: int
    path: str


def is_async_callable(function: Callable) -> bool:
    function = inspect.unwrap(function)
    if inspect.iscoroutinefunction(function):
        return True
    call = getattr(function, "__call__", None)
    if call is not None and inspect.iscoroutinefunction(inspect.unwrap(call)):
        return True
    return False


class CustomRewardResolver:
    """Per-process cache of custom reward callables keyed by (path,
    generation)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], ResolvedCustomReward] = {}
        self._load_counts: dict[str, int] = {}

    def load_count(self, path: str) -> int:
        return self._load_counts.get(path, 0)

    def ensure_loaded(self, path: str, generation: int, *, refresh_modules: bool = False) -> ResolvedCustomReward:
        key = (path, generation)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if refresh_modules:
            from relax.utils.reload_utils import reload_function

            function = reload_function(path)
            if function is None:
                raise RuntimeError(f"Failed to refresh custom reward module for path={path!r}")
        else:
            function = load_function(path)
        self._load_counts[path] = self._load_counts.get(path, 0) + 1
        resolved = ResolvedCustomReward(
            function=function,
            is_async=is_async_callable(function),
            generation=generation,
            path=path,
        )
        self._cache[key] = resolved
        return resolved

    def invalidate(self, path: str) -> None:
        for key in list(self._cache):
            if key[0] == path:
                self._cache.pop(key, None)

    def invalidate_all(self) -> None:
        self._cache.clear()


def sample_identity(sample: Sample) -> dict[str, Any]:
    return {
        "index": getattr(sample, "index", None),
        "group_index": getattr(sample, "group_index", None),
        "session_id": getattr(sample, "session_id", None),
    }


def format_sample_identity(sample: Sample) -> str:
    ident = sample_identity(sample)
    return f"index={ident['index']!r}, group_index={ident['group_index']!r}, session_id={ident['session_id']!r}"


def format_group_identity(samples: list[Sample]) -> str:
    parts = [format_sample_identity(sample) for sample in samples[:8]]
    more = "" if len(samples) <= 8 else f", ... (+{len(samples) - 8} more)"
    return f"n={len(samples)} [{'; '.join(parts)}{more}]"


def wrap_custom_reward_error(
    error: BaseException,
    *,
    path: str,
    sample: Sample | None = None,
    samples: list[Sample] | None = None,
) -> CustomRewardError:
    if sample is not None:
        detail = format_sample_identity(sample)
    elif samples is not None:
        detail = format_group_identity(samples)
    else:
        detail = "unknown sample"
    message = f"Custom reward failed for path={path!r} ({detail}): {type(error).__name__}: {error}"
    return CustomRewardError(message)
