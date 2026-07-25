# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping


RewardMode = Literal["sync", "async"]
RewardSource = Literal["metadata", "global", "label", "none"]
RewardRouteReason = Literal["unknown", "missing", "conflict"]
RewardRouteStage = Literal["metadata", "label", "rm_type"]
LabelMatcher = Callable[[object, dict], bool]

DEFAULT_REWARD_ROUTE_PRIORITY: tuple[RewardRouteStage, ...] = ("metadata", "label", "rm_type")
_REWARD_ROUTE_STAGES = frozenset(DEFAULT_REWARD_ROUTE_PRIORITY)
_REWARD_ROUTE_CONFIG_KEYS = frozenset(("priority",))


@dataclass(frozen=True)
class RewardSpec:
    mode: RewardMode
    handler: Callable
    label_matcher: LabelMatcher | None = None


@dataclass(frozen=True)
class RewardRouteConfig:
    priority: tuple[RewardRouteStage, ...] = DEFAULT_REWARD_ROUTE_PRIORITY


@dataclass(frozen=True)
class ResolvedRewardRoute:
    reward_type: str | None
    source: RewardSource
    boxed: bool = False
    reason: RewardRouteReason | None = None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewardRoutePreflightReport:
    total: int
    assignments: dict[str, int]
    fallback_count: int
    unresolved_count: int
    conflict_count: int
    unresolved_indices: tuple[int, ...]
    conflict_indices: tuple[int, ...]

    @property
    def is_valid(self) -> bool:
        return self.unresolved_count == 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "assignments": dict(self.assignments),
            "fallback_count": self.fallback_count,
            "unresolved_count": self.unresolved_count,
            "conflict_count": self.conflict_count,
            "unresolved_indices": list(self.unresolved_indices),
            "conflict_indices": list(self.conflict_indices),
        }


_MULTIPLE_CHOICE_LABEL = re.compile(r"\s*<answer>\s*[A-Za-z]\s*</answer>\s*", re.DOTALL)
_MATH_LABEL = re.compile(
    r"""
    \s*(?:
        [-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+))?
        | \\frac\s*\{\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*\}\s*\{\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*\}
        | \\boxed\s*\{\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+))?\s*\}
    )\s*
    """,
    re.VERBOSE,
)


def match_multiple_choice_label(label: object, metadata: dict) -> bool:
    del metadata
    return isinstance(label, str) and _MULTIPLE_CHOICE_LABEL.fullmatch(label) is not None


def match_math_label(label: object, metadata: dict) -> bool:
    del metadata
    return isinstance(label, str) and _MATH_LABEL.fullmatch(label) is not None


def build_reward_registry(entries: Iterable[tuple[str, RewardSpec]]) -> Mapping[str, RewardSpec]:
    registry: dict[str, RewardSpec] = {}
    for name, spec in entries:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError(f"Reward registry names must be non-empty normalized strings, got {name!r}.")
        if name in registry:
            raise ValueError(f"Duplicate reward registry entry: {name!r}.")
        if not isinstance(spec, RewardSpec):
            raise TypeError(f"Reward registry entry {name!r} must be a RewardSpec, got {type(spec).__name__}.")
        if spec.mode not in ("sync", "async"):
            raise ValueError(f"Reward registry entry {name!r} has invalid mode {spec.mode!r}.")
        if not callable(spec.handler):
            raise TypeError(f"Reward registry handler for {name!r} must be callable.")
        if spec.label_matcher is not None and not callable(spec.label_matcher):
            raise TypeError(f"Reward registry label matcher for {name!r} must be callable.")
        registry[name] = spec
    return MappingProxyType(registry)


def _registered_reward_type(value: object, registry: Mapping[str, RewardSpec]) -> tuple[str, bool] | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    boxed = normalized.startswith("boxed_")
    reward_type = normalized[len("boxed_") :] if boxed else normalized
    if reward_type not in registry:
        return None
    return reward_type, boxed


def normalize_reward_route_priority(priority: Iterable[str] | None) -> tuple[RewardRouteStage, ...]:
    if priority is None:
        return DEFAULT_REWARD_ROUTE_PRIORITY
    normalized = tuple(priority)
    if len(normalized) != len(_REWARD_ROUTE_STAGES) or set(normalized) != _REWARD_ROUTE_STAGES:
        raise ValueError(
            f"Reward route priority must contain metadata, label, and rm_type exactly once, got {normalized!r}."
        )
    return normalized


def normalize_reward_route_config(config: object) -> RewardRouteConfig:
    if config is None:
        return RewardRouteConfig()
    if not isinstance(config, Mapping):
        raise ValueError(f"reward_route must be a mapping, got {type(config).__name__}.")

    unknown_keys = set(config) - _REWARD_ROUTE_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown reward_route config keys: {sorted(unknown_keys)!r}.")

    priority = normalize_reward_route_priority(config.get("priority"))
    return RewardRouteConfig(priority=priority)


def resolve_reward_route(
    metadata: object,
    label: object,
    explicit_rm_type: object,
    registry: Mapping[str, RewardSpec],
    priority: Iterable[str] | None = None,
) -> ResolvedRewardRoute:
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    metadata_value = metadata_dict.get("rm_type")
    metadata_missing = metadata_value is None or (isinstance(metadata_value, str) and not metadata_value.strip())
    explicit_missing = explicit_rm_type is None or (isinstance(explicit_rm_type, str) and not explicit_rm_type.strip())
    first_problem: tuple[RewardRouteReason, tuple[str, ...]] | None = None

    for stage in normalize_reward_route_priority(priority):
        if stage == "metadata":
            if metadata_missing:
                continue
            registered = _registered_reward_type(metadata_value, registry)
            if registered is not None:
                reward_type, boxed = registered
                reason, candidates = first_problem or (None, ())
                return ResolvedRewardRoute(
                    reward_type=reward_type,
                    source="metadata",
                    boxed=boxed,
                    reason=reason,
                    candidates=candidates,
                )
            if first_problem is None:
                first_problem = ("unknown", (repr(metadata_value),))
            continue

        if stage == "label":
            matches = tuple(
                name
                for name, spec in registry.items()
                if spec.label_matcher is not None and spec.label_matcher(label, metadata_dict)
            )
            if len(matches) == 1:
                reason, candidates = first_problem or (None, ())
                return ResolvedRewardRoute(
                    reward_type=matches[0],
                    source="label",
                    reason=reason,
                    candidates=candidates,
                )
            if len(matches) > 1 and first_problem is None:
                first_problem = ("conflict", matches)
            continue

        if explicit_missing:
            continue
        registered = _registered_reward_type(explicit_rm_type, registry)
        if registered is not None:
            reward_type, boxed = registered
            reason, candidates = first_problem or (None, ())
            return ResolvedRewardRoute(
                reward_type=reward_type,
                source="global",
                boxed=boxed,
                reason=reason,
                candidates=candidates,
            )
        if first_problem is None:
            first_problem = ("unknown", (repr(explicit_rm_type),))

    reason, candidates = first_problem or ("missing", ())
    return ResolvedRewardRoute(
        reward_type=None,
        source="none",
        reason=reason,
        candidates=candidates,
    )


def preflight_reward_routes(
    records: Iterable[tuple[int, object, object]],
    explicit_rm_type: object,
    registry: Mapping[str, RewardSpec],
    priority: Iterable[str] | None = None,
    max_reported_indices: int = 10,
) -> RewardRoutePreflightReport:
    normalized_priority = normalize_reward_route_priority(priority)

    assignments: Counter[str] = Counter()
    total = 0
    fallback_count = 0
    unresolved_count = 0
    conflict_count = 0
    unresolved_indices: list[int] = []
    conflict_indices: list[int] = []

    for index, label, metadata in records:
        total += 1
        route = resolve_reward_route(metadata, label, explicit_rm_type, registry, normalized_priority)
        if route.reason == "conflict":
            conflict_count += 1
            if len(conflict_indices) < max_reported_indices:
                conflict_indices.append(index)
        if route.reward_type is not None:
            assignments[f"{route.source}/{route.reward_type}"] += 1
            if route.reason is not None:
                fallback_count += 1
            continue

        unresolved_count += 1
        if len(unresolved_indices) < max_reported_indices:
            unresolved_indices.append(index)

    return RewardRoutePreflightReport(
        total=total,
        assignments=dict(sorted(assignments.items())),
        fallback_count=fallback_count,
        unresolved_count=unresolved_count,
        conflict_count=conflict_count,
        unresolved_indices=tuple(unresolved_indices),
        conflict_indices=tuple(conflict_indices),
    )
