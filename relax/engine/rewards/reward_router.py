# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping


RewardMode = Literal["sync", "async"]
RewardSource = Literal["metadata", "global", "label", "fallback", "none"]
RewardRouteReason = Literal["unknown", "missing", "conflict"]
LabelMatcher = Callable[[object, dict], bool]


@dataclass(frozen=True)
class RewardSpec:
    mode: RewardMode
    handler: Callable
    label_matcher: LabelMatcher | None = None


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


def validate_explicit_reward_type(value: object, registry: Mapping[str, RewardSpec]) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return
    if _registered_reward_type(value, registry) is None:
        raise ValueError(f"Explicit --rm-type {value!r} is not registered.")


def resolve_reward_route(
    metadata: object,
    label: object,
    explicit_rm_type: object,
    registry: Mapping[str, RewardSpec],
) -> ResolvedRewardRoute:
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    metadata_value = metadata_dict.get("rm_type")
    metadata_missing = metadata_value is None or (isinstance(metadata_value, str) and not metadata_value.strip())

    if not metadata_missing:
        registered = _registered_reward_type(metadata_value, registry)
        if registered is not None:
            reward_type, boxed = registered
            return ResolvedRewardRoute(reward_type=reward_type, source="metadata", boxed=boxed)

        fallback = _registered_reward_type(explicit_rm_type, registry)
        if fallback is not None:
            reward_type, boxed = fallback
            return ResolvedRewardRoute(
                reward_type=reward_type,
                source="fallback",
                boxed=boxed,
                reason="unknown",
                candidates=(repr(metadata_value),),
            )
        return ResolvedRewardRoute(
            reward_type=None,
            source="none",
            reason="unknown",
            candidates=(repr(metadata_value),),
        )

    if explicit_rm_type is not None and not (isinstance(explicit_rm_type, str) and not explicit_rm_type.strip()):
        registered = _registered_reward_type(explicit_rm_type, registry)
        if registered is not None:
            reward_type, boxed = registered
            return ResolvedRewardRoute(reward_type=reward_type, source="global", boxed=boxed)
        return ResolvedRewardRoute(
            reward_type=None,
            source="none",
            reason="unknown",
            candidates=(repr(explicit_rm_type),),
        )

    matches = tuple(
        name
        for name, spec in registry.items()
        if spec.label_matcher is not None and spec.label_matcher(label, metadata_dict)
    )
    if len(matches) == 1:
        return ResolvedRewardRoute(reward_type=matches[0], source="label")
    if len(matches) > 1:
        return ResolvedRewardRoute(
            reward_type=None,
            source="none",
            reason="conflict",
            candidates=matches,
        )
    return ResolvedRewardRoute(reward_type=None, source="none", reason="missing")


def preflight_reward_routes(
    records: Iterable[tuple[int, object, object]],
    explicit_rm_type: object,
    registry: Mapping[str, RewardSpec],
    max_reported_indices: int = 10,
) -> RewardRoutePreflightReport:
    validate_explicit_reward_type(explicit_rm_type, registry)

    assignments: Counter[str] = Counter()
    total = 0
    fallback_count = 0
    unresolved_count = 0
    conflict_count = 0
    unresolved_indices: list[int] = []
    conflict_indices: list[int] = []

    for index, label, metadata in records:
        total += 1
        route = resolve_reward_route(metadata, label, explicit_rm_type, registry)
        if route.reward_type is not None:
            assignments[f"{route.source}/{route.reward_type}"] += 1
            if route.source == "fallback":
                fallback_count += 1
            continue

        unresolved_count += 1
        if len(unresolved_indices) < max_reported_indices:
            unresolved_indices.append(index)
        if route.reason == "conflict":
            conflict_count += 1
            if len(conflict_indices) < max_reported_indices:
                conflict_indices.append(index)

    return RewardRoutePreflightReport(
        total=total,
        assignments=dict(sorted(assignments.items())),
        fallback_count=fallback_count,
        unresolved_count=unresolved_count,
        conflict_count=conflict_count,
        unresolved_indices=tuple(unresolved_indices),
        conflict_indices=tuple(conflict_indices),
    )
