# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import grade_answer_verl
from .multiple_choice import get_multiple_choice_reward
from .openr1mm import get_openr1mm_rule_based_reward


LabelMatcher = Callable[[Any], bool]
RouteSource = Literal["metadata", "label", "fallback", "unresolved"]
ExecutionMode = Literal["sync", "async"]


@dataclass(frozen=True)
class RewardSpec:
    handler: Callable[..., Any]
    label_matcher: LabelMatcher | None = None
    execution: ExecutionMode = "sync"


@dataclass(frozen=True)
class RewardRoute:
    rm_type: str | None
    source: RouteSource
    strip_boxed: bool = False
    warnings: tuple[str, ...] = ()


class RewardRegistry:
    """Registry for reward handlers and optional label matchers."""

    def __init__(self):
        self._specs: dict[str, RewardSpec] = {}

    def register(
        self,
        rm_type: str,
        handler: Callable[..., Any],
        label_matcher: LabelMatcher | None = None,
        execution: ExecutionMode = "sync",
    ) -> None:
        name = rm_type.strip()
        if not name:
            raise ValueError("Reward type must be a non-empty string.")
        if name in self._specs:
            raise ValueError(f"Reward type {name!r} is already registered.")
        if execution not in ("sync", "async"):
            raise ValueError(f"Unsupported reward execution mode: {execution!r}.")
        self._specs[name] = RewardSpec(handler=handler, label_matcher=label_matcher, execution=execution)

    def get(self, rm_type: str) -> RewardSpec | None:
        return self._specs.get(rm_type)

    def __contains__(self, rm_type: object) -> bool:
        return rm_type in self._specs

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def matching_types(self, label: Any) -> tuple[str, ...]:
        return tuple(
            rm_type
            for rm_type, spec in self._specs.items()
            if spec.label_matcher is not None and spec.label_matcher(label)
        )


def _normalize_rm_type(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _resolve_registered_type(
    value: object,
    registry: RewardRegistry,
) -> tuple[str | None, bool]:
    rm_type = _normalize_rm_type(value)
    if not rm_type:
        return None, False

    strip_boxed = rm_type.startswith("boxed_")
    base_type = rm_type[len("boxed_") :] if strip_boxed else rm_type
    if base_type in registry:
        return base_type, strip_boxed
    return None, strip_boxed


def resolve_reward_route(
    metadata: object,
    label: object,
    fallback_rm_type: object,
    registry: RewardRegistry,
) -> RewardRoute:
    warnings = []
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    metadata_rm_type = metadata_dict.get("rm_type")
    resolved_type, strip_boxed = _resolve_registered_type(metadata_rm_type, registry)
    if resolved_type is not None:
        return RewardRoute(resolved_type, "metadata", strip_boxed=strip_boxed)

    raw_metadata_type = _normalize_rm_type(metadata_rm_type)
    if "rm_type" in metadata_dict and metadata_rm_type not in (None, ""):
        display_type = raw_metadata_type or metadata_rm_type
        warnings.append(f"Unknown metadata rm_type {display_type!r}; trying label routing and fallback.")

    matching_types = registry.matching_types(label)
    if len(matching_types) > 1:
        warnings.append(f"Conflicting reward types inferred from label: {', '.join(matching_types)}.")

    fallback_type, fallback_strip_boxed = _resolve_registered_type(
        fallback_rm_type,
        registry,
    )
    if fallback_type is not None:
        return RewardRoute(
            fallback_type,
            "fallback",
            strip_boxed=fallback_strip_boxed,
            warnings=tuple(warnings),
        )

    raw_fallback_type = _normalize_rm_type(fallback_rm_type)
    if raw_fallback_type:
        warnings.append(f"Unknown fallback rm_type {raw_fallback_type!r}.")
    if len(matching_types) == 1:
        return RewardRoute(matching_types[0], "label", warnings=tuple(warnings))

    warnings.append("Unable to resolve a reward type; returning a zero reward.")
    return RewardRoute(None, "unresolved", warnings=tuple(warnings))


_MULTIPLE_CHOICE_LABEL = re.compile(r"<answer>\s*[A-Za-z]\s*</answer>", re.DOTALL)
_NUMERIC_LABEL = re.compile(
    r"""
    [+-]?
    (?:
        (?:\d[\d,]*)(?:\.\d+)?
        |
        (?:\d[\d,]*)?\.\d+
    )
    (?:\s*/\s*[+-]?\d[\d,]*(?:\.\d+)?)?
    """,
    re.VERBOSE,
)
_LATEX_FRACTION_LABEL = re.compile(r"\\(?:d)?frac\{[+-]?\d+\}\{[+-]?\d+\}")


def _is_multiple_choice_label(label: Any) -> bool:
    return isinstance(label, str) and _MULTIPLE_CHOICE_LABEL.fullmatch(label.strip()) is not None


def _is_math_label(label: Any) -> bool:
    if not isinstance(label, (str, int, float)) or isinstance(label, bool):
        return False
    text = str(label).strip()
    if text.startswith("\\boxed{") and text.endswith("}"):
        text = text[len("\\boxed{") : -1].strip()
    return _NUMERIC_LABEL.fullmatch(text) is not None or _LATEX_FRACTION_LABEL.fullmatch(text) is not None


def _deepscaler_reward(response: str, label: Any, metadata: dict) -> Any:
    return get_deepscaler_rule_based_reward(response, label)


def _geo3k_reward(response: str, label: Any, metadata: dict) -> Any:
    from .geo3k import get_geo3k_reward

    return get_geo3k_reward(response, label)


def _openr1mm_reward(response: str, label: Any, metadata: dict) -> Any:
    return get_openr1mm_rule_based_reward(response, label)


def _multiple_choice_reward(response: str, label: Any, metadata: dict) -> Any:
    return get_multiple_choice_reward(response, label)


def _dapo_reward(response: str, label: Any, metadata: dict) -> Any:
    return compute_score_dapo(response, label)


def _math_reward(response: str, label: Any, metadata: dict) -> Any:
    return 1 if grade_answer_verl(response, label) else 0


def _mopd_reward(response: str, label: Any, metadata: dict) -> Any:
    from .mopd import get_mopd_reward

    return get_mopd_reward(response, label, metadata)


def _f1_reward(response: str, label: Any, metadata: dict) -> Any:
    return f1_score(response, label)[0]


def _gpqa_reward(response: str, label: Any, metadata: dict) -> Any:
    return compute_gpqa_reward(response, label, metadata=metadata)


def _ifbench_reward(response: str, label: Any, metadata: dict) -> Any:
    from .ifbench import compute_ifbench_reward

    return compute_ifbench_reward(response, label, metadata=metadata)


def _random_reward(response: str, label: Any, metadata: dict) -> Any:
    return random.randint(0, 1)


REWARD_REGISTRY = RewardRegistry()
REWARD_REGISTRY.register("deepscaler", _deepscaler_reward)
REWARD_REGISTRY.register("geo3k", _geo3k_reward)
REWARD_REGISTRY.register("openr1mm", _openr1mm_reward)
REWARD_REGISTRY.register("multiple_choice", _multiple_choice_reward, _is_multiple_choice_label)
REWARD_REGISTRY.register("dapo", _dapo_reward)
REWARD_REGISTRY.register("math", _math_reward, _is_math_label)
REWARD_REGISTRY.register("mopd", _mopd_reward)
REWARD_REGISTRY.register("f1", _f1_reward)
REWARD_REGISTRY.register("gpqa", _gpqa_reward)
REWARD_REGISTRY.register("ifbench", _ifbench_reward)
REWARD_REGISTRY.register("random", _random_reward)
