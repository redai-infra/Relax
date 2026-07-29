# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal


RewardMode = Literal["sync", "async"]
RewardAction = Literal["dispatch", "zero", "error"]
RewardHandler = Callable[..., Any]
LabelMatcher = Callable[[Any, dict[str, Any]], bool]


@dataclass(frozen=True)
class RewardSpec:
    """A registered reward handler and its execution requirements."""

    name: str
    mode: RewardMode
    handler: RewardHandler
    label_matcher: LabelMatcher | None = None


@dataclass(frozen=True)
class RouteDecision:
    """The deterministic result of resolving one sample's reward route."""

    action: RewardAction
    reward_type: str | None
    source: str
    reason: str | None = None
    requested_type: str | None = None
    boxed: bool = False


class RewardRegistry:
    """Import-time reward registrations, read-only during reward execution."""

    def __init__(self) -> None:
        self._specs: dict[str, RewardSpec] = {}

    def register(self, spec: RewardSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ValueError("Reward type name must be non-empty.")
        if spec.mode not in ("sync", "async"):
            raise ValueError(f"Reward {name!r} has invalid execution mode {spec.mode!r}.")
        if name in self._specs:
            raise ValueError(f"Reward type {name!r} is already registered.")
        self._specs[name] = spec

    def get(self, name: str | None) -> RewardSpec | None:
        if not isinstance(name, str):
            return None
        return self._specs.get(name.strip())

    def match_label(self, label: Any, metadata: dict[str, Any] | None = None) -> list[str]:
        metadata = metadata if isinstance(metadata, dict) else {}
        matches = [
            name
            for name, spec in self._specs.items()
            if spec.label_matcher is not None and spec.label_matcher(label, metadata)
        ]
        return sorted(matches)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)


_ANSWER_TAG = re.compile(r"^\s*<answer>\s*(.*?)\s*</answer>\s*$", re.IGNORECASE | re.DOTALL)
_BOXED = re.compile(r"^\s*\\boxed\{(.*)\}\s*$", re.DOTALL)
_MULTIPLE_CHOICE_LABEL = re.compile(r"^[A-Z]$", re.IGNORECASE)
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?$")
_LATEX_MATH = re.compile(r"(?:\\(?:frac|sqrt|pi|cdot|times|left|right|begin|end|sin|cos|tan|log|ln)\b|[\^_=])")


def _unwrap_label(label: Any) -> str | None:
    if not isinstance(label, str):
        return None
    value = label.strip()
    answer_match = _ANSWER_TAG.fullmatch(value)
    if answer_match is not None:
        value = answer_match.group(1).strip()
    boxed_match = _BOXED.fullmatch(value)
    if boxed_match is not None:
        value = boxed_match.group(1).strip()
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1].strip()
    return value


def match_multiple_choice_label(label: Any, metadata: dict[str, Any]) -> bool:
    """Match explicit choices metadata or an answer-tagged single-letter
    label."""

    choices = metadata.get("choices", metadata.get("options"))
    if isinstance(choices, (list, tuple)) and len(choices) >= 2:
        return True
    if not isinstance(label, str):
        return False
    answer_match = _ANSWER_TAG.fullmatch(label)
    return answer_match is not None and _MULTIPLE_CHOICE_LABEL.fullmatch(answer_match.group(1).strip()) is not None


def match_math_label(label: Any, metadata: dict[str, Any]) -> bool:
    """Conservatively recognize numeric, fractional, or explicit LaTeX math
    labels."""

    choices = metadata.get("choices", metadata.get("options"))
    if isinstance(choices, (list, tuple)) and len(choices) >= 2:
        return False
    value = _unwrap_label(label)
    if not value or _MULTIPLE_CHOICE_LABEL.fullmatch(value):
        return False
    if _NUMBER.fullmatch(value):
        return True
    return bool(_LATEX_MATH.search(value) and (re.search(r"\d", value) or "\\pi" in value))


def _split_boxed_prefix(raw_type: Any) -> tuple[str | None, bool]:
    if not isinstance(raw_type, str):
        return None, False
    name = raw_type.strip()
    if not name:
        return None, False
    if name.startswith("boxed_"):
        return name[len("boxed_") :], True
    return name, False


def validate_reward_fallback(registry: RewardRegistry, fallback_type: str | None) -> None:
    fallback = "zero" if fallback_type is None else fallback_type.strip()
    if fallback in ("zero", "error"):
        return
    reward_type, _ = _split_boxed_prefix(fallback)
    if reward_type is None or registry.get(reward_type) is None:
        raise ValueError(
            f"Unknown --rm-type-fallback value {fallback_type!r}; expected 'zero', 'error', "
            f"or one of {registry.names}."
        )


def _fallback_decision(
    *,
    registry: RewardRegistry,
    fallback_type: str | None,
    source: str,
    reason: str,
    requested_type: str | None,
) -> RouteDecision:
    fallback = "zero" if fallback_type is None else fallback_type.strip()
    if fallback == "zero":
        return RouteDecision(
            action="zero",
            reward_type=None,
            source=source,
            reason=reason,
            requested_type=requested_type,
        )
    if fallback == "error":
        return RouteDecision(
            action="error",
            reward_type=None,
            source=source,
            reason=reason,
            requested_type=requested_type,
        )

    reward_type, boxed = _split_boxed_prefix(fallback)
    assert reward_type is not None and registry.get(reward_type) is not None
    return RouteDecision(
        action="dispatch",
        reward_type=reward_type,
        source=f"fallback:{source}",
        reason=reason,
        requested_type=requested_type,
        boxed=boxed,
    )


def resolve_reward_route(
    *,
    registry: RewardRegistry,
    metadata: dict[str, Any] | None,
    label: Any,
    explicit_rm_type: str | None,
    fallback_type: str | None = "zero",
) -> RouteDecision:
    """Resolve one reward route without performing logging or computation."""

    validate_reward_fallback(registry, fallback_type)
    metadata = metadata if isinstance(metadata, dict) else {}

    if "rm_type" in metadata and metadata["rm_type"] is not None:
        metadata_type, boxed = _split_boxed_prefix(metadata["rm_type"])
        requested_type = metadata["rm_type"] if isinstance(metadata["rm_type"], str) else repr(metadata["rm_type"])
        if metadata_type is None:
            return _fallback_decision(
                registry=registry,
                fallback_type=fallback_type,
                source="metadata",
                reason="invalid_metadata_type",
                requested_type=requested_type,
            )
        metadata_spec = registry.get(metadata_type)
        if metadata_spec is None:
            return _fallback_decision(
                registry=registry,
                fallback_type=fallback_type,
                source="metadata",
                reason="unknown_metadata_type",
                requested_type=requested_type,
            )
        if metadata_spec.label_matcher is not None:
            label_matches = registry.match_label(label, metadata)
            if len(label_matches) > 1 or (len(label_matches) == 1 and label_matches[0] != metadata_type):
                return _fallback_decision(
                    registry=registry,
                    fallback_type=fallback_type,
                    source="metadata+label",
                    reason="conflicting_reward_types",
                    requested_type=requested_type,
                )
        return RouteDecision(
            action="dispatch",
            reward_type=metadata_type,
            source="metadata",
            requested_type=requested_type,
            boxed=boxed,
        )

    if explicit_rm_type is not None:
        cli_type, boxed = _split_boxed_prefix(explicit_rm_type)
        if cli_type is None:
            return _fallback_decision(
                registry=registry,
                fallback_type=fallback_type,
                source="cli",
                reason="invalid_cli_type",
                requested_type=explicit_rm_type,
            )
        if registry.get(cli_type) is None:
            return _fallback_decision(
                registry=registry,
                fallback_type=fallback_type,
                source="cli",
                reason="unknown_cli_type",
                requested_type=explicit_rm_type,
            )
        return RouteDecision(
            action="dispatch",
            reward_type=cli_type,
            source="cli",
            requested_type=explicit_rm_type,
            boxed=boxed,
        )

    label_matches = registry.match_label(label, metadata)
    if len(label_matches) == 1:
        return RouteDecision(
            action="dispatch",
            reward_type=label_matches[0],
            source="label",
            requested_type=label_matches[0],
        )
    if len(label_matches) > 1:
        return _fallback_decision(
            registry=registry,
            fallback_type=fallback_type,
            source="label",
            reason="ambiguous_label_type",
            requested_type=",".join(label_matches),
        )
    return _fallback_decision(
        registry=registry,
        fallback_type=fallback_type,
        source="label",
        reason="missing_reward_type",
        requested_type=None,
    )
