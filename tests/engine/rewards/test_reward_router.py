# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.engine.rewards.registry import (
    REWARD_REGISTRY,
    RewardRegistry,
    resolve_reward_route,
)


def _constant_reward(response: str, label: object, metadata: dict) -> float:
    return 1.0


def test_registry_registers_reward_without_router_changes():
    registry = RewardRegistry()

    registry.register("custom", _constant_reward)

    spec = registry.get("custom")
    assert spec is not None
    assert spec.handler("response", "label", {}) == 1.0
    route = resolve_reward_route({}, None, "custom", registry)
    assert route.rm_type == "custom"
    assert route.source == "fallback"


def test_registry_rejects_duplicate_reward_type():
    registry = RewardRegistry()
    registry.register("custom", _constant_reward)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("custom", _constant_reward)


def test_registry_rejects_unknown_execution_mode():
    registry = RewardRegistry()

    with pytest.raises(ValueError, match="Unsupported reward execution mode"):
        registry.register("custom", _constant_reward, execution="thread")


def test_metadata_route_has_highest_priority():
    route = resolve_reward_route(
        metadata={"rm_type": "math"},
        label="<answer>B</answer>",
        fallback_rm_type="multiple_choice",
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == "math"
    assert route.source == "metadata"
    assert route.warnings == ()


def test_unknown_metadata_type_uses_unique_label_match():
    route = resolve_reward_route(
        metadata={"rm_type": "unknown"},
        label="<answer>B</answer>",
        fallback_rm_type=None,
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == "multiple_choice"
    assert route.source == "label"
    assert "Unknown metadata rm_type" in route.warnings[0]


def test_unknown_metadata_type_uses_configured_fallback_and_warns():
    route = resolve_reward_route(
        metadata={"rm_type": 123},
        label="free-form answer",
        fallback_rm_type="math",
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == "math"
    assert route.source == "fallback"
    assert "Unknown metadata rm_type" in route.warnings[0]


def test_whitespace_metadata_type_is_treated_as_missing():
    route = resolve_reward_route(
        metadata={"rm_type": " \t\n "},
        label="free-form answer",
        fallback_rm_type="math",
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == "math"
    assert route.source == "fallback"
    assert route.warnings == ()


@pytest.mark.parametrize(
    "label",
    ["42", "-3.5", "1 / 2", "\\frac{1}{2}", "\\boxed{7}", "\\boxed{\\frac{1}{2}}", 12, 0.5],
)
def test_math_label_matcher_routes_unambiguous_labels(label):
    route = resolve_reward_route(
        metadata={},
        label=label,
        fallback_rm_type=None,
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == "math"
    assert route.source == "label"


def test_multiple_choice_label_matcher_requires_answer_tag():
    tagged_route = resolve_reward_route(
        metadata={},
        label="<answer> B </answer>",
        fallback_rm_type=None,
        registry=REWARD_REGISTRY,
    )
    bare_route = resolve_reward_route(
        metadata={},
        label="B",
        fallback_rm_type=None,
        registry=REWARD_REGISTRY,
    )

    assert tagged_route.rm_type == "multiple_choice"
    assert tagged_route.source == "label"
    assert bare_route.rm_type is None
    assert bare_route.source == "unresolved"


@pytest.mark.parametrize("label", ["\\boxed{B}", "\\boxed{Paris}", True])
def test_math_label_matcher_rejects_non_math_boxed_and_boolean_labels(label):
    route = resolve_reward_route(
        metadata={},
        label=label,
        fallback_rm_type=None,
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type is None
    assert route.source == "unresolved"


def test_explicit_fallback_takes_priority_over_label_match():
    route = resolve_reward_route(
        metadata={},
        label="<answer>C</answer>",
        fallback_rm_type="math",
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == "math"
    assert route.source == "fallback"


@pytest.mark.parametrize("rm_type", ["dapo", "dapo-genrm", "dummy", "remote_rm", "openr1mm"])
def test_explicit_reward_type_is_not_overridden_by_numeric_label(rm_type):
    route = resolve_reward_route(
        metadata={},
        label="42",
        fallback_rm_type=rm_type,
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == rm_type
    assert route.source == "fallback"


def test_missing_label_match_uses_valid_fallback_without_warning():
    route = resolve_reward_route(
        metadata={},
        label="free-form answer",
        fallback_rm_type="openr1mm",
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == "openr1mm"
    assert route.source == "fallback"
    assert route.warnings == ()


def test_conflicting_label_matches_use_fallback_and_warn():
    registry = RewardRegistry()
    registry.register("first", _constant_reward, lambda label: label == "conflict")
    registry.register("second", _constant_reward, lambda label: label == "conflict")
    registry.register("fallback", _constant_reward)

    route = resolve_reward_route(
        metadata={},
        label="conflict",
        fallback_rm_type="fallback",
        registry=registry,
    )

    assert route.rm_type == "fallback"
    assert route.source == "fallback"
    assert "Conflicting reward types" in route.warnings[0]


def test_conflicting_label_matches_without_fallback_are_unresolved():
    registry = RewardRegistry()
    registry.register("first", _constant_reward, lambda label: label == "conflict")
    registry.register("second", _constant_reward, lambda label: label == "conflict")

    route = resolve_reward_route(
        metadata={},
        label="conflict",
        fallback_rm_type=None,
        registry=registry,
    )

    assert route.rm_type is None
    assert route.source == "unresolved"
    assert "Conflicting reward types" in route.warnings[0]
    assert "returning a zero reward" in route.warnings[-1]


@pytest.mark.parametrize(
    ("metadata", "label", "fallback"),
    [
        ({}, None, None),
        (None, {}, None),
        ({"rm_type": 123}, "free-form answer", None),
        ({}, "free-form answer", "unknown"),
    ],
)
def test_unresolved_route_returns_zero_target_and_warning(metadata, label, fallback):
    route = resolve_reward_route(
        metadata=metadata,
        label=label,
        fallback_rm_type=fallback,
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type is None
    assert route.source == "unresolved"
    assert "returning a zero reward" in route.warnings[-1]


def test_boxed_metadata_and_fallback_keep_compatibility():
    metadata_route = resolve_reward_route(
        metadata={"rm_type": "boxed_math"},
        label="free-form answer",
        fallback_rm_type=None,
        registry=REWARD_REGISTRY,
    )
    fallback_route = resolve_reward_route(
        metadata={},
        label="42",
        fallback_rm_type="boxed_math",
        registry=REWARD_REGISTRY,
    )

    assert metadata_route.rm_type == "math"
    assert metadata_route.strip_boxed is True
    assert fallback_route.rm_type == "math"
    assert fallback_route.strip_boxed is True


def test_async_reward_type_is_registered_for_metadata_routing():
    route = resolve_reward_route(
        metadata={"rm_type": "remote_rm"},
        label=None,
        fallback_rm_type=None,
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == "remote_rm"
    assert route.source == "metadata"
    spec = REWARD_REGISTRY.get("remote_rm")
    assert spec is not None
    assert spec.execution == "async"


def test_async_reward_type_can_be_used_as_fallback():
    route = resolve_reward_route(
        metadata={},
        label=None,
        fallback_rm_type="remote_rm",
        registry=REWARD_REGISTRY,
    )

    assert route.rm_type == "remote_rm"
    assert route.source == "fallback"


def test_mixed_math_and_multiple_choice_batch_routes_per_sample():
    samples = [
        ({}, "42"),
        ({}, "<answer>A</answer>"),
        ({"rm_type": "math"}, "<answer>B</answer>"),
        ({"rm_type": "multiple_choice"}, "not inferable"),
    ]

    routes = [
        resolve_reward_route(
            metadata=metadata,
            label=label,
            fallback_rm_type=None,
            registry=REWARD_REGISTRY,
        )
        for metadata, label in samples
    ]

    assert [route.rm_type for route in routes] == [
        "math",
        "multiple_choice",
        "math",
        "multiple_choice",
    ]
