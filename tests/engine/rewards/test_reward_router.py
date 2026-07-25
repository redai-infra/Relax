# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.rewards.registry import (
    RewardRegistry,
    RewardSpec,
    match_math_label,
    match_multiple_choice_label,
    resolve_reward_route,
)


def _sync_handler(response, label, metadata):
    return response, label, metadata


async def _async_handler(args, sample):
    return args, sample


def _registry() -> RewardRegistry:
    registry = RewardRegistry()
    registry.register(RewardSpec("math", "sync", _sync_handler, match_math_label))
    registry.register(RewardSpec("multiple_choice", "sync", _sync_handler, match_multiple_choice_label))
    registry.register(RewardSpec("remote_rm", "async", _async_handler))
    return registry


def test_registry_registers_sync_and_async_rewards():
    registry = _registry()

    assert registry.names == ("math", "multiple_choice", "remote_rm")
    assert registry.get("math").handler is _sync_handler
    assert registry.get("remote_rm").mode == "async"
    assert registry.get("missing") is None


def test_registry_rejects_duplicate_empty_and_invalid_registrations():
    registry = _registry()

    with pytest.raises(ValueError, match="already registered"):
        registry.register(RewardSpec("math", "sync", _sync_handler))
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(RewardSpec(" ", "sync", _sync_handler))
    with pytest.raises(ValueError, match="invalid execution mode"):
        registry.register(RewardSpec("bad", "thread", _sync_handler))


@pytest.mark.parametrize(
    ("label", "metadata", "expected"),
    [
        ("9", {}, ["math"]),
        ("<answer> -3/4 </answer>", {}, ["math"]),
        ("\\boxed{\\frac{1}{2}}", {}, ["math"]),
        ("<answer>B</answer>", {}, ["multiple_choice"]),
        ("B", {"choices": ["A", "B", "C"]}, ["multiple_choice"]),
        ("1", {"choices": ["1", "2", "3"]}, ["multiple_choice"]),
        ("B", {}, []),
        ("plain text", {}, []),
    ],
)
def test_registry_matches_only_conservative_label_formats(label, metadata, expected):
    assert _registry().match_label(label, metadata) == expected


def test_metadata_routes_each_supported_reward_type():
    registry = _registry()

    math = resolve_reward_route(
        registry=registry,
        metadata={"rm_type": "math"},
        label="9",
        explicit_rm_type=None,
    )
    multiple_choice = resolve_reward_route(
        registry=registry,
        metadata={"rm_type": "multiple_choice"},
        label="<answer>B</answer>",
        explicit_rm_type=None,
    )

    assert (math.action, math.reward_type, math.source) == ("dispatch", "math", "metadata")
    assert (multiple_choice.action, multiple_choice.reward_type, multiple_choice.source) == (
        "dispatch",
        "multiple_choice",
        "metadata",
    )


def test_explicit_cli_preserves_legacy_global_route():
    registry = _registry()

    def unexpected_matcher(label, metadata):
        raise AssertionError("Label matchers must not run for an explicit CLI route.")

    registry.register(RewardSpec("unrelated", "sync", _sync_handler, unexpected_matcher))
    decision = resolve_reward_route(
        registry=registry,
        metadata={},
        label="<answer>B</answer>",
        explicit_rm_type="math",
    )

    assert decision.action == "dispatch"
    assert decision.reward_type == "math"
    assert decision.source == "cli"


def test_metadata_route_takes_precedence_over_cli():
    decision = resolve_reward_route(
        registry=_registry(),
        metadata={"rm_type": "multiple_choice"},
        label="<answer>B</answer>",
        explicit_rm_type="math",
    )

    assert decision.action == "dispatch"
    assert decision.reward_type == "multiple_choice"
    assert decision.source == "metadata"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("9", "math"),
        ("<answer>B</answer>", "multiple_choice"),
    ],
)
def test_label_routes_when_metadata_and_cli_are_missing(label, expected):
    decision = resolve_reward_route(
        registry=_registry(),
        metadata={},
        label=label,
        explicit_rm_type=None,
    )

    assert decision.action == "dispatch"
    assert decision.reward_type == expected
    assert decision.source == "label"


def test_metadata_and_label_conflict_returns_zero():
    decision = resolve_reward_route(
        registry=_registry(),
        metadata={"rm_type": "math"},
        label="<answer>B</answer>",
        explicit_rm_type=None,
    )

    assert decision.action == "zero"
    assert decision.reason == "conflicting_reward_types"
    assert decision.source == "metadata+label"


def test_multiple_label_matches_return_zero():
    registry = _registry()
    registry.register(RewardSpec("numeric", "sync", _sync_handler, match_math_label))

    decision = resolve_reward_route(
        registry=registry,
        metadata={},
        label="1",
        explicit_rm_type=None,
    )

    assert decision.action == "zero"
    assert decision.reason == "ambiguous_label_type"
    assert decision.requested_type == "math,numeric"


@pytest.mark.parametrize(
    ("metadata", "explicit_rm_type", "reason"),
    [
        ({"rm_type": "unknown"}, None, "unknown_metadata_type"),
        ({"rm_type": 123}, None, "invalid_metadata_type"),
        ({}, "unknown", "unknown_cli_type"),
        ({}, None, "missing_reward_type"),
    ],
)
def test_unknown_invalid_and_missing_routes_return_zero(metadata, explicit_rm_type, reason):
    decision = resolve_reward_route(
        registry=_registry(),
        metadata=metadata,
        label="not typed",
        explicit_rm_type=explicit_rm_type,
    )

    assert decision.action == "zero"
    assert decision.reason == reason


def test_configured_reward_fallback_dispatches():
    decision = resolve_reward_route(
        registry=_registry(),
        metadata={"rm_type": "unknown"},
        label="not typed",
        explicit_rm_type=None,
        fallback_type="math",
    )

    assert decision.action == "dispatch"
    assert decision.reward_type == "math"
    assert decision.source == "fallback:metadata"
    assert decision.reason == "unknown_metadata_type"


def test_error_fallback_returns_error_decision():
    decision = resolve_reward_route(
        registry=_registry(),
        metadata={},
        label="not typed",
        explicit_rm_type=None,
        fallback_type="error",
    )

    assert decision.action == "error"
    assert decision.reason == "missing_reward_type"


@pytest.mark.parametrize(
    ("metadata", "label"),
    [
        ({}, "not typed"),
        ({"rm_type": "math"}, "9"),
    ],
)
def test_invalid_fallback_fails_before_routing(metadata, label):
    with pytest.raises(ValueError, match="Unknown --rm-type-fallback"):
        resolve_reward_route(
            registry=_registry(),
            metadata=metadata,
            label=label,
            explicit_rm_type=None,
            fallback_type="missing",
        )


def test_boxed_prefix_is_preserved_in_route_decision():
    decision = resolve_reward_route(
        registry=_registry(),
        metadata={},
        label="9",
        explicit_rm_type="boxed_math",
    )

    assert decision.action == "dispatch"
    assert decision.reward_type == "math"
    assert decision.boxed is True


def test_registering_one_reward_adds_routing_without_router_changes():
    registry = _registry()
    registry.register(RewardSpec("boolean", "sync", _sync_handler, lambda label, metadata: label in {True, False}))

    decision = resolve_reward_route(
        registry=registry,
        metadata={},
        label=True,
        explicit_rm_type=None,
    )

    assert decision.action == "dispatch"
    assert decision.reward_type == "boolean"
    assert decision.source == "label"


def test_route_decision_is_compatible_with_namespace_configuration():
    args = SimpleNamespace(rm_type=None, rm_type_fallback="zero")

    decision = resolve_reward_route(
        registry=_registry(),
        metadata={},
        label=None,
        explicit_rm_type=args.rm_type,
        fallback_type=args.rm_type_fallback,
    )

    assert decision.action == "zero"
