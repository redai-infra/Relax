# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from relax.engine.rewards import REWARD_REGISTRY, RewardExecutor, _zero_reward
from relax.engine.rewards.reward_router import (
    RewardSpec,
    build_reward_registry,
    normalize_reward_route_config,
    normalize_reward_route_priority,
    preflight_reward_routes,
    resolve_reward_route,
)
from relax.utils.types import Sample


def _handler(*args):
    del args
    return 0.0


def _registry(*entries):
    return build_reward_registry(entries)


def _issue_reasons(route):
    return tuple(issue.reason for issue in route.issues)


def test_metadata_route_keeps_existing_priority_and_skips_matcher():
    def unexpected_matcher(label, metadata):
        del label, metadata
        raise AssertionError("label matcher must not run")

    registry = _registry(
        ("metadata_reward", RewardSpec("sync", _handler)),
        ("global_reward", RewardSpec("sync", _handler)),
        ("label_reward", RewardSpec("sync", _handler, unexpected_matcher)),
    )

    route = resolve_reward_route(
        {"rm_type": "metadata_reward"},
        "label",
        "global_reward",
        registry,
    )

    assert route.reward_type == "metadata_reward"
    assert route.source == "metadata"
    assert _issue_reasons(route) == ()


def test_label_route_precedes_global_route_by_default():
    registry = _registry(
        ("global_reward", RewardSpec("sync", _handler)),
        ("label_reward", RewardSpec("sync", _handler, lambda label, metadata: True)),
    )

    route = resolve_reward_route({}, "label", "global_reward", registry)

    assert route.reward_type == "label_reward"
    assert route.source == "label"


def test_custom_priority_can_fall_back_to_global_before_label_matcher():
    def unexpected_matcher(label, metadata):
        del label, metadata
        raise AssertionError("label matcher must not run")

    registry = _registry(
        ("fallback", RewardSpec("sync", _handler)),
        ("label_reward", RewardSpec("sync", _handler, unexpected_matcher)),
    )

    route = resolve_reward_route(
        {},
        "label",
        "fallback",
        registry,
        ("metadata", "rm_type", "label"),
    )

    assert route.reward_type == "fallback"
    assert route.source == "global"
    assert _issue_reasons(route) == ("missing",)


@pytest.mark.parametrize("priority", tuple(itertools.permutations(("metadata", "label", "rm_type"))))
def test_every_priority_permutation_selects_first_valid_source(priority):
    registry = _registry(
        ("metadata_reward", RewardSpec("sync", _handler)),
        ("label_reward", RewardSpec("sync", _handler, lambda label, metadata: label == "match")),
        ("global_reward", RewardSpec("sync", _handler)),
    )
    expected = {
        "metadata": ("metadata_reward", "metadata"),
        "label": ("label_reward", "label"),
        "rm_type": ("global_reward", "global"),
    }[priority[0]]

    route = resolve_reward_route(
        {"rm_type": "metadata_reward"},
        "match",
        "global_reward",
        registry,
        priority,
    )

    assert (route.reward_type, route.source) == expected


def test_missing_metadata_and_label_use_later_global_route_by_default():
    registry = _registry(("global_reward", RewardSpec("sync", _handler)))

    route = resolve_reward_route({}, "unsupported", "global_reward", registry)

    assert route.reward_type == "global_reward"
    assert route.source == "global"
    assert _issue_reasons(route) == ("missing", "missing")
    assert [(issue.stage, issue.reason) for issue in route.issues] == [
        ("metadata", "missing"),
        ("label", "missing"),
    ]


def test_missing_metadata_does_not_warn_when_label_matches():
    registry = _registry(
        ("label_reward", RewardSpec("sync", _handler, lambda label, metadata: label == "match")),
    )

    route = resolve_reward_route({}, "match", "unused", registry)

    assert route.reward_type == "label_reward"
    assert route.source == "label"
    assert _issue_reasons(route) == ()
    assert route.issues == ()


def test_global_first_is_explicit_selection_not_fallback():
    registry = _registry(("global_reward", RewardSpec("sync", _handler)))

    route = resolve_reward_route(
        {},
        "unsupported",
        "global_reward",
        registry,
        ("rm_type", "metadata", "label"),
    )

    assert route.reward_type == "global_reward"
    assert route.source == "global"
    assert _issue_reasons(route) == ()
    assert route.issues == ()


def test_unknown_metadata_uses_label_before_global_by_default():
    registry = _registry(
        ("global_reward", RewardSpec("sync", _handler)),
        ("label_reward", RewardSpec("sync", _handler, lambda label, metadata: True)),
    )

    route = resolve_reward_route({"rm_type": "unknown"}, "label", "global_reward", registry)

    assert route.reward_type == "label_reward"
    assert route.source == "label"
    assert _issue_reasons(route) == ("unknown",)


def test_unknown_metadata_uses_global_fallback_when_configured_before_label():
    registry = _registry(
        ("global_reward", RewardSpec("sync", _handler)),
        ("label_reward", RewardSpec("sync", _handler, lambda label, metadata: True)),
    )

    route = resolve_reward_route(
        {"rm_type": "unknown"},
        "label",
        "global_reward",
        registry,
        ("metadata", "rm_type", "label"),
    )

    assert route.reward_type == "global_reward"
    assert route.source == "global"
    assert _issue_reasons(route) == ("unknown",)


def test_unknown_metadata_without_fallback_returns_unresolved_route():
    registry = _registry(("math", RewardSpec("sync", _handler, lambda label, metadata: label == "42")))

    route = resolve_reward_route({"rm_type": "unknown"}, "unsupported", None, registry)

    assert route.reward_type is None
    assert route.source == "none"
    assert _issue_reasons(route) == ("unknown", "missing", "missing")


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("42", "math"),
        ("-3.5", "math"),
        ("1/2", "math"),
        (r"\frac{1}{2}", "math"),
        (r"\boxed{7}", "math"),
        ("<answer>B</answer>", "multiple_choice"),
    ],
)
def test_strict_label_matchers_select_expected_reward(label, expected):
    route = resolve_reward_route({}, label, None, REWARD_REGISTRY)

    assert route.reward_type == expected
    assert route.source == "label"


@pytest.mark.parametrize("label", ["B", r"\boxed{Paris}", "free-form answer", None])
def test_ambiguous_or_unsupported_label_is_not_guessed(label):
    route = resolve_reward_route({}, label, None, REWARD_REGISTRY)

    assert route.reward_type is None
    assert _issue_reasons(route) == ("missing", "missing", "missing")


def test_multiple_label_matches_are_reported_as_conflict():
    registry = _registry(
        ("first", RewardSpec("sync", _handler, lambda label, metadata: True)),
        ("second", RewardSpec("sync", _handler, lambda label, metadata: True)),
    )

    route = resolve_reward_route({}, "label", None, registry)

    assert route.reward_type is None
    assert _issue_reasons(route) == ("missing", "conflict", "missing")
    conflict = next(issue for issue in route.issues if issue.reason == "conflict")
    assert conflict.candidates == ("first", "second")


def test_label_conflict_route_keeps_later_global_result_for_diagnostics():
    registry = _registry(
        ("first", RewardSpec("sync", _handler, lambda label, metadata: True)),
        ("second", RewardSpec("sync", _handler, lambda label, metadata: True)),
        ("global_reward", RewardSpec("sync", _handler)),
    )

    route = resolve_reward_route({}, "label", "global_reward", registry)

    assert route.reward_type == "global_reward"
    assert route.source == "global"
    assert _issue_reasons(route) == ("missing", "conflict")
    assert route.issues[-1].candidates == ("first", "second")


def test_unknown_metadata_and_label_conflict_both_remain_observable_after_global_fallback():
    registry = _registry(
        ("first", RewardSpec("sync", _handler, lambda label, metadata: True)),
        ("second", RewardSpec("sync", _handler, lambda label, metadata: True)),
        ("global_reward", RewardSpec("sync", _handler)),
    )

    route = resolve_reward_route({"rm_type": "unknown"}, "label", "global_reward", registry)
    report = preflight_reward_routes([(7, "label", {"rm_type": "unknown"})], "global_reward", registry)

    assert route.reward_type == "global_reward"
    assert [(issue.stage, issue.reason) for issue in route.issues] == [
        ("metadata", "unknown"),
        ("label", "conflict"),
    ]
    assert _issue_reasons(route) == ("unknown", "conflict")
    assert report.fallback_count == 1
    assert report.conflict_count == 1
    assert report.conflict_indices == (7,)
    assert report.is_valid is False


def test_registry_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate reward registry entry"):
        _registry(
            ("duplicate", RewardSpec("sync", _handler)),
            ("duplicate", RewardSpec("async", _handler)),
        )


@pytest.mark.parametrize(
    ("entry", "error_type"),
    [
        (("bad", RewardSpec("invalid", _handler)), ValueError),
        (("bad", RewardSpec("sync", None)), TypeError),
        (("bad", RewardSpec("sync", _handler, "not-callable")), TypeError),
    ],
)
def test_registry_rejects_invalid_specs(entry, error_type):
    with pytest.raises(error_type):
        _registry(entry)


@pytest.mark.asyncio
async def test_single_registry_entry_routes_and_executes_without_router_branch():
    calls = []

    async def score_registered_reward(args, sample):
        calls.append((args, sample))
        return 0.75

    registry = _registry(
        ("registered_reward", RewardSpec("async", score_registered_reward, lambda label, metadata: label == "R"))
    )
    executor = RewardExecutor(max_concurrency=1, num_workers=1)
    args = SimpleNamespace(custom_rm_path=None, rm_type=None, reward_key=None)
    sample = Sample(response="response", label="R", metadata={})

    with patch("relax.engine.rewards.REWARD_REGISTRY", registry):
        reward = await executor.execute(args, sample)

    assert reward == 0.75
    assert calls == [(args, sample)]


def test_route_priority_requires_each_stage_exactly_once():
    with pytest.raises(ValueError, match="exactly once"):
        normalize_reward_route_priority(("metadata", "label", "label"))


def test_reward_route_config_uses_default_priority():
    config = normalize_reward_route_config(None)

    assert config.priority == ("metadata", "label", "rm_type")


def test_reward_route_config_accepts_yaml_mapping():
    config = normalize_reward_route_config({"priority": ["metadata", "rm_type", "label"]})

    assert config.priority == ("metadata", "rm_type", "label")


def test_reward_route_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown reward_route config keys"):
        normalize_reward_route_config({"priority": ["metadata", "label", "rm_type"], "typo": True})


def test_boxed_prefix_preserves_registered_base_reward():
    route = resolve_reward_route({"rm_type": "boxed_math"}, "42", None, REWARD_REGISTRY)

    assert route.reward_type == "math"
    assert route.source == "metadata"
    assert route.boxed is True


def test_invalid_explicit_reward_type_does_not_block_higher_priority_label():
    report = preflight_reward_routes([(0, "42", {})], "unknown", REWARD_REGISTRY)

    assert report.assignments == {"label/math": 1}
    assert report.fallback_count == 0
    assert report.is_valid is True


def test_preflight_reports_mixed_routes_fallback_and_unresolved_indices():
    records = [
        (0, "42", {"rm_type": "math"}),
        (1, "<answer>B</answer>", {}),
        (2, "42", {"rm_type": "unknown"}),
        (3, "unsupported", {}),
    ]

    report = preflight_reward_routes(records, None, REWARD_REGISTRY)

    assert report.total == 4
    assert report.assignments == {"label/math": 1, "label/multiple_choice": 1, "metadata/math": 1}
    assert report.fallback_count == 1
    assert report.unresolved_count == 1
    assert report.conflict_count == 0
    assert report.unresolved_indices == (3,)


def test_preflight_counts_unknown_metadata_global_fallback():
    records = [(0, "unsupported", {"rm_type": "unknown"})]

    report = preflight_reward_routes(records, "math", REWARD_REGISTRY)

    assert report.assignments == {"global/math": 1}
    assert report.fallback_count == 1
    assert report.is_valid is True


def test_preflight_counts_missing_sample_route_global_fallback():
    report = preflight_reward_routes([(0, "unsupported", {})], "math", REWARD_REGISTRY)

    assert report.assignments == {"global/math": 1}
    assert report.fallback_count == 1
    assert report.is_valid is True


def test_preflight_rejects_label_conflict_even_when_global_route_resolves():
    registry = _registry(
        ("first", RewardSpec("sync", _handler, lambda label, metadata: True)),
        ("second", RewardSpec("sync", _handler, lambda label, metadata: True)),
        ("global_reward", RewardSpec("sync", _handler)),
    )

    report = preflight_reward_routes([(7, "label", {})], "global_reward", registry)

    assert report.assignments == {"global/global_reward": 1}
    assert report.fallback_count == 1
    assert report.unresolved_count == 0
    assert report.conflict_count == 1
    assert report.conflict_indices == (7,)
    assert report.is_valid is False


def test_zero_reward_preserves_reward_key_shape():
    assert _zero_reward(SimpleNamespace(reward_key=None)) == 0.0
    assert _zero_reward(SimpleNamespace(reward_key="score")) == {"score": 0.0}


@pytest.mark.asyncio
async def test_runtime_unresolved_route_raises_by_default_without_warning():
    executor = RewardExecutor(max_concurrency=1, num_workers=1)
    args = SimpleNamespace(custom_rm_path=None, rm_type=None, reward_key=None)
    sample = Sample(label="unsupported", metadata={})

    with patch("relax.engine.rewards.logger.warning") as warning:
        with pytest.raises(NotImplementedError, match="could not be resolved"):
            await executor.execute(args, sample)

    warning.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_uses_configured_priority():
    executor = RewardExecutor(max_concurrency=1, num_workers=1)
    args = SimpleNamespace(
        custom_rm_path=None,
        rm_type="openr1mm",
        reward_key=None,
        reward_route={"priority": ["metadata", "rm_type", "label"]},
    )
    sample = Sample(response="<answer>hello</answer>", label="hello", metadata={})

    async def completed_reward():
        return 1.0

    with patch.object(executor, "_ensure_workers"), patch.object(executor, "_next_worker") as next_worker:
        worker = next_worker.return_value
        worker.compute.remote.return_value = completed_reward()
        await executor.execute(args, sample)

    worker.compute.remote.assert_called_once_with("openr1mm", "<answer>hello</answer>", "hello", metadata={})


@pytest.mark.asyncio
async def test_runtime_warns_when_invalid_metadata_falls_back_to_global_route():
    executor = RewardExecutor(max_concurrency=1, num_workers=1)
    args = SimpleNamespace(
        custom_rm_path=None,
        rm_type="openr1mm",
        reward_key=None,
        reward_route={"priority": ["metadata", "rm_type", "label"]},
    )
    sample = Sample(response="answer", label="unsupported", metadata={"rm_type": "unknown"})

    async def completed_reward():
        return 1.0

    with (
        patch.object(executor, "_ensure_workers"),
        patch.object(executor, "_next_worker") as next_worker,
        patch("relax.engine.rewards.logger.warning") as warning,
    ):
        worker = next_worker.return_value
        worker.compute.remote.return_value = completed_reward()
        await executor.execute(args, sample)

    warning.assert_called_once()
    assert tuple(issue.reason for issue in warning.call_args.args[3]) == ("unknown",)
    worker.compute.remote.assert_called_once_with(
        "openr1mm",
        "answer",
        "unsupported",
        metadata={"rm_type": "unknown"},
    )


@pytest.mark.asyncio
async def test_runtime_warns_when_missing_sample_route_falls_back_to_global_route():
    executor = RewardExecutor(max_concurrency=1, num_workers=1)
    args = SimpleNamespace(custom_rm_path=None, rm_type="openr1mm", reward_key=None)
    sample = Sample(response="answer", label="unsupported", metadata={})

    async def completed_reward():
        return 1.0

    with (
        patch.object(executor, "_ensure_workers"),
        patch.object(executor, "_next_worker") as next_worker,
        patch("relax.engine.rewards.logger.warning") as warning,
    ):
        worker = next_worker.return_value
        worker.compute.remote.return_value = completed_reward()
        await executor.execute(args, sample)

    warning.assert_called_once()
    assert tuple(issue.reason for issue in warning.call_args.args[3]) == ("missing", "missing")
