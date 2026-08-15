# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import math

import pytest

from examples.graphgpo.custom_advantage import (
    compute_custom_advantage,
    compute_group_advantages,
)
from examples.graphgpo.graph_credit import SUCCESS


def _slot(
    trajectory_id: str,
    transitions: list[tuple[str, str, str]],
    *,
    task_id: str = "task",
    success: bool = False,
    invalid_turns: set[int] | None = None,
    rollout_group_id: str = "group-0",
    policy_version: int = 0,
) -> dict[str, dict[str, object]]:
    invalid_turns = invalid_turns or set()
    declared_return = 10.0 * float(success) - 0.1 * len(invalid_turns)
    result: dict[str, dict[str, object]] = {}
    for turn_index, (state, action, next_state) in enumerate(transitions):
        is_final = turn_index == len(transitions) - 1
        result[f"turn_{turn_index:03d}"] = {
            "row_id": f"{rollout_group_id}:{trajectory_id}:{turn_index}",
            "rollout_group_id": rollout_group_id,
            "policy_version": policy_version,
            "task_id": task_id,
            "trajectory_id": trajectory_id,
            "turn_index": turn_index,
            "state_key": state,
            "action": action,
            "next_state_key": SUCCESS if success and is_final else next_state,
            "is_action_valid": turn_index not in invalid_turns,
            "success": success,
            "terminal": success and is_final,
            "truncated": not success and is_final,
            "episode_return": declared_return,
        }
    return result


def _golden_group() -> list[dict[str, dict[str, object]]]:
    return [
        _slot(
            "tau0",
            [("A", "to-b", "B"), ("B", "finish", "natural-terminal")],
            success=True,
        ),
        _slot(
            "tau1",
            [("A", "to-c-1", "C"), ("C", "to-d", "D")],
        ),
        _slot(
            "tau2",
            [
                ("A", "to-c-2", "C"),
                ("C", "to-b", "B"),
                ("B", "back-to-c", "C"),
            ],
        ),
    ]


def test_custom_advantage_three_method_golden_vectors():
    group = _golden_group()

    grpo = compute_group_advantages(group, method="grpo", expected_group_size=3)
    gigpo = compute_group_advantages(group, method="gigpo", expected_group_size=3)
    graphgpo = compute_group_advantages(group, method="graphgpo", expected_group_size=3)

    episode_high = 1.1547003383792862
    episode_low = -0.5773501691896431
    gigpo_a_high = 1.1547003278529744
    gigpo_a_low = -0.5773501639264869
    gigpo_b_high = 0.7071066811865616
    gigpo_b_low = -0.7071066811865616
    graph_a_high = 1.154698316161306
    graph_a_low = -0.577349158080653
    graph_b_high = 0.707106680176461
    graph_b_low = -0.707106680176461
    c_graph_high = 0.7071057710869802
    c_graph_low = -0.7071057710869802

    assert grpo == [
        {
            "turn_000": pytest.approx(episode_high),
            "turn_001": pytest.approx(episode_high),
        },
        {
            "turn_000": pytest.approx(episode_low),
            "turn_001": pytest.approx(episode_low),
        },
        {
            "turn_000": pytest.approx(episode_low),
            "turn_001": pytest.approx(episode_low),
            "turn_002": pytest.approx(episode_low),
        },
    ]
    assert gigpo == [
        {
            "turn_000": pytest.approx(episode_high + gigpo_a_high),
            "turn_001": pytest.approx(episode_high + gigpo_b_high),
        },
        {
            "turn_000": pytest.approx(episode_low + gigpo_a_low),
            "turn_001": pytest.approx(episode_low),
        },
        {
            "turn_000": pytest.approx(episode_low + gigpo_a_low),
            "turn_001": pytest.approx(episode_low),
            "turn_002": pytest.approx(episode_low + gigpo_b_low),
        },
    ]
    assert graphgpo == [
        {
            "turn_000": pytest.approx(episode_high + graph_a_high),
            "turn_001": pytest.approx(episode_high + graph_b_high),
        },
        {
            "turn_000": pytest.approx(episode_low + graph_a_low),
            "turn_001": pytest.approx(episode_low + c_graph_low),
        },
        {
            "turn_000": pytest.approx(episode_low + graph_a_low),
            "turn_001": pytest.approx(episode_low + c_graph_high),
            "turn_002": pytest.approx(episode_low + graph_b_low),
        },
    ]


def test_custom_advantage_preserves_variable_length_slot_shape():
    group = [
        _slot("short", [("A", "wait", "A")]),
        _slot(
            "long",
            [("A", "go", "B"), ("B", "go", "C"), ("C", "wait", "C")],
        ),
    ]

    result = compute_group_advantages(group, method="graphgpo", expected_group_size=2)

    assert [list(slot) for slot in result] == [
        ["turn_000"],
        ["turn_000", "turn_001", "turn_002"],
    ]
    assert all(math.isfinite(value) for slot in result for value in slot.values())


@pytest.mark.parametrize("method", ["grpo", "gigpo", "graphgpo"])
def test_custom_advantage_all_fail_group_is_finite_zero(method):
    group = [
        _slot("tau0", [("A", "wait-0", "A")]),
        _slot("tau1", [("A", "wait-1", "A")]),
    ]

    result = compute_group_advantages(group, method=method, expected_group_size=2)

    assert result == [{"turn_000": 0.0}, {"turn_000": 0.0}]


def test_custom_advantage_uses_slot_position_not_slot_metadata():
    group = _golden_group()
    for slot_index, slot in enumerate(group):
        for metadata in slot.values():
            metadata["slot_index"] = 99 - slot_index
            metadata["group_generation"] = f"irrelevant-{slot_index}"

    result = compute_group_advantages(group, method="grpo", expected_group_size=3)

    assert len(result) == 3
    assert result[0]["turn_000"] > result[1]["turn_000"]


def test_custom_advantage_default_environment_entry_is_group_of_eight(
    monkeypatch,
):
    for name in (
        "GRAPHGPO_METHOD",
        "METHOD",
        "GRAPHGPO_EXPECTED_GROUP_SIZE",
        "GROUP_SIZE",
        "GRAPHGPO_OMEGA",
        "OMEGA",
        "GRAPHGPO_GAMMA",
        "GAMMA",
        "GRAPHGPO_BETA",
        "BETA",
        "GRAPHGPO_BETA_EPISODE",
        "BETA_EPISODE",
        "GRAPHGPO_EPISODE_WEIGHTING",
        "EPISODE_WEIGHTING",
    ):
        monkeypatch.delenv(name, raising=False)
    group = [_slot(f"tau{index}", [("A", f"wait-{index}", "A")]) for index in range(8)]

    result = compute_custom_advantage(group)

    assert result == [{"turn_000": 0.0} for _ in range(8)]


def test_custom_advantage_environment_entry_accepts_recipe_short_names(
    monkeypatch,
):
    monkeypatch.setenv("METHOD", "grpo")
    monkeypatch.setenv("GROUP_SIZE", "3")
    monkeypatch.setenv("BETA_EPISODE", "2")
    monkeypatch.setenv("EPISODE_WEIGHTING", "trajectory_once")

    result = compute_custom_advantage(_golden_group())

    assert result[0]["turn_000"] == pytest.approx(2.3080, rel=1e-3)
    assert result[1]["turn_000"] == pytest.approx(-1.1547, rel=1e-3)


def test_custom_advantage_environment_defaults_to_trajectory_once(
    monkeypatch,
):
    monkeypatch.setenv("METHOD", "grpo")
    monkeypatch.setenv("GROUP_SIZE", "2")
    monkeypatch.delenv("GRAPHGPO_EPISODE_WEIGHTING", raising=False)
    monkeypatch.delenv("EPISODE_WEIGHTING", raising=False)
    group = [
        _slot(
            "success",
            [("A", "a", "B"), ("B", "b", "C"), ("C", "finish", "done")],
            success=True,
        ),
        _slot("failure", [("A", "fail", "Z")]),
    ]

    result = compute_custom_advantage(group)

    expected = 5.0 / (math.sqrt(50.0) + 1e-6)
    assert result[0] == {
        "turn_000": pytest.approx(expected),
        "turn_001": pytest.approx(expected),
        "turn_002": pytest.approx(expected),
    }
    assert result[1] == {"turn_000": pytest.approx(-expected)}


def test_custom_advantage_reference_cross_steps_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("METHOD", "grpo")
    monkeypatch.setenv("GROUP_SIZE", "2")
    monkeypatch.setenv("EPISODE_WEIGHTING", "reference_cross_steps")
    group = [
        _slot(
            "success",
            [("A", "a", "B"), ("B", "b", "C"), ("C", "finish", "done")],
            success=True,
        ),
        _slot("failure", [("A", "fail", "Z")]),
    ]

    result = compute_custom_advantage(group)

    assert result[0] == {
        "turn_000": pytest.approx(2.5 / (5.0 + 1e-6)),
        "turn_001": pytest.approx(2.5 / (5.0 + 1e-6)),
        "turn_002": pytest.approx(2.5 / (5.0 + 1e-6)),
    }
    assert result[1] == {"turn_000": pytest.approx(-7.5 / (5.0 + 1e-6))}


def test_custom_advantage_rejects_missing_metadata():
    group = _golden_group()
    del group[0]["turn_000"]["state_key"]

    with pytest.raises(ValueError, match="missing required metadata field 'state_key'"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_extra_slot():
    group = _golden_group()
    group.append(_slot("tau3", [("A", "wait", "A")]))

    with pytest.raises(ValueError, match="rollout group size mismatch"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_missing_slot():
    group = _golden_group()[:-1]

    with pytest.raises(ValueError, match="rollout group size mismatch"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_duplicate_trajectory_id():
    group = _golden_group()
    for metadata in group[1].values():
        metadata["trajectory_id"] = "tau0"

    with pytest.raises(ValueError, match="appears in more than one slot"):
        compute_group_advantages(group, expected_group_size=3)


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_custom_advantage_rejects_non_finite_episode_return(non_finite):
    group = _golden_group()
    group[0]["turn_000"]["episode_return"] = non_finite

    with pytest.raises(ValueError, match="episode_return must be a finite"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_non_finite_cost():
    group = _golden_group()
    group[0]["turn_000"]["cost"] = math.nan

    with pytest.raises(ValueError, match="cost must be a finite"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_turn_gap():
    group = _golden_group()
    moved = group[2].pop("turn_001")
    moved["turn_index"] = 3
    group[2]["turn_003"] = moved

    with pytest.raises(ValueError, match="non-contiguous turn indices"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_noncanonical_turn_name():
    group = _golden_group()
    group[0]["edge_000"] = group[0].pop("turn_000")

    with pytest.raises(ValueError, match="canonical name"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_task_mix():
    group = _golden_group()
    for metadata in group[2].values():
        metadata["task_id"] = "different-task"

    with pytest.raises(ValueError, match="cannot mix multiple task_id"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_identity_mix_and_duplicate_row():
    mixed_group = _golden_group()
    mixed_group[1]["turn_000"]["rollout_group_id"] = "other-group"
    with pytest.raises(ValueError, match="multiple rollout_group_id"):
        compute_group_advantages(mixed_group, expected_group_size=3)

    mixed_policy = _golden_group()
    mixed_policy[1]["turn_000"]["policy_version"] = 1
    with pytest.raises(ValueError, match="multiple policy_version"):
        compute_group_advantages(mixed_policy, expected_group_size=3)

    duplicate_row = _golden_group()
    duplicate_row[1]["turn_000"]["row_id"] = duplicate_row[0]["turn_000"]["row_id"]
    with pytest.raises(ValueError, match="duplicate row_id"):
        compute_group_advantages(duplicate_row, expected_group_size=3)


def test_custom_advantage_rejects_inconsistent_episode_return():
    group = _golden_group()
    group[0]["turn_001"]["episode_return"] = 9.9

    with pytest.raises(ValueError, match="inconsistent episode_return"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_episode_return_mismatch():
    group = _golden_group()
    for metadata in group[0].values():
        metadata["episode_return"] = 9.9

    with pytest.raises(ValueError, match="episode_return mismatch"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_inconsistent_success_flags():
    group = _golden_group()
    group[0]["turn_000"]["success"] = False

    with pytest.raises(ValueError, match="inconsistent success flags"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_early_terminal():
    group = _golden_group()
    group[0]["turn_000"]["terminal"] = True

    with pytest.raises(ValueError, match="terminates before its final turn"):
        compute_group_advantages(group, expected_group_size=3)


def test_custom_advantage_rejects_unknown_method_without_mutating_input():
    group = _golden_group()
    original = copy.deepcopy(group)

    with pytest.raises(ValueError, match="method must be one of"):
        compute_group_advantages(group, method="unknown", expected_group_size=3)

    assert group == original
