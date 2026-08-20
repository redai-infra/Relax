# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from examples.graphgpo.graph_credit import (
    SUCCESS,
    Turn,
    build_occurrence_graph,
    compute_method_advantages,
    episode_advantages,
    episode_return,
    finalize_trajectory,
    gigpo_discounted_returns,
    graph_advantages,
    graph_raw_returns,
    reverse_shortest_distances,
    standardize_by_key,
)


def _trajectory(
    trajectory_id: str,
    transitions: list[tuple[str, str, str]],
    *,
    success: bool,
    task_id: str = "task",
    invalid_turns: set[int] | None = None,
) -> tuple[Turn, ...]:
    invalid_turns = invalid_turns or set()
    turns = []
    for turn_index, (state, action, next_state) in enumerate(transitions):
        is_final = turn_index == len(transitions) - 1
        turns.append(
            Turn(
                task_id=task_id,
                trajectory_id=trajectory_id,
                turn_index=turn_index,
                state_key=state,
                action=action,
                next_state_key=next_state,
                success=success,
                terminal=success and is_final,
                truncated=not success and is_final,
                is_action_valid=turn_index not in invalid_turns,
            )
        )
    return finalize_trajectory(turns, success=success)


def _golden_turns() -> tuple[Turn, ...]:
    return (
        *_trajectory(
            "tau0",
            [("A", "to-b", "B"), ("B", "finish", "natural-terminal")],
            success=True,
        ),
        *_trajectory(
            "tau1",
            [("A", "to-c-1", "C"), ("C", "to-d", "D")],
            success=False,
        ),
        *_trajectory(
            "tau2",
            [("A", "to-c-2", "C"), ("C", "to-b", "B"), ("B", "back-to-c", "C")],
            success=False,
        ),
    )


def test_graph_credit_reference_golden_vector():
    graph = build_occurrence_graph(_golden_turns())
    distances = reverse_shortest_distances(graph)

    assert len(graph.occurrences) == 7
    assert sum(edge.source == "A" and edge.target == "C" for edge in graph.occurrences) == 2
    assert distances.has_success
    assert distances.max_finite_distance == pytest.approx(2.0)
    assert distances.distances == {
        SUCCESS: 0.0,
        "A": 2.0,
        "B": 1.0,
        "C": 2.0,
        "D": 3.0,
    }

    raw = graph_raw_returns(graph, distances)
    assert raw == pytest.approx(
        {
            ("tau0", 0): 1.0,
            ("tau0", 1): 10.0,
            ("tau1", 0): 0.1,
            ("tau1", 1): 0.01,
            ("tau2", 0): 0.1,
            ("tau2", 1): 1.0,
            ("tau2", 2): 0.1,
        }
    )

    advantages = graph_advantages(graph, distances)
    assert advantages == pytest.approx(
        {
            ("tau0", 0): 1.15469831616131,
            ("tau0", 1): 0.707106680176461,
            ("tau1", 0): -0.577349158080653,
            ("tau1", 1): -0.70710577108698,
            ("tau2", 0): -0.577349158080653,
            ("tau2", 1): 0.70710577108698,
            ("tau2", 2): -0.707106680176461,
        }
    )


def test_graph_credit_paper_convention_keeps_formula_difference_explicit():
    graph = build_occurrence_graph(_golden_turns())
    distances = reverse_shortest_distances(graph)
    reference = graph_raw_returns(graph, distances, convention="reference")
    paper = graph_raw_returns(graph, distances, convention="paper")

    assert paper == pytest.approx({key: value * 0.1 for key, value in reference.items()})


def test_graph_credit_same_source_distance_oracle_is_10_1_point_1():
    turns = (
        *_trajectory("distance-0", [("S", "finish-now", "done")], success=True),
        *_trajectory(
            "distance-1",
            [("S", "to-b", "B"), ("B", "finish", "done")],
            success=True,
        ),
        *_trajectory(
            "distance-2",
            [
                ("S", "to-c", "C"),
                ("C", "to-b", "B"),
                ("B", "finish", "done"),
            ],
            success=True,
        ),
    )
    graph = build_occurrence_graph(turns)
    distances = reverse_shortest_distances(graph)
    raw = graph_raw_returns(graph, distances, convention="reference")

    assert [
        distances.distances[turns[0].next_state_key],
        distances.distances[turns[1].next_state_key],
        distances.distances[turns[3].next_state_key],
    ] == pytest.approx([0.0, 1.0, 2.0])
    assert [
        raw[("distance-0", 0)],
        raw[("distance-1", 0)],
        raw[("distance-2", 0)],
    ] == pytest.approx([10.0, 1.0, 0.1])


def test_graph_credit_all_fail_group_has_zero_graph_advantage():
    turns = (
        *_trajectory("tau0", [("A", "left", "B")], success=False),
        *_trajectory("tau1", [("A", "right", "C")], success=False),
    )
    graph = build_occurrence_graph(turns)
    distances = reverse_shortest_distances(graph)

    assert not distances.has_success
    assert distances.max_finite_distance == 0.0
    assert set(distances.distances.values()) == {1.0}
    assert graph_advantages(graph, distances) == {("tau0", 0): 0.0, ("tau1", 0): 0.0}
    assert compute_method_advantages("graphgpo", turns, expected_group_size=2) == {
        ("tau0", 0): 0.0,
        ("tau1", 0): 0.0,
    }


def test_graph_credit_all_success_zero_variance_is_finite_zero():
    turns = (
        *_trajectory("tau0", [("S", "finish-0", "done")], success=True),
        *_trajectory("tau1", [("S", "finish-1", "done")], success=True),
    )

    result = compute_method_advantages(
        "graphgpo",
        turns,
        expected_group_size=2,
    )

    assert result == {("tau0", 0): 0.0, ("tau1", 0): 0.0}
    assert all(math.isfinite(value) for value in result.values())


def test_graph_credit_cycle_self_loop_and_success_edge():
    turns = _trajectory(
        "tau0",
        [("S", "wait", "S"), ("S", "finish", "natural-terminal")],
        success=True,
    )
    graph = build_occurrence_graph(turns)
    distances = reverse_shortest_distances(graph)

    assert distances.distances["S"] == pytest.approx(1.0)
    assert graph_raw_returns(graph, distances) == pytest.approx({("tau0", 0): 1.0, ("tau0", 1): 10.0})
    assert graph_advantages(graph, distances) == pytest.approx(
        {("tau0", 0): -0.707106680176461, ("tau0", 1): 0.707106680176461}
    )


def test_graph_credit_repeated_occurrences_change_group_statistics():
    graph = build_occurrence_graph(_golden_turns())
    distances = reverse_shortest_distances(graph)
    advantages = graph_advantages(graph, distances)

    assert advantages[("tau0", 0)] == pytest.approx(1.15469831616131)
    assert advantages[("tau1", 0)] == pytest.approx(-0.577349158080653)
    assert advantages[("tau2", 0)] == pytest.approx(-0.577349158080653)


def test_finalize_trajectory_replaces_last_real_target_without_adding_a_row():
    unfinished = (
        Turn("task", "tau", 0, "A", "to-b", "B", False, False, False, True),
        Turn(
            "task",
            "tau",
            1,
            "B",
            "finish",
            "natural-terminal",
            False,
            False,
            True,
            True,
        ),
    )

    finalized = finalize_trajectory(unfinished, success=True)

    assert len(finalized) == len(unfinished)
    assert finalized[0].next_state_key == "B"
    assert finalized[-1].next_state_key == SUCCESS
    assert finalized[-1].terminal
    assert not finalized[-1].truncated
    assert all(turn.success for turn in finalized)
    assert finalize_trajectory(finalized, success=True) == finalized


def test_episode_return_counts_invalid_actions_once_per_trajectory():
    clean = _trajectory(
        "clean",
        [("A", "a", "B"), ("B", "b", "C"), ("C", "c", "done")],
        success=True,
    )
    invalid = _trajectory(
        "invalid",
        [("A", "a", "B"), ("B", "b", "C"), ("C", "c", "done")],
        success=True,
        invalid_turns={0, 2},
    )

    assert episode_return(clean) == pytest.approx(10.0)
    assert episode_return(invalid) == pytest.approx(9.8)
    normalized = episode_advantages(
        {"clean": episode_return(clean), "invalid": episode_return(invalid)},
        {"clean": "task", "invalid": "task"},
    )
    expected = 0.1 / (math.sqrt(0.02) + 1e-6)
    assert normalized == pytest.approx({"clean": expected, "invalid": -expected})

    clean_raw = graph_raw_returns(
        build_occurrence_graph(clean),
        reverse_shortest_distances(build_occurrence_graph(clean)),
    )
    invalid_graph = build_occurrence_graph(invalid)
    invalid_raw = graph_raw_returns(invalid_graph, reverse_shortest_distances(invalid_graph))
    assert list(clean_raw.values()) == pytest.approx(list(invalid_raw.values()))


def test_grpo_uses_one_value_per_trajectory_and_broadcasts_to_turns():
    success = _trajectory(
        "success",
        [("A", "a", "B"), ("B", "b", "C"), ("C", "finish", "done")],
        success=True,
    )
    failure = _trajectory("failure", [("A", "fail", "Z")], success=False)

    result = compute_method_advantages(
        "grpo",
        (*success, *failure),
        expected_group_size=2,
    )
    expected = 5.0 / (math.sqrt(50.0) + 1e-6)
    assert result == pytest.approx(
        {
            ("failure", 0): -expected,
            ("success", 0): expected,
            ("success", 1): expected,
            ("success", 2): expected,
        }
    )


def test_grpo_reference_cross_steps_matches_frozen_length_weighting():
    success = _trajectory(
        "success",
        [("A", "a", "B"), ("B", "b", "C"), ("C", "finish", "done")],
        success=True,
    )
    failure = _trajectory("failure", [("A", "fail", "Z")], success=False)

    result = compute_method_advantages(
        "grpo",
        (*success, *failure),
        expected_group_size=2,
        episode_weighting="reference_cross_steps",
    )

    assert result == pytest.approx(
        {
            ("failure", 0): -7.5 / (5.0 + 1e-6),
            ("success", 0): 2.5 / (5.0 + 1e-6),
            ("success", 1): 2.5 / (5.0 + 1e-6),
            ("success", 2): 2.5 / (5.0 + 1e-6),
        }
    )


def test_episode_reference_cross_steps_requires_valid_lengths():
    returns = {"success": 10.0, "failure": 0.0}
    tasks = {"success": "task", "failure": "task"}

    with pytest.raises(ValueError, match="trajectory_lengths are required"):
        episode_advantages(
            returns,
            tasks,
            weighting="reference_cross_steps",
        )
    with pytest.raises(ValueError, match="positive integers"):
        episode_advantages(
            returns,
            tasks,
            weighting="reference_cross_steps",
            trajectory_lengths={"success": 0, "failure": 1},
        )
    with pytest.raises(ValueError, match="unknown episode weighting"):
        episode_advantages(returns, tasks, weighting="unknown")


def test_grpo_eight_trajectory_golden_vector():
    returns = {f"tau{index}": 10.0 if index == 0 else 0.0 for index in range(8)}
    tasks = {trajectory_id: "task" for trajectory_id in returns}

    result = episode_advantages(returns, tasks, expected_group_size=8)

    assert result["tau0"] == pytest.approx(2.47487303415311)
    assert [result[f"tau{index}"] for index in range(1, 8)] == pytest.approx([-0.353553290593302] * 7)


def test_gigpo_discounted_return_and_same_state_advantage():
    success = _trajectory(
        "success",
        [("S", "to-x", "X"), ("X", "finish", "done")],
        success=True,
    )
    failure = _trajectory(
        "failure",
        [("S", "to-y", "Y"), ("Y", "fail", "Z")],
        success=False,
    )
    standalone = _trajectory(
        "standalone",
        [("A", "a", "B"), ("B", "invalid", "C"), ("C", "finish", "done")],
        success=True,
        invalid_turns={1},
    )
    immediate = {
        ("standalone", 0): 0.0,
        ("standalone", 1): -0.1,
        ("standalone", 2): 10.0,
    }
    discounted = gigpo_discounted_returns(standalone, immediate, gamma=0.95)
    assert discounted == pytest.approx(
        {
            ("standalone", 0): 8.93,
            ("standalone", 1): 9.4,
            ("standalone", 2): 10.0,
        }
    )

    step_only = compute_method_advantages(
        "gigpo",
        (*success, *failure),
        expected_group_size=2,
        beta_episode=0.0,
    )
    expected = 4.75 / (math.sqrt(45.125) + 1e-6)
    assert step_only == pytest.approx(
        {
            ("failure", 0): -expected,
            ("failure", 1): 0.0,
            ("success", 0): expected,
            ("success", 1): 0.0,
        }
    )


def test_graphgpo_isolates_identical_state_keys_between_tasks():
    turns = (
        *_trajectory(
            "task1-success",
            [("A", "to-b", "B"), ("B", "finish", "done")],
            success=True,
            task_id="task1",
        ),
        *_trajectory(
            "task1-fail",
            [("A", "to-c", "C")],
            success=False,
            task_id="task1",
        ),
        *_trajectory(
            "task2-success",
            [("A", "to-d", "D"), ("D", "finish", "done")],
            success=True,
            task_id="task2",
        ),
        *_trajectory(
            "task2-fail",
            [("A", "to-e", "E")],
            success=False,
            task_id="task2",
        ),
    )

    graph_only = compute_method_advantages(
        "graphgpo",
        turns,
        expected_group_size=2,
        beta_episode=0.0,
    )

    expected = 0.495 / (math.sqrt(0.49005) + 1e-6)
    assert graph_only[("task1-success", 0)] == pytest.approx(expected)
    assert graph_only[("task1-fail", 0)] == pytest.approx(-expected)
    assert graph_only[("task2-success", 0)] == pytest.approx(expected)
    assert graph_only[("task2-fail", 0)] == pytest.approx(-expected)


def test_compute_advantages_is_permutation_invariant_and_key_complete():
    turns = _golden_turns()
    expected = compute_method_advantages(
        "graphgpo",
        turns,
        expected_group_size=3,
    )
    shuffled = compute_method_advantages(
        "graphgpo",
        tuple(reversed(turns)),
        expected_group_size=3,
    )

    assert shuffled == expected
    assert set(expected) == {turn.key for turn in turns}


def test_standardize_singleton_and_zero_variance_are_zero():
    assert standardize_by_key({"only": 3.0}, {"only": "group"}) == {"only": 0.0}
    assert standardize_by_key({"a": 3.0, "b": 3.0}, {"a": "group", "b": "group"}) == {
        "a": 0.0,
        "b": 0.0,
    }


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda turns: (turns[0], replace(turns[1], turn_index=2)),
            "non-contiguous",
        ),
        (
            lambda turns: (turns[0], replace(turns[1], task_id="other-task")),
            "multiple tasks|span multiple tasks",
        ),
        (
            lambda turns: (replace(turns[0], next_state_key="wrong"), turns[1]),
            "broken state transition",
        ),
        (
            lambda turns: (turns[0], replace(turns[1], next_state_key="not-success")),
            "final real action",
        ),
        (
            lambda turns: (replace(turns[0], cost=math.inf), turns[1]),
            "finite real",
        ),
        (
            lambda turns: (replace(turns[0], terminal=True), turns[1]),
            "before its final turn",
        ),
        (
            lambda turns: (
                turns[0],
                replace(turns[1], terminal=False, truncated=False),
            ),
            "must end",
        ),
    ],
)
def test_graph_credit_rejects_malformed_trajectories(mutator, match):
    turns = _trajectory(
        "tau",
        [("A", "to-b", "B"), ("B", "finish", "done")],
        success=True,
    )
    malformed = mutator(turns)

    with pytest.raises(ValueError, match=match):
        build_occurrence_graph(malformed)


def test_graph_credit_rejects_cross_task_graph_and_wrong_group_size():
    task1 = _trajectory("tau1", [("A", "a", "B")], success=False, task_id="task1")
    task2 = _trajectory("tau2", [("A", "a", "B")], success=False, task_id="task2")

    with pytest.raises(ValueError, match="exactly one task"):
        build_occurrence_graph((*task1, *task2))
    with pytest.raises(ValueError, match="group size mismatch"):
        compute_method_advantages("grpo", (*task1, *task2), expected_group_size=2)


def test_graph_credit_rejects_unknown_method_bad_rewards_and_non_unit_reference_cost():
    turns = _trajectory("tau", [("A", "a", "B")], success=False)

    with pytest.raises(ValueError, match="unknown method"):
        compute_method_advantages("unknown", turns)
    with pytest.raises(ValueError, match="exactly one value per turn"):
        gigpo_discounted_returns(turns, {})
    with pytest.raises(ValueError, match="finite real"):
        gigpo_discounted_returns(turns, {("tau", 0): math.nan})

    non_unit = (replace(turns[0], cost=2.0),)
    graph = build_occurrence_graph(non_unit)
    with pytest.raises(ValueError, match="unit edge costs"):
        graph_raw_returns(graph, reverse_shortest_distances(graph), convention="reference")


def test_standardize_rejects_key_mismatch_and_non_finite_values():
    with pytest.raises(ValueError, match="same keys"):
        standardize_by_key({"a": 1.0}, {"b": "group"})
    with pytest.raises(ValueError, match="finite real"):
        standardize_by_key({"a": math.nan}, {"a": "group"})


def test_graph_credit_rejects_tampered_distance_results():
    graph = build_occurrence_graph(_golden_turns())
    distances = reverse_shortest_distances(graph)

    with pytest.raises(ValueError, match="keys do not match"):
        graph_raw_returns(graph, replace(distances, distances={SUCCESS: 0.0}))
    with pytest.raises(ValueError, match="distance zero"):
        graph_advantages(
            graph,
            replace(
                distances,
                distances={**distances.distances, SUCCESS: 1.0},
            ),
        )
