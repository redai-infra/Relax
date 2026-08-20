# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure-Python credit assignment kernels for the GraphGPO example.

The implementation intentionally has no dependency on Ray, PyTorch, or the
ALFWorld environment.  It operates on finalized turn transitions so the same
kernel can be exercised by unit tests and by the custom-advantage adapter.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import count
from time import perf_counter_ns
from typing import Hashable, Literal, Mapping, Sequence, TypeVar

from examples.graphgpo.diagnostics import (
    DiagnosticsCallback,
    build_graph_diagnostics,
)


SUCCESS = "__GRAPHGPO_SUCCESS__"
# TurnKey stays compact because every public computation first validates one
# group-local identity namespace; rollout_group_id and policy_version are
# checked by the custom adapter before these keys are constructed.
TurnKey = tuple[Hashable, int]
TrajectoryId = Hashable
EpisodeWeighting = Literal["trajectory_once", "reference_cross_steps"]
DEFAULT_EPISODE_WEIGHTING: EpisodeWeighting = "trajectory_once"
EPISODE_WEIGHTINGS = frozenset(("trajectory_once", "reference_cross_steps"))
K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True)
class Turn:
    """One real environment action and its resulting state transition."""

    task_id: Hashable
    trajectory_id: TrajectoryId
    turn_index: int
    state_key: Hashable
    action: str
    next_state_key: Hashable
    success: bool
    terminal: bool
    truncated: bool
    is_action_valid: bool
    cost: float = 1.0

    @property
    def key(self) -> TurnKey:
        return (self.trajectory_id, self.turn_index)


@dataclass(frozen=True)
class EdgeOccurrence:
    """An observed graph edge; repeated transitions remain repeated."""

    turn_key: TurnKey
    source: Hashable
    action: str
    target: Hashable
    cost: float


@dataclass(frozen=True)
class OccurrenceGraph:
    """One task-local graph with both topology and occurrence information."""

    task_id: Hashable
    nodes: frozenset[Hashable]
    occurrences: tuple[EdgeOccurrence, ...]
    reverse_edges: tuple[tuple[Hashable, Hashable, float], ...]
    has_success: bool


@dataclass(frozen=True)
class DistanceResult:
    """Finite distances used by the reference GraphGPO reward convention."""

    distances: Mapping[Hashable, float]
    max_finite_distance: float
    has_success: bool


def _stable_key(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"


def _turn_sort_key(turn: Turn) -> tuple[str, str, int]:
    return (_stable_key(turn.task_id), _stable_key(turn.trajectory_id), turn.turn_index)


def _require_hashable(name: str, value: object) -> None:
    try:
        hash(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be hashable") from exc


def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _canonicalize_turns(turns: Sequence[Turn], *, require_finalized: bool) -> tuple[Turn, ...]:
    if not turns:
        raise ValueError("at least one turn is required")

    by_trajectory: dict[TrajectoryId, list[Turn]] = defaultdict(list)
    task_by_trajectory: dict[TrajectoryId, Hashable] = {}
    seen_keys: set[TurnKey] = set()

    for turn in turns:
        if not isinstance(turn, Turn):
            raise ValueError("all entries must be Turn instances")
        for name, value in (
            ("task_id", turn.task_id),
            ("trajectory_id", turn.trajectory_id),
            ("state_key", turn.state_key),
            ("next_state_key", turn.next_state_key),
        ):
            _require_hashable(name, value)
        if turn.state_key == SUCCESS:
            raise ValueError("SUCCESS is a sink and cannot be a source state")
        if isinstance(turn.turn_index, bool) or not isinstance(turn.turn_index, int) or turn.turn_index < 0:
            raise ValueError("turn_index must be a non-negative integer")
        if not isinstance(turn.action, str):
            raise ValueError("action must be a string")
        if not all(
            isinstance(flag, bool)
            for flag in (
                turn.success,
                turn.terminal,
                turn.truncated,
                turn.is_action_valid,
            )
        ):
            raise ValueError("success, terminal, truncated, and is_action_valid must be booleans")
        if _require_finite("cost", turn.cost) <= 0.0:
            raise ValueError("cost must be greater than zero")
        if turn.key in seen_keys:
            raise ValueError(f"duplicate turn key: {turn.key!r}")
        seen_keys.add(turn.key)

        previous_task = task_by_trajectory.setdefault(turn.trajectory_id, turn.task_id)
        if previous_task != turn.task_id:
            raise ValueError("a trajectory_id cannot span multiple tasks")
        by_trajectory[turn.trajectory_id].append(turn)

    for trajectory_id, trajectory in by_trajectory.items():
        trajectory.sort(key=lambda item: item.turn_index)
        expected_indices = list(range(len(trajectory)))
        actual_indices = [turn.turn_index for turn in trajectory]
        if actual_indices != expected_indices:
            raise ValueError(f"trajectory {trajectory_id!r} has non-contiguous turn indices")

        success_values = {turn.success for turn in trajectory}
        if len(success_values) != 1:
            raise ValueError(f"trajectory {trajectory_id!r} has inconsistent success flags")

        for current, following in zip(trajectory, trajectory[1:]):
            if current.next_state_key != following.state_key:
                raise ValueError(f"trajectory {trajectory_id!r} has a broken state transition chain")
            if current.terminal or current.truncated:
                raise ValueError(f"trajectory {trajectory_id!r} terminates before its final turn")

        final_turn = trajectory[-1]
        if final_turn.terminal and final_turn.truncated:
            raise ValueError(f"trajectory {trajectory_id!r} cannot be both terminal and truncated")
        if not final_turn.terminal and not final_turn.truncated:
            raise ValueError(f"trajectory {trajectory_id!r} must end as terminal or truncated")

        if require_finalized:
            success = final_turn.success
            success_targets = [turn.turn_index for turn in trajectory if turn.next_state_key == SUCCESS]
            if success and success_targets != [final_turn.turn_index]:
                raise ValueError("a successful trajectory must point its final real action to SUCCESS")
            if not success and success_targets:
                raise ValueError("an unsuccessful trajectory cannot point to SUCCESS")

    return tuple(sorted(turns, key=_turn_sort_key))


def finalize_trajectory(
    turns: Sequence[Turn],
    *,
    success: bool,
    success_key: Hashable = SUCCESS,
) -> tuple[Turn, ...]:
    """Finalize one trajectory without adding a synthetic transition.

    On success, only the last real action's ``next_state_key`` is replaced by
    ``success_key``.  The row count and all earlier transitions remain intact.
    """

    if not isinstance(success, bool):
        raise ValueError("success must be a boolean")
    _require_hashable("success_key", success_key)
    if success_key != SUCCESS:
        raise ValueError("the graph kernel requires the canonical SUCCESS key")
    if not turns:
        raise ValueError("at least one turn is required")

    trajectory_ids = {turn.trajectory_id for turn in turns if isinstance(turn, Turn)}
    task_ids = {turn.task_id for turn in turns if isinstance(turn, Turn)}
    if len(trajectory_ids) != 1 or len(task_ids) != 1 or len(trajectory_ids) != len(task_ids):
        raise ValueError("finalize_trajectory accepts exactly one task-local trajectory")

    rewritten = [replace(turn, success=success) for turn in turns]
    rewritten.sort(key=lambda item: item.turn_index)
    last_index = len(rewritten) - 1
    for index, turn in enumerate(rewritten):
        if turn.next_state_key == SUCCESS and index != last_index:
            raise ValueError("SUCCESS may only appear on the final real action")

    if success:
        rewritten[-1] = replace(
            rewritten[-1],
            next_state_key=SUCCESS,
            success=True,
            terminal=True,
            truncated=False,
        )
    elif rewritten[-1].next_state_key == SUCCESS:
        raise ValueError("an unsuccessful trajectory cannot point to SUCCESS")

    return _canonicalize_turns(rewritten, require_finalized=True)


def episode_return(
    turns: Sequence[Turn],
    *,
    success_reward: float = 10.0,
    invalid_penalty: float = 0.1,
) -> float:
    """Return ``success_reward * success - invalid_penalty * invalid_count``."""

    canonical = _canonicalize_turns(turns, require_finalized=True)
    if len({turn.trajectory_id for turn in canonical}) != 1:
        raise ValueError("episode_return accepts exactly one trajectory")
    success_reward = _require_finite("success_reward", success_reward)
    invalid_penalty = _require_finite("invalid_penalty", invalid_penalty)
    if success_reward < 0.0 or invalid_penalty < 0.0:
        raise ValueError("reward and penalty magnitudes must be non-negative")

    success = canonical[0].success
    invalid_count = sum(not turn.is_action_valid for turn in canonical)
    result = success_reward * float(success) - invalid_penalty * invalid_count
    return _require_finite("episode return", result)


def build_occurrence_graph(turns: Sequence[Turn]) -> OccurrenceGraph:
    """Build one task-local graph while preserving every sampled edge."""

    canonical = _canonicalize_turns(turns, require_finalized=True)
    task_ids = {turn.task_id for turn in canonical}
    if len(task_ids) != 1:
        raise ValueError("build_occurrence_graph accepts turns from exactly one task")
    task_id = next(iter(task_ids))

    occurrences = tuple(
        EdgeOccurrence(
            turn_key=turn.key,
            source=turn.state_key,
            action=turn.action,
            target=turn.next_state_key,
            cost=float(turn.cost),
        )
        for turn in canonical
    )
    nodes = frozenset(node for occurrence in occurrences for node in (occurrence.source, occurrence.target))

    minimum_costs: dict[tuple[Hashable, Hashable], float] = {}
    for occurrence in occurrences:
        topology_key = (occurrence.source, occurrence.target)
        old_cost = minimum_costs.get(topology_key)
        if old_cost is None or occurrence.cost < old_cost:
            minimum_costs[topology_key] = occurrence.cost

    reverse_edges = tuple(
        sorted(
            ((target, source, cost) for (source, target), cost in minimum_costs.items()),
            key=lambda item: (_stable_key(item[0]), _stable_key(item[1]), item[2]),
        )
    )
    return OccurrenceGraph(
        task_id=task_id,
        nodes=nodes,
        occurrences=occurrences,
        reverse_edges=reverse_edges,
        has_success=SUCCESS in nodes,
    )


def reverse_shortest_distances(graph: OccurrenceGraph) -> DistanceResult:
    """Compute distances to SUCCESS with a finite unreachable replacement."""

    if not isinstance(graph, OccurrenceGraph):
        raise ValueError("graph must be an OccurrenceGraph")
    if not graph.nodes:
        raise ValueError("graph must contain at least one node")

    if not graph.has_success:
        return DistanceResult(
            distances={node: 1.0 for node in sorted(graph.nodes, key=_stable_key)},
            max_finite_distance=0.0,
            has_success=False,
        )

    reverse_adjacency: dict[Hashable, list[tuple[Hashable, float]]] = defaultdict(list)
    for target, source, cost in graph.reverse_edges:
        reverse_adjacency[target].append((source, cost))

    distances: dict[Hashable, float] = {SUCCESS: 0.0}
    tie_breaker = count()
    queue: list[tuple[float, int, Hashable]] = [(0.0, next(tie_breaker), SUCCESS)]
    while queue:
        distance, _, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        for predecessor, edge_cost in reverse_adjacency.get(node, ()):
            candidate = distance + edge_cost
            if candidate < distances.get(predecessor, math.inf):
                distances[predecessor] = candidate
                heapq.heappush(queue, (candidate, next(tie_breaker), predecessor))

    max_finite = max(distances.values())
    unreachable_distance = max_finite + 1.0
    effective_distances = {
        node: distances.get(node, unreachable_distance) for node in sorted(graph.nodes, key=_stable_key)
    }
    return DistanceResult(
        distances=effective_distances,
        max_finite_distance=max_finite,
        has_success=True,
    )


def _validate_distance_result(graph: OccurrenceGraph, distance_result: DistanceResult) -> None:
    if distance_result.has_success != graph.has_success:
        raise ValueError("distance result success flag does not match the graph")
    if set(distance_result.distances) != set(graph.nodes):
        raise ValueError("distance result keys do not match graph nodes")
    max_finite = _require_finite("max_finite_distance", distance_result.max_finite_distance)
    if max_finite < 0.0:
        raise ValueError("max_finite_distance must be non-negative")
    for node, value in distance_result.distances.items():
        if _require_finite(f"distance for {node!r}", value) < 0.0:
            raise ValueError("node distances must be non-negative")
    if graph.has_success and distance_result.distances[SUCCESS] != 0.0:
        raise ValueError("SUCCESS must have distance zero")


def graph_raw_returns(
    graph: OccurrenceGraph,
    distance_result: DistanceResult,
    *,
    omega: float = 0.1,
    success_reward: float = 10.0,
    convention: Literal["reference", "paper"] = "reference",
) -> dict[TurnKey, float]:
    """Compute raw graph returns for each observed transition occurrence.

    ``reference`` implements the frozen recipe used by the proposal:
    ``success_reward * omega ** d(next)`` with unit edge costs.  ``paper``
    implements GraphGPO Eq. 4: ``success_reward * omega ** (d(next) + cost)``.
    """

    if not isinstance(graph, OccurrenceGraph) or not isinstance(distance_result, DistanceResult):
        raise ValueError("graph and distance_result have invalid types")
    _validate_distance_result(graph, distance_result)
    omega = _require_finite("omega", omega)
    success_reward = _require_finite("success_reward", success_reward)
    if not 0.0 < omega < 1.0:
        raise ValueError("omega must be between zero and one")
    if success_reward < 0.0:
        raise ValueError("success_reward must be non-negative")
    if convention not in {"reference", "paper"}:
        raise ValueError(f"unknown graph reward convention: {convention!r}")

    result: dict[TurnKey, float] = {}
    for occurrence in graph.occurrences:
        if occurrence.target not in distance_result.distances:
            raise ValueError(f"missing distance for node {occurrence.target!r}")
        if convention == "reference":
            if occurrence.cost != 1.0:
                raise ValueError("the reference convention requires unit edge costs")
            exponent = distance_result.distances[occurrence.target]
        else:
            exponent = distance_result.distances[occurrence.target] + occurrence.cost
        value = success_reward * omega**exponent
        result[occurrence.turn_key] = _require_finite("graph raw return", value)

    expected_keys = {occurrence.turn_key for occurrence in graph.occurrences}
    if set(result) != expected_keys:
        raise ValueError("graph raw return keys do not match edge occurrence keys")
    return result


def standardize_by_key(
    values: Mapping[K, float],
    group_keys: Mapping[K, Hashable],
    *,
    eps: float = 1e-6,
    ddof: int = 1,
    singleton_zero: bool = True,
) -> dict[K, float]:
    """Standardize values independently inside each group.

    The computation uses ``math.fsum`` and a deterministic key order.  With
    ``ddof=1`` it matches the frozen recipe's sample standard deviation.
    """

    if set(values) != set(group_keys):
        raise ValueError("values and group_keys must contain exactly the same keys")
    eps = _require_finite("eps", eps)
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    if not isinstance(singleton_zero, bool):
        raise ValueError("singleton_zero must be a boolean")

    grouped: dict[Hashable, list[K]] = defaultdict(list)
    finite_values: dict[K, float] = {}
    for key in sorted(values, key=_stable_key):
        value = _require_finite(f"value for {key!r}", values[key])
        group_key = group_keys[key]
        _require_hashable("group key", group_key)
        grouped[group_key].append(key)
        finite_values[key] = value

    result: dict[K, float] = {}
    for group_key in sorted(grouped, key=_stable_key):
        members = grouped[group_key]
        if len(members) <= ddof:
            if not singleton_zero:
                raise ValueError(f"group {group_key!r} is too small for ddof={ddof}")
            result.update((key, 0.0) for key in members)
            continue

        mean = math.fsum(finite_values[key] for key in members) / len(members)
        squared_deviations = math.fsum((finite_values[key] - mean) ** 2 for key in members)
        variance = squared_deviations / (len(members) - ddof)
        if variance <= 0.0:
            result.update((key, 0.0) for key in members)
            continue

        denominator = math.sqrt(variance) + eps
        for key in members:
            result[key] = _require_finite(
                f"standardized value for {key!r}",
                (finite_values[key] - mean) / denominator,
            )

    if set(result) != set(values):
        raise ValueError("standardization did not return every input key")
    return result


def graph_advantages(
    graph: OccurrenceGraph,
    distance_result: DistanceResult,
    *,
    omega: float = 0.1,
    success_reward: float = 10.0,
    eps: float = 1e-6,
    convention: Literal["reference", "paper"] = "reference",
) -> dict[TurnKey, float]:
    """Return occurrence-weighted, same-source graph advantages."""

    omega = _require_finite("omega", omega)
    success_reward = _require_finite("success_reward", success_reward)
    eps = _require_finite("eps", eps)
    if not 0.0 < omega < 1.0:
        raise ValueError("omega must be between zero and one")
    if success_reward < 0.0 or eps < 0.0:
        raise ValueError("success_reward and eps must be non-negative")
    if convention not in {"reference", "paper"}:
        raise ValueError(f"unknown graph reward convention: {convention!r}")

    _validate_distance_result(graph, distance_result)
    keys = {occurrence.turn_key for occurrence in graph.occurrences}
    if not distance_result.has_success:
        return {key: 0.0 for key in sorted(keys, key=_stable_key)}

    raw_returns = graph_raw_returns(
        graph,
        distance_result,
        omega=omega,
        success_reward=success_reward,
        convention=convention,
    )
    sources = {occurrence.turn_key: occurrence.source for occurrence in graph.occurrences}
    return standardize_by_key(raw_returns, sources, eps=eps, ddof=1, singleton_zero=True)


def episode_advantages(
    episode_returns: Mapping[TrajectoryId, float],
    task_ids: Mapping[TrajectoryId, Hashable],
    *,
    expected_group_size: int | None = None,
    eps: float = 1e-6,
    weighting: EpisodeWeighting = DEFAULT_EPISODE_WEIGHTING,
    trajectory_lengths: Mapping[TrajectoryId, int] | None = None,
) -> dict[TrajectoryId, float]:
    """Normalize episode returns independently for every task.

    ``trajectory_once`` is the usual GRPO convention: every trajectory
    contributes one value to the group statistics.  ``reference_cross_steps``
    reproduces the frozen GraphGPO helper: a trajectory's episode return
    contributes once for every real turn before the result is broadcast back to
    all turns in that trajectory.
    """

    if not episode_returns:
        raise ValueError("at least one episode return is required")
    if set(episode_returns) != set(task_ids):
        raise ValueError("episode_returns and task_ids must contain exactly the same trajectories")
    if weighting not in EPISODE_WEIGHTINGS:
        raise ValueError(f"unknown episode weighting: {weighting!r}")
    if trajectory_lengths is not None and set(trajectory_lengths) != set(episode_returns):
        raise ValueError("trajectory_lengths must contain exactly the same trajectories")

    finite_returns = {
        trajectory_id: _require_finite(f"episode return for {trajectory_id!r}", episode_return)
        for trajectory_id, episode_return in episode_returns.items()
    }
    finite_eps = _require_finite("eps", eps)
    if finite_eps < 0.0:
        raise ValueError("eps must be non-negative")

    normalized_lengths: dict[TrajectoryId, int] | None = None
    if weighting == "reference_cross_steps":
        if trajectory_lengths is None:
            raise ValueError("trajectory_lengths are required for reference_cross_steps")
        normalized_lengths = {}
        for trajectory_id, length in trajectory_lengths.items():
            if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
                raise ValueError("trajectory lengths must be positive integers")
            normalized_lengths[trajectory_id] = length

    if expected_group_size is not None:
        if (
            isinstance(expected_group_size, bool)
            or not isinstance(expected_group_size, int)
            or expected_group_size <= 0
        ):
            raise ValueError("expected_group_size must be a positive integer")
        counts: dict[Hashable, int] = defaultdict(int)
        for task_id in task_ids.values():
            _require_hashable("task_id", task_id)
            counts[task_id] += 1
        wrong_sizes = {task_id: count for task_id, count in counts.items() if count != expected_group_size}
        if wrong_sizes:
            raise ValueError(f"trajectory group size mismatch: {wrong_sizes!r}")

    if weighting == "trajectory_once":
        return standardize_by_key(
            finite_returns,
            task_ids,
            eps=finite_eps,
            ddof=1,
            singleton_zero=True,
        )

    assert normalized_lengths is not None
    trajectories_by_task: dict[Hashable, list[TrajectoryId]] = defaultdict(list)
    for trajectory_id in sorted(finite_returns, key=_stable_key):
        task_id = task_ids[trajectory_id]
        _require_hashable("task_id", task_id)
        trajectories_by_task[task_id].append(trajectory_id)

    result: dict[TrajectoryId, float] = {}
    for task_id in sorted(trajectories_by_task, key=_stable_key):
        trajectories = trajectories_by_task[task_id]
        expanded_returns = [
            finite_returns[trajectory_id]
            for trajectory_id in trajectories
            for _ in range(normalized_lengths[trajectory_id])
        ]
        if len(expanded_returns) == 1:
            # Matches the frozen helper's explicit singleton branch.
            mean = 0.0
            standard_deviation = 1.0
        else:
            mean = math.fsum(expanded_returns) / len(expanded_returns)
            squared_deviations = math.fsum((value - mean) ** 2 for value in expanded_returns)
            variance = squared_deviations / (len(expanded_returns) - 1)
            standard_deviation = math.sqrt(max(variance, 0.0))

        denominator = standard_deviation + finite_eps
        for trajectory_id in trajectories:
            result[trajectory_id] = _require_finite(
                f"episode advantage for {trajectory_id!r}",
                (finite_returns[trajectory_id] - mean) / denominator,
            )

    if set(result) != set(finite_returns):
        raise ValueError("episode normalization did not return every trajectory")
    return result


def gigpo_discounted_returns(
    turns: Sequence[Turn],
    immediate_rewards: Mapping[TurnKey, float],
    *,
    gamma: float = 0.95,
) -> dict[TurnKey, float]:
    """Compute GiGPO's discounted return from every real environment turn."""

    canonical = _canonicalize_turns(turns, require_finalized=True)
    expected_keys = {turn.key for turn in canonical}
    if set(immediate_rewards) != expected_keys:
        raise ValueError("immediate_rewards must contain exactly one value per turn")
    gamma = _require_finite("gamma", gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between zero and one")
    finite_rewards = {
        key: _require_finite(f"immediate reward for {key!r}", value) for key, value in immediate_rewards.items()
    }

    by_trajectory: dict[TrajectoryId, list[Turn]] = defaultdict(list)
    for turn in canonical:
        by_trajectory[turn.trajectory_id].append(turn)

    result: dict[TurnKey, float] = {}
    for trajectory_id in sorted(by_trajectory, key=_stable_key):
        running_return = 0.0
        for turn in reversed(by_trajectory[trajectory_id]):
            running_return = finite_rewards[turn.key] + gamma * running_return
            result[turn.key] = _require_finite("discounted return", running_return)

    if set(result) != expected_keys:
        raise ValueError("discounted returns do not match input turn keys")
    return result


def _default_immediate_rewards(
    turns: Sequence[Turn],
    *,
    success_reward: float,
    invalid_penalty: float,
) -> dict[TurnKey, float]:
    result: dict[TurnKey, float] = {}
    for turn in turns:
        reward = -invalid_penalty if not turn.is_action_valid else 0.0
        if turn.success and turn.next_state_key == SUCCESS:
            reward += success_reward
        result[turn.key] = reward
    return result


def compute_method_advantages(
    method: Literal["grpo", "gigpo", "graphgpo"] | str,
    turns: Sequence[Turn],
    *,
    expected_group_size: int | None = None,
    success_reward: float = 10.0,
    invalid_penalty: float = 0.1,
    omega: float = 0.1,
    gamma: float = 0.95,
    beta_episode: float = 1.0,
    beta_graph: float = 1.0,
    beta_step: float = 1.0,
    eps: float = 1e-6,
    immediate_rewards: Mapping[TurnKey, float] | None = None,
    episode_weighting: EpisodeWeighting = DEFAULT_EPISODE_WEIGHTING,
    graph_diagnostics_callback: DiagnosticsCallback | None = None,
    diagnostic_rollout_group_id: object | None = None,
    diagnostic_policy_version: object | None = None,
) -> dict[TurnKey, float]:
    """Compute GRPO, GiGPO, or GraphGPO scalar advantages for every turn."""

    if method not in {"grpo", "gigpo", "graphgpo"}:
        raise ValueError(f"unknown method: {method!r}")
    canonical = _canonicalize_turns(turns, require_finalized=True)
    success_reward = _require_finite("success_reward", success_reward)
    invalid_penalty = _require_finite("invalid_penalty", invalid_penalty)
    beta_episode = _require_finite("beta_episode", beta_episode)
    beta_graph = _require_finite("beta_graph", beta_graph)
    beta_step = _require_finite("beta_step", beta_step)
    if success_reward < 0.0 or invalid_penalty < 0.0:
        raise ValueError("reward and penalty magnitudes must be non-negative")

    by_trajectory: dict[TrajectoryId, list[Turn]] = defaultdict(list)
    task_by_trajectory: dict[TrajectoryId, Hashable] = {}
    for turn in canonical:
        by_trajectory[turn.trajectory_id].append(turn)
        task_by_trajectory[turn.trajectory_id] = turn.task_id

    returns = {
        trajectory_id: episode_return(
            trajectory,
            success_reward=success_reward,
            invalid_penalty=invalid_penalty,
        )
        for trajectory_id, trajectory in by_trajectory.items()
    }
    per_trajectory_episode = episode_advantages(
        returns,
        task_by_trajectory,
        expected_group_size=expected_group_size,
        eps=eps,
        weighting=episode_weighting,
        trajectory_lengths={trajectory_id: len(trajectory) for trajectory_id, trajectory in by_trajectory.items()},
    )
    episode_by_turn = {turn.key: per_trajectory_episode[turn.trajectory_id] for turn in canonical}

    if method == "grpo":
        result = {key: beta_episode * value for key, value in episode_by_turn.items()}
    elif method == "graphgpo":
        graph_by_turn: dict[TurnKey, float] = {}
        turns_by_task: dict[Hashable, list[Turn]] = defaultdict(list)
        for turn in canonical:
            turns_by_task[turn.task_id].append(turn)
        for task_id in sorted(turns_by_task, key=_stable_key):
            build_started_ns = perf_counter_ns() if graph_diagnostics_callback is not None else 0
            graph = build_occurrence_graph(turns_by_task[task_id])
            graph_build_ns = perf_counter_ns() - build_started_ns if graph_diagnostics_callback is not None else 0
            dijkstra_started_ns = perf_counter_ns() if graph_diagnostics_callback is not None else 0
            distance_result = reverse_shortest_distances(graph)
            reverse_dijkstra_ns = (
                perf_counter_ns() - dijkstra_started_ns if graph_diagnostics_callback is not None else 0
            )
            advantage_started_ns = perf_counter_ns() if graph_diagnostics_callback is not None else 0
            task_graph_advantages = graph_advantages(
                graph,
                distance_result,
                omega=omega,
                success_reward=success_reward,
                eps=eps,
                convention="reference",
            )
            graph_advantage_ns = (
                perf_counter_ns() - advantage_started_ns if graph_diagnostics_callback is not None else 0
            )
            graph_by_turn.update(task_graph_advantages)
            if graph_diagnostics_callback is not None:
                graph_diagnostics_callback(
                    build_graph_diagnostics(
                        task_id=task_id,
                        nodes=graph.nodes,
                        occurrences=graph.occurrences,
                        distances=distance_result.distances,
                        max_finite_distance=distance_result.max_finite_distance,
                        has_success=distance_result.has_success,
                        advantages=task_graph_advantages,
                        graph_build_ns=graph_build_ns,
                        reverse_dijkstra_ns=reverse_dijkstra_ns,
                        graph_advantage_ns=graph_advantage_ns,
                        rollout_group_id=diagnostic_rollout_group_id,
                        policy_version=diagnostic_policy_version,
                    )
                )
        result = {
            turn.key: beta_episode * episode_by_turn[turn.key] + beta_graph * graph_by_turn[turn.key]
            for turn in canonical
        }
    else:
        if immediate_rewards is None:
            immediate_rewards = _default_immediate_rewards(
                canonical,
                success_reward=success_reward,
                invalid_penalty=invalid_penalty,
            )
        discounted_returns = gigpo_discounted_returns(canonical, immediate_rewards, gamma=gamma)
        state_groups = {turn.key: (turn.task_id, turn.state_key) for turn in canonical}
        step_by_turn = standardize_by_key(
            discounted_returns,
            state_groups,
            eps=eps,
            ddof=1,
            singleton_zero=True,
        )
        result = {
            turn.key: beta_episode * episode_by_turn[turn.key] + beta_step * step_by_turn[turn.key]
            for turn in canonical
        }

    expected_keys = {turn.key for turn in canonical}
    if set(result) != expected_keys:
        raise ValueError("computed advantages do not match input turn keys")
    return {key: _require_finite(f"advantage for {key!r}", result[key]) for key in sorted(result, key=_stable_key)}
