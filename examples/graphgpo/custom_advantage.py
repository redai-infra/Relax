# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Custom-advantage adapter for the GraphGPO reproduction recipe.

Relax calls :func:`compute_custom_advantage` with the value returned by
``RewardWaitingGroup.metadata_by_slot()``.  Each slot is one trajectory and
each named unit is one environment turn.  This module validates that contract,
delegates the numerical work to :mod:`examples.graphgpo.graph_credit`, and
returns one scalar advantage for every input unit.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from examples.graphgpo.diagnostics import diagnostics_callback_from_environment
from examples.graphgpo.graph_credit import (
    DEFAULT_EPISODE_WEIGHTING,
    EPISODE_WEIGHTINGS,
    EpisodeWeighting,
    Turn,
    compute_method_advantages,
    episode_return,
)


Method = Literal["grpo", "gigpo", "graphgpo"]
_METHODS = frozenset(("grpo", "gigpo", "graphgpo"))
_TURN_NAME = re.compile(r"^turn_(\d{3,})$")
_REQUIRED_FIELDS = (
    "row_id",
    "rollout_group_id",
    "policy_version",
    "task_id",
    "trajectory_id",
    "turn_index",
    "state_key",
    "action",
    "next_state_key",
    "is_action_valid",
    "success",
    "terminal",
    "truncated",
    "episode_return",
)


@dataclass(frozen=True)
class _UnitRef:
    slot_index: int
    unit_name: str
    row_id: Hashable
    rollout_group_id: Hashable
    policy_version: Hashable
    turn: Turn
    declared_episode_return: float


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_non_negative_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return result


def _require_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _require_hashable(name: str, value: object) -> Hashable:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a hashable non-boolean value")
    try:
        hash(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be hashable") from exc
    return value  # type: ignore[return-value]


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _required(metadata: Mapping[str, Any], field: str, context: str) -> Any:
    if field not in metadata:
        raise ValueError(f"{context} is missing required metadata field {field!r}")
    return metadata[field]


def _parse_unit(slot_index: int, unit_name: object, metadata: object) -> _UnitRef:
    if not isinstance(unit_name, str):
        raise ValueError(f"slot {slot_index} contains a non-string unit name")
    context = f"slot {slot_index} unit {unit_name!r}"
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{context} metadata must be a mapping")
    for field in _REQUIRED_FIELDS:
        _required(metadata, field, context)

    turn_index = _required(metadata, "turn_index", context)
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError(f"{context} turn_index must be a non-negative integer")
    match = _TURN_NAME.fullmatch(unit_name)
    if match is None or int(match.group(1)) != turn_index:
        raise ValueError(f"{context} must use the canonical name {f'turn_{turn_index:03d}'!r}")

    action = _required(metadata, "action", context)
    if not isinstance(action, str):
        raise ValueError(f"{context} action must be a string")
    cost = _require_finite(f"{context} cost", metadata.get("cost", 1.0))
    if cost <= 0.0:
        raise ValueError(f"{context} cost must be greater than zero")

    turn = Turn(
        task_id=_require_hashable(f"{context} task_id", _required(metadata, "task_id", context)),
        trajectory_id=_require_hashable(
            f"{context} trajectory_id",
            _required(metadata, "trajectory_id", context),
        ),
        turn_index=turn_index,
        state_key=_require_hashable(f"{context} state_key", _required(metadata, "state_key", context)),
        action=action,
        next_state_key=_require_hashable(
            f"{context} next_state_key",
            _required(metadata, "next_state_key", context),
        ),
        success=_require_bool(f"{context} success", _required(metadata, "success", context)),
        terminal=_require_bool(f"{context} terminal", _required(metadata, "terminal", context)),
        truncated=_require_bool(f"{context} truncated", _required(metadata, "truncated", context)),
        is_action_valid=_require_bool(
            f"{context} is_action_valid",
            _required(metadata, "is_action_valid", context),
        ),
        cost=cost,
    )
    declared_episode_return = _require_finite(
        f"{context} episode_return",
        _required(metadata, "episode_return", context),
    )
    return _UnitRef(
        slot_index=slot_index,
        unit_name=unit_name,
        row_id=_require_hashable(f"{context} row_id", _required(metadata, "row_id", context)),
        rollout_group_id=_require_hashable(
            f"{context} rollout_group_id",
            _required(metadata, "rollout_group_id", context),
        ),
        policy_version=_require_hashable(
            f"{context} policy_version",
            _required(metadata, "policy_version", context),
        ),
        turn=turn,
        declared_episode_return=declared_episode_return,
    )


def _parse_group(
    metadata_by_slot: object,
    *,
    expected_group_size: int,
    success_reward: float,
    invalid_penalty: float,
) -> tuple[tuple[_UnitRef, ...], ...]:
    if isinstance(metadata_by_slot, (str, bytes)) or not isinstance(metadata_by_slot, Sequence):
        raise ValueError("metadata_by_slot must be a sequence of slot mappings")
    if len(metadata_by_slot) != expected_group_size:
        raise ValueError(f"rollout group size mismatch: expected {expected_group_size}, got {len(metadata_by_slot)}")

    parsed_slots: list[tuple[_UnitRef, ...]] = []
    seen_trajectory_ids: set[Hashable] = set()
    seen_row_ids: set[Hashable] = set()
    unset_task_id = object()
    task_id: Hashable | object = unset_task_id
    unset_rollout_group_id = object()
    rollout_group_id: Hashable | object = unset_rollout_group_id
    unset_policy_version = object()
    policy_version: Hashable | object = unset_policy_version
    for slot_index, slot in enumerate(metadata_by_slot):
        if not isinstance(slot, Mapping):
            raise ValueError(f"slot {slot_index} must be a unit metadata mapping")
        if not slot:
            raise ValueError(f"slot {slot_index} must contain at least one turn unit")

        refs = tuple(_parse_unit(slot_index, unit_name, metadata) for unit_name, metadata in slot.items())
        for ref in refs:
            if ref.row_id in seen_row_ids:
                raise ValueError(f"duplicate row_id in rollout group: {ref.row_id!r}")
            seen_row_ids.add(ref.row_id)
            if rollout_group_id is unset_rollout_group_id:
                rollout_group_id = ref.rollout_group_id
            elif ref.rollout_group_id != rollout_group_id:
                raise ValueError("one custom-advantage call cannot mix multiple rollout_group_id values")
            if policy_version is unset_policy_version:
                policy_version = ref.policy_version
            elif ref.policy_version != policy_version:
                raise ValueError("one custom-advantage call cannot mix multiple policy_version values")

        trajectory_ids = {ref.turn.trajectory_id for ref in refs}
        if len(trajectory_ids) != 1:
            raise ValueError(f"slot {slot_index} mixes multiple trajectory_id values")
        trajectory_id = next(iter(trajectory_ids))
        if trajectory_id in seen_trajectory_ids:
            raise ValueError(f"trajectory_id {trajectory_id!r} appears in more than one slot")
        seen_trajectory_ids.add(trajectory_id)

        task_ids = {ref.turn.task_id for ref in refs}
        if len(task_ids) != 1:
            raise ValueError(f"slot {slot_index} mixes multiple task_id values")
        slot_task_id = next(iter(task_ids))
        if task_id is unset_task_id:
            task_id = slot_task_id
        elif slot_task_id != task_id:
            raise ValueError("one rollout group cannot mix multiple task_id values")

        refs_by_turn = tuple(sorted(refs, key=lambda ref: ref.turn.turn_index))
        actual_indices = [ref.turn.turn_index for ref in refs_by_turn]
        expected_indices = list(range(len(refs_by_turn)))
        if actual_indices != expected_indices:
            raise ValueError(f"slot {slot_index} has non-contiguous turn indices: {actual_indices!r}")
        expected_names = [f"turn_{turn_index:03d}" for turn_index in expected_indices]
        actual_names = [ref.unit_name for ref in refs_by_turn]
        if actual_names != expected_names:
            raise ValueError(f"slot {slot_index} turn unit names do not match their indices")

        declared_returns = {ref.declared_episode_return for ref in refs_by_turn}
        if len(declared_returns) != 1:
            raise ValueError(f"slot {slot_index} has inconsistent episode_return values")
        computed_return = episode_return(
            [ref.turn for ref in refs_by_turn],
            success_reward=success_reward,
            invalid_penalty=invalid_penalty,
        )
        declared_return = next(iter(declared_returns))
        if not math.isclose(declared_return, computed_return, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"slot {slot_index} episode_return mismatch: declared {declared_return}, computed {computed_return}"
            )
        parsed_slots.append(refs_by_turn)

    return tuple(parsed_slots)


def compute_group_advantages(
    metadata_by_slot: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    method: Method | str = "graphgpo",
    expected_group_size: int = 8,
    omega: float = 0.1,
    gamma: float = 0.95,
    beta: float = 1.0,
    beta_episode: float = 1.0,
    success_reward: float = 10.0,
    invalid_penalty: float = 0.1,
    eps: float = 1e-6,
    episode_weighting: EpisodeWeighting | str = DEFAULT_EPISODE_WEIGHTING,
) -> list[dict[str, float]]:
    """Validate one complete rollout group and return per-unit advantages.

    ``beta`` weights the method-specific graph or step term.  ``beta_episode``
    weights the episode-level GRPO term.  For ``method="grpo"`` only
    ``beta_episode`` is used.

    Invalid input raises a descriptive exception.  This function never returns
    ``None`` because the recipe does not silently drop malformed groups.
    """

    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)!r}, got {method!r}")
    if not isinstance(episode_weighting, str) or episode_weighting not in EPISODE_WEIGHTINGS:
        raise ValueError(f"episode_weighting must be one of {sorted(EPISODE_WEIGHTINGS)!r}, got {episode_weighting!r}")
    expected_group_size = _require_positive_int("expected_group_size", expected_group_size)
    omega = _require_finite("omega", omega)
    gamma = _require_finite("gamma", gamma)
    beta = _require_finite("beta", beta)
    beta_episode = _require_finite("beta_episode", beta_episode)
    success_reward = _require_non_negative_finite("success_reward", success_reward)
    invalid_penalty = _require_non_negative_finite("invalid_penalty", invalid_penalty)
    eps = _require_non_negative_finite("eps", eps)

    parsed_slots = _parse_group(
        metadata_by_slot,
        expected_group_size=expected_group_size,
        success_reward=success_reward,
        invalid_penalty=invalid_penalty,
    )
    turns = [ref.turn for refs in parsed_slots for ref in refs]
    advantages = compute_method_advantages(
        method,
        turns,
        expected_group_size=expected_group_size,
        success_reward=success_reward,
        invalid_penalty=invalid_penalty,
        omega=omega,
        gamma=gamma,
        beta_episode=beta_episode,
        beta_graph=beta,
        beta_step=beta,
        eps=eps,
        episode_weighting=episode_weighting,
        graph_diagnostics_callback=diagnostics_callback_from_environment(),
        diagnostic_rollout_group_id=parsed_slots[0][0].rollout_group_id,
        diagnostic_policy_version=parsed_slots[0][0].policy_version,
    )

    result: list[dict[str, float]] = []
    for refs in parsed_slots:
        slot_result: dict[str, float] = {}
        for ref in refs:
            value = _require_finite(
                f"advantage for slot {ref.slot_index} unit {ref.unit_name!r}",
                advantages[ref.turn.key],
            )
            slot_result[ref.unit_name] = value
        result.append(slot_result)

    expected_shape = [set(slot) for slot in metadata_by_slot]
    actual_shape = [set(slot) for slot in result]
    if actual_shape != expected_shape:
        raise ValueError("custom advantage output shape does not match its input")
    return result


def _environment_value(primary: str, fallback: str | None, default: str) -> str:
    if primary in os.environ:
        return os.environ[primary]
    if fallback is not None and fallback in os.environ:
        return os.environ[fallback]
    return default


def _environment_int(primary: str, fallback: str | None, default: int) -> int:
    raw = _environment_value(primary, fallback, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{primary} must be an integer, got {raw!r}") from exc
    return _require_positive_int(primary, value)


def _environment_float(primary: str, fallback: str | None, default: float) -> float:
    raw = _environment_value(primary, fallback, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{primary} must be a real number, got {raw!r}") from exc
    return _require_finite(primary, value)


def compute_custom_advantage(
    metadata_by_slot: Sequence[Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, float]]:
    """Environment-configured hook loaded by ``--agentic-custom-advantage-
    path``.

    Recipe-specific variables take precedence over the short names used by the
    launch script.  Defaults reproduce the proposal's GraphGPO group-of-eight
    configuration, including one vote per trajectory for episode statistics.
    """

    method = _environment_value("GRAPHGPO_METHOD", "METHOD", "graphgpo")
    episode_weighting = _environment_value(
        "GRAPHGPO_EPISODE_WEIGHTING",
        "EPISODE_WEIGHTING",
        DEFAULT_EPISODE_WEIGHTING,
    )
    return compute_group_advantages(
        metadata_by_slot,
        method=method,
        episode_weighting=episode_weighting,
        expected_group_size=_environment_int("GRAPHGPO_EXPECTED_GROUP_SIZE", "GROUP_SIZE", 8),
        omega=_environment_float("GRAPHGPO_OMEGA", "OMEGA", 0.1),
        gamma=_environment_float("GRAPHGPO_GAMMA", "GAMMA", 0.95),
        beta=_environment_float("GRAPHGPO_BETA", "BETA", 1.0),
        beta_episode=_environment_float("GRAPHGPO_BETA_EPISODE", "BETA_EPISODE", 1.0),
    )
