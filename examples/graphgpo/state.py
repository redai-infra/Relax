# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deterministic ALFWorld state anchors without an environment dependency."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence


STATE_ANCHOR_VERSION = "reference_anchor_v1"
ALFWORLD_TRACKER_FIELDS = (
    "location",
    "holding",
    "history_items",
    "item_location",
)
_ALFWORLD_TRACKER_FIELD_SET = frozenset(ALFWORLD_TRACKER_FIELDS)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("tracker values must be JSON-serializable and finite") from exc


@dataclass(frozen=True)
class TrackerState:
    """An immutable snapshot of wrapper-maintained world-state fields."""

    encoded_fields: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "TrackerState":
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping")

        supplied_fields: set[str] = set()
        for key in values:
            if not isinstance(key, str) or not key:
                raise ValueError("tracker field names must be non-empty strings")
            supplied_fields.add(key)

        missing = _ALFWORLD_TRACKER_FIELD_SET - supplied_fields
        unexpected = supplied_fields - _ALFWORLD_TRACKER_FIELD_SET
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={sorted(missing)!r}")
            if unexpected:
                details.append(f"unexpected={sorted(unexpected)!r}")
            raise ValueError(
                f"tracker fields must be exactly {list(ALFWORLD_TRACKER_FIELDS)!r} ({', '.join(details)})"
            )

        encoded_fields = tuple((field, _canonical_json(values[field])) for field in ALFWORLD_TRACKER_FIELDS)
        return cls(encoded_fields)

    def to_mapping(self) -> dict[str, object]:
        return {key: json.loads(value) for key, value in self.encoded_fields}

    def render(self) -> str:
        return "\n".join(f"{key}={value}" for key, value in self.encoded_fields)


def update_tracker(
    tracker: TrackerState,
    *,
    raw_observation: str,
    updates: Mapping[str, object],
) -> TrackerState:
    """Apply structured wrapper updates unless the action changed nothing."""

    if not isinstance(tracker, TrackerState):
        raise TypeError("tracker must be a TrackerState")
    if not isinstance(raw_observation, str):
        raise TypeError("raw_observation must be a string")
    if not isinstance(updates, Mapping):
        raise TypeError("updates must be a mapping")

    update_fields: set[str] = set()
    for key in updates:
        if not isinstance(key, str) or not key:
            raise ValueError("tracker update field names must be non-empty strings")
        update_fields.add(key)
    unexpected = update_fields - _ALFWORLD_TRACKER_FIELD_SET
    if unexpected:
        raise ValueError(f"tracker updates contain unexpected fields: {sorted(unexpected)!r}")

    if "nothing happens" in raw_observation.casefold():
        return tracker

    merged = tracker.to_mapping()
    merged.update(updates)
    return TrackerState.from_mapping(merged)


def _length_prefixed(value: str) -> str:
    return f"{len(value)}:{value}"


def _active_item_states(
    item: str,
    history_items: Mapping[str, object],
) -> str:
    state_value = history_items.get(item, {})
    if not isinstance(state_value, Mapping):
        raise TypeError("every history_items value must be a mapping")
    active_states: list[str] = []
    for state_name, active in state_value.items():
        if not isinstance(state_name, str) or not state_name:
            raise TypeError("item state names must be non-empty strings")
        if not isinstance(active, bool):
            raise TypeError("item state values must be booleans")
        if active:
            active_states.append(state_name)
    if active_states:
        return f"({' '.join(active_states)})"
    return "(unprocessed)"


def format_reference_observation(
    raw_observation: str,
    tracker: TrackerState,
) -> str:
    """Append only tracker context displayed by the frozen ALFWorld worker."""

    if not isinstance(raw_observation, str):
        raise TypeError("raw_observation must be a string")
    if not isinstance(tracker, TrackerState):
        raise TypeError("tracker must be a TrackerState")
    values = tracker.to_mapping()

    location = values.get("location")
    holding = values.get("holding")
    history_items = values.get("history_items")
    item_location = values.get("item_location")
    if not isinstance(location, str) or not isinstance(holding, str):
        raise TypeError("tracker location and holding must be strings")
    if not isinstance(history_items, Mapping):
        raise TypeError("tracker history_items must be a mapping")
    if not isinstance(item_location, Mapping):
        raise TypeError("tracker item_location must be a mapping")

    holding_states: list[str] = []
    for item in history_items:
        if not isinstance(item, str):
            raise TypeError("history item names must be strings")
        if item in holding:
            active = _active_item_states(item, history_items)
            if active != "(unprocessed)":
                holding_states.append(active[1:-1])
    holding_suffix = ""
    if holding_states:
        holding_suffix = f"({' '.join(holding_states)})"
    elif holding != "nothing":
        holding_suffix = "(unprocessed)"

    movement_entries: list[str] = []
    for item, movement in item_location.items():
        if not isinstance(item, str):
            raise TypeError("item_location names must be strings")
        if not isinstance(movement, Mapping):
            raise TypeError("every item_location value must be a mapping")
        old_location = movement.get("old_location")
        new_location = movement.get("new_location")
        if not isinstance(old_location, str) or not isinstance(new_location, str):
            raise TypeError("item locations must be strings")
        item_state = _active_item_states(item, history_items)
        if old_location != new_location or (item_state != "(unprocessed)" and holding != item):
            movement_entries.append(f"{item}{item_state} from {old_location} move to {new_location};")

    movement_history = ""
    if movement_entries:
        movement_history = "History moving list by you: " + "".join(movement_entries)
    location_holding = f"Location: {location}. Items in hand (status): {holding}{holding_suffix}."
    return f"{raw_observation} {location_holding} {movement_history}"


def reference_anchor_v1(
    raw_observation: str,
    tracker: TrackerState,
    admissible_commands: Sequence[str],
) -> str:
    """Build the frozen reference-style, task-local state anchor.

    The anchor uses exactly the tracker-derived context displayed to the model,
    rather than internal bookkeeping fields that the reference worker filters
    out. It also sorts all admissible commands. In particular, ``help`` remains
    part of the state even though it is hidden from the model prompt. Length
    prefixes make the representation collision resistant without depending on
    process-specific ``hash()``.
    """

    if not isinstance(raw_observation, str):
        raise TypeError("raw_observation must be a string")
    if not isinstance(tracker, TrackerState):
        raise TypeError("tracker must be a TrackerState")
    if isinstance(admissible_commands, (str, bytes)) or not isinstance(admissible_commands, Sequence):
        raise TypeError("admissible_commands must be a sequence of strings")

    commands: list[str] = []
    for command in admissible_commands:
        if not isinstance(command, str):
            raise TypeError("every admissible command must be a string")
        commands.append(command)
    commands.sort()

    display_observation = format_reference_observation(raw_observation, tracker)
    parts = [
        STATE_ANCHOR_VERSION,
        f"observation={_length_prefixed(display_observation)}",
        f"command_count={len(commands)}",
    ]
    parts.extend(f"command={_length_prefixed(command)}" for command in commands)
    return "\n".join(parts)


def state_key_v1(
    raw_observation: str,
    tracker: TrackerState,
    admissible_commands: Sequence[str],
) -> str:
    """Return a stable SHA256 key for ``reference_anchor_v1``."""

    anchor = reference_anchor_v1(raw_observation, tracker, admissible_commands)
    return hashlib.sha256(anchor.encode("utf-8")).hexdigest()
