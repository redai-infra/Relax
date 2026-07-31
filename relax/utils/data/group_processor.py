# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Helpers for safely reusing multimodal processor outputs within a prompt
group."""

from collections.abc import Sequence
from typing import Any


MM_GROUP_ID_KEY = "__relax_mm_group_id__"
MM_GROUP_OWNER_KEY = "__relax_mm_group_owner__"


def get_reusable_group_processor_input(samples: Sequence[Any]) -> tuple[Any, dict] | None:
    """Return shared prompt/media when a whole sample group is safe to
    preprocess once."""
    if not samples:
        return None

    first_prompt = samples[0].prompt
    first_mm = samples[0].multimodal_inputs
    if first_mm is None or not any(first_mm.get(key) for key in ("images", "videos", "audio")):
        return None
    if any(sample.prompt != first_prompt or sample.multimodal_inputs is not first_mm for sample in samples[1:]):
        return None
    return first_prompt, first_mm


def copy_group_processor_output(
    prompt_ids: list[int],
    multimodal_train_inputs: dict[str, Any] | None,
) -> tuple[list[int], dict[str, Any] | None]:
    """Copy mutable containers per sample while retaining shared tensor
    storage."""
    copied_train_inputs = None
    if multimodal_train_inputs is not None:
        copied_train_inputs = {
            key: list(value) if isinstance(value, list) else value for key, value in multimodal_train_inputs.items()
        }
    return list(prompt_ids), copied_train_inputs


def _copy_multimodal_train_inputs(multimodal_train_inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: list(value) if isinstance(value, list) else value for key, value in multimodal_train_inputs.items()}


def _shares_payload_storage(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_shares_payload_storage(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_shares_payload_storage(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, (str, bytes, int, float, bool, type(None))):
        return left == right
    return False


def _as_cpu_scalar(value: Any, name: str) -> int | bool:
    if isinstance(value, (int, bool)):
        return value

    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.numel() != 1:
            raise ValueError(f"{name} must be a CPU scalar")
        value = int(value) if value.dtype != torch.bool else bool(value)
    if not isinstance(value, (int, bool)):
        raise ValueError(f"{name} must be an integer or boolean scalar")
    return value


def get_group_multimodal_transport_marker(item: dict[str, Any]) -> tuple[int, bool] | None:
    """Return the group id and owner flag carried by a packed row."""
    has_group_id = MM_GROUP_ID_KEY in item
    has_owner = MM_GROUP_OWNER_KEY in item
    if has_group_id != has_owner:
        raise ValueError("multimodal group transport marker is incomplete")
    if not has_group_id:
        return None

    group_index = _as_cpu_scalar(item[MM_GROUP_ID_KEY], MM_GROUP_ID_KEY)
    is_owner = _as_cpu_scalar(item[MM_GROUP_OWNER_KEY], MM_GROUP_OWNER_KEY)
    return int(group_index), bool(is_owner)


def _is_empty_transport_placeholder(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    if isinstance(value, dict):
        return all(_is_empty_transport_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_empty_transport_placeholder(item) for item in value)
    return False


def validate_group_multimodal_transport_ref(item: dict[str, Any]) -> None:
    """Reject ref rows that would silently discard a non-empty payload."""
    payload = {key: value for key, value in item.items() if key not in {MM_GROUP_ID_KEY, MM_GROUP_OWNER_KEY}}
    if payload and not all(_is_empty_transport_placeholder(value) for value in payload.values()):
        raise ValueError("multimodal group ref contains a non-empty payload")


def pack_group_multimodal_train_inputs(
    samples: Sequence[Any],
    group_size: int,
) -> tuple[list[dict[str, Any] | None], int, int]:
    """Store one tensor payload per complete prompt group and lightweight refs
    for the remaining samples."""
    if group_size <= 1:
        raise ValueError(f"group_size must be greater than 1, got {group_size}")

    original = [sample.multimodal_train_inputs for sample in samples]
    if not samples or len(samples) % group_size != 0:
        return original, 0, 0

    positions_by_group: dict[int, list[int]] = {}
    for position, sample in enumerate(samples):
        group_index = getattr(sample, "group_index", None)
        if not isinstance(group_index, int):
            return original, 0, 0
        positions_by_group.setdefault(group_index, []).append(position)

    if len(positions_by_group) * group_size != len(samples):
        return original, 0, 0

    source_positions: dict[int, int] = {}
    for group_index, positions in positions_by_group.items():
        if len(positions) != group_size:
            return original, 0, 0
        source = original[positions[0]]
        if not isinstance(source, dict):
            return original, 0, 0
        if any(
            isinstance(original[position], dict)
            and (MM_GROUP_ID_KEY in original[position] or MM_GROUP_OWNER_KEY in original[position])
            for position in positions
        ):
            raise ValueError("multimodal processor output collides with Relax group transfer marker keys")
        if any(not _shares_payload_storage(source, original[position]) for position in positions[1:]):
            return original, 0, 0
        source_positions[group_index] = positions[0]

    packed: list[dict[str, Any]] = []
    for position, sample in enumerate(samples):
        is_owner = source_positions[sample.group_index] == position
        item = _copy_multimodal_train_inputs(original[position]) if is_owner else {}
        item[MM_GROUP_ID_KEY] = sample.group_index
        item[MM_GROUP_OWNER_KEY] = is_owner
        packed.append(item)
    source_count = len(source_positions)
    return packed, source_count, len(samples) - source_count


def unpack_group_multimodal_train_inputs(
    packed_inputs: Sequence[dict[str, Any] | None],
    group_size: int,
) -> list[dict[str, Any] | None]:
    """Expand group refs after transport while retaining shared tensor
    storage."""
    if group_size <= 1:
        raise ValueError(f"group_size must be greater than 1, got {group_size}")
    if len(packed_inputs) % group_size != 0:
        raise ValueError(
            f"multimodal transfer batch size {len(packed_inputs)} is not divisible by group_size {group_size}"
        )

    sources: dict[int, dict[str, Any]] = {}
    marker_counts: dict[int, int] = {}
    for item in packed_inputs:
        if not isinstance(item, dict):
            continue
        marker = get_group_multimodal_transport_marker(item)
        if marker is not None:
            group_index, is_owner = marker
            marker_counts[group_index] = marker_counts.get(group_index, 0) + 1
        else:
            continue
        if is_owner:
            if group_index in sources:
                raise ValueError(f"duplicate multimodal group source for group {group_index}")
            source = {key: value for key, value in item.items() if key not in {MM_GROUP_ID_KEY, MM_GROUP_OWNER_KEY}}
            if not source:
                raise ValueError(f"empty multimodal group source for group {group_index}")
            sources[group_index] = source
        else:
            validate_group_multimodal_transport_ref(item)

    for group_index, count in marker_counts.items():
        if count != group_size:
            raise ValueError(
                f"multimodal transfer group {group_index} has {count} rows, expected group_size {group_size}"
            )

    unpacked: list[dict[str, Any] | None] = []
    for item in packed_inputs:
        if not isinstance(item, dict):
            unpacked.append(item)
            continue
        marker = get_group_multimodal_transport_marker(item)
        if marker is not None:
            group_index, _is_owner = marker
            if group_index not in sources:
                raise ValueError(f"missing multimodal group source for group {group_index}")
            unpacked.append(_copy_multimodal_train_inputs(sources[group_index]))
        else:
            unpacked.append(item)
    return unpacked
