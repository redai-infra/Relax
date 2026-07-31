# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Helpers for safely reusing multimodal processor outputs within a prompt
group."""

from collections.abc import Sequence
from typing import Any


MM_GROUP_SOURCE_KEY = "__relax_mm_group_source__"
MM_GROUP_REF_KEY = "__relax_mm_group_ref__"


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


def pack_group_multimodal_train_inputs(
    samples: Sequence[Any],
    group_size: int,
) -> tuple[list[dict[str, Any] | None], int, int]:
    """Store one tensor payload per complete prompt group and lightweight refs
    for the remaining samples."""
    if group_size <= 1:
        raise ValueError(f"group_size must be greater than 1, got {group_size}")

    packed = [sample.multimodal_train_inputs for sample in samples]
    positions_by_group: dict[int, list[int]] = {}
    for position, sample in enumerate(samples):
        group_index = getattr(sample, "group_index", None)
        if group_index is not None:
            positions_by_group.setdefault(group_index, []).append(position)

    source_count = 0
    ref_count = 0
    for group_index, positions in positions_by_group.items():
        if len(positions) != group_size:
            continue
        source = packed[positions[0]]
        if not isinstance(source, dict):
            continue
        if any(
            isinstance(packed[position], dict)
            and (MM_GROUP_SOURCE_KEY in packed[position] or MM_GROUP_REF_KEY in packed[position])
            for position in positions
        ):
            raise ValueError("multimodal processor output collides with Relax group transfer marker keys")
        if any(not _shares_payload_storage(source, packed[position]) for position in positions[1:]):
            continue

        source_payload = _copy_multimodal_train_inputs(source)
        source_payload[MM_GROUP_SOURCE_KEY] = group_index
        packed[positions[0]] = source_payload
        for position in positions[1:]:
            packed[position] = {MM_GROUP_REF_KEY: group_index}
        source_count += 1
        ref_count += len(positions) - 1

    return packed, source_count, ref_count


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
        is_source = MM_GROUP_SOURCE_KEY in item
        is_ref = MM_GROUP_REF_KEY in item
        if is_source and is_ref:
            raise ValueError("multimodal group source cannot also be a group ref")
        if is_source:
            group_index = item[MM_GROUP_SOURCE_KEY]
            if group_index in sources:
                raise ValueError(f"duplicate multimodal group source for group {group_index}")
            source = {key: value for key, value in item.items() if key != MM_GROUP_SOURCE_KEY}
            if not source:
                raise ValueError(f"empty multimodal group source for group {group_index}")
            sources[group_index] = source
            marker_counts[group_index] = marker_counts.get(group_index, 0) + 1
        elif is_ref:
            group_index = item[MM_GROUP_REF_KEY]
            marker_counts[group_index] = marker_counts.get(group_index, 0) + 1

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
        if MM_GROUP_REF_KEY in item:
            if item.keys() != {MM_GROUP_REF_KEY}:
                raise ValueError("multimodal group ref contains unexpected payload fields")
            group_index = item[MM_GROUP_REF_KEY]
            if group_index not in sources:
                raise ValueError(f"missing multimodal group source for group {group_index}")
            unpacked.append(_copy_multimodal_train_inputs(sources[group_index]))
        elif MM_GROUP_SOURCE_KEY in item:
            unpacked.append(_copy_multimodal_train_inputs(sources[item[MM_GROUP_SOURCE_KEY]]))
        else:
            unpacked.append(item)
    return unpacked
