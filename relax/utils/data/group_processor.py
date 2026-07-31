# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Helpers for safely reusing multimodal processor outputs within a prompt
group."""

from collections.abc import Sequence
from typing import Any


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
