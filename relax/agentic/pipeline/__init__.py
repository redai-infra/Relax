# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from typing import Any


SampleKey = tuple[str, int]
GroupKey = tuple[SampleKey, ...]


def sample_key(sample: Any) -> SampleKey:
    return sample.session_id, sample.index


def sample_group_key(group: list[Any]) -> GroupKey:
    return tuple(sample_key(sample) for sample in group)
