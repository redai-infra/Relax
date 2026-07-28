# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from relax.utils.types import Sample


SampleKey = tuple[Any, Any, Any]
GroupKey = tuple[SampleKey, ...]


class ExportMode(str, Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


@dataclass
class PendingExportUnit:
    name: str | None
    sample: Sample
    assistant_token_spans: list[tuple[int, int]]
    mode: ExportMode


def sample_key(sample: Any) -> SampleKey:
    return sample.session_id, sample.index, getattr(sample, "_agentic_export_name", None)


def sample_group_key(group: list[Any]) -> GroupKey:
    return tuple(sample_key(sample) for sample in group)
