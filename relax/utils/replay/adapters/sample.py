# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Sample-stage adapter (data-layer integrity)."""

from __future__ import annotations

from typing import Any

from relax.utils.replay.bundle import LoadedBundle
from relax.utils.replay.report import FieldDivergence, StageResult, StageStatus
from relax.utils.replay.schema import StageId
from relax.utils.replay.stages import register_adapter


@register_adapter(StageId.SAMPLE)
def replay_sample(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Verify sample records are internally consistent.

    This is the data-layer ``recompute``: it does not produce a number but
    checks that every record is complete and well-formed, mirroring what the
    validator enforces at load time.
    """
    result = StageResult(stage=StageId.SAMPLE.value, status=StageStatus.PASS)
    for record in bundle.index.samples:
        if record.response_length < 0 or record.total_length < record.response_length:
            result.status = StageStatus.FAIL
            result.divergences.append(
                FieldDivergence(field="length", sample_id=record.sample_id, expected=None, actual=None)
            )
        if len(record.loss_mask) != record.response_length:
            result.status = StageStatus.FAIL
            result.divergences.append(
                FieldDivergence(field="loss_mask_length", sample_id=record.sample_id, expected=None, actual=None)
            )
    if result.status == StageStatus.FAIL:
        result.message = f"{len(result.divergences)} sample record(s) are inconsistent"
    return result
