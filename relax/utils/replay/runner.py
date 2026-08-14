# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline replay runner.

Executes the pipeline stages in DAG order. Only stages that (a) resolve to
recompute under the frozen V1 matrix for the bundle's topology and (b) are
declared recompute in the manifest are executed; everything else is reported as
skipped and never counted toward the first-divergent-stage determination.
"""

from __future__ import annotations

from pathlib import Path

from relax.utils.logging_utils import get_logger
from relax.utils.replay.adapters import register_all
from relax.utils.replay.bundle import BundleReader, LoadedBundle
from relax.utils.replay.report import ReplayReport, StageResult, StageStatus
from relax.utils.replay.schema import STAGE_ORDER, StageCapability, StageId
from relax.utils.replay.selection import select_bundle
from relax.utils.replay.stages import capability_for, get_adapter


logger = get_logger(__name__)

# Stages whose output is a reduction over the entire cohort. They cannot be
# faithfully recomputed for a partial selection (single sample/group/batch), so
# a partial replay reports them as skipped rather than comparing a subset scalar
# against a full-cohort expected value.
_COHORT_LEVEL_STAGES = frozenset({StageId.LOSS_POLICY, StageId.LOSS_VALUE})


def replay_bundle(bundle: LoadedBundle, *, partial_selection: bool = False) -> ReplayReport:
    """Replay every declared stage and return the divergence report."""
    register_all()
    report = ReplayReport(bundle_id=bundle.index.bundle_id)
    config = bundle.index.config
    context_parallel = bundle.index.identity.rank.get("cp", 1)
    ctx: dict[str, object] = {}

    for stage in STAGE_ORDER:
        if partial_selection and stage in _COHORT_LEVEL_STAGES:
            report.add(
                StageResult(
                    stage=stage.value,
                    status=StageStatus.SKIPPED,
                    message="cohort-level stage is not replayable under a partial selection",
                )
            )
            continue

        contract = bundle.manifest.stage_contracts.get(stage)
        if contract is None:
            report.add(StageResult(stage=stage.value, status=StageStatus.SKIPPED, message="no stage contract"))
            continue

        capability = capability_for(
            stage, advantage_estimator=config.advantage_estimator, context_parallel=context_parallel
        )
        if capability != StageCapability.RECOMPUTE:
            report.add(
                StageResult(stage=stage.value, status=StageStatus.SKIPPED, message=f"capability={capability.value}")
            )
            continue
        if contract.capability != StageCapability.RECOMPUTE:
            report.add(
                StageResult(
                    stage=stage.value,
                    status=StageStatus.SKIPPED,
                    message=f"manifest capability={contract.capability.value}",
                )
            )
            continue

        adapter = get_adapter(stage)
        report.add(adapter(bundle, ctx))

    return report


def replay(
    path: str | Path,
    *,
    sample_ids: list[str] | None = None,
    group_ids: list[str] | None = None,
    batch_ids: list[str] | None = None,
) -> ReplayReport:
    """Load a bundle from path and replay it (optionally a selection)."""
    bundle = BundleReader(path).load()

    has_selection = any((sample_ids, group_ids, batch_ids))
    partial_selection = False
    if has_selection:
        full_count = len(bundle.index.samples)
        bundle = select_bundle(bundle, sample_ids=sample_ids, group_ids=group_ids, batch_ids=batch_ids)
        partial_selection = len(bundle.index.samples) != full_count

    logger.info("replaying bundle %s (%d samples)", bundle.index.bundle_id, len(bundle.index.samples))
    return replay_bundle(bundle, partial_selection=partial_selection)
