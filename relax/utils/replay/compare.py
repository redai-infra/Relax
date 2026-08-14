# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Numerical comparison helpers for replay adapters."""

from __future__ import annotations

import torch

from relax.utils.replay.report import FieldDivergence
from relax.utils.replay.schema import Manifest, StageId


def tolerance_for(
    manifest: Manifest, stage: StageId, *, default_atol: float = 1e-6, default_rtol: float = 1e-5
) -> tuple[float, float]:
    """Return ``(atol, rtol)`` for ``stage`` from the manifest, with
    defaults."""
    entry = manifest.comparison_policy.tolerances.get(stage.value, {})
    return entry.get("atol", default_atol), entry.get("rtol", default_rtol)


def compare_scalar(expected: float, actual: float, *, atol: float, rtol: float) -> bool:
    """True when ``actual`` matches ``expected`` within tolerance."""
    return bool(torch.isclose(torch.tensor(actual), torch.tensor(expected), atol=atol, rtol=rtol))


def scalar_error(expected: float, actual: float) -> float:
    return abs(actual - expected)


def compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    field: str,
    atol: float,
    rtol: float,
    sample_ids: list[str] | None = None,
    response_lengths: list[int] | None = None,
) -> tuple[list[FieldDivergence], float, int, int]:
    """Elementwise-compare two 1-D tensors and localize divergences.

    Returns ``(divergences, max_abs_error, mismatch_count, non_finite_count)``.
    Flat token index is mapped back to ``(sample_id, token_offset)`` when
    ``sample_ids``/``response_lengths`` are provided.
    """
    expected = expected.float().reshape(-1)
    actual = actual.float().reshape(-1)
    if expected.numel() != actual.numel():
        raise ValueError(f"shape mismatch for {field!r}: {tuple(expected.shape)} vs {tuple(actual.shape)}")

    diff = (actual - expected).abs()
    threshold = atol + rtol * expected.abs()
    mismatched = (diff > threshold).nonzero(as_tuple=False).flatten().tolist()

    non_finite = int((~torch.isfinite(actual)).sum().item())
    max_abs_error = float(diff.max().item()) if diff.numel() else 0.0

    divergences: list[FieldDivergence] = []
    for token_index in mismatched[:10]:  # cap the report at 10 concrete examples
        sample_id, token_offset = _locate(token_index, sample_ids, response_lengths)
        divergences.append(
            FieldDivergence(
                field=field,
                sample_id=sample_id,
                token_offset=token_offset,
                expected=float(expected[token_index].item()),
                actual=float(actual[token_index].item()),
                abs_error=float(diff[token_index].item()),
            )
        )
    return divergences, max_abs_error, len(mismatched), non_finite


def _locate(
    token_index: int,
    sample_ids: list[str] | None,
    response_lengths: list[int] | None,
) -> tuple[str | None, int | None]:
    if not sample_ids or not response_lengths:
        return None, None
    offset = token_index
    for sample_id, length in zip(sample_ids, response_lengths, strict=False):
        if offset < length:
            return sample_id, offset
        offset -= length
    return None, None
