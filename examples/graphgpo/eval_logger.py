# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Episode-level evaluation metrics for variable-row agentic rollouts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Hashable


def _sample_metadata(sample: object, *, context: str) -> Mapping[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{context} metadata must be a mapping")
    return metadata


def _required(metadata: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in metadata:
        raise ValueError(f"{context} metadata is missing {key!r}")
    return metadata[key]


def _trajectory_id(metadata: Mapping[str, Any], *, context: str) -> Hashable:
    value = _required(metadata, "trajectory_id", context=context)
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"{context} trajectory_id must be hashable") from exc
    return value


def _bool(metadata: Mapping[str, Any], key: str, *, context: str) -> bool:
    value = _required(metadata, key, context=context)
    if not isinstance(value, bool):
        raise ValueError(f"{context} {key} must be a boolean")
    return value


def _finite_return(metadata: Mapping[str, Any], *, context: str) -> float:
    value = _required(metadata, "episode_return", context=context)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} episode_return must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{context} episode_return must be a finite number")
    return value


def episode_metrics(samples: Sequence[object]) -> dict[str, float | int]:
    """Compute one vote per trajectory, regardless of exported turn count."""

    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    if not samples:
        raise ValueError("episode-level eval metrics require at least one sample")

    by_trajectory: dict[Hashable, list[Mapping[str, Any]]] = {}
    for index, sample in enumerate(samples):
        context = f"sample {index}"
        metadata = _sample_metadata(sample, context=context)
        by_trajectory.setdefault(_trajectory_id(metadata, context=context), []).append(metadata)

    successes: list[bool] = []
    episode_returns: list[float] = []
    truncated_episodes: list[bool] = []
    for trajectory_id, rows in by_trajectory.items():
        context = f"trajectory {trajectory_id!r}"
        success_values = {_bool(row, "success", context=context) for row in rows}
        return_values = {_finite_return(row, context=context) for row in rows}
        if len(success_values) != 1:
            raise ValueError(f"{context} has inconsistent success values")
        if len(return_values) != 1:
            raise ValueError(f"{context} has inconsistent episode_return values")
        successes.append(next(iter(success_values)))
        episode_returns.append(next(iter(return_values)))
        truncated_episodes.append(any(_bool(row, "truncated", context=context) for row in rows))

    episode_count = len(by_trajectory)
    return {
        "episode_count": episode_count,
        "success_rate": sum(successes) / episode_count,
        "episode_return_mean": sum(episode_returns) / episode_count,
        "truncated_rate": sum(truncated_episodes) / episode_count,
    }


def build_eval_metrics(
    data: Mapping[str, Mapping[str, Any]],
    extra_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build namespaced episode metrics for every configured eval dataset."""

    if not isinstance(data, Mapping) or not data:
        raise ValueError("eval data must be a non-empty mapping")
    metrics: dict[str, Any] = dict(extra_metrics or {})
    for dataset_name, dataset in data.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("eval dataset names must be non-empty strings")
        if not isinstance(dataset, Mapping):
            raise ValueError(f"eval dataset {dataset_name!r} must be a mapping")
        samples = dataset.get("samples")
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise ValueError(f"eval dataset {dataset_name!r} must include a samples sequence")
        for metric_name, value in episode_metrics(samples).items():
            metrics[f"eval/{dataset_name}/{metric_name}"] = value
    return metrics


def _emit_metrics(rollout_id: int, args: Any, metrics: dict[str, Any]) -> None:
    from relax.utils import tracking_utils
    from relax.utils.logging_utils import get_logger
    from relax.utils.metrics.metric_utils import compute_rollout_step

    step = compute_rollout_step(args, rollout_id)
    metrics["eval/step"] = step
    get_logger(__name__).info("eval %s: %s", rollout_id, metrics)
    tracking_utils.log(args, metrics, step_key="eval/step")
    tracking_utils.flush_metrics(args, step)


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: Mapping[str, Mapping[str, Any]],
    extra_metrics: Mapping[str, Any] | None = None,
) -> bool:
    """Relax custom eval hook that replaces row-weighted default logging."""

    metrics = build_eval_metrics(data, extra_metrics)
    _emit_metrics(rollout_id, args, metrics)
    return True
