# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in JSONL diagnostics for the pure-Python GraphGPO graph kernel."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from statistics import median
from typing import Callable, Hashable, Mapping, Protocol, Sequence


DIAGNOSTICS_PATH_ENV = "GRAPHGPO_DIAGNOSTICS_JSONL"
DIAGNOSTICS_SCHEMA = "graphgpo-diagnostics-v1"
DIAGNOSTICS_SUMMARY_SCHEMA = "graphgpo-diagnostics-summary-v1"
TurnKey = tuple[Hashable, int]


class EdgeOccurrenceLike(Protocol):
    """Structural subset used to keep this module independent of
    graph_credit."""

    turn_key: TurnKey
    source: Hashable
    action: str
    target: Hashable


@dataclass(frozen=True)
class GraphDiagnostics:
    """One task-local graph-credit diagnostic record."""

    schema: str
    task_id: str
    rollout_group_id: str | None
    policy_version: str | None
    graph_build_ns: int
    reverse_dijkstra_ns: int
    graph_advantage_ns: int
    graph_credit_total_ns: int
    node_count: int
    edge_occurrence_count: int
    duplicate_edge_count: int
    duplicate_edge_rate: float
    shared_state_count: int
    shared_state_rate: float
    singleton_source_count: int
    singleton_source_rate: float
    nonzero_graph_advantage_count: int
    nonzero_graph_advantage_rate: float
    all_fail: bool
    unreachable_node_count: int
    unreachable_node_rate: float
    max_finite_distance: float
    distance_histogram: Mapping[str, int]

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


DiagnosticsCallback = Callable[[GraphDiagnostics], None]


@dataclass(frozen=True)
class GraphDiagnosticsSummary:
    """Aggregate median/p95 evidence derived from raw diagnostic records."""

    schema: str
    record_count: int
    timing_ns: Mapping[str, Mapping[str, float | int]]
    graph_metrics: Mapping[str, Mapping[str, float]]
    all_fail_count: int
    all_fail_rate: float
    distance_histogram: Mapping[str, int]

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def _stable_identifier(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _require_duration(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _distance_label(value: float) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("distances must be finite real numbers")
    finite_value = float(value)
    if not math.isfinite(finite_value) or finite_value < 0.0:
        raise ValueError("distances must be finite non-negative real numbers")
    return format(finite_value, ".17g")


def build_graph_diagnostics(
    *,
    task_id: object,
    nodes: Sequence[Hashable] | frozenset[Hashable],
    occurrences: Sequence[EdgeOccurrenceLike],
    distances: Mapping[Hashable, float],
    max_finite_distance: float,
    has_success: bool,
    advantages: Mapping[TurnKey, float],
    graph_build_ns: int,
    reverse_dijkstra_ns: int,
    graph_advantage_ns: int,
    rollout_group_id: object | None = None,
    policy_version: object | None = None,
) -> GraphDiagnostics:
    """Derive deterministic graph metrics from one completed task-local
    graph."""

    graph_build_ns = _require_duration("graph_build_ns", graph_build_ns)
    reverse_dijkstra_ns = _require_duration("reverse_dijkstra_ns", reverse_dijkstra_ns)
    graph_advantage_ns = _require_duration("graph_advantage_ns", graph_advantage_ns)
    if not isinstance(has_success, bool):
        raise ValueError("has_success must be a boolean")

    node_set = frozenset(nodes)
    if not node_set or set(distances) != set(node_set):
        raise ValueError("distances must contain exactly one value per graph node")

    occurrence_keys = {occurrence.turn_key for occurrence in occurrences}
    if len(occurrence_keys) != len(occurrences):
        raise ValueError("edge occurrence turn keys must be unique")
    if set(advantages) != occurrence_keys:
        raise ValueError("advantages must contain exactly one value per edge occurrence")

    edge_counts = Counter((occurrence.source, occurrence.action, occurrence.target) for occurrence in occurrences)
    duplicate_edge_count = sum(count - 1 for count in edge_counts.values())

    trajectories_by_source: dict[Hashable, set[Hashable]] = defaultdict(set)
    occurrences_by_source: Counter[Hashable] = Counter()
    for occurrence in occurrences:
        trajectories_by_source[occurrence.source].add(occurrence.turn_key[0])
        occurrences_by_source[occurrence.source] += 1
    shared_state_count = sum(len(trajectory_ids) > 1 for trajectory_ids in trajectories_by_source.values())
    singleton_source_count = sum(count == 1 for count in occurrences_by_source.values())

    finite_advantages: list[float] = []
    for value in advantages.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("graph advantages must be finite real numbers")
        finite_value = float(value)
        if not math.isfinite(finite_value):
            raise ValueError("graph advantages must be finite real numbers")
        finite_advantages.append(finite_value)
    nonzero_count = sum(value != 0.0 for value in finite_advantages)

    _distance_label(max_finite_distance)
    distance_values = {node: float(value) for node, value in distances.items()}
    if has_success:
        unreachable_count = sum(value > float(max_finite_distance) for value in distance_values.values())
    else:
        unreachable_count = len(node_set)
    histogram = Counter(_distance_label(value) for value in distance_values.values())

    edge_count = len(occurrences)
    source_count = len(occurrences_by_source)
    node_count = len(node_set)
    return GraphDiagnostics(
        schema=DIAGNOSTICS_SCHEMA,
        task_id=_stable_identifier(task_id),
        rollout_group_id=(None if rollout_group_id is None else _stable_identifier(rollout_group_id)),
        policy_version=(None if policy_version is None else _stable_identifier(policy_version)),
        graph_build_ns=graph_build_ns,
        reverse_dijkstra_ns=reverse_dijkstra_ns,
        graph_advantage_ns=graph_advantage_ns,
        graph_credit_total_ns=(graph_build_ns + reverse_dijkstra_ns + graph_advantage_ns),
        node_count=node_count,
        edge_occurrence_count=edge_count,
        duplicate_edge_count=duplicate_edge_count,
        duplicate_edge_rate=_rate(duplicate_edge_count, edge_count),
        shared_state_count=shared_state_count,
        shared_state_rate=_rate(shared_state_count, source_count),
        singleton_source_count=singleton_source_count,
        singleton_source_rate=_rate(singleton_source_count, source_count),
        nonzero_graph_advantage_count=nonzero_count,
        nonzero_graph_advantage_rate=_rate(nonzero_count, edge_count),
        all_fail=not has_success,
        unreachable_node_count=unreachable_count,
        unreachable_node_rate=_rate(unreachable_count, node_count),
        max_finite_distance=float(max_finite_distance),
        distance_histogram=dict(sorted(histogram.items())),
    )


class JsonlDiagnosticsWriter:
    """Append one compact JSON object per graph when explicitly configured."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def __call__(self, record: GraphDiagnostics) -> None:
        if not isinstance(record, GraphDiagnostics):
            raise TypeError("record must be a GraphDiagnostics instance")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            record.to_mapping(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")


def diagnostics_callback_from_environment() -> DiagnosticsCallback | None:
    """Return a writer only when ``GRAPHGPO_DIAGNOSTICS_JSONL`` is non-
    empty."""

    path = os.environ.get(DIAGNOSTICS_PATH_ENV, "").strip()
    if not path:
        return None
    return JsonlDiagnosticsWriter(path)


def _nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(values)
    return ordered[ceil(0.95 * len(ordered)) - 1]


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("at least one value is required")
    return {
        "median": float(median(values)),
        "p95": float(_nearest_rank_p95(values)),
    }


def summarize_graph_diagnostics(
    records: Sequence[GraphDiagnostics],
) -> GraphDiagnosticsSummary:
    """Summarize raw task-local records with median and nearest-rank p95."""

    if not records:
        raise ValueError("at least one graph diagnostic record is required")
    if any(not isinstance(record, GraphDiagnostics) for record in records):
        raise TypeError("all records must be GraphDiagnostics instances")

    timing_fields = (
        "graph_build_ns",
        "reverse_dijkstra_ns",
        "graph_advantage_ns",
        "graph_credit_total_ns",
    )
    timing_ns: dict[str, dict[str, float | int]] = {}
    for field in timing_fields:
        values = [getattr(record, field) for record in records]
        timing_ns[field] = {
            **_distribution(values),
            "total": sum(values),
        }

    metric_fields = (
        "node_count",
        "edge_occurrence_count",
        "duplicate_edge_rate",
        "shared_state_rate",
        "singleton_source_rate",
        "nonzero_graph_advantage_rate",
        "unreachable_node_rate",
    )
    graph_metrics = {
        field: _distribution([float(getattr(record, field)) for record in records]) for field in metric_fields
    }

    all_fail_count = sum(record.all_fail for record in records)
    distance_histogram: Counter[str] = Counter()
    for record in records:
        distance_histogram.update(record.distance_histogram)
    return GraphDiagnosticsSummary(
        schema=DIAGNOSTICS_SUMMARY_SCHEMA,
        record_count=len(records),
        timing_ns=timing_ns,
        graph_metrics=graph_metrics,
        all_fail_count=all_fail_count,
        all_fail_rate=_rate(all_fail_count, len(records)),
        distance_histogram=dict(sorted(distance_histogram.items(), key=lambda item: float(item[0]))),
    )


def load_graph_diagnostics_jsonl(
    path: str | os.PathLike[str],
) -> tuple[GraphDiagnostics, ...]:
    """Load and schema-check raw diagnostic JSONL records."""

    records: list[GraphDiagnostics] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid diagnostics JSON on line {line_number}") from exc
            if not isinstance(payload, dict) or payload.get("schema") != DIAGNOSTICS_SCHEMA:
                raise ValueError(f"unexpected diagnostics schema on line {line_number}")
            try:
                records.append(GraphDiagnostics(**payload))
            except TypeError as exc:
                raise ValueError(f"invalid diagnostics record on line {line_number}") from exc
    if not records:
        raise ValueError("diagnostics JSONL contains no records")
    return tuple(records)


def write_graph_diagnostics_summary(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> GraphDiagnosticsSummary:
    """Write an immutable JSON summary next to retained raw JSONL evidence."""

    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("diagnostics summary output must differ from its input")
    summary = summarize_graph_diagnostics(load_graph_diagnostics_jsonl(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        summary.to_mapping(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
    return summary
