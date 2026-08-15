# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
from dataclasses import replace

from examples.graphgpo.diagnostics import (
    DIAGNOSTICS_PATH_ENV,
    DIAGNOSTICS_SCHEMA,
    JsonlDiagnosticsWriter,
    build_graph_diagnostics,
    diagnostics_callback_from_environment,
    summarize_graph_diagnostics,
    write_graph_diagnostics_summary,
)
from examples.graphgpo.graph_credit import (
    SUCCESS,
    DistanceResult,
    EdgeOccurrence,
    Turn,
    compute_method_advantages,
    finalize_trajectory,
)


def test_graph_diagnostics_fixed_metric_oracle():
    occurrences = (
        EdgeOccurrence(("tau0", 0), "S", "to-b", "B", 1.0),
        EdgeOccurrence(("tau1", 0), "S", "to-b", "B", 1.0),
        EdgeOccurrence(("tau2", 0), "S", "to-u", "U", 1.0),
        EdgeOccurrence(("tau0", 1), "B", "finish", SUCCESS, 1.0),
    )
    distances = DistanceResult(
        distances={SUCCESS: 0.0, "B": 1.0, "S": 2.0, "U": 3.0},
        max_finite_distance=2.0,
        has_success=True,
    )

    record = build_graph_diagnostics(
        task_id="task",
        nodes=frozenset((SUCCESS, "B", "S", "U")),
        occurrences=occurrences,
        distances=distances.distances,
        max_finite_distance=distances.max_finite_distance,
        has_success=distances.has_success,
        advantages={
            ("tau0", 0): 1.0,
            ("tau1", 0): -1.0,
            ("tau2", 0): 0.0,
            ("tau0", 1): 0.0,
        },
        graph_build_ns=11,
        reverse_dijkstra_ns=13,
        graph_advantage_ns=17,
        rollout_group_id="group-7",
        policy_version=42,
    )

    assert record.schema == DIAGNOSTICS_SCHEMA
    assert record.rollout_group_id == "builtins.str:'group-7'"
    assert record.policy_version == "builtins.int:42"
    assert record.graph_credit_total_ns == 41
    assert record.node_count == 4
    assert record.edge_occurrence_count == 4
    assert record.duplicate_edge_count == 1
    assert record.duplicate_edge_rate == 0.25
    assert record.shared_state_count == 1
    assert record.shared_state_rate == 0.5
    assert record.singleton_source_count == 1
    assert record.singleton_source_rate == 0.5
    assert record.nonzero_graph_advantage_count == 2
    assert record.nonzero_graph_advantage_rate == 0.5
    assert not record.all_fail
    assert record.unreachable_node_count == 1
    assert record.unreachable_node_rate == 0.25
    assert record.distance_histogram == {"0": 1, "1": 1, "2": 1, "3": 1}


def test_diagnostics_are_default_off(monkeypatch):
    monkeypatch.delenv(DIAGNOSTICS_PATH_ENV, raising=False)

    assert diagnostics_callback_from_environment() is None


def test_jsonl_writer_is_enabled_only_by_explicit_path(monkeypatch, tmp_path):
    output_path = tmp_path / "nested" / "diagnostics.jsonl"
    monkeypatch.setenv(DIAGNOSTICS_PATH_ENV, str(output_path))
    writer = diagnostics_callback_from_environment()
    assert isinstance(writer, JsonlDiagnosticsWriter)

    turn = Turn(
        "task",
        "tau",
        0,
        "S",
        "finish",
        "done",
        True,
        True,
        False,
        True,
    )
    turns = finalize_trajectory((turn,), success=True)
    compute_method_advantages(
        "graphgpo",
        turns,
        expected_group_size=1,
        graph_diagnostics_callback=writer,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == DIAGNOSTICS_SCHEMA
    assert payload["node_count"] == 2
    assert payload["edge_occurrence_count"] == 1
    assert payload["all_fail"] is False
    assert payload["graph_build_ns"] >= 0
    assert payload["reverse_dijkstra_ns"] >= 0
    assert payload["graph_advantage_ns"] >= 0


def test_diagnostics_summary_reports_median_and_nearest_rank_p95(tmp_path):
    occurrence = EdgeOccurrence(("tau", 0), "S", "finish", SUCCESS, 1.0)
    base = build_graph_diagnostics(
        task_id="task",
        nodes=frozenset(("S", SUCCESS)),
        occurrences=(occurrence,),
        distances={"S": 1.0, SUCCESS: 0.0},
        max_finite_distance=1.0,
        has_success=True,
        advantages={("tau", 0): 0.0},
        graph_build_ns=10,
        reverse_dijkstra_ns=20,
        graph_advantage_ns=30,
    )
    records = (
        base,
        replace(
            base,
            graph_build_ns=20,
            reverse_dijkstra_ns=30,
            graph_advantage_ns=40,
            graph_credit_total_ns=90,
        ),
        replace(
            base,
            graph_build_ns=100,
            reverse_dijkstra_ns=200,
            graph_advantage_ns=300,
            graph_credit_total_ns=600,
            all_fail=True,
        ),
    )

    summary = summarize_graph_diagnostics(records)

    assert summary.record_count == 3
    assert summary.timing_ns["graph_build_ns"] == {
        "median": 20.0,
        "p95": 100.0,
        "total": 130,
    }
    assert summary.all_fail_count == 1
    assert summary.all_fail_rate == 1 / 3
    assert summary.distance_histogram == {"0": 3, "1": 3}

    raw_path = tmp_path / "raw.jsonl"
    writer = JsonlDiagnosticsWriter(raw_path)
    for record in records:
        writer(record)
    output_path = tmp_path / "summary.json"
    written = write_graph_diagnostics_summary(raw_path, output_path)

    assert written == summary
    assert json.loads(output_path.read_text(encoding="utf-8"))["timing_ns"]["graph_build_ns"]["p95"] == 100.0
