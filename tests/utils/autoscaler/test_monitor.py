# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the autoscaler monitor TUI data layer.

The Textual widgets/rendering need a real terminal and are verified visually on
a cluster; here we cover the pure data-accumulation path that feeds the charts
(``MetricsHistory``), including the engine-count history added for the "engine
count over time" chart.
"""

from relax.utils.autoscaler.monitor import MetricsHistory, _render_chart


def test_metrics_history_tracks_engine_count():
    """MetricsHistory must record the current engine count per point (so the
    TUI can chart how the engine count fluctuates over time), and clear() must
    drop it alongside the other series."""
    h = MetricsHistory(max_points=3)
    h.add_point(throughput=100.0, token_usage=0.1, running_reqs=5, queue_reqs=0, timestamp="t0", engines=1)
    h.add_point(throughput=120.0, token_usage=0.2, running_reqs=8, queue_reqs=2, timestamp="t1", engines=2)
    assert list(h.engines) == [1.0, 2.0]
    # honors max_points ring buffer like the other series
    h.add_point(throughput=90.0, token_usage=0.05, running_reqs=3, queue_reqs=0, timestamp="t2", engines=3)
    h.add_point(throughput=80.0, token_usage=0.04, running_reqs=2, queue_reqs=0, timestamp="t3", engines=1)
    assert list(h.engines) == [2.0, 3.0, 1.0]
    h.clear()
    assert list(h.engines) == []


def test_render_chart_renders_engine_series():
    """The chart renderer produces a titled chart for a small integer engine
    series without error."""
    out = _render_chart(
        [1.0, 2.0, 1.0, 3.0, 2.0],
        width=20,
        height=5,
        color="magenta",
        title="Engines",
        unit="",
        fmt=".0f",
        y_min=0.0,
    )
    assert isinstance(out, str)
    assert "Engines" in out


def _yaxis_labels(chart: str) -> list[str]:
    """Extract the y-axis tick labels (text after the trailing ``¦`` on chart
    body rows) from a rendered chart."""
    labels: list[str] = []
    for line in chart.splitlines():
        if not line.startswith("¦"):
            continue
        parts = line.rsplit("¦", 1)
        if len(parts) == 2 and parts[1].strip():
            labels.append(parts[1].strip())
    return labels


def test_render_chart_integer_yaxis_has_no_fractional_ticks():
    """Engine count is an integer quantity, so with integer_yaxis=True the
    y-axis tick labels must be whole numbers (no 0.2 / 0.8 fractional ticks
    even when the value range is small, e.g. a constant 1 engine with
    y_min=0)."""
    out = _render_chart(
        [1.0, 1.0, 1.0],
        width=20,
        height=6,
        color="magenta",
        title="Engines",
        unit="",
        fmt=".0f",
        y_min=0.0,
        integer_yaxis=True,
    )
    labels = _yaxis_labels(out)
    assert labels, "expected some y-axis tick labels"
    for lbl in labels:
        assert "." not in lbl, f"integer y-axis should not emit a fractional tick: {lbl!r}"
