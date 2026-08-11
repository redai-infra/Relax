# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Autoscaler monitoring TUI for real-time metrics and scaling actions.

This module provides a Textual-based terminal dashboard for monitoring the
Relax autoscaler service. It polls the autoscaler HTTP endpoints and displays
engine metrics, scaling conditions, and operation history in real-time.

Usage:
    python -m relax.utils.autoscaler.monitor
    python -m relax.utils.autoscaler.monitor --url http://localhost:8000/autoscaler

Keyboard Controls:
    q - Quit the monitor
    r - Force refresh (bypass 5s interval)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Union


try:
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from aiohttp import ClientSession


JsonDict = dict[str, object]

MAX_HISTORY_POINTS = 60


def _as_dict(v: object) -> JsonDict:
    return v if isinstance(v, dict) else {}


def _as_list(v: object) -> list[object]:
    return v if isinstance(v, list) else []


def _as_str(v: object, default: str = "-") -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return default
    return str(v)


def _as_int(v: object, default: int = 0) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        try:
            return int(float(v))
        except Exception:
            return default
    return default


def _as_float(v: object, default: float = 0.0) -> float:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return default
    return default


def _fmt_ts(ts: Union[float, None]) -> str:
    if ts is None or ts == 0:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _fmt_ts_short(ts: Union[float, None]) -> str:
    if ts is None or ts == 0:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        return "-"


def _coalesce(*values: object, default: object = "-") -> object:
    for v in values:
        if v is not None:
            return v
    return default


def _norm_base_url(url: str) -> str:
    url = url.strip()
    while url.endswith("/"):
        url = url[:-1]
    return url


def _fmt_delta_int(current: int, previous: Union[int, None]) -> str:
    if previous is None:
        return "-"
    delta = current - previous
    if delta > 0:
        return f"[green]+{delta}[/green]"
    elif delta < 0:
        return f"[red]{delta}[/red]"
    return "[dim]0[/dim]"


def _fmt_delta_float(current: float, previous: Union[float, None], precision: int = 1) -> str:
    if previous is None or previous == 0:
        return "-"
    delta = current - previous
    if abs(delta) < 1e-9:
        return "[dim]0[/dim]"
    if delta > 0:
        return f"[green]+{delta:.{precision}f}[/green]"
    return f"[red]{delta:.{precision}f}[/red]"


def _fmt_delta_percent(current: float, previous: Union[float, None], precision: int = 1) -> str:
    if previous is None or previous == 0:
        return "-"
    if current == 0 and previous == 0:
        return "[dim]0%[/dim]"
    pct_change = ((current - previous) / previous) * 100 if previous != 0 else 0
    if abs(pct_change) < 0.1:
        return "[dim]~0%[/dim]"
    if pct_change > 0:
        return f"[green]+{pct_change:.{precision}f}%[/green]"
    return f"[red]{pct_change:.{precision}f}%[/red]"


@dataclass
class AutoscalerSnapshot:
    status: Union[JsonDict, None] = None
    conditions: Union[JsonDict, None] = None
    history: Union[JsonDict, None] = None
    last_refresh_ts: Union[float, None] = None
    last_error: Union[str, None] = None

    def copy(self) -> "AutoscalerSnapshot":
        import copy

        return AutoscalerSnapshot(
            status=copy.deepcopy(self.status),
            conditions=copy.deepcopy(self.conditions),
            history=copy.deepcopy(self.history),
            last_refresh_ts=self.last_refresh_ts,
            last_error=self.last_error,
        )


@dataclass
class MetricRow:
    label: str
    current: str
    delta: str = "-"
    category: str = ""


class MetricsHistory:
    def __init__(self, max_points: int = MAX_HISTORY_POINTS) -> None:
        self.max_points = max_points
        self.throughput: deque[float] = deque(maxlen=max_points)
        self.token_usage: deque[float] = deque(maxlen=max_points)
        self.running_reqs: deque[float] = deque(maxlen=max_points)
        self.queue_reqs: deque[float] = deque(maxlen=max_points)
        self.engines: deque[float] = deque(maxlen=max_points)
        self.timestamps: deque[str] = deque(maxlen=max_points)

    def add_point(
        self,
        throughput: float,
        token_usage: float,
        running_reqs: int,
        queue_reqs: int,
        timestamp: str,
        engines: int = 0,
    ) -> None:
        self.throughput.append(throughput)
        self.token_usage.append(token_usage)
        self.running_reqs.append(float(running_reqs))
        self.queue_reqs.append(float(queue_reqs))
        self.engines.append(float(engines))
        self.timestamps.append(timestamp)

    def clear(self) -> None:
        self.throughput.clear()
        self.token_usage.clear()
        self.running_reqs.clear()
        self.queue_reqs.clear()
        self.engines.clear()
        self.timestamps.clear()


def _render_chart(
    values: list[float],
    width: int,
    height: int,
    color: str,
    title: str,
    unit: str,
    fmt: str = ".1f",
    y_min: float | None = None,
    y_max: float | None = None,
    integer_yaxis: bool = False,
) -> str:
    """使用 Half-block 半方块绘制连续折线图，包含坐标轴与边框。"""
    if not values:
        return f"[bold]{title}[/bold]\n[dim]No data[/dim]"

    real_values = [v for v in values if v is not None]
    if not real_values:
        return f"[bold]{title}[/bold]\n[dim]No data[/dim]"

    display_min = y_min if y_min is not None else min(real_values)
    display_max = y_max if y_max is not None else max(real_values)

    if display_max == display_min:
        display_max += 1.0
        display_min -= 1.0

    # 重采样以适应网格宽度
    if len(values) > width:
        step = len(values) / width
        display = []
        for i in range(width):
            idx = int(i * step)
            end_idx = int((i + 1) * step)
            chunk = values[idx:end_idx] if end_idx > idx else [values[idx]]
            display.append(sum(chunk) / len(chunk) if chunk else 0.0)
    elif len(values) < width:
        display = [None] * (width - len(values)) + list(values)
    else:
        display = list(values)

    vmin = display_min
    vmax = display_max
    vrange = vmax - vmin if vmax > vmin else 1.0

    # 值映射到虚拟行 (0 到 2*height - 1)
    def to_vr(v: float) -> float:
        return (v - vmin) / vrange * (2 * height - 1)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    # 1. 绘制 y=0 水平基准线
    y0_r = -1
    if vmin <= 0 <= vmax:
        y0_r = int(round(to_vr(0))) // 2
        if 0 <= y0_r < height:
            for c in range(width):
                grid[y0_r][c] = "-"

    # 2. 绘制数据曲线 (使用 Half-blocks)
    for c in range(width):
        if display[c] is None:
            continue
        v_curr = display[c]
        vr_curr = to_vr(v_curr)

        # 计算该列的左右连接点高度，形成平滑走势
        v_left = v_curr
        if c > 0 and display[c - 1] is not None:
            v_left = display[c - 1]
        vr_left = to_vr((v_curr + v_left) / 2)

        v_right = v_curr
        if c < width - 1 and display[c + 1] is not None:
            v_right = display[c + 1]
        vr_right = to_vr((v_curr + v_right) / 2)

        min_vri = int(round(min(vr_curr, vr_left, vr_right)))
        max_vri = int(round(max(vr_curr, vr_left, vr_right)))

        # 填充当前列跨越的虚行
        for r in range(height):
            bottom_active = min_vri <= 2 * r <= max_vri
            top_active = min_vri <= 2 * r + 1 <= max_vri

            if bottom_active and top_active:
                grid[r][c] = "█"
            elif top_active:
                grid[r][c] = "▀"
            elif bottom_active:
                grid[r][c] = "▄"

    # 生成优雅的 y 轴刻度
    def nice_ticks(ymin, ymax):
        rng = ymax - ymin
        if rng == 0:
            return [ymin]
        space = 10 ** math.floor(math.log10(rng))
        if rng / space < 2:
            space /= 5
        elif rng / space < 5:
            space /= 2
        if integer_yaxis:
            # Integer quantities (e.g. engine count) must not get sub-integer
            # ticks like 0.2 / 0.8 -- force a whole-number step of at least 1.
            space = max(1.0, float(round(space)))
        start = math.ceil(ymin / space) * space
        t, v = [], start
        while v <= ymax + 1e-9:
            t.append(v)
            v += space
        return t

    ticks = nice_ticks(vmin, vmax)
    row_labels = {}
    for t in ticks:
        r = int(round(to_vr(t))) // 2
        if 0 <= r < height:
            if abs(t) < 1e-5:
                t = 0.0
            row_labels[r] = f"{int(t) if abs(t - round(t)) < 1e-5 else round(t, 1)}"

    lines = []

    # 居中 Title
    pad = max(0, (width - len(title)) // 2)
    lines.append(" " * pad + f"[bold]{title}[/bold]")

    # 顶部边框
    lines.append("┌" + "-" * width + "┐")

    # 图表主体与右侧 Y 轴刻度
    for r in range(height - 1, -1, -1):
        row_str = "".join(grid[r])
        colored_row = f"[{color}]{row_str}[/{color}]"
        label = row_labels.get(r, "")
        if label:
            lines.append(f"¦{colored_row}¦ {label}")
        else:
            lines.append(f"¦{colored_row}¦")

    # 底部边框
    lines.append("└" + "-" * width + "┘")

    # 底部 X 轴刻度
    num_vals = len(values)
    if num_vals > 1:
        xt = nice_ticks(0, num_vals - 1)
        x_line = [" "] * width
        for t in xt:
            idx = int(t / (num_vals - 1) * (width - 1))
            if idx >= width:
                idx = width - 1
            lbl = str(int(t))
            start = idx - len(lbl) // 2
            for j, ch in enumerate(lbl):
                if 0 <= start + j < width:
                    x_line[start + j] = ch
        lines.append(" " + "".join(x_line))

    return "\n".join(lines)


class AutoscalerApiClient:
    base_url: str

    def __init__(self, base_url: str) -> None:
        self.base_url = _norm_base_url(base_url)
        self._session: Union["ClientSession", None] = None

    async def start(self) -> None:
        if self._session is not None:
            return
        try:
            import aiohttp

            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0))
        except ImportError as e:
            logger.error("aiohttp is required for autoscaler monitor")
            raise e

    async def stop(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            finally:
                self._session = None

    async def _get_json(self, path: str) -> JsonDict:
        if self._session is None:
            raise RuntimeError("HTTP session not started")
        url = f"{self.base_url}{path}"
        async with self._session.get(url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text}")
            data = await resp.json()
            return _as_dict(data)

    async def fetch_all(self) -> tuple[JsonDict, JsonDict, JsonDict]:
        status_task = asyncio.create_task(self._get_json("/status"))
        conditions_task = asyncio.create_task(self._get_json("/conditions"))
        history_task = asyncio.create_task(self._get_json("/scale_history?limit=10"))
        return await asyncio.gather(status_task, conditions_task, history_task)


def _get_prev_val(prev_metrics: JsonDict, key: str, default: float = 0.0) -> Union[float, None]:
    if not prev_metrics:
        return None
    val = prev_metrics.get(key)
    return _as_float(val, default) if val is not None else None


def _get_prev_int(prev_metrics: JsonDict, key: str, default: int = 0) -> Union[int, None]:
    if not prev_metrics:
        return None
    val = prev_metrics.get(key)
    return _as_int(val, default) if val is not None else None


def _build_metrics_rows(
    snapshot: AutoscalerSnapshot,
    prev: Union[AutoscalerSnapshot, None],
) -> list[MetricRow]:
    rows: list[MetricRow] = []
    status = _as_dict(snapshot.status)
    metrics = _as_dict(status.get("recent_metrics"))
    prev_status = _as_dict(prev.status) if prev else {}
    prev_metrics = _as_dict(prev_status.get("recent_metrics")) if prev_status else {}
    has_prev = bool(prev_metrics)

    cur_engines = _as_int(_coalesce(status.get("current_engines"), metrics.get("num_engines"), default=0))
    prev_engines = _get_prev_int(prev_metrics, "num_engines") if has_prev else None
    if prev_engines is None and prev_status:
        prev_engines = _get_prev_int(prev_status, "current_engines")

    rows.append(
        MetricRow("Engines", f"[bold]{cur_engines}[/bold]", _fmt_delta_int(cur_engines, prev_engines), "Engine")
    )

    min_e = _as_int(status.get("min_engines"))
    max_e = _as_int(status.get("max_engines"))
    rows.append(MetricRow("  Min/Max", f"{min_e} / {max_e}", "-", "Engine"))

    queue_reqs = _as_int(metrics.get("total_queue_reqs"))
    prev_queue = _get_prev_int(prev_metrics, "total_queue_reqs") if has_prev else None
    rows.append(
        MetricRow("Queue Requests", f"[bold]{queue_reqs}[/bold]", _fmt_delta_int(queue_reqs, prev_queue), "Requests")
    )

    running_reqs = _as_int(metrics.get("total_running_reqs"))
    prev_running = _get_prev_int(prev_metrics, "total_running_reqs") if has_prev else None
    rows.append(
        MetricRow(
            "Running Requests", f"[bold]{running_reqs}[/bold]", _fmt_delta_int(running_reqs, prev_running), "Requests"
        )
    )

    token_usage = _as_float(metrics.get("avg_token_usage"))
    prev_token = _get_prev_val(prev_metrics, "avg_token_usage") if has_prev else None
    token_color = "green" if token_usage < 0.7 else ("yellow" if token_usage < 0.9 else "red")
    token_str = f"[{token_color}]{token_usage * 100:.1f}%[/{token_color}]"
    rows.append(MetricRow("Token Usage", token_str, _fmt_delta_percent(token_usage, prev_token), "Resources"))

    used_tokens = _as_int(metrics.get("num_used_tokens", 0))
    max_tokens = _as_int(metrics.get("max_total_num_tokens", 0))
    if max_tokens > 0:
        rows.append(MetricRow("  KV Cache", f"{used_tokens:,} / {max_tokens:,}", "-", "Resources"))

    throughput = _as_float(metrics.get("total_throughput"))
    prev_throughput = _get_prev_val(prev_metrics, "total_throughput") if has_prev else None
    rows.append(
        MetricRow(
            "Throughput",
            f"[bold]{throughput:.1f}[/bold] tok/s",
            _fmt_delta_percent(throughput, prev_throughput),
            "Performance",
        )
    )

    variance = _as_float(metrics.get("throughput_variance"))
    prev_variance = _get_prev_val(prev_metrics, "throughput_variance") if has_prev else None
    rows.append(
        MetricRow("  Variance", f"{variance * 100:.1f}%", _fmt_delta_percent(variance, prev_variance), "Performance")
    )

    q_p95 = _as_float(metrics.get("max_queue_time_p95"))
    prev_q = _get_prev_val(prev_metrics, "max_queue_time_p95") if has_prev else None
    rows.append(MetricRow("Queue Time P95", f"{q_p95:.3f}s", _fmt_delta_float(q_p95, prev_q, 3), "Latency"))

    ttft_p95 = _as_float(metrics.get("max_ttft_p95"))
    prev_ttft = _get_prev_val(prev_metrics, "max_ttft_p95") if has_prev else None
    rows.append(MetricRow("TTFT P95", f"{ttft_p95:.3f}s", _fmt_delta_float(ttft_p95, prev_ttft, 3), "Latency"))

    itl_p95 = _as_float(metrics.get("max_itl_p95"))
    prev_itl = _get_prev_val(prev_metrics, "max_itl_p95") if has_prev else None
    rows.append(MetricRow("ITL P95", f"{itl_p95:.4f}s", _fmt_delta_float(itl_p95, prev_itl, 4), "Latency"))

    e2e_p95 = _as_float(metrics.get("e2e_latency_p95"))
    prev_e2e = _get_prev_val(prev_metrics, "e2e_latency_p95") if has_prev else None
    rows.append(MetricRow("E2E Latency P95", f"{e2e_p95:.3f}s", _fmt_delta_float(e2e_p95, prev_e2e, 3), "Latency"))

    config = _as_dict(status.get("config"))
    if config:
        eval_interval = _as_float(config.get("evaluation_interval_secs"))
        rows.append(MetricRow("Eval Interval", f"{eval_interval:.0f}s", "-", "Config"))

        scale_out_cd = _as_float(config.get("scale_out_cooldown_secs"))
        scale_in_cd = _as_float(config.get("scale_in_cooldown_secs"))
        rows.append(MetricRow("  Cooldowns", f"out:{scale_out_cd:.0f}s / in:{scale_in_cd:.0f}s", "-", "Config"))

        scale_out_policy = _as_dict(config.get("scale_out_policy"))
        if scale_out_policy:
            threshold = _as_float(scale_out_policy.get("token_usage_threshold"))
            queue_depth = _as_int(scale_out_policy.get("queue_depth_per_engine"))
            rows.append(
                MetricRow("  Scale-Out", f"usage>{threshold:.0%} or queue>{queue_depth}/engine", "-", "Config")
            )

        scale_in_policy = _as_dict(config.get("scale_in_policy"))
        if scale_in_policy:
            threshold = _as_float(scale_in_policy.get("token_usage_threshold"))
            rows.append(MetricRow("  Scale-In", f"usage<{threshold:.0%} & queue=0", "-", "Config"))

    total_ops = _as_int(status.get("total_scale_operations"))
    prev_ops = _get_prev_int(prev_status, "total_scale_operations") if prev_status else None
    rows.append(MetricRow("Total Scale Ops", f"{total_ops}", _fmt_delta_int(total_ops, prev_ops), "Stats"))

    return rows


def _render_status_panel(snapshot: AutoscalerSnapshot, _prev: Union[AutoscalerSnapshot, None]) -> str:
    if snapshot.last_error:
        return (
            f"[bold red]Connection error[/bold red]\n[red]{snapshot.last_error}[/red]\n\n[dim]Press r to retry[/dim]"
        )

    status = _as_dict(snapshot.status)
    pending_requests = _as_list(status.get("pending_requests"))

    enabled = bool(status.get("enabled", False))
    running = bool(status.get("running", False))
    enabled_str = "[green]ENABLED[/green]" if enabled else "[dim]DISABLED[/dim]"
    running_str = "[green]RUNNING[/green]" if running else "[red]STOPPED[/red]"

    last_action = _as_str(status.get("last_scale_action"), default="")
    if last_action == "scale_out":
        last_action_str = "[bold green]scale_out[/bold green]"
    elif last_action == "scale_in":
        last_action_str = "[bold blue]scale_in[/bold blue]"
    else:
        last_action_str = "-"

    last_scale_time = _fmt_ts(_as_float(status.get("last_scale_time"), default=0.0))
    pending_count = len(pending_requests)

    refresh_time = _fmt_ts_short(snapshot.last_refresh_ts) if snapshot.last_refresh_ts else "-"

    lines = [
        f"Status: {enabled_str} | {running_str}",
        f"Pending: [bold]{pending_count}[/bold]",
        f"Last scale: {last_action_str}",
        f"  @ [dim]{last_scale_time}[/dim]",
        f"Refresh: [dim]{refresh_time}[/dim]",
    ]
    return "\n".join(lines)


def _render_scale_out_panel(snapshot: AutoscalerSnapshot) -> str:
    conditions_payload = _as_dict(snapshot.conditions)
    conditions = _as_dict(conditions_payload.get("conditions"))

    lines: list[str] = ["[bold yellow]Scale Out[/bold yellow]"]
    if not conditions:
        lines.append("[dim]-[/dim]")
        return "\n".join(lines)

    for name in sorted([_as_str(k) for k in conditions.keys()]):
        item = _as_dict(conditions.get(name))
        ctype = _as_str(item.get("type"), default="unknown")
        if ctype != "scale_out":
            continue
        trig = bool(item.get("triggered", False))
        mark = "✓" if trig else "✗"
        color = "orange3" if trig else "dim"
        lines.append(f"[{color}]{mark}[/{color}] [bold]{name}[/bold]")

    if len(lines) == 1:
        lines.append("[dim]-[/dim]")
    return "\n".join(lines)


def _render_scale_in_panel(snapshot: AutoscalerSnapshot) -> str:
    conditions_payload = _as_dict(snapshot.conditions)
    conditions = _as_dict(conditions_payload.get("conditions"))

    lines: list[str] = ["[bold blue]Scale In[/bold blue]"]
    if not conditions:
        lines.append("[dim]-[/dim]")
        return "\n".join(lines)

    for name in sorted([_as_str(k) for k in conditions.keys()]):
        item = _as_dict(conditions.get(name))
        ctype = _as_str(item.get("type"), default="unknown")
        if ctype != "scale_in":
            continue
        trig = bool(item.get("triggered", False))
        mark = "✓" if trig else "✗"
        color = "cyan" if trig else "dim"
        lines.append(f"[{color}]{mark}[/{color}] [bold]{name}[/bold]")

    if len(lines) == 1:
        lines.append("[dim]-[/dim]")
    return "\n".join(lines)


def _history_rows(snapshot: AutoscalerSnapshot) -> list[tuple[str, str, str, str]]:
    payload = _as_dict(snapshot.history)
    history = _as_list(payload.get("history"))
    rows: list[tuple[str, str, str, str]] = []
    for raw in history[:10]:
        item = _as_dict(raw)
        triggered_at = _fmt_ts(_as_float(item.get("triggered_at"), default=0.0))
        action = _as_str(item.get("action"), default="-")
        from_e = item.get("from_engines")
        to_e = item.get("to_engines")
        if from_e is None or to_e is None:
            from_to = "-"
        else:
            from_to = f"{_as_int(from_e)}→{_as_int(to_e)}"
        reason = _as_str(item.get("reason"), default="")
        if len(reason) > 80:
            reason = reason[:77] + "..."
        rows.append((triggered_at, action, from_to, reason))
    return rows


class AutoscalerMonitorApp:
    base_url: str

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def run(self) -> None:
        try:
            from rich.text import Text
            from textual.app import App, ComposeResult
            from textual.binding import Binding
            from textual.containers import Container, Horizontal, Vertical
            from textual.widgets import DataTable, Footer, Header, Static
        except Exception as e:
            logger.error(f"Textual/Rich is required to run autoscaler monitor: {e}")
            raise

        base_url = self.base_url
        logger_local = logger

        class _App(App[None]):
            TITLE = "Relax Autoscaler Monitor"

            CSS = """
            Screen {
                layout: vertical;
            }
            #body {
                height: 1fr;
                layout: vertical;
                overflow: hidden;
            }
            .panel {
                border: round $panel;
                padding: 1 2;
                height: 1fr;
            }
            #top {
                height: 8;
                min-height: 8;
                max-height: 8;
            }
            #top Horizontal {
                height: 1fr;
            }
            #status_panel { width: 1fr; }
            #scale_out_panel { width: 1fr; }
            #scale_in_panel { width: 1fr; }
            #middle {
                height: 1fr;
            }
            #middle Horizontal {
                height: 1fr;
            }
            #metrics_panel { width: 1fr; }
            #history_panel { width: 1fr; }
            #bottom {
                height: 14;
                min-height: 14;
                max-height: 14;
            }
            #bottom Horizontal {
                height: 1fr;
            }
            #chart_throughput { width: 1fr; }
            #chart_token { width: 1fr; }
            #chart_running { width: 1fr; }
            #chart_engines { width: 1fr; }

            """

            BINDINGS = [
                Binding("q", "quit", "Quit"),
                Binding("r", "refresh", "Refresh"),
            ]

            def __init__(self) -> None:
                super().__init__()
                self.snapshot: AutoscalerSnapshot = AutoscalerSnapshot()
                self.previous_snapshot: Union[AutoscalerSnapshot, None] = None
                self._client: AutoscalerApiClient = AutoscalerApiClient(base_url)
                self._poll_task: Union[asyncio.Task[None], None] = None
                self._history: MetricsHistory = MetricsHistory()
                self._refresh_interval: float = 5.0
                self._countdown_seconds: float = self._refresh_interval

            def compose(self) -> ComposeResult:
                yield Header()
                with Container(id="body"):
                    with Container(id="top"):
                        with Horizontal():
                            with Vertical(id="status_panel", classes="panel"):
                                yield Static(id="status_text")
                            with Vertical(id="scale_out_panel", classes="panel"):
                                yield Static(id="scale_out_text")
                            with Vertical(id="scale_in_panel", classes="panel"):
                                yield Static(id="scale_in_text")
                    with Container(id="middle"):
                        with Horizontal():
                            with Vertical(id="metrics_panel", classes="panel"):
                                yield DataTable(id="metrics_table")
                            with Vertical(id="history_panel", classes="panel"):
                                yield DataTable(id="history_table")
                    with Container(id="bottom"):
                        with Horizontal():
                            with Vertical(id="chart_throughput", classes="panel"):
                                yield Static(id="chart_throughput_text")
                            with Vertical(id="chart_token", classes="panel"):
                                yield Static(id="chart_token_text")
                            with Vertical(id="chart_running", classes="panel"):
                                yield Static(id="chart_running_text")
                            with Vertical(id="chart_engines", classes="panel"):
                                yield Static(id="chart_engines_text")
                yield Footer()

            async def on_mount(self) -> None:
                metrics_table = self.query_one("#metrics_table", DataTable)
                _ = metrics_table.add_columns("Metric", "Current", "Change")
                metrics_table.cursor_type = "row"

                history_table = self.query_one("#history_table", DataTable)
                _ = history_table.add_columns("Time", "Action", "From→To", "Reason")
                history_table.cursor_type = "row"

                self.query_one("#status_text", Static).update(f"[dim]Connecting to {base_url}...[/dim]")

                await self._client.start()
                await self._refresh_once()
                _ = self.set_interval(self._refresh_interval, self._refresh_once)
                _ = self.set_interval(1.0, self._update_countdown)

            async def on_shutdown(self) -> None:
                await self._client.stop()

            async def action_refresh(self) -> None:
                await self._refresh_once(force=True)

            def _update_countdown(self) -> None:
                self._countdown_seconds -= 1.0
                if self._countdown_seconds < 0:
                    self._countdown_seconds = 0
                self._update_status_panel()

            def _update_status_panel(self) -> None:
                system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                countdown = int(self._countdown_seconds)
                countdown_str = f"[bold cyan]{countdown}s[/bold cyan]" if countdown > 0 else "[dim]0s[/dim]"
                status_content = _render_status_panel(self.snapshot, self.previous_snapshot)
                header = f"[dim]System:[/dim] {system_time}  [dim]Refresh in:[/dim] {countdown_str}\n"
                self.query_one("#status_text", Static).update(header + status_content)

            async def _refresh_once(self, force: bool = False) -> None:
                if self._poll_task is not None and not self._poll_task.done():
                    if force:
                        logger_local.info("Refresh requested while refresh in progress")
                    return

                async def _do() -> None:
                    try:
                        status, conditions, history = await self._client.fetch_all()
                        self.previous_snapshot = self.snapshot.copy()
                        self.snapshot.status = status
                        self.snapshot.conditions = conditions
                        self.snapshot.history = history
                        self.snapshot.last_refresh_ts = asyncio.get_event_loop().time()
                        self.snapshot.last_error = None

                        metrics = _as_dict(status.get("recent_metrics"))
                        throughput = _as_float(metrics.get("total_throughput"))
                        token_usage = _as_float(metrics.get("avg_token_usage"))
                        running_reqs = _as_int(metrics.get("total_running_reqs"))
                        queue_reqs = _as_int(metrics.get("total_queue_reqs"))
                        engines = _as_int(
                            _coalesce(status.get("current_engines"), metrics.get("num_engines"), default=0)
                        )
                        timestamp = _fmt_ts_short(self.snapshot.last_refresh_ts)
                        self._history.add_point(
                            throughput, token_usage, running_reqs, queue_reqs, timestamp, engines=engines
                        )
                    except Exception as e:
                        self.snapshot.last_error = str(e)
                        logger_local.warning(f"Autoscaler monitor refresh failed: {e}")

                    self._render()

                self._poll_task = asyncio.create_task(_do())
                await self._poll_task
                self._countdown_seconds = self._refresh_interval

            def _render(self) -> None:
                self._update_status_panel()

                self.query_one("#scale_out_text", Static).update(_render_scale_out_panel(self.snapshot))
                self.query_one("#scale_in_text", Static).update(_render_scale_in_panel(self.snapshot))

                metrics_table = self.query_one("#metrics_table", DataTable)
                _ = metrics_table.clear(columns=False)
                rows = _build_metrics_rows(self.snapshot, self.previous_snapshot)
                for row in rows:
                    current_text = Text.from_markup(row.current) if "[" in row.current else Text(row.current)
                    delta_text = Text.from_markup(row.delta) if "[" in row.delta else Text(row.delta)
                    _ = metrics_table.add_row(Text(row.label), current_text, delta_text)

                history_table = self.query_one("#history_table", DataTable)
                _ = history_table.clear(columns=False)
                for t, action, from_to, reason in _history_rows(self.snapshot):
                    if action == "scale_out":
                        action_cell = Text.from_markup("[green]scale_out[/green]")
                    elif action == "scale_in":
                        action_cell = Text.from_markup("[blue]scale_in[/blue]")
                    else:
                        action_cell = Text(action)
                    _ = history_table.add_row(Text(t), action_cell, Text(from_to), Text(reason))

                # 获取面板的实际可用宽度，减去左右边框和Y轴标签的预估宽度(约8个字符)
                try:
                    panel = self.query_one("#chart_throughput")
                    available_width = panel.content_size.width
                    # 如果一开始没获取到尺寸，给个默认值 35
                    chart_width = max(20, available_width - 8) if available_width > 0 else 35
                except Exception:
                    chart_width = 35
                # ==========================================

                # 更新为较适宜的半方块画图尺寸并渲染
                self.query_one("#chart_throughput_text", Static).update(
                    _render_chart(
                        list(self._history.throughput),
                        width=chart_width,
                        height=6,
                        color="cyan",
                        title="Throughput",
                        unit=" tok/s",
                        fmt=".0f",
                        y_min=0.0,
                    )
                )

                token_pct = [v * 100 for v in self._history.token_usage]
                self.query_one("#chart_token_text", Static).update(
                    _render_chart(
                        token_pct,
                        width=chart_width,
                        height=6,
                        color="yellow",
                        title="Token Usage",
                        unit="%",
                        fmt=".1f",
                        y_min=0.0,
                    )
                )

                self.query_one("#chart_running_text", Static).update(
                    _render_chart(
                        list(self._history.running_reqs),
                        width=chart_width,
                        height=6,
                        color="green",
                        title="Running Reqs",
                        unit="",
                        fmt=".0f",
                        y_min=0.0,
                    )
                )

                self.query_one("#chart_engines_text", Static).update(
                    _render_chart(
                        list(self._history.engines),
                        width=chart_width,
                        height=6,
                        color="magenta",
                        title="Engines",
                        unit="",
                        fmt=".0f",
                        y_min=0.0,
                        integer_yaxis=True,
                    )
                )

        _App().run()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relax autoscaler Textual monitor")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/autoscaler",
        help="Base autoscaler URL (default: http://localhost:8000/autoscaler)",
    )
    return parser


def main(argv: Union[list[str], None] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    AutoscalerMonitorApp(base_url=str(args.url)).run()


if __name__ == "__main__":
    main()
