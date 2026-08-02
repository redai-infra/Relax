# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from datetime import datetime
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "task22_hybrid_async_text" / "analyze.py"
SPEC = importlib.util.spec_from_file_location("task22_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_parse_metric_dict_accepts_known_infinity() -> None:
    record = analysis.parse_metric_dict("{'perf/step_time': 2.5, 'perf/device_peak_tflops': inf}")

    assert record == {"perf/step_time": 2.5, "perf/device_peak_tflops": float("inf")}


def test_nonfinite_counts_separates_peak_sentinel() -> None:
    known, unexpected = analysis.nonfinite_counts(
        [{"perf/device_peak_tflops": float("inf"), "train/loss": 0.0}, {"train/tis": float("nan")}]
    )

    assert known == 1
    assert unexpected == 1


def test_metric_records_filters_same_prefix_by_required_key(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "2026-08-02 10:00:00 | INFO | perf 2: {'perf/rollout_time': 4.0}\n"
        "2026-08-02 10:00:01 | INFO | perf 2: {'perf/step_time': 5.0}\n"
    )

    actor = analysis.metric_records(log, "perf", required_key="perf/step_time")
    rollout = analysis.metric_records(log, "perf", required_key="perf/rollout_time")

    assert actor == {2: {"perf/step_time": 5.0}}
    assert rollout == {2: {"perf/rollout_time": 4.0}}


def test_actor_completion_timestamps_parse_steps(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "2026-08-02 10:00:01 | INFO | Actor training completed step 1/10\n"
        "2026-08-02 10:00:03.250 | INFO | Actor training completed step 2/10\n"
    )

    timestamps = analysis.actor_completion_timestamps(log)

    assert timestamps[1] == datetime.fromisoformat("2026-08-02 10:00:01")
    assert timestamps[2] == datetime.fromisoformat("2026-08-02 10:00:03.250")


def test_gpu_stats_filters_window_and_roles(tmp_path: Path) -> None:
    gpu_log = tmp_path / "gpu.csv"
    gpu_log.write_text(
        "2026-08-02T10:00:00+08:00,2,10,100,1000\n"
        "2026-08-02T10:00:01+08:00,2,80,700,1000\n"
        "2026-08-02T10:00:01+08:00,3,40,600,1000\n"
        "2026-08-02T10:00:01+08:00,1,99,999,1000\n"
        "2026-08-02T10:00:02+08:00,2,100,800,1000\n"
        "2026-08-02T10:00:02+08:00,3,60,650,1000\n"
    )

    stats = analysis.gpu_stats(
        gpu_log,
        datetime.fromisoformat("2026-08-02 10:00:01"),
        datetime.fromisoformat("2026-08-02 10:00:02"),
        actor_gpu=2,
        rollout_gpu=3,
    )

    assert stats == {
        "gpu_util_pct": 70.0,
        "actor_gpu_util_pct": 90.0,
        "rollout_gpu_util_pct": 50.0,
        "peak_memory_mib": 800.0,
    }
