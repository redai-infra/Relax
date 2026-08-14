# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "analyze_hybrid_pipeline_benchmark.py"
SPEC = importlib.util.spec_from_file_location("analyze_hybrid_pipeline_benchmark", SCRIPT_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

FINGERPRINT_0 = "00000000000000000000000000000001"
FINGERPRINT_1 = "00000000000000000000000000000002"
FULL_FINGERPRINT = "00000000000000000000000000000003"


def test_parser_help_renders_percent_targets(capsys):
    with pytest.raises(SystemExit) as exc_info:
        analyzer.build_parser().parse_args(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--enforce-targets" in output
    assert "5% throughput, 15% phase-1, and 80% overlap" in " ".join(output.split())


def _install_fake_matplotlib(monkeypatch):
    """Install the plotting API subset used by the analyzer for minimal CI."""
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []
    matplotlib.use = MagicMock()
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.tight_layout = MagicMock()
    pyplot.close = MagicMock()
    pyplot.figure = MagicMock()
    pyplot.plot = MagicMock()
    pyplot.xlabel = MagicMock()
    pyplot.ylabel = MagicMock()
    pyplot.legend = MagicMock()
    pyplot.title = MagicMock()
    pyplot.savefig = MagicMock(side_effect=lambda path, **_kwargs: Path(path).write_bytes(b"deterministic plot"))

    def axes(count):
        return [MagicMock() for _ in range(count)]

    class AxesGrid:
        def __init__(self, rows, columns):
            self._axes = {(row, column): MagicMock() for row in range(rows) for column in range(columns)}
            self._rows = [[self._axes[row, column] for column in range(columns)] for row in range(rows)]
            for axis in self._axes.values():
                axis.get_legend_handles_labels.return_value = ([], [])

        def __getitem__(self, key):
            return self._axes[key]

        def __iter__(self):
            return iter(self._rows)

    pyplot.subplots = MagicMock(
        side_effect=[
            (MagicMock(), axes(2)),
            (MagicMock(), axes(3)),
            (MagicMock(), AxesGrid(2, 2)),
            (MagicMock(), AxesGrid(3, 2)),
            (MagicMock(), axes(2)),
        ]
    )
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)


def _event(
    event,
    monotonic_ns,
    *,
    role,
    rollout_id=4,
    chunk_index=None,
    sample_count=None,
    fingerprint=None,
    hostname="test-host",
    pid=None,
    global_rank=None,
    **details,
):
    row = {
        "event": event,
        "monotonic_ns": monotonic_ns,
        "rollout_id": rollout_id,
        "chunk_index": chunk_index,
        "sample_count": sample_count,
        "total_tokens": sample_count * 10 if sample_count is not None else None,
        "response_tokens": sample_count * 4 if sample_count is not None else None,
        "multimodal_tensor_bytes": sample_count * 100 if sample_count is not None else None,
        "role": role,
        "hostname": hostname,
        "pid": (10 if role == "rollout" else 20) if pid is None else pid,
        "global_rank": (-1 if role == "rollout" else 0) if global_rank is None else global_rank,
        "cuda_visible_devices": "0,1,2,3",
        "cuda_max_allocated_bytes": 1024,
        "cuda_max_reserved_bytes": 2048,
        "global_indexes_fingerprint": fingerprint,
    }
    row.update(details)
    return row


def _write_run(
    tmp_path,
    *,
    name="run",
    pipeline_enabled=True,
    pipeline_overlap=None,
    hostname="test-host",
    missing_event=None,
    nonfinite=False,
    producer_put_count=2,
    producer_last_start_ns=700,
):
    if pipeline_overlap is None:
        pipeline_overlap = pipeline_enabled
    run_dir = tmp_path / name
    timeline = run_dir / "timeline"
    timeline.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "condition": "experiment" if pipeline_enabled and pipeline_overlap else "baseline",
                "hostname": hostname,
                "order": "B1" if pipeline_enabled else "A1",
                "seed": 7,
                "rollout_seed": 7,
                "num_rollout": 20,
                "max_staleness": 2,
                "global_batch_size": 4,
                "rollout_batch_size": 2,
                "n_samples_per_prompt": 2,
                "num_iters_per_train_update": 2,
                "hybrid_pipeline_forward": int(pipeline_enabled),
                "hybrid_pipeline_overlap": int(pipeline_overlap),
                "hybrid_pipeline_trace_dir": str(timeline),
                "hybrid_pipeline_fetch_timeout_s": 600,
                "git_commit": "a" * 40,
                "git_branch": "perf/task21",
                "git_status_porcelain": "",
                "image_archive_sha256": "b" * 64,
                "image_manifest_digest": "sha256:" + "c" * 64,
                "image_id": "sha256:" + "d" * 64,
                "transferqueue_commit": "e" * 40,
                "python": "3.12.3",
                "entrypoint": "bash target.sh hybrid-async",
            }
        ),
        encoding="utf-8",
    )

    producer_shapes = {
        1: [(4, FULL_FINGERPRINT)],
        2: [(2, FINGERPRINT_0), (2, FINGERPRINT_1)],
        # Fingerprints are additive, so three value-1 digests preserve
        # the same full digest as the actor's value-1 + value-2 chunks.
        3: [(1, FINGERPRINT_0), (1, FINGERPRINT_0), (2, FINGERPRINT_0)],
    }
    if producer_put_count not in producer_shapes:
        raise ValueError(f"unsupported producer_put_count={producer_put_count}")
    producer = []
    for chunk_index, (sample_count, _) in enumerate(producer_shapes[producer_put_count]):
        start_ns = producer_last_start_ns if chunk_index == producer_put_count - 1 else 100 + chunk_index * 100
        producer.append(
            _event(
                "tq_put_start",
                start_ns,
                role="rollout",
                chunk_index=chunk_index,
                sample_count=sample_count,
                hostname=hostname,
                event_id=f"put-{chunk_index}",
                is_last=chunk_index == producer_put_count - 1,
            )
        )
    for chunk_index, (sample_count, fingerprint) in enumerate(producer_shapes[producer_put_count]):
        done_ns = 800 if chunk_index == producer_put_count - 1 else 150 + chunk_index * 100
        producer.append(
            _event(
                "tq_put_done",
                done_ns,
                role="rollout",
                chunk_index=chunk_index,
                sample_count=sample_count,
                fingerprint=fingerprint,
                hostname=hostname,
                event_id=f"put-{chunk_index}",
                is_last=chunk_index == producer_put_count - 1,
            )
        )
    producer.sort(key=lambda row: row["monotonic_ns"])
    if nonfinite:
        producer[0]["total_tokens"] = float("nan")

    if pipeline_enabled and pipeline_overlap:
        actor = [
            _event("actor_restore_start", 210, role="actor", chunk_index=0, sample_count=4),
            _event("actor_restore_end", 300, role="actor", chunk_index=0, sample_count=4),
            _event("chunk_fetch_start", 310, role="actor", chunk_index=0, sample_count=2),
            _event(
                "chunk_fetch_end",
                350,
                role="actor",
                chunk_index=0,
                sample_count=2,
                fingerprint=FINGERPRINT_0,
            ),
            _event(
                "actor_forward_start",
                400,
                role="actor",
                chunk_index=0,
                sample_count=2,
                fingerprint=FINGERPRINT_0,
            ),
            _event(
                "actor_forward_end",
                600,
                role="actor",
                chunk_index=0,
                sample_count=2,
                fingerprint=FINGERPRINT_0,
            ),
            _event("chunk_fetch_start", 810, role="actor", chunk_index=1, sample_count=2),
            _event(
                "chunk_fetch_end",
                850,
                role="actor",
                chunk_index=1,
                sample_count=2,
                fingerprint=FINGERPRINT_1,
            ),
            _event(
                "actor_forward_start",
                860,
                role="actor",
                chunk_index=1,
                sample_count=2,
                fingerprint=FINGERPRINT_1,
            ),
            _event(
                "actor_forward_end",
                1000,
                role="actor",
                chunk_index=1,
                sample_count=2,
                fingerprint=FINGERPRINT_1,
            ),
        ]
    elif pipeline_enabled:
        actor = [
            _event("actor_restore_start", 210, role="actor", chunk_index=0, sample_count=4),
            _event("actor_restore_end", 300, role="actor", chunk_index=0, sample_count=4),
            _event("chunk_fetch_start", 310, role="actor", chunk_index=0, sample_count=2),
            _event(
                "chunk_fetch_end",
                350,
                role="actor",
                chunk_index=0,
                sample_count=2,
                fingerprint=FINGERPRINT_0,
            ),
            _event("chunk_fetch_start", 810, role="actor", chunk_index=1, sample_count=2),
            _event(
                "chunk_fetch_end",
                850,
                role="actor",
                chunk_index=1,
                sample_count=2,
                fingerprint=FINGERPRINT_1,
            ),
            _event(
                "actor_forward_start",
                860,
                role="actor",
                chunk_index=0,
                sample_count=2,
                fingerprint=FINGERPRINT_0,
            ),
            _event(
                "actor_forward_end",
                900,
                role="actor",
                chunk_index=0,
                sample_count=2,
                fingerprint=FINGERPRINT_0,
            ),
            _event(
                "actor_forward_start",
                910,
                role="actor",
                chunk_index=1,
                sample_count=2,
                fingerprint=FINGERPRINT_1,
            ),
            _event(
                "actor_forward_end",
                1000,
                role="actor",
                chunk_index=1,
                sample_count=2,
                fingerprint=FINGERPRINT_1,
            ),
        ]
    else:
        actor = [
            _event("chunk_fetch_start", 810, role="actor", chunk_index=0, sample_count=4),
            _event(
                "chunk_fetch_end",
                850,
                role="actor",
                chunk_index=0,
                sample_count=4,
                fingerprint=FULL_FINGERPRINT,
            ),
            _event("actor_restore_start", 860, role="actor", chunk_index=0, sample_count=4),
            _event("actor_restore_end", 900, role="actor", chunk_index=0, sample_count=4),
            _event(
                "actor_forward_start",
                910,
                role="actor",
                chunk_index=0,
                sample_count=4,
                fingerprint=FULL_FINGERPRINT,
            ),
            _event(
                "actor_forward_end",
                1000,
                role="actor",
                chunk_index=0,
                sample_count=4,
                fingerprint=FULL_FINGERPRINT,
            ),
        ]
    actor += [
        _event("advantages_start", 1010, role="actor", sample_count=4),
        _event("advantages_end", 1020, role="actor", sample_count=4),
        _event("optimizer_start", 1030, role="actor", sample_count=4),
        _event("optimizer_end", 1040, role="actor", sample_count=4),
    ]
    if missing_event is not None:
        actor = [row for row in actor if row["event"] != missing_event]

    for filename, rows in (("rollout.jsonl", producer), ("actor.jsonl", actor)):
        (timeline / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    return run_dir


def test_complete_pipeline_trace_is_validated_and_summarized(tmp_path):
    run_dir = _write_run(tmp_path)

    analysis = analyzer.analyze_run(
        run_dir,
        windows=((4, 4),),
        expected_samples=4,
        expected_actor_chunks=2,
    )

    assert analysis.summary["validation"] == "passed"
    assert analysis.summary["trace"]["producer_overlap_rollout_ratio"] == 1.0
    assert analysis.trace_rows[0]["actor_fetch_count"] == 2
    assert analysis.trace_rows[0]["actor_restore_count"] == 1
    assert analysis.trace_rows[0]["actor_forward_samples"] == 4
    assert (run_dir / "analysis" / "summary.json").is_file()
    assert (run_dir / "analysis" / "trace_events.csv").is_file()


@pytest.mark.parametrize("producer_put_count", [1, 2, 3])
def test_producer_put_grouping_is_independent_from_actor_chunks(tmp_path, producer_put_count):
    run_dir = _write_run(tmp_path, producer_put_count=producer_put_count)

    analysis = analyzer.analyze_run(
        run_dir,
        windows=((4, 4),),
        expected_samples=4,
        expected_actor_chunks=2,
    )

    assert analysis.trace_rows[0]["producer_put_count"] == producer_put_count
    assert analysis.trace_rows[0]["actor_fetch_count"] == 2
    assert analysis.trace_rows[0]["actor_forward_count"] == 2


def test_delayed_put_done_is_not_strict_producer_overlap(tmp_path):
    run_dir = _write_run(tmp_path, producer_last_start_ns=150)

    analysis = analyzer.analyze_run(
        run_dir,
        windows=((4, 4),),
        expected_samples=4,
        expected_actor_chunks=2,
    )

    row = analysis.trace_rows[0]
    assert row["first_forward_before_last_put_start"] is False
    assert row["producer_overlap_s"] == 0
    assert row["first_forward_before_last_put_done"] is True
    assert row["transfer_overlap_s"] > 0


def test_baseline_trace_preserves_one_full_fetch_and_no_overlap(tmp_path):
    run_dir = _write_run(tmp_path, pipeline_enabled=False)

    analysis = analyzer.analyze_run(
        run_dir,
        windows=((4, 4),),
        expected_samples=4,
        expected_actor_chunks=2,
    )

    assert analysis.trace_rows[0]["actor_fetch_count"] == 1
    assert analysis.trace_rows[0]["actor_restore_count"] == 1
    assert analysis.trace_rows[0]["first_forward_before_last_put_start"] is False


def test_schedule_matched_baseline_fetches_all_chunks_before_forward(tmp_path):
    run_dir = _write_run(tmp_path, pipeline_enabled=True, pipeline_overlap=False)

    analysis = analyzer.analyze_run(
        run_dir,
        windows=((4, 4),),
        expected_samples=4,
        expected_actor_chunks=2,
    )

    assert analysis.trace_rows[0]["actor_fetch_count"] == 2
    assert analysis.trace_rows[0]["all_chunks_fetched_before_first_forward"] is True
    assert analysis.summary["hybrid_pipeline_overlap"] is False


def test_measurement_scope_distinguishes_first_step_from_steady_state():
    assert analyzer._measurement_scope(((0, 0),)) == "fresh_process_first_step"
    assert analyzer._measurement_scope(((4, 8), (9, 13))) == "steady_state"
    assert analyzer._measurement_scope(((0, 2),)) == "selected_steps"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"missing_event": "optimizer_end"}, "optimizer_end"),
        ({"nonfinite": True}, "non-finite"),
    ],
)
def test_invalid_trace_returns_nonzero_validation(tmp_path, kwargs, error, capsys):
    run_dir = _write_run(tmp_path, **kwargs)

    status = analyzer.main(
        [
            "--run-dir",
            str(run_dir),
            "--steady-windows",
            "4-4",
            "--expected-samples",
            "4",
            "--expected-actor-chunks",
            "2",
            "--validate-only",
        ]
    )

    assert status == 2
    assert error in capsys.readouterr().err


def test_cross_hostname_trace_is_rejected(tmp_path):
    run_dir = _write_run(tmp_path)
    actor_path = run_dir / "timeline" / "actor.jsonl"
    rows = [json.loads(line) for line in actor_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["hostname"] = "another-host"
    actor_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(analyzer.BenchmarkValidationError, match="multiple hostnames"):
        analyzer.analyze_run(
            run_dir,
            windows=((4, 4),),
            expected_samples=4,
            expected_actor_chunks=2,
        )


def test_dirty_manifest_and_missing_reproducibility_artifacts_fail_closed(tmp_path):
    dirty_run = _write_run(tmp_path, name="dirty")
    manifest_path = dirty_run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git_status_porcelain"] = " M relax/file.py"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(analyzer.BenchmarkValidationError, match="dirty working tree"):
        analyzer.analyze_run(
            dirty_run,
            windows=((4, 4),),
            expected_samples=4,
            expected_actor_chunks=2,
        )

    incomplete_run = _write_run(tmp_path, name="missing-artifacts")
    with pytest.raises(analyzer.BenchmarkValidationError, match="reproducibility artifacts"):
        analyzer.analyze_run(
            incomplete_run,
            windows=((4, 4),),
            expected_samples=4,
            expected_actor_chunks=2,
            require_reproducibility_artifacts=True,
        )


def test_invalid_or_overlapping_windows_are_rejected():
    with pytest.raises(analyzer.BenchmarkValidationError, match="overlap"):
        analyzer._parse_windows("4-8,8-10")
    with pytest.raises(analyzer.BenchmarkValidationError, match="0 <= START <= END"):
        analyzer._parse_windows("5-4")


def test_global_index_fingerprint_combination_ignores_chunk_grouping():
    producer = [
        {"global_indexes_fingerprint": FINGERPRINT_0},
        {"global_indexes_fingerprint": FINGERPRINT_1},
    ]
    regrouped_fetch = [{"global_indexes_fingerprint": FULL_FINGERPRINT}]

    assert analyzer._combine_global_index_fingerprints(
        producer,
        context="producer",
    ) == analyzer._combine_global_index_fingerprints(
        regrouped_fetch,
        context="fetch",
    )


def test_aggregate_throughput_uses_token_sum_over_time_sum():
    rows = [
        {"tag": "perf/step_time", "step": 4, "value": 3.0},
        {"tag": "perf/step_time", "step": 5, "value": 1.0},
        {"tag": "perf/step_token_per_s", "step": 4, "value": 100.0},
        {"tag": "perf/step_token_per_s", "step": 5, "value": 200.0},
    ]

    result = analyzer._aggregate_throughput(
        rows,
        "perf/step_token_per_s",
        ((4, 5),),
    )

    assert result == 125.0
    assert analyzer._aggregate_samples_per_second(rows, ((4, 5),), samples_per_step=256) == 128.0


def test_nvml_utilization_uses_registered_steady_step_wall_time():
    scalar_rows = [
        {
            "tag": "perf/step_time",
            "step": 4,
            "value": 2.0,
            "wall_time": 100.0,
        }
    ]
    intervals = analyzer._steady_wall_time_intervals(scalar_rows, ((4, 4),))
    nvml_rows = [
        {
            "gpu_index": 0.0,
            "wall_time": wall_time,
            "gpu_util_percent": utilization,
            "memory_used_mib": memory,
            "power_w": 100.0,
        }
        for wall_time, utilization, memory in (
            (97.0, 0.0, 1000.0),
            (99.0, 80.0, 2000.0),
            (101.0, 0.0, 3000.0),
        )
    ]

    summary = analyzer._nvml_summary(nvml_rows, steady_intervals=intervals)

    assert intervals == [(98.0, 100.0)]
    assert summary["per_gpu"]["0"]["steady_sample_count"] == 1
    assert summary["steady_mean_gpu_util_percent"] == 80.0
    assert summary["steady_idle_ratio_below_10_percent"] == 0.0
    # Peak memory deliberately remains a full-run safety metric.
    assert summary["peak_memory_used_mib"] == 3000.0


def _comparison_analysis(
    condition,
    seed,
    throughput,
    phase1,
    *,
    overlap=None,
    accuracy=0.5,
    peak_vram_mib=10_000,
):
    if overlap is None:
        overlap = condition == "experiment"
    metrics = {
        "perf/step_token_per_s": {"aggregate": throughput},
        "perf/step_resp_token_per_s": {"aggregate": throughput / 2},
        "perf/wall_clock_samples_per_s": {"aggregate": 12.8},
        "perf/step_time": {"mean": 20.0, "p95": 21.0},
        "perf/hybrid_phase1_time": {"mean": phase1},
        "rollout/raw_reward": {"mean": accuracy},
        "rollout/truncated_ratio": {"mean": 0.0},
        "train/loss": {"mean": 1.0},
        "train/grad_norm": {"mean": 0.5},
        "train/ppo_kl": {"mean": 0.0},
        "train/pg_clipfrac": {"mean": 0.0},
    }
    trace_rows = [
        {
            "rollout_id": rollout_id,
            "first_forward_before_last_put_start": overlap,
            "first_forward_before_last_put_done": overlap,
            "producer_overlap_s": 1.0 if overlap else 0.0,
            "transfer_overlap_s": 1.0 if overlap else 0.0,
            "actor_fetch_samples": 256,
            "actor_total_tokens": 10_000 + seed,
            "actor_response_tokens": 5_000 + seed,
            "actor_multimodal_tensor_bytes": 1_000_000 + seed,
            "producer_lead_at_first_forward": 1.0,
            "producer_global_indexes_fingerprint": f"{seed:032x}",
        }
        for rollout_id in range(4, 19)
    ]
    scalar_values = {
        "perf/step_token_per_s": throughput,
        "perf/step_resp_token_per_s": throughput / 2,
        "perf/step_time": 20.0,
        "perf/hybrid_phase1_time": phase1,
        "rollout/raw_reward": accuracy,
        "rollout/truncated_ratio": 0.0,
        "train/loss": 1.0,
        "train/grad_norm": 0.5,
        "train/ppo_kl": 0.0,
        "train/pg_clipfrac": 0.0,
    }
    scalar_rows = [
        {"tag": tag, "step": step, "value": value} for step in range(4, 19) for tag, value in scalar_values.items()
    ]
    return analyzer.RunAnalysis(
        run_dir=Path(f"/{condition}-{seed}"),
        manifest={
            "hostname": "test-host",
            "condition": condition,
            "order": ("A" if condition == "baseline" else "B") + str(seed),
            "seed": seed,
            "rollout_seed": seed,
            "num_rollout": 20,
            "max_staleness": 2,
            "global_batch_size": 256,
            "rollout_batch_size": 32,
            "n_samples_per_prompt": 8,
            "num_iters_per_train_update": 2,
            "hybrid_pipeline_forward": True,
            "hybrid_pipeline_overlap": condition == "experiment",
            "hybrid_pipeline_fetch_timeout_s": 600,
            "git_commit": "a" * 40,
            "git_branch": "perf/task21",
            "image_archive_sha256": "b" * 64,
            "image_manifest_digest": "sha256:" + "c" * 64,
            "image_id": "sha256:" + "d" * 64,
            "transferqueue_commit": "e" * 40,
            "python": "3.12.3",
            "entrypoint": "bash target.sh hybrid-async",
            "schema_version": 1,
            "baseline_commit": "f" * 40,
            "model_variant": "qwen3vl8b",
            "model_name": "Qwen3-VL-8B-Instruct",
            "model_dir": "/data01/LWX/Qwen3-VL-8B-Instruct",
            "model_config_file": "/workspace/Relax/scripts/models/qwen3-vl-8B.sh",
            "data_file": "/data01/LWX/openr1mm/train.parquet",
            "rollout_max_response_len": 1024,
            "rollout_max_prompt_len": 2048,
            "rollout_max_context_len": 3072,
            "actor_max_tokens_per_gpu": 6144,
            "resource": {"actor": [1, 4], "rollout": [1, 1]},
            "rollout_num_gpus_per_engine": 1,
            "physical_gpu_indices": [0, 1, 2, 3, 5],
            "container_cuda_visible_devices": [0, 1, 2, 3, 4],
            "gpu_hardware_fingerprint": "9" * 64,
            "checkpoint_save": False,
            "sglang_deterministic_inference": 1,
            "sglang_mem_fraction_static": 0.8,
            "load_debug_rollout_data": None,
            "save_debug_rollout_data": None,
            "save_debug_train_data": None,
        },
        trace_rows=trace_rows,
        actor_rank_rows=[],
        scalar_rows=scalar_rows,
        nvml_rows=[],
        summary={
            "metrics": metrics,
            "nvml": {
                "gpu_count": 8,
                "per_gpu": {
                    str(gpu_index): {
                        "steady_sample_count": 10,
                        "steady_mean_gpu_util_percent": 80.0,
                        "steady_idle_ratio_below_10_percent": 0.05,
                    }
                    for gpu_index in range(8)
                },
                "steady_mean_gpu_util_percent": 80.0,
                "steady_idle_ratio_below_10_percent": 0.05,
                "peak_memory_used_mib": peak_vram_mib,
            },
        },
    )


def test_preregistered_targets_pass_and_fail_deterministically():
    passing = [
        _comparison_analysis("baseline", 1, 100, 10),
        _comparison_analysis("experiment", 1, 106, 8),
        _comparison_analysis("baseline", 2, 102, 10.2),
        _comparison_analysis("experiment", 2, 108.12, 8.16),
    ]

    comparison = analyzer._build_comparison(
        passing,
        windows=((4, 8), (9, 13), (14, 18)),
        enforce_targets=True,
    )

    assert comparison["token_throughput_geomean_speedup"] == pytest.approx(0.06)
    assert comparison["hybrid_phase1_geomean_reduction"] == pytest.approx(0.2)
    assert comparison["experiment_steady_producer_overlap_ratio"] == 1.0
    assert len(comparison["window_speedups"]) == 6
    assert comparison["distributions"]["paired_step_token_per_s_speedup"]["count"] == 2
    assert comparison["distributions"]["baseline_step_token_per_s"]["coefficient_of_variation"] > 0

    failing = [
        _comparison_analysis("baseline", 1, 100, 10),
        _comparison_analysis("experiment", 1, 104, 8),
        _comparison_analysis("baseline", 2, 102, 10.2),
        _comparison_analysis("experiment", 2, 106.08, 8.16),
    ]
    with pytest.raises(analyzer.BenchmarkValidationError, match="below 5%"):
        analyzer._build_comparison(
            failing,
            windows=((4, 8), (9, 13), (14, 18)),
            enforce_targets=True,
        )


def test_performance_targets_require_schedule_matched_overlap_modes():
    analyses = [
        _comparison_analysis("baseline", 1, 100, 10),
        _comparison_analysis("experiment", 1, 106, 8),
        _comparison_analysis("baseline", 2, 102, 10.2),
        _comparison_analysis("experiment", 2, 108.12, 8.16),
    ]
    analyses[0].manifest["hybrid_pipeline_forward"] = False

    with pytest.raises(analyzer.BenchmarkValidationError, match="schedule-matched chunk forwarding"):
        analyzer._build_comparison(
            analyses,
            windows=((4, 8), (9, 13), (14, 18)),
            enforce_targets=True,
        )


def test_comparison_rejects_mixed_candidate_commits():
    analyses = [
        _comparison_analysis("baseline", 1, 100, 10),
        _comparison_analysis("experiment", 1, 106, 8),
        _comparison_analysis("baseline", 2, 102, 10.2),
        _comparison_analysis("experiment", 2, 108.12, 8.16),
    ]
    analyses[1].manifest["git_commit"] = "f" * 40

    with pytest.raises(analyzer.BenchmarkValidationError, match="git_commit"):
        analyzer._build_comparison(
            analyses,
            windows=((4, 8), (9, 13), (14, 18)),
            enforce_targets=True,
        )


def test_comparison_rejects_mixed_or_incomplete_workload_manifests():
    analyses = [
        _comparison_analysis("baseline", 1, 100, 10),
        _comparison_analysis("experiment", 1, 106, 8),
    ]
    analyses[1].manifest["rollout_max_response_len"] = 2048

    with pytest.raises(analyzer.BenchmarkValidationError, match="rollout_max_response_len"):
        analyzer._build_comparison(
            analyses,
            windows=((4, 8), (9, 13), (14, 18)),
            enforce_targets=False,
        )

    analyses[1].manifest["rollout_max_response_len"] = 1024
    del analyses[1].manifest["physical_gpu_indices"]
    with pytest.raises(analyzer.BenchmarkValidationError, match="missing workload fields"):
        analyzer._build_comparison(
            analyses,
            windows=((4, 8), (9, 13), (14, 18)),
            enforce_targets=False,
        )


def test_comparison_rejects_changed_global_index_fingerprint():
    analyses = [
        _comparison_analysis("baseline", 1, 100, 10),
        _comparison_analysis("experiment", 1, 106, 8),
    ]
    analyses[1].trace_rows[0]["producer_global_indexes_fingerprint"] = "f" * 32

    with pytest.raises(analyzer.BenchmarkValidationError, match="global-index fingerprints differ"):
        analyzer._build_comparison(
            analyses,
            windows=((4, 8), (9, 13), (14, 18)),
            enforce_targets=False,
        )


def test_comparison_plot_bundle_is_generated(tmp_path, monkeypatch):
    analyses = [
        _comparison_analysis("baseline", 1, 100, 10),
        _comparison_analysis("experiment", 1, 106, 8),
        _comparison_analysis("baseline", 2, 102, 10.2),
        _comparison_analysis("experiment", 2, 108.12, 8.16),
    ]
    comparison = analyzer._build_comparison(
        analyses,
        windows=((4, 8), (9, 13), (14, 18)),
        enforce_targets=True,
    )

    if importlib.util.find_spec("matplotlib") is None:
        _install_fake_matplotlib(monkeypatch)

    generated = analyzer._plot_comparison(
        analyses,
        comparison,
        tmp_path,
        ((4, 8), (9, 13), (14, 18)),
    )

    assert {Path(path).name for path in generated} == {
        "task21_correctness_quality.png",
        "task21_gpu_util_vram.png",
        "task21_phase1_overlap.png",
        "task21_step_throughput.png",
        "task21_window_summary.png",
    }
    assert all(Path(path).stat().st_size > 0 for path in generated)


def test_comparison_plot_bundle_fails_fast_without_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    with pytest.raises(
        analyzer.BenchmarkValidationError,
        match="plot generation requires the optional dependency 'matplotlib'",
    ):
        analyzer._plot_comparison([], {}, tmp_path, ())


@pytest.mark.parametrize(
    ("guardrail", "error"),
    [
        ("overlap", "at least 80% steady producer overlap"),
        ("accuracy", "accuracy drop exceeds 3 percentage points"),
        ("truncation", "truncation-rate increase exceeds 2 percentage points"),
        ("staleness", "average producer-lead increase exceeds 0.25"),
        ("staleness_max", "observed producer lead exceeds configured max_staleness=2"),
        ("vram", "peak VRAM increased"),
        ("workload", "actor_total_tokens must match exactly"),
        ("ppo_kl", "same-weight train/ppo_kl exceeds"),
        ("pg_clipfrac", "same-weight train/pg_clipfrac exceeds"),
    ],
)
def test_preregistered_guardrails_fail_closed(guardrail, error):
    analyses = [
        _comparison_analysis("baseline", 1, 100, 10),
        _comparison_analysis("experiment", 1, 106, 8),
        _comparison_analysis("baseline", 2, 102, 10.2),
        _comparison_analysis("experiment", 2, 108.12, 8.16),
    ]
    experiment = analyses[1]
    if guardrail == "overlap":
        for row in experiment.trace_rows:
            row["first_forward_before_last_put_start"] = False
    elif guardrail == "accuracy":
        experiment.summary["metrics"]["rollout/raw_reward"]["mean"] = 0.46
    elif guardrail == "truncation":
        experiment.summary["metrics"]["rollout/truncated_ratio"]["mean"] = 0.03
    elif guardrail == "staleness":
        for row in experiment.trace_rows:
            row["producer_lead_at_first_forward"] = 1.3
    elif guardrail == "staleness_max":
        experiment.trace_rows[0]["producer_lead_at_first_forward"] = 3.0
    elif guardrail == "vram":
        experiment.summary["nvml"]["peak_memory_used_mib"] = 12_000
    elif guardrail == "workload":
        for row in experiment.trace_rows:
            row["actor_total_tokens"] *= 2
    elif guardrail == "ppo_kl":
        experiment.summary["metrics"]["train/ppo_kl"]["mean"] = 1e-4
    elif guardrail == "pg_clipfrac":
        experiment.summary["metrics"]["train/pg_clipfrac"]["mean"] = 1e-4

    with pytest.raises(analyzer.BenchmarkValidationError, match=error):
        analyzer._build_comparison(
            analyses,
            windows=((4, 8), (9, 13), (14, 18)),
            enforce_targets=True,
        )


def test_reproducibility_artifacts_require_all_zero_exit_statuses(tmp_path):
    run_dir = tmp_path / "run"
    for relative_path in analyzer.REPRODUCIBILITY_ARTIFACTS:
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("captured\n", encoding="utf-8")
    for relative_path in analyzer.EXIT_STATUS_ARTIFACTS:
        (run_dir / relative_path).write_text("0\n", encoding="utf-8")

    analyzer._require_reproducibility_artifacts(run_dir)

    (run_dir / "validation_exit_status.txt").write_text("2\n", encoding="utf-8")
    with pytest.raises(analyzer.BenchmarkValidationError, match="non-zero or invalid"):
        analyzer._require_reproducibility_artifacts(run_dir)
