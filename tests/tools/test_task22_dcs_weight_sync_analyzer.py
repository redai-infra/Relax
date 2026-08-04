# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "task22" / "analyze_dcs_weight_sync.py"
SPEC = importlib.util.spec_from_file_location("task22_dcs_weight_sync_analyzer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)
analyze = ANALYZER.analyze


def _marker(
    step: int,
    *,
    reused: bool = True,
    world_size: int = 3,
    receivers: int = 2,
    h2d_bytes: int = 0,
    expired_requests: int = 2,
    safe_requests: int = 10,
) -> str:
    broadcast_bytes = 1000 + max(step, 0)
    return (
        "prefix TASK22_DCS_WEIGHT_SYNC "
        f"logical_step={step} weight_version={step + 2} "
        f"group_reused={str(reused).lower()} group_world_size={world_size} rollout_receivers={receivers} "
        "topology_seconds=0.010000 group_setup_seconds=0.020000 "
        f"source_materialize_seconds=0.030000 source_h2d_bytes={h2d_bytes} source_local_bytes=2000 "
        "tp_gather_seconds=0.040000 hf_conversion_seconds=0.050000 "
        "lock_wait_seconds=0.001000 broadcast_seconds=0.060000 receiver_finalize_seconds=0.070000 "
        "pause_flush_seconds=0.080000 continue_seconds=0.090000 "
        f"targeted_prepare_seconds=0.010000 targeted_active_requests={expired_requests + safe_requests} "
        f"targeted_expired_requests={expired_requests} targeted_safe_requests={safe_requests} "
        f"broadcast_buckets=4 broadcast_tensors=398 broadcast_bytes={broadcast_bytes} "
        f"fanout_bytes={2 * broadcast_bytes} backend_total_seconds=0.400000 client_total_seconds=0.430000"
    )


def _retirement(step: int) -> list[str]:
    version = step + 2
    publication_id = f"pub-{version}"
    return [
        (
            "prefix TASK22_TARGETED_RETIRE event=prepare "
            f"publication_id={publication_id} target_version={version} "
            "active_requests=12 expired_requests=2 safe_requests=10 workers=2 prepare_seconds=0.010000"
        ),
        (
            "prefix TASK22_TARGETED_RETIRE event=commit "
            f"publication_id={publication_id} target_version={version} expired_requests=2"
        ),
    ]


def _snapshot(step: int, *, on_device: bool = True) -> str:
    return (
        "prefix TASK22_WEIGHT_SNAPSHOT "
        f"logical_step={step} on_device={str(on_device).lower()} "
        "local_tensors=398 local_bytes=4000000000 elapsed_seconds=0.120000"
    )


def _write_log(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "driver.log"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_analyzer_accepts_cold_init_and_steady_four_gpu_dcs_markers(tmp_path) -> None:
    rows = [_snapshot(-1), *_retirement(-1), _marker(-1, reused=False)]
    for step in range(11):
        rows.extend([_snapshot(step), *_retirement(step), _marker(step)])

    result = analyze(_write_log(tmp_path, rows))

    assert result["verdict"] == "PASS"
    assert result["errors"] == []
    assert result["init"][0]["group_reused"] is False
    assert result["headline_summary"]["client_total_seconds"]["count"] == 8
    assert result["headline_summary"]["snapshot_plus_client_seconds"]["mean"] == 0.55
    assert result["headline_summary"]["source_h2d_bytes"]["sum"] == 0
    assert (
        result["headline_summary"]["fanout_bytes"]["sum"] == 2 * result["headline_summary"]["broadcast_bytes"]["sum"]
    )


def test_analyzer_rejects_missing_gpu_snapshot(tmp_path) -> None:
    rows = [_snapshot(-1), *_retirement(-1), _marker(-1, reused=False)]
    for step in range(11):
        rows.extend([_snapshot(step), *_retirement(step), _marker(step, h2d_bytes=1024 if step == 4 else 0)])

    result = analyze(_write_log(tmp_path, rows))

    assert result["verdict"] == "INVALID"
    assert "gpu_snapshot_missed" in "\n".join(result["errors"])


def test_analyzer_rejects_wrong_dcs_topology(tmp_path) -> None:
    rows = [_snapshot(-1), *_retirement(-1), _marker(-1, reused=False)]
    for step in range(11):
        rows.extend([_snapshot(step), *_retirement(step), _marker(step, world_size=2 if step == 3 else 3)])

    result = analyze(_write_log(tmp_path, rows))

    assert result["verdict"] == "INVALID"
    assert "expected=3/2" in "\n".join(result["errors"])


def test_analyzer_rejects_missing_training_step(tmp_path) -> None:
    rows = [_snapshot(-1), *_retirement(-1), _marker(-1, reused=False)]
    for step in range(11):
        rows.append(_snapshot(step))
        rows.extend(_retirement(step))
        if step != 8:
            rows.append(_marker(step))

    result = analyze(_write_log(tmp_path, rows))

    assert result["verdict"] == "INVALID"
    assert "missing=[8]" in "\n".join(result["errors"])


def test_analyzer_rejects_cpu_snapshot_even_when_dcs_h2d_is_zero(tmp_path) -> None:
    rows = [_snapshot(-1), *_retirement(-1), _marker(-1, reused=False)]
    for step in range(11):
        rows.extend([_snapshot(step, on_device=step != 5), *_retirement(step), _marker(step)])

    result = analyze(_write_log(tmp_path, rows))

    assert result["verdict"] == "INVALID"
    assert "snapshot_not_on_device" in "\n".join(result["errors"])


def test_analyzer_rejects_unexercised_targeted_retirement(tmp_path) -> None:
    rows = [_snapshot(-1), *_retirement(-1), _marker(-1, reused=False)]
    for step in range(11):
        rows.extend([_snapshot(step), *_retirement(step), _marker(step, expired_requests=0)])

    result = analyze(_write_log(tmp_path, rows))

    assert result["verdict"] == "INVALID"
    assert "targeted_expired_requests_not_exercised" in result["errors"]
