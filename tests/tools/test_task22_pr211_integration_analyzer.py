# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "task22" / "analyze_pr211_integration_run.py"
SPEC = importlib.util.spec_from_file_location("task22_pr211_integration_analyzer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def _contract(tmp_path: Path, *, total_gpus: int, per_engine: int, interval: int = 1, kv: bool = True) -> Path:
    path = tmp_path / "run_contract.json"
    expected = {
        "rollout_total_gpus": total_gpus,
        "rollout_gpus_per_engine": per_engine,
        "rollout_receiver_count": total_gpus // per_engine,
        "group_world_size": 1 + total_gpus,
    }
    path.write_text(
        json.dumps(
            {
                "num_rollout": 3,
                "update_weights_interval": interval,
                "cross_version_kv": kv,
                "train_argv": [
                    "--resource",
                    json.dumps({"actor": [1, 2], "rollout": [1, total_gpus]}),
                    "--num-rollout",
                    "3",
                    "--update-weights-interval",
                    str(interval),
                    "--rollout-num-gpus-per-engine",
                    str(per_engine),
                    *(["--enable-cross-version-kv-continuation"] if kv else []),
                ],
                "expected_topology": expected,
            }
        ),
        encoding="utf-8",
    )
    return path


def _marker(step: int, *, world_size: int, receivers: int, reused: bool) -> str:
    return (
        "prefix DCS_WEIGHT_SYNC "
        f"logical_step={step} weight_version={step + 2} group_reused={str(reused).lower()} "
        f"group_world_size={world_size} rollout_receivers={receivers} trailing_metric=1"
    )


def _log(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "driver.log"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_analyzer_derives_single_gpu_engine_topology_from_run_contract(tmp_path) -> None:
    contract = _contract(tmp_path, total_gpus=4, per_engine=1)
    rows = [_marker(-1, world_size=5, receivers=4, reused=False)]
    rows.extend(_marker(step, world_size=5, receivers=4, reused=True) for step in range(3))

    result = analyzer.analyze(contract, _log(tmp_path, rows))

    assert result["verdict"] == "VALID"
    assert result["contract"]["expected_topology"]["rollout_receiver_count"] == 4
    assert result["contract"]["expected_topology"]["group_world_size"] == 5


def test_analyzer_derives_tensor_parallel_engine_count_not_raw_gpu_count(tmp_path) -> None:
    contract = _contract(tmp_path, total_gpus=4, per_engine=2, interval=2)
    rows = [
        _marker(-1, world_size=5, receivers=2, reused=False),
        _marker(1, world_size=5, receivers=2, reused=True),
        _marker(2, world_size=5, receivers=2, reused=True),
    ]

    result = analyzer.analyze(contract, _log(tmp_path, rows))

    assert result["verdict"] == "VALID"
    assert result["contract"]["expected_publication_steps"] == [-1, 1, 2]


def test_analyzer_rejects_receiver_loss_on_any_publication(tmp_path) -> None:
    contract = _contract(tmp_path, total_gpus=2, per_engine=1)
    rows = [_marker(-1, world_size=3, receivers=2, reused=False)]
    rows.extend(
        _marker(step, world_size=2 if step == 1 else 3, receivers=1 if step == 1 else 2, reused=True)
        for step in range(3)
    )

    result = analyzer.analyze(contract, _log(tmp_path, rows))

    assert result["verdict"] == "INVALID_INPUT"
    assert "logical_step=1:topology=2/1:expected=3/2" in result["errors"]


def test_analyzer_rejects_missing_publication_and_group_rebuild(tmp_path) -> None:
    contract = _contract(tmp_path, total_gpus=2, per_engine=1)
    rows = [
        _marker(-1, world_size=3, receivers=2, reused=False),
        _marker(0, world_size=3, receivers=2, reused=True),
        _marker(2, world_size=3, receivers=2, reused=False),
    ]

    result = analyzer.analyze(contract, _log(tmp_path, rows))

    assert result["verdict"] == "INVALID_INPUT"
    assert any(error.startswith("publication_step_coverage:") for error in result["errors"])
    assert "logical_step=2:group_not_reused" in result["errors"]


def test_kv_off_arm_is_valid_only_without_dcs_publications(tmp_path) -> None:
    contract = _contract(tmp_path, total_gpus=2, per_engine=1, kv=False)

    result = analyzer.analyze(contract, _log(tmp_path, []))

    assert result["verdict"] == "VALID"
    assert result["applicable"] is False


def test_analyzer_can_derive_legacy_contract_fields_from_train_argv(tmp_path) -> None:
    contract = _contract(tmp_path, total_gpus=2, per_engine=1)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    del payload["num_rollout"]
    del payload["update_weights_interval"]
    del payload["cross_version_kv"]
    contract.write_text(json.dumps(payload), encoding="utf-8")
    rows = [_marker(-1, world_size=3, receivers=2, reused=False)]
    rows.extend(_marker(step, world_size=3, receivers=2, reused=True) for step in range(3))

    result = analyzer.analyze(contract, _log(tmp_path, rows))

    assert result["verdict"] == "VALID"
    assert result["contract"]["expected_publication_steps"] == [-1, 0, 1, 2]
