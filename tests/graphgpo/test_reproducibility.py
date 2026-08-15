# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.graphgpo.reproducibility import (
    DECLARED_VERSION_FIELDS,
    build_seed_mapping,
    build_test_report,
    build_version_inventory,
    main,
    parse_junit_report,
)


BASELINE_COMMIT = "8a54679e971087566bfda939a5f75649e07fb861"


def _write_junit(path: Path, cases: str) -> None:
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?>\n<testsuites><testsuite>{cases}</testsuite></testsuites>\n',
        encoding="utf-8",
    )


def test_registered_seed_mapping_is_a_plan_for_all_twelve_runs():
    mapping = build_seed_mapping()

    assert mapping["mapping_scope"] == "registered_plan_not_execution_evidence"
    runs = mapping["runs"]
    assert len(runs) == 12
    assert len({run["run_id"] for run in runs}) == 12
    assert {run["condition"] for run in runs} == {
        "grpo",
        "gigpo",
        "graphgpo",
        "graph_only",
    }
    for condition in {run["condition"] for run in runs}:
        assert {run["registered_seed"] for run in runs if run["condition"] == condition} == {0, 1, 2}
    assert all(run["execution_status"] == "planned" for run in runs)
    assert all(run["execution_evidence"] is None for run in runs)
    assert all(run["launcher_environment"]["EPISODE_WEIGHTING"] == "trajectory_once" for run in runs)
    graph_only = next(run for run in runs if run["condition"] == "graph_only")
    assert graph_only["launcher_environment"] == {
        "METHOD": "graphgpo",
        "BETA": "1",
        "BETA_EPISODE": "0",
        "EPISODE_WEIGHTING": "trajectory_once",
        "SEED": "0",
    }


def test_junit_counts_are_parsed_but_plain_log_outcome_is_not_inferred(tmp_path):
    junit = tmp_path / "pytest.xml"
    _write_junit(
        junit,
        """
        <testcase classname="suite" name="pass" time="0.1" />
        <testcase classname="suite" name="skip" time="0.2"><skipped /></testcase>
        <testcase classname="suite" name="fail" time="0.3"><failure>bad</failure></testcase>
        <testcase classname="suite" name="error" time="0.4"><error>boom</error></testcase>
        """,
    )
    log = tmp_path / "pre-commit.log"
    log.write_text("Passed\n", encoding="utf-8")

    summary = parse_junit_report(junit)
    assert summary == {
        "tests": 4,
        "passed": 1,
        "failures": 1,
        "errors": 1,
        "skipped": 1,
        "duration_seconds": pytest.approx(1.0),
        "outcome": "failed",
    }
    report = build_test_report({"pytest": junit}, {"pre_commit": log})
    assert report["overall_outcome_from_junit_only"] == "failed"
    assert report["complete_text_logs"][0]["outcome"] == "not_inferred_from_unstructured_log"
    assert report["complete_text_logs"][0]["artifact"]["sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()


def test_cli_writes_content_addressed_bundle_without_fabricating_versions(tmp_path):
    evidence = tmp_path / "expanded_command.sh"
    evidence.write_text("python relax/entrypoints/train.py --seed 0\n", encoding="utf-8")
    junit = tmp_path / "pytest.xml"
    _write_junit(
        junit,
        '<testcase classname="suite" name="pass" time="0.125" />',
    )
    log = tmp_path / "pre-commit.log"
    log.write_text("ruff................................Passed\n", encoding="utf-8")
    output_dir = tmp_path / "bundle"

    argv = [
        "--output-dir",
        str(output_dir),
        "--artifact",
        f"expanded_command={evidence}",
        "--junit",
        f"pytest_graphgpo={junit}",
        "--log",
        f"pre_commit={log}",
        "--version",
        f"relax_baseline_commit={BASELINE_COMMIT}",
    ]
    assert main(argv) == 0
    assert main(argv) == 0

    seed_path = output_dir / "seed_mapping.json"
    test_path = output_dir / "test_report.json"
    manifest_path = output_dir / "reproducibility.manifest.json"
    assert seed_path.is_file()
    assert test_path.is_file()
    assert manifest_path.is_file()

    test_report = json.loads(test_path.read_text(encoding="utf-8"))
    assert test_report["overall_outcome_from_junit_only"] == "passed"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declarations = manifest["versions"]["declared"]
    assert set(declarations) == set(DECLARED_VERSION_FIELDS)
    assert declarations["relax_baseline_commit"] == {
        "value": BASELINE_COMMIT,
        "source": "operator_supplied",
        "verification": "not_verified_by_generator",
    }
    assert declarations["candidate_commit"] == {
        "value": None,
        "source": None,
        "verification": "not_claimed",
    }
    assert manifest["input_artifacts"] == [
        {
            "label": "expanded_command",
            "file_name": evidence.name,
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "size_bytes": evidence.stat().st_size,
        }
    ]
    generated = {entry["label"]: entry for entry in manifest["generated_artifacts"]}
    assert generated["seed_mapping"]["sha256"] == hashlib.sha256(seed_path.read_bytes()).hexdigest()
    assert generated["test_report"]["sha256"] == hashlib.sha256(test_path.read_bytes()).hexdigest()

    serialized_bundle = "".join(path.read_text(encoding="utf-8") for path in (seed_path, test_path, manifest_path))
    assert str(tmp_path) not in serialized_bundle


def test_version_inventory_rejects_values_that_look_verified_but_are_ambiguous():
    with pytest.raises(ValueError, match="full lowercase 40-character commit"):
        build_version_inventory({"candidate_commit": "deadbeef"})
    with pytest.raises(ValueError, match="pinned by a sha256 digest"):
        build_version_inventory({"container_image_digest": "relax:latest"})
