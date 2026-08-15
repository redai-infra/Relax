# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build auditable Task 37 reproducibility evidence without inventing facts.

The command writes three deterministic JSON files:

``seed_mapping.json``
    The registered four-condition, three-seed plan.  Entries are explicitly
    labelled as planned runs and are not execution evidence.
``test_report.json``
    Counts parsed from JUnit XML plus content hashes for complete text logs.
    A text log is never interpreted as a passing result.
``reproducibility.manifest.json``
    Content hashes for supplied artifacts, the two generated files, declared
    source revisions, and versions observed from the current interpreter.

Unknown revisions remain JSON ``null`` with ``verification="not_claimed"``.
Operator-supplied revisions are labelled as declarations rather than as facts
verified by this generator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path


REPRODUCIBILITY_SCHEMA_VERSION = "task37-reproducibility-manifest-v1"
SEED_MAPPING_SCHEMA_VERSION = "task37-seed-mapping-v1"
TEST_REPORT_SCHEMA_VERSION = "task37-test-report-v1"

DECLARED_VERSION_FIELDS = (
    "relax_baseline_commit",
    "candidate_commit",
    "graphgpo_reference_commit",
    "container_image_digest",
    "model_revision",
    "alfworld_version",
)

RUNTIME_DISTRIBUTIONS = (
    ("relax", "relax"),
    ("torch", "torch"),
    ("megatron_core", "megatron-core"),
    ("ray", "ray"),
    ("sglang", "sglang"),
    ("transformers", "transformers"),
    ("alfworld", "alfworld"),
)

_COMMIT_FIELDS = frozenset(("relax_baseline_commit", "candidate_commit", "graphgpo_reference_commit"))
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")

_REGISTERED_CONDITIONS = (
    (
        "grpo",
        {
            "METHOD": "grpo",
            "BETA_EPISODE": "1",
        },
    ),
    (
        "gigpo",
        {
            "METHOD": "gigpo",
            "BETA": "1",
            "BETA_EPISODE": "1",
        },
    ),
    (
        "graphgpo",
        {
            "METHOD": "graphgpo",
            "BETA": "1",
            "BETA_EPISODE": "1",
        },
    ),
    (
        "graph_only",
        {
            "METHOD": "graphgpo",
            "BETA": "1",
            "BETA_EPISODE": "0",
        },
    ),
)


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(label: str, path: Path) -> dict[str, object]:
    if not isinstance(label, str) or not label.strip():
        raise ValueError("artifact label must be a non-empty string")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"artifact must be a regular file: {path}")
    return {
        "label": label.strip(),
        "file_name": resolved.name,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validate_declared_version(field: str, value: object) -> str:
    if field not in DECLARED_VERSION_FIELDS:
        raise ValueError(f"unknown version field {field!r}; expected one of {list(DECLARED_VERSION_FIELDS)!r}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if field in _COMMIT_FIELDS and _FULL_COMMIT.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a full lowercase 40-character commit")
    if field == "container_image_digest" and _IMAGE_DIGEST.fullmatch(normalized) is None:
        raise ValueError("container_image_digest must be an image reference pinned by a sha256 digest")
    return normalized


def build_version_inventory(
    declared_versions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Record declarations separately from versions observed at runtime."""

    supplied = dict(declared_versions or {})
    unexpected = sorted(set(supplied) - set(DECLARED_VERSION_FIELDS))
    if unexpected:
        raise ValueError(f"unknown version fields: {unexpected!r}")

    declarations: dict[str, dict[str, str | None]] = {}
    for field in DECLARED_VERSION_FIELDS:
        if field not in supplied:
            declarations[field] = {
                "value": None,
                "source": None,
                "verification": "not_claimed",
            }
            continue
        declarations[field] = {
            "value": _validate_declared_version(field, supplied[field]),
            "source": "operator_supplied",
            "verification": "not_verified_by_generator",
        }

    packages: dict[str, dict[str, str | None]] = {}
    for label, distribution in RUNTIME_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[label] = {
                "value": None,
                "source": f"importlib.metadata:{distribution}",
                "verification": "not_installed_in_generator_runtime",
            }
        else:
            packages[label] = {
                "value": version,
                "source": f"importlib.metadata:{distribution}",
                "verification": "observed_in_generator_runtime",
            }

    return {
        "declared": declarations,
        "runtime_observed": {
            "python": {
                "value": platform.python_version(),
                "source": "platform.python_version",
                "verification": "observed_in_generator_runtime",
            },
            "operating_system": {
                "value": platform.platform(),
                "source": "platform.platform",
                "verification": "observed_in_generator_runtime",
            },
            "packages": packages,
            "scope_note": (
                "Runtime observations describe only the interpreter running this generator; "
                "they are not evidence for a remote training container."
            ),
        },
    }


def build_seed_mapping(
    seeds: Sequence[int] = (0, 1, 2),
    *,
    episode_weighting: str = "trajectory_once",
) -> dict[str, object]:
    """Return the registered 4 x N run grid without claiming it was
    executed."""

    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds must be a sequence of non-negative integers")
    normalized_seeds: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("every seed must be a non-negative integer")
        normalized_seeds.append(seed)
    if not normalized_seeds:
        raise ValueError("at least one seed is required")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must not contain duplicates")
    if episode_weighting not in {"trajectory_once", "reference_cross_steps"}:
        raise ValueError("unknown episode weighting")

    runs: list[dict[str, object]] = []
    for condition, condition_environment in _REGISTERED_CONDITIONS:
        for seed in normalized_seeds:
            environment = {
                **condition_environment,
                "EPISODE_WEIGHTING": episode_weighting,
                "SEED": str(seed),
            }
            runs.append(
                {
                    "run_id": f"{condition}-seed-{seed}",
                    "condition": condition,
                    "registered_seed": seed,
                    "launcher_environment": environment,
                    "execution_status": "planned",
                    "execution_evidence": None,
                }
            )

    return {
        "schema_version": SEED_MAPPING_SCHEMA_VERSION,
        "mapping_scope": "registered_plan_not_execution_evidence",
        "seed_semantics": {
            "declared_entrypoint": "launcher environment variable SEED",
            "declared_training_argument": "--seed",
            "qualification": (
                "The mapping records the registered launcher input only; it does not claim that "
                "every third-party component consumes that seed."
            ),
        },
        "runs": runs,
    }


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_junit_report(path: Path) -> dict[str, object]:
    """Parse outcomes from test cases instead of trusting suite totals."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"JUnit report must be a regular file: {path}")
    root = ET.parse(resolved).getroot()
    cases = [element for element in root.iter() if _xml_local_name(element.tag) == "testcase"]

    failures = 0
    errors = 0
    skipped = 0
    duration_seconds = 0.0
    for case in cases:
        raw_duration = case.attrib.get("time", "0")
        try:
            duration = float(raw_duration)
        except ValueError as exc:
            raise ValueError(f"invalid JUnit testcase duration {raw_duration!r}") from exc
        if duration < 0:
            raise ValueError("JUnit testcase duration must be non-negative")
        duration_seconds += duration

        child_names = {_xml_local_name(child.tag) for child in case}
        if "error" in child_names:
            errors += 1
        elif "failure" in child_names:
            failures += 1
        elif "skipped" in child_names:
            skipped += 1

    tests = len(cases)
    passed = tests - failures - errors - skipped
    if tests == 0:
        outcome = "empty"
    elif failures or errors:
        outcome = "failed"
    else:
        outcome = "passed"
    return {
        "tests": tests,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "duration_seconds": duration_seconds,
        "outcome": outcome,
    }


def build_test_report(
    junit_reports: Mapping[str, Path] | None = None,
    text_logs: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    junit_entries: list[dict[str, object]] = []
    for label, path in sorted((junit_reports or {}).items()):
        junit_entries.append(
            {
                "artifact": _artifact_record(label, path),
                "summary": parse_junit_report(path),
            }
        )

    log_entries: list[dict[str, object]] = []
    for label, path in sorted((text_logs or {}).items()):
        log_entries.append(
            {
                "artifact": _artifact_record(label, path),
                "outcome": "not_inferred_from_unstructured_log",
            }
        )

    junit_outcomes = [entry["summary"]["outcome"] for entry in junit_entries]
    if not junit_outcomes:
        overall_outcome = "not_evaluated"
    elif any(outcome == "failed" for outcome in junit_outcomes):
        overall_outcome = "failed"
    elif any(outcome == "empty" for outcome in junit_outcomes):
        overall_outcome = "incomplete"
    else:
        overall_outcome = "passed"
    return {
        "schema_version": TEST_REPORT_SCHEMA_VERSION,
        "overall_outcome_from_junit_only": overall_outcome,
        "junit_reports": junit_entries,
        "complete_text_logs": log_entries,
    }


def _write_idempotent(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file():
            raise ValueError(f"output path is not a regular file: {path}")
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace different evidence file: {path}")
        return
    path.write_bytes(content)


def write_reproducibility_bundle(
    output_dir: Path,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    episode_weighting: str = "trajectory_once",
    artifacts: Mapping[str, Path] | None = None,
    junit_reports: Mapping[str, Path] | None = None,
    text_logs: Mapping[str, Path] | None = None,
    declared_versions: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Write a seed plan, parsed test report, and content-addressed
    manifest."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError("output_dir must be a directory")

    seed_path = output_dir / "seed_mapping.json"
    test_path = output_dir / "test_report.json"
    manifest_path = output_dir / "reproducibility.manifest.json"

    seed_payload = build_seed_mapping(seeds, episode_weighting=episode_weighting)
    test_payload = build_test_report(junit_reports, text_logs)
    _write_idempotent(seed_path, _json_bytes(seed_payload))
    _write_idempotent(test_path, _json_bytes(test_payload))

    artifact_entries = [_artifact_record(label, path) for label, path in sorted((artifacts or {}).items())]
    manifest_payload = {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "claim_scope": {
            "seed_mapping": "plan_only",
            "test_results": "parsed_only_from_supplied_junit",
            "text_logs": "content_addressed_without_outcome_inference",
            "versions": "missing_values_are_not_claimed",
        },
        "versions": build_version_inventory(declared_versions),
        "input_artifacts": artifact_entries,
        "generated_artifacts": [
            _artifact_record("seed_mapping", seed_path),
            _artifact_record("test_report", test_path),
        ],
    }
    _write_idempotent(manifest_path, _json_bytes(manifest_payload))
    return {
        "seed_mapping": seed_path,
        "test_report": test_path,
        "manifest": manifest_path,
    }


def _assignments(values: Sequence[str], *, kind: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} must use LABEL=VALUE syntax: {value!r}")
        label, assigned_value = value.split("=", 1)
        label = label.strip()
        assigned_value = assigned_value.strip()
        if not label or not assigned_value:
            raise ValueError(f"{kind} must use non-empty LABEL=VALUE syntax")
        if label in result:
            raise ValueError(f"duplicate {kind} label: {label!r}")
        result[label] = assigned_value
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(
        "--episode-weighting",
        choices=("trajectory_once", "reference_cross_steps"),
        default="trajectory_once",
    )
    parser.add_argument("--artifact", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--junit", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--log", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument(
        "--version",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help=f"declared version; FIELD is one of {', '.join(DECLARED_VERSION_FIELDS)}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    artifact_values = _assignments(args.artifact, kind="artifact")
    junit_values = _assignments(args.junit, kind="junit")
    log_values = _assignments(args.log, kind="log")
    version_values = _assignments(args.version, kind="version")
    write_reproducibility_bundle(
        args.output_dir,
        seeds=args.seeds or (0, 1, 2),
        episode_weighting=args.episode_weighting,
        artifacts={label: Path(path) for label, path in artifact_values.items()},
        junit_reports={label: Path(path) for label, path in junit_values.items()},
        text_logs={label: Path(path) for label, path in log_values.items()},
        declared_versions=version_values,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
