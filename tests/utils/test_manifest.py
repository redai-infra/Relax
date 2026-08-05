# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the compact experiment manifest implementation."""

import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from relax.utils import manifest


class _FakeRay:
    def __init__(self, node_count: int) -> None:
        self.node_count = node_count

    def is_initialized(self) -> bool:
        return True

    def nodes(self) -> list:
        return [
            {
                "Alive": True,
                "IsHeadNode": index == 0,
                "NodeManagerAddress": f"10.0.0.{index + 1}",
                "Resources": {"CPU": 8, "GPU": 1, "accelerator_type:A100": 1, f"node:10.0.0.{index + 1}": 1},
            }
            for index in range(self.node_count)
        ]

    def cluster_resources(self) -> dict:
        return {"CPU": 8 * self.node_count, "GPU": self.node_count, "node:10.0.0.1": 1}


def test_sanitizer_covers_reviewed_secret_shapes() -> None:
    """Nested JSON, credentials, internal hosts and private IPv6 never
    survive."""
    raw = {
        "argv": [
            "--runtime-env-json",
            '{"HF_TOKEN":"super-secret-value"}',
            "--header",
            "Authorization: Bearer token-abc-123",
        ],
        "database": "postgresql://alice:s3cr3t@example.com/db",
        "basic": "Authorization: Basic dXNlcjpwYXNz",
        "quoted": 'PASSWORD="top secret"',
        "ray": "ray-head.prod.internal:6379",
        "bare_ray": "ray-head:6379",
        "ipv6": "fd00::1234",
        "patch": "+ HF_TOKEN=patch-secret-value",
        "tokenizer": "qwen-tokenizer",
    }

    sanitized = manifest.sanitize_value(raw)
    serialized = json.dumps(sanitized)

    for leaked in (
        "super-secret-value",
        "token-abc-123",
        "s3cr3t",
        "ray-head.prod.internal",
        "ray-head:6379",
        "fd00::1234",
        "patch-secret-value",
        "dXNlcjpwYXNz",
        "top secret",
    ):
        assert leaked not in serialized
    assert sanitized["tokenizer"] == "qwen-tokenizer"
    assert manifest.sanitize_value("ray-head", "master_addr") == "<internal-host>"
    assert manifest.sanitize_value("ray-head", "ray_head_ip") == "<internal-host>"
    assert manifest.sanitize_value("ray-head", "node_address") == "<internal-host>"
    assert manifest.sanitize_value("ray-head", "NodeManagerAddress") == "<internal-host>"
    assert manifest.sanitize_value(["ray-head"], "host") == "<internal-host>"
    assert manifest.sanitize_value("ray-head", "worker_hostname") == "<internal-host>"
    assert manifest.sanitize_value(["ray-head:8000"], "rollout_external_engine_addrs") == "<internal-host>"
    assert manifest.sanitize_value("alice@example.com", "email_address") == "alice@example.com"
    manifest.verify_no_secrets(sanitized)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_sanitizer_converts_non_finite_floats_to_standard_json(value: float) -> None:
    sanitized = manifest.sanitize_value({"value": value})

    assert sanitized["value"].startswith("<non-finite:")
    json.dumps(sanitized, allow_nan=False)


def test_sanitize_argv_understands_separate_and_inline_flags() -> None:
    argv = manifest.sanitize_argv(
        [
            "python",
            "train.py",
            "--api-token",
            "secret-one",
            "--password=secret-two",
            "--head-node",
            "ray-head",
            "--ray-address=ray-head:6379",
        ]
    )

    assert argv == [
        "python",
        "train.py",
        "--api-token",
        manifest.REDACTED,
        f"--password={manifest.REDACTED}",
        "--head-node",
        "<internal-host>",
        "--ray-address=<internal-host>",
    ]


def test_canonical_argv_preserves_module_invocation() -> None:
    with patch.object(sys, "orig_argv", [sys.executable, "-m", "relax.entrypoints.train", "--help"]):
        assert manifest._canonical_argv() == ["python", "-m", "relax.entrypoints.train", "--help"]


@pytest.mark.parametrize("section", ["code", "command", "config", "environment", "hardware", "runtime"])
def test_normalize_rejects_malformed_sections(section: str) -> None:
    with pytest.raises(manifest.ManifestError, match="must be an object"):
        manifest.normalize_manifest({"schema_version": "1.0", section: []})


def test_normalize_rejects_null_command() -> None:
    with pytest.raises(manifest.ManifestError, match="must be an object"):
        manifest.normalize_manifest({"schema_version": "1.0", "command": None})


@pytest.mark.parametrize(
    "value, field",
    [({"code": {"relax": None}}, "code.relax"), ({"environment": {"packages": []}}, "environment.packages")],
)
def test_normalize_rejects_malformed_nested_sections(value: dict, field: str) -> None:
    with pytest.raises(manifest.ManifestError, match=re.escape(field)):
        manifest.normalize_manifest({"schema_version": "1.0", **value})


def test_normalize_accepts_future_v1_minor_and_legacy_cli_args() -> None:
    normalized = manifest.normalize_manifest({"schema_version": "1.7", "cli_args": ["python", "train.py"]})

    assert normalized["schema_version"] == "1.0"
    assert normalized["command"]["argv"] == ["python", "train.py"]


@pytest.mark.parametrize("version", [None, 1, "1", "1.future", "v1.0"])
def test_normalize_rejects_malformed_schema_versions(version: object) -> None:
    with pytest.raises(manifest.ManifestError, match="Invalid schema version"):
        manifest.normalize_manifest({"schema_version": version})


def test_documented_schema_matches_runtime_v1_contract() -> None:
    schema_path = Path(manifest.__file__).resolve().parents[2] / "docs/schema/experiment-manifest-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    version_pattern = schema["properties"]["schema_version"]["pattern"]
    assert re.fullmatch(version_pattern, manifest.SCHEMA_VERSION)
    assert re.fullmatch(version_pattern, "1.7")
    assert not re.fullmatch(version_pattern, "2.0")
    assert set(schema["required"]) == {"schema_version", "run_id", "generated_at", "command"}
    assert schema["additionalProperties"] is True


@pytest.mark.parametrize(
    "node_count, expected_mode",
    [(0, "ray_initialized_unknown"), (1, "single_node_ray"), (2, "multi_node_ray")],
)
def test_ray_runtime_modes_exclude_node_identity(node_count: int, expected_mode: str) -> None:
    with patch.dict(sys.modules, {"ray": _FakeRay(node_count)}):
        runtime = manifest._collect_runtime()

    assert runtime["mode"] == expected_mode
    assert runtime["node_count"] == node_count
    serialized = json.dumps(runtime)
    assert "NodeManagerAddress" not in serialized
    assert "node:10.0.0" not in serialized
    assert ("accelerator_type:A100" in serialized) is (node_count > 0)
    if node_count:
        assert runtime["nodes"][0]["role"] == "head"
        assert all(node["role"] == "worker" for node in runtime["nodes"][1:])


def test_training_metadata_has_inputs_and_parallel_topology() -> None:
    args = SimpleNamespace(
        hf_checkpoint="/home/alice/models/qwen",
        tokenizer_model="/home/alice/models/tokenizer",
        prompt_data=["train", "/lustre/home/alice/data/train.jsonl"],
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        global_batch_size=64,
        micro_batch_size=2,
        algorithm="grpo",
    )

    collected = manifest.collect_manifest(args=args, argv=["python", "train.py"], include_runtime=False)

    assert collected["inputs"]["model"] == "<home>/models/qwen"
    assert collected["inputs"]["tokenizer"] == "<home>/models/tokenizer"
    assert collected["inputs"]["dataset"] == ["train", "<home>/data/train.jsonl"]
    assert collected["training"]["parallel_topology"] == {
        "tensor": 2,
        "pipeline": 4,
        "context": 1,
        "expert": 8,
    }
    assert "runtime" not in collected


def test_run_identity_is_anonymized() -> None:
    with (
        patch("socket.gethostname", return_value="gpu-node-07.prod.internal"),
        patch.dict("os.environ", {"USER": "alice"}, clear=True),
    ):
        collected = manifest.collect_manifest(argv=["python", "train.py"], include_runtime=False)

    assert collected["user"] == "<user>"
    assert collected["hostname"] == "<host>"


def test_megatron_git_is_discovered_from_pythonpath(tmp_path: Path) -> None:
    (tmp_path / "megatron").mkdir()
    with (
        patch.dict("os.environ", {"PYTHONPATH": str(tmp_path)}, clear=True),
        patch.object(manifest, "_git_repository", side_effect=lambda path: {"path": str(path)}),
    ):
        code = manifest._collect_code()

    assert code["megatron"]["path"] == str(tmp_path)


def test_relax_git_is_discovered_from_source_outside_repository_cwd(tmp_path: Path) -> None:
    source_root = Path(manifest.__file__).resolve().parents[2]

    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(manifest, "_git_repository", return_value={"commit": "source-commit"}) as repository,
        patch("pathlib.Path.cwd", return_value=tmp_path),
    ):
        code = manifest._collect_code()

    assert code["relax"] == {"commit": "source-commit"}
    repository.assert_called_once_with(source_root)


def test_environment_cuda_fallback_without_importing_torch() -> None:
    with (
        patch.dict(sys.modules, {"torch": None}),
        patch.object(manifest, "_run", return_value="Driver Version: 580.0 CUDA Version: 12.9"),
    ):
        environment = manifest._collect_environment()

    assert environment["cuda_version"] == "12.9"


def test_input_metadata_hashes_only_small_files(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt":"hello"}', encoding="utf-8")

    inputs = manifest._collect_inputs(SimpleNamespace(prompt_data=str(dataset)))

    assert inputs["dataset_metadata"]["size_bytes"] == dataset.stat().st_size
    assert len(inputs["dataset_metadata"]["sha256"]) == 64


def test_input_metadata_uses_one_bounded_probe_for_lists() -> None:
    metadata = json.dumps([{"size_bytes": None} for _ in range(32)])
    with patch.object(manifest, "_run", return_value=metadata) as run:
        details = manifest._input_metadata([f"dataset-{index}" for index in range(100)])

    run.assert_called_once()
    assert details[-1] == {"truncated_items": 68}


def test_local_runtime_mode() -> None:
    with patch.dict(sys.modules, {"ray": None}):
        assert manifest._collect_runtime()["mode"] == "local"


def test_ray_node_role_supports_current_ray_resource_marker() -> None:
    head = {"Resources": {"CPU": 8, "node:head-marker": 1, "node:__internal_head__": 1}}
    worker = {"Resources": {"CPU": 8, "node:worker-marker": 1}}

    assert manifest._ray_node_role(head) == "head"
    assert manifest._ray_node_role(worker) == "worker"


def test_diff_ignores_run_identity() -> None:
    old = {"run_id": "one", "generated_at": "old", "config": {"arguments": {"lr": 1e-5}}}
    new = {"run_id": "two", "generated_at": "new", "config": {"arguments": {"lr": 2e-5}}}

    assert manifest.diff_manifests(old, new) == [{"field": "config.arguments.lr", "old": 1e-5, "new": 2e-5}]
    assert "| `config.arguments.lr` |" in manifest._render_diff(manifest.diff_manifests(old, new))


def test_validation_reports_environment_drift() -> None:
    recorded = {"schema_version": "1.0", "environment": {"python_version": "3.12.0"}}
    current = {
        "schema_version": "1.0",
        "environment": {"python_version": "3.11.0"},
        "command": {"argv": []},
    }

    with patch.object(manifest, "collect_manifest", return_value=current):
        report = manifest.validate_environment(recorded)

    assert report["status"] == "FAIL"
    assert report["match_status"] == "DIFF"
    assert report["differences"][0]["field"] == "environment.python_version"


def test_validation_reports_match_and_missing() -> None:
    recorded = {"schema_version": "1.0", "environment": {"python_version": "3.12.0"}}
    matching = {
        "schema_version": "1.0",
        "environment": {"python_version": "3.12.0"},
        "command": {"argv": []},
    }
    missing = {"schema_version": "1.0", "environment": {}, "command": {"argv": []}}

    with patch.object(manifest, "collect_manifest", return_value=matching):
        assert manifest.validate_environment(recorded)["match_status"] == "MATCH"
    with patch.object(manifest, "collect_manifest", return_value=missing):
        assert manifest.validate_environment(recorded)["match_status"] == "MISSING"


def test_validation_skips_unavailable_ray_runtime() -> None:
    recorded = {"schema_version": "1.0", "runtime": {"mode": "multi_node_ray", "node_count": 2}}
    current = {"schema_version": "1.0", "runtime": {"mode": "local", "node_count": 1}, "command": {"argv": []}}

    with patch.object(manifest, "collect_manifest", return_value=current):
        report = manifest.validate_environment(recorded)

    assert report["status"] == "WARN"
    assert report["match_status"] == "MISSING"


def test_validation_reports_unobservable_runtime_environment() -> None:
    recorded = {
        "schema_version": "1.0",
        "config": {"runtime_env": {"env_vars": {"TQ_PRE_ALLOC_SAMPLE_NUM": "16"}}},
    }
    current = {
        "schema_version": "1.0",
        "command": {"argv": []},
        "config": {"environment": {}},
    }

    with patch.object(manifest, "collect_manifest", return_value=current):
        report = manifest.validate_environment(recorded)

    assert report["status"] == "WARN"
    assert report["match_status"] == "MISSING"
    assert report["differences"][0]["field"] == "runtime_environment_observation.TQ_PRE_ALLOC_SAMPLE_NUM"


def test_validation_ignores_its_own_untracked_manifest() -> None:
    output = Path.cwd() / "manifest_for_validation.json"
    recorded = {
        "schema_version": "1.0",
        "code": {"relax": {"dirty": False, "tracked_dirty": False, "untracked_files": []}},
    }
    current = {
        "schema_version": "1.0",
        "command": {"argv": []},
        "code": {
            "relax": {
                "dirty": True,
                "tracked_dirty": False,
                "untracked_files": [output.name],
            }
        },
    }

    with (
        patch.object(manifest, "collect_manifest", return_value=current),
        patch.object(manifest, "_run", return_value=str(Path.cwd())),
    ):
        report = manifest.validate_environment(recorded, str(output))

    assert report["match_status"] == "MATCH"


def test_validation_warns_for_compatible_patch_version() -> None:
    recorded = {"schema_version": "1.0", "environment": {"python_version": "3.12.1"}}
    current = {
        "schema_version": "1.0",
        "environment": {"python_version": "3.12.9"},
        "command": {"argv": []},
    }

    with (
        patch.object(manifest, "collect_manifest", return_value=current),
        patch.object(manifest, "load_manifest", return_value=recorded),
    ):
        report = manifest.validate_environment(recorded)
        exit_code = manifest.main(["check", "ignored.json"])

    assert report["status"] == "WARN"
    assert report["match_status"] == "DIFF"
    assert exit_code == 0


def test_reproduction_script_is_complete_and_rejects_redacted_commands() -> None:
    recorded = {
        "schema_version": "1.0",
        "command": {"argv": ["python", "-m", "relax.entrypoints.train", "--help"]},
        "code": {"relax": {"commit": "abc123"}},
        "environment": {"packages": {"torch": "2.11.0"}},
        "config": {
            "environment": {
                "CUDA_VISIBLE_DEVICES": "0",
                "API_TOKEN": "plain-secret",
                "RAY_ADDRESS": "<internal-host>",
                "PATH": "/tmp/untrusted-bin",
                "LD_PRELOAD": "/tmp/untrusted.so",
                "TQ_PRE_ALLOC_SAMPLE_NUM": "16",
                "PYTHONPATH": "<home>/Megatron-LM:<home>/Relax",
                "PYTHONBUFFERED": "16",
                "PYTHONHOME": "/tmp/untrusted-python",
            }
        },
    }

    script = manifest.build_reproduction_script(recorded, "manifest.json")

    assert "relax.utils.manifest validate manifest.json" in script
    assert "git checkout abc123" in script
    assert "torch==2.11.0" in script
    assert "export CUDA_VISIBLE_DEVICES=0" in script
    assert "export API_TOKEN" not in script
    assert "export RAY_ADDRESS" not in script
    assert "export PATH" not in script
    assert "export LD_PRELOAD" not in script
    assert "export TQ_PRE_ALLOC_SAMPLE_NUM=16" in script
    assert "export PYTHONPATH=" in script
    assert "export PYTHONBUFFERED=16" in script
    assert "export PYTHONHOME" not in script
    assert "python -m relax.entrypoints.train --help" in script
    recorded["command"]["argv"].append("<home>/models/qwen")
    assert '"${HOME}"/models/qwen' in manifest.build_reproduction_script(recorded, "manifest.json")
    recorded["command"]["argv"].pop()
    recorded["command"]["argv"].append(manifest.REDACTED)
    with pytest.raises(manifest.ManifestError, match="redacted"):
        manifest.build_reproduction_script(recorded, "manifest.json")
    recorded["command"]["argv"][-1] = "bad\x00argument"
    with pytest.raises(manifest.ManifestError, match="redacted"):
        manifest.build_reproduction_script(recorded, "manifest.json")


@pytest.mark.parametrize(
    "value",
    [":", "<home>/Relax:", ":<home>/Relax", "<home>/Relax::<home>/Megatron-LM", "<home>/../../tmp/evil"],
)
def test_reproduction_rejects_unsafe_pythonpath(value: str) -> None:
    recorded = {
        "schema_version": "1.0",
        "command": {"argv": ["python", "-c", "pass"]},
        "config": {"environment": {"PYTHONPATH": value}},
    }

    assert "export PYTHONPATH" not in manifest.build_reproduction_script(recorded, "manifest.json")


def test_default_output_paths_are_unique(tmp_path: Path) -> None:
    args = SimpleNamespace(tensorboard_dir=str(tmp_path))
    base = {"schema_version": "1.0", "command": {"argv": []}}
    first = {**base, "run_id": "run-one"}
    second = {**base, "run_id": "run-two"}

    with patch.object(manifest, "collect_manifest", side_effect=[first, second]):
        first_path = manifest.collect_and_save_manifest(args)
        second_path = manifest.collect_and_save_manifest(args)

    assert first_path != second_path
    assert Path(first_path).exists()
    assert Path(second_path).exists()


def test_runtime_update_preserves_run_and_adds_ray_topology(tmp_path: Path) -> None:
    args = SimpleNamespace(tensorboard_dir=str(tmp_path), tensor_model_parallel_size=2)
    path = manifest.collect_and_save_manifest(args, include_runtime=False)

    assert path is not None
    assert "runtime" not in json.loads(Path(path).read_text(encoding="utf-8"))
    with patch.dict(sys.modules, {"ray": _FakeRay(2)}):
        assert manifest.update_manifest_runtime(path, args) == path

    updated = json.loads(Path(path).read_text(encoding="utf-8"))
    assert updated["runtime"]["mode"] == "multi_node_ray"
    assert updated["runtime"]["parallel_topology"]["tensor"] == 2


def test_issue_cli_entrypoint_uses_manifest_cli() -> None:
    from relax.entrypoints import reproducibility

    assert reproducibility.main is manifest.main


def test_cli_returns_controlled_error_for_malformed_manifest(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":"1.0","command":null}', encoding="utf-8")

    assert manifest.main(["reproduce", str(path), "--dry-run"]) == 2
    path.write_text('{"schema_version":"1.0","code":{"relax":null}}', encoding="utf-8")
    assert manifest.main(["reproduce", str(path), "--dry-run"]) == 2
