# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from relax.utils.reproducibility import (
    REDACTED,
    ManifestError,
    build_manifest,
    compare_environment,
    load_manifest,
    replay_command,
    sanitize_argv,
    write_experiment_manifest,
    write_manifest,
)


_PRIVATE_IP_A = ".".join(("10", "0", "0", "1"))
_PRIVATE_IP_B = ".".join(("10", "0", "0", "2"))
_PRIVATE_IP_C = ".".join(("10", "0", "0", "8"))


class _FakeRay:
    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def nodes():
        return [
            {
                "Alive": True,
                "NodeID": "node-a",
                "NodeManagerAddress": _PRIVATE_IP_A,
                "Resources": {"CPU": 8, "GPU": 4, f"node:{_PRIVATE_IP_A}": 1},
            },
            {
                "Alive": True,
                "NodeID": "node-b",
                "NodeManagerAddress": _PRIVATE_IP_B,
                "Resources": {"CPU": 8, "GPU": 4, "accelerator_type:L40": 1},
            },
        ]

    @staticmethod
    def cluster_resources():
        return {"CPU": 16, "GPU": 8, f"node:{_PRIVATE_IP_A}": 1}


class _SingleNodeRay(_FakeRay):
    @staticmethod
    def nodes():
        return _FakeRay.nodes()[:1]


def test_manifest_redacts_secrets_addresses_and_ray_node_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTAINER_IMAGE", "registry.corp.internal/team/relax:latest")
    args = SimpleNamespace(
        api_token="plain-secret-token",
        tokenizer_path="/models/tokenizer.json",
        rollout_external_engine_addrs=[f"{_PRIVATE_IP_C}:8000"],
        nested={"password": "plain-password", "safe": "kept"},
        tensor_model_parallel_size=4,
        actor_num_nodes=2,
        actor_num_gpus_per_node=4,
    )
    runtime_env = {
        "env_vars": {
            "HF_TOKEN": "runtime-secret-token",
            "SERVICE_URL": "ray://head.internal:10001",
            "SAFE_SETTING": "kept",
        }
    }
    manifest = build_manifest(
        args,
        runtime_env,
        argv=[
            "python",
            "train.py",
            "--api-token",
            "argv-secret-token",
            "--tokenizer-path=/models/tokenizer.json",
        ],
        cwd=tmp_path,
        ray_module=_FakeRay,
    )

    payload = json.dumps(manifest, sort_keys=True)
    for sensitive_value in (
        "plain-secret-token",
        "plain-password",
        "runtime-secret-token",
        "argv-secret-token",
        _PRIVATE_IP_A,
        _PRIVATE_IP_B,
        _PRIVATE_IP_C,
        "head.internal",
        "node-a",
        "node-b",
    ):
        assert sensitive_value not in payload
    assert "/models/tokenizer.json" in payload
    assert manifest["runtime"]["mode"] == "multi_node_ray"
    assert manifest["runtime"]["node_count"] == 2
    assert manifest["runtime"]["cluster_resources"]["GPU"] == 8
    assert manifest["config"]["arguments"]["api_token"] == REDACTED
    assert manifest["environment"]["container_image"] == f"{REDACTED}/team/relax:latest"


def test_sanitize_argv_does_not_confuse_tokenizer_with_token():
    argv = sanitize_argv(
        ["train.py", "--token", "secret", "--tokenizer-path", "/models/tokenizer", "--endpoint=ray://head:1"]
    )

    assert argv == [
        "train.py",
        "--token",
        REDACTED,
        "--tokenizer-path",
        "/models/tokenizer",
        f"--endpoint={REDACTED}",
    ]


def test_runtime_mode_covers_local_and_single_node_ray(tmp_path):
    local_manifest = build_manifest(cwd=tmp_path)
    ray_manifest = build_manifest(cwd=tmp_path, ray_module=_SingleNodeRay)

    assert local_manifest["runtime"]["mode"] == "local"
    assert local_manifest["runtime"]["node_count"] == 1
    assert ray_manifest["runtime"]["mode"] == "single_node_ray"
    assert ray_manifest["runtime"]["node_count"] == 1


def test_v1_reader_accepts_integer_and_future_minor_versions(tmp_path):
    for version in (1, "1.9"):
        path = tmp_path / f"manifest-{version}.json"
        path.write_text(json.dumps({"schema_version": version, "command": {"argv": ["true"]}}), encoding="utf-8")

        manifest = load_manifest(path)

        assert manifest["schema_version"] == str(version)
        assert manifest["runtime"] == {}


def test_reader_rejects_unknown_major_version(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "2.0"}), encoding="utf-8")

    with pytest.raises(ManifestError, match="Unsupported manifest schema"):
        load_manifest(path)


def test_write_manifest_is_atomic_and_replay_rejects_redacted_command(tmp_path):
    path = write_manifest({"schema_version": "1.0", "command": {"argv": ["python", "-V"]}}, tmp_path / "m.json")

    assert load_manifest(path)["command"]["argv"] == ["python", "-V"]
    with pytest.raises(ManifestError, match="redacted values"):
        replay_command({"schema_version": "1.0", "command": {"argv": ["python", REDACTED]}})


def test_compare_environment_reports_actionable_drift():
    expected = {
        "schema_version": "1.0",
        "code": {"commit": "abc"},
        "environment": {"python": "3.10.1", "packages": {"ray": "2.1"}},
    }
    actual = {
        "schema_version": "1.0",
        "code": {"commit": "def"},
        "environment": {"python": "3.11.1", "packages": {"ray": "2.2"}},
    }

    differences = compare_environment(expected, actual)

    assert {difference["field"] for difference in differences} == {
        "code.commit",
        "environment.python",
        "environment.packages",
    }
    assert all(difference["suggestion"] for difference in differences)


def test_training_hook_failure_does_not_escape(tmp_path):
    args = SimpleNamespace(tb_experiment_name="test")

    assert write_experiment_manifest(args, {}, path=tmp_path) is None
