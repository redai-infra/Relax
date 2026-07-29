# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "recipes" / "r2e-gym" / "eval_r2e_gym_maas.py"
SPEC = importlib.util.spec_from_file_location("eval_r2e_gym_maas", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _task() -> dict:
    return {
        "responses_create_params": {
            "metadata": {
                "instance_id": "owner__repo-commit",
                "instance_dict": {
                    "instance_id": "owner__repo-commit",
                },
            }
        }
    }


def test_maas_trial_payload_uses_custom_auth_headers():
    payload = MODULE._trial_payload(
        _task(),
        request_id="request-one",
        model_base_url="https://maas.example/v1/",
        model="tool-model",
        api_key="secret",
        user_email="user@example.com",
        app_id="app",
        max_tokens=1024,
        temperature=0.2,
        deadline_s=300,
        lease_s=60,
    )

    assert payload["model_endpoint"] == {
        "base_url": "https://maas.example/v1",
        "api_key": "secret",
        "model": "tool-model",
        "api_key_header": "api-key",
        "api_key_prefix": "",
        "headers": {
            "x-maas-user-email": "user@example.com",
            "x-maas-app-id": "app",
        },
    }
    assert payload["generation"]["sampling_params"] == {
        "max_tokens": 1024,
        "temperature": 0.2,
    }
    assert payload["metadata"]["capture_artifacts"] is True


def test_read_tasks_respects_limit_and_parses_string_instance_dict(tmp_path):
    task = _task()
    task["responses_create_params"]["metadata"].pop("instance_id")
    task["responses_create_params"]["metadata"]["instance_dict"] = json.dumps({"instance_id": "owner__repo-commit"})
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(
        "\n".join(json.dumps(task) for _ in range(3)) + "\n",
        encoding="utf-8",
    )

    tasks = MODULE._read_tasks(task_path, limit=2)

    assert len(tasks) == 2
    assert MODULE._instance_id(tasks[0]) == "owner__repo-commit"


def test_scalar_reward_accepts_protocol_reward_object():
    assert MODULE._scalar_reward({"reward": {"scalar": 0.75, "components": {"tests": 1}}}) == 0.75


def test_read_artifact_loads_persisted_manifest(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        json.dumps({"evaluation": {"patch_exists": False}, "response_output": []}),
        encoding="utf-8",
    )

    artifact = MODULE._read_artifact(str(artifact_path))

    assert artifact["evaluation"]["patch_exists"] is False
