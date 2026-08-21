# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import os
import subprocess
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_DIR.parents[1]


def test_shared_scripts_remain_under_example_scripts() -> None:
    scripts_dir = EXAMPLE_DIR / "scripts"
    expected = {
        "run-qwen3-4B-8xgpu-nemo-gym.sh",
        "run_agent_app.sh",
        "run_gateway.sh",
        "run_training.sh",
        "convert_dataset.py",
    }

    assert expected <= {path.name for path in scripts_dir.iterdir()}


def test_each_recipe_owns_its_scripts() -> None:
    expected = {
        "calendar": {
            "prepare_calendar.sh",
            "run-qwen3-4B-8xgpu-nemo-gym-calendar.sh",
            "start_calendar_gym.sh",
            "verify_calendar.py",
        },
        "gsm8k": {
            "prepare_gsm8k.sh",
            "run-qwen3-4B-8xgpu-nemo-gym.sh",
            "start_gsm8k_gym.sh",
            "verify_gsm8k.py",
        },
        "workplace-assistant": {
            "prepare_workplace_assistant.sh",
            "run-qwen3-4B-2xgpu-nemo-gym-workplace.sh",
            "run-qwen3-4B-8xgpu-nemo-gym-workplace.sh",
            "start_workplace_assistant_gym.sh",
            "start_workplace_assistant_gym_remote.sh",
            "verify_workplace_assistant.py",
            "verify_workplace_assistant_trial.py",
        },
        "r2e-gym": {
            "prepare_r2e_gym.py",
            "prepare_r2e_gym.sh",
            "run-qwen35-9B-8xgpu-nemo-gym-r2e.sh",
            "start_r2e_gym_local.sh",
            "start_r2e_gym_remote.sh",
            "submit_r2e_gym.sh",
            "verify_r2e_gym_trial.py",
        },
    }

    for recipe_name, expected_files in expected.items():
        recipe_dir = EXAMPLE_DIR / "recipes" / recipe_name
        assert expected_files <= {path.name for path in recipe_dir.iterdir()}


def test_generic_recipe_local_entrypoint_exists() -> None:
    entrypoint = EXAMPLE_DIR / "scripts" / "../../../scripts/entrypoint/local.sh"

    assert entrypoint.resolve() == REPO_ROOT / "scripts/entrypoint/local.sh"
    assert entrypoint.is_file()


def test_each_recipe_has_chinese_runbook_and_pitfall_record() -> None:
    recipes_dir = EXAMPLE_DIR / "recipes"

    for recipe_name in ("calendar", "gsm8k", "workplace-assistant", "r2e-gym"):
        recipe_dir = recipes_dir / recipe_name
        readme = (recipe_dir / "README.md").read_text(encoding="utf-8")
        pitfail = (recipe_dir / "PITFAIL.md").read_text(encoding="utf-8")

        assert "从零" in readme or "准备变量" in readme
        assert "PITFAIL.md" in readme
        assert "踩坑" in pitfail


def test_each_recipe_training_script_configures_tracking_names() -> None:
    training_scripts = sorted((EXAMPLE_DIR / "recipes").glob("*/run-*.sh"))

    assert len(training_scripts) == 5
    for script in training_scripts:
        content = script.read_text()
        assert 'PROJECT_NAME="${PROJECT_NAME:-Relax/dev/nemo-gym}"' in content
        assert 'EXP_NAME="${EXP_NAME:-' in content

    tracking_implementations = [
        EXAMPLE_DIR / "scripts" / "run-qwen3-4B-8xgpu-nemo-gym.sh",
        EXAMPLE_DIR / "recipes" / "r2e-gym" / "run-qwen35-9B-8xgpu-nemo-gym-r2e.sh",
        EXAMPLE_DIR / "recipes" / "workplace-assistant" / "run-qwen3-4B-2xgpu-nemo-gym-workplace.sh",
    ]
    for script in tracking_implementations:
        content = script.read_text()
        assert "--use-clearml" in content
        assert "--use-metrics-service" in content
        assert '--tb-project-name "${PROJECT_NAME}"' in content
        assert '--tb-experiment-name "${EXP_NAME}-${now}"' in content


def test_workplace_training_uses_shared_checkout_without_runtime_upload() -> None:
    shared_script = (EXAMPLE_DIR / "scripts" / "run-qwen3-4B-8xgpu-nemo-gym.sh").read_text()
    two_gpu_script = (
        EXAMPLE_DIR / "recipes" / "workplace-assistant" / "run-qwen3-4B-2xgpu-nemo-gym-workplace.sh"
    ).read_text()
    workplace_readme = (EXAMPLE_DIR / "recipes" / "workplace-assistant" / "README.md").read_text()
    gsm8k_readme = (EXAMPLE_DIR / "recipes" / "gsm8k" / "README.md").read_text()

    assert 'RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.."' in shared_script
    assert '--agent-cwd "${RELAX_ROOT}"' in shared_script
    assert '-- bash "${RUN_TRAINING_SCRIPT}"' in shared_script
    assert "--working-dir" not in shared_script
    assert 'DASHBOARD_ADDRESS="http://${BASH_REMATCH[1]}:${RAY_DASHBOARD_PORT}"' in shared_script
    assert '--address="${DASHBOARD_ADDRESS}"' in shared_script
    assert "export WORKING_DIR" not in workplace_readme
    assert 'RELAX_SUBMISSION_ID="relax-nemo-workplace-smoke-001"' not in workplace_readme
    assert "export WORKING_DIR" not in gsm8k_readme
    assert 'RELAX_SUBMISSION_ID="relax-nemo-gsm8k-smoke-001"' not in gsm8k_readme
    assert '--resource \'{"actor":[1,2],"rollout":[1,2]}\'' in two_gpu_script
    assert "--tensor-model-parallel-size 2" in two_gpu_script
    assert "--rollout-num-gpus-per-engine 2" in two_gpu_script
    assert "--num-rollout 1" in two_gpu_script
    assert "--n-samples-per-prompt 2" in two_gpu_script
    assert "--global-batch-size 2" in two_gpu_script
    assert "--rollout-max-context-len 8192" in two_gpu_script
    assert "exec bash" not in two_gpu_script


def test_shared_nemo_gym_training_enforces_context_budget() -> None:
    shared_script = (EXAMPLE_DIR / "scripts" / "run-qwen3-4B-8xgpu-nemo-gym.sh").read_text()

    assert "NEMO_GYM_ROLLOUT_MAX_CONTEXT_LEN - 1" in shared_script
    assert '--rollout-max-prompt-len "${NEMO_GYM_ROLLOUT_MAX_PROMPT_LEN}"' in shared_script
    assert "NEMO_GYM_ROLLOUT_MAX_PROMPT_LEN must be smaller" in shared_script
    assert "NEMO_GYM_ROLLOUT_MAX_RESPONSE_LEN cannot exceed" in shared_script
    assert "NEMO_GYM_MAX_TOKENS_PER_GPU must be an integer no smaller" in shared_script
    assert 'NEMO_GYM_GATEWAY_PORT="${NEMO_GYM_GATEWAY_PORT:-28100}"' in shared_script
    assert 'GATEWAY_URL="http://${GYM_HOST}:${NEMO_GYM_GATEWAY_PORT}"' in shared_script


def test_workplace_cleanup_contract_is_wired_into_recipe() -> None:
    recipe_dir = EXAMPLE_DIR / "recipes" / "workplace-assistant"
    start_script = (recipe_dir / "start_workplace_assistant_gym.sh").read_text()
    remote_script = (recipe_dir / "start_workplace_assistant_gym_remote.sh").read_text()
    trial_verifier = (recipe_dir / "verify_workplace_assistant_trial.py").read_text()
    cleanup_patch = (EXAMPLE_DIR / "service" / "patches" / "workplace_assistant_cleanup.patch").read_text()

    assert 'WORKPLACE_ASSISTANT_PORT_BASE="${WORKPLACE_ASSISTANT_PORT_BASE:-29000}"' in start_script
    assert 'GYM_RAY_PORT="${GYM_RAY_PORT:-6382}"' in start_script
    assert 'GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS:-8}"' in start_script
    assert '--num-cpus="${GYM_RAY_NUM_CPUS}"' in start_script
    assert 'GYM_RAY_MIN_WORKER_PORT="${GYM_RAY_MIN_WORKER_PORT:-50000}"' in start_script
    assert '--min-worker-port="${GYM_RAY_MIN_WORKER_PORT}"' in start_script
    assert '--object-manager-port="${GYM_RAY_OBJECT_MANAGER_PORT}"' in start_script
    assert '"force_cleanup_url": f"http://{host}:{resource_port}/cleanup/{{rollout_id}}"' in start_script
    assert '"cleanup_probe_url": f"http://{host}:{resource_port}/cleanup/{{rollout_id}}"' in start_script
    assert '--env WORKPLACE_ASSISTANT_PORT_BASE="${WORKPLACE_ASSISTANT_PORT_BASE}"' in remote_script
    assert '--env GYM_RAY_PORT="${GYM_RAY_PORT}"' in remote_script
    assert 'WORKPLACE_ASSISTANT_PORT_BASE="${WORKPLACE_ASSISTANT_PORT_BASE:-29000}"' in remote_script
    assert 'GYM_RAY_PORT="${GYM_RAY_PORT:-6382}"' in remote_script
    assert '--env GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS}"' in remote_script
    assert 'docker rm -f "${NEMO_GYM_CONTAINER}"' in remote_script
    assert '--mount "type=bind,source=${RELAX_REPO_ROOT},target=/repo,readonly"' in remote_script
    assert "test -s /repo/examples/nemo_gym_agentic/recipes/workplace-assistant/" in remote_script
    assert 'app.post("/cleanup/{rollout_id}")(self.cleanup_session)' in cleanup_patch
    assert 'app.get("/cleanup/{rollout_id}")(self.probe_cleanup)' in cleanup_patch
    assert 'health.json().get("active_trials") != 0' in trial_verifier


def test_dual_use_image_preserves_relax_python_environment() -> None:
    dockerfile = (EXAMPLE_DIR / "service" / "Dockerfile").read_text()

    assert "ENV PATH=/opt/nemo-gym/.venv/bin:" not in dockerfile
    assert "ENV PYTHONPATH=/opt/relax-integration:/opt/nemo-gym" not in dockerfile
    assert "RELAX_RAY_VERSION=\"$(python -c 'import ray; print(ray.__version__)')\"" in dockerfile
    assert '"ray[default]==${RELAX_RAY_VERSION}"' in dockerfile
    assert 'test "${GYM_RAY_VERSION}" = "${RELAX_RAY_VERSION}"' in dockerfile


def test_r2e_recipe_uses_configured_ray_and_apptainer() -> None:
    recipe_dir = EXAMPLE_DIR / "recipes" / "r2e-gym"
    start_script = (recipe_dir / "start_r2e_gym_local.sh").read_text()
    remote_script = (recipe_dir / "start_r2e_gym_remote.sh").read_text()
    prepare_script = (recipe_dir / "prepare_r2e_gym.sh").read_text()
    submit_script = (recipe_dir / "submit_r2e_gym.sh").read_text()
    swe_agents_patch = (EXAMPLE_DIR / "service" / "patches" / "swe_agents_r2e.patch").read_text()
    assert '+ray_head_node_address="${GYM_RAY_ADDRESS}"' in start_script
    assert "container_formatter='${R2E_GYM_SIF_DIR}/${R2E_GYM_SIF_PREFIX}{instance_id}.sif'" in start_script
    assert "ray start" not in start_script
    assert "ray stop" not in start_script
    assert "pkill" not in start_script
    assert "docker create" not in start_script
    assert 'command_name in "${RAY_CLI}" jq hostname ss apptainer' not in start_script
    assert 'ray_address="${RAY_ADDRESS:-auto}"' in start_script
    assert '"${RAY_CLI}" list nodes --address="${discovery_ray_address}" --format json' in start_script
    assert ".is_head_node == true" in start_script
    assert 'GYM_BIND_HOST="${GYM_BIND_HOST:-0.0.0.0}"' in start_script
    assert 'NEMO_GYM_ARTIFACT_ROOT="${NEMO_GYM_ARTIFACT_ROOT:-${R2E_DATA_DIR}/artifacts}"' in start_script
    assert 'apptainer exec "${first_sif}" true' in start_script
    assert "responses_api_models/relax_gateway_model/.venv/bin/python" in start_script
    assert "responses_api_agents/swe_agents/.venv/bin/python" in start_script
    assert '"ray[default]==${cluster_ray_version}"' in start_script
    assert "Ray versions aligned with cluster Python" in start_script
    assert 'psutil.net_connections(kind="tcp")' in start_script
    assert "process.terminate()" in start_script
    assert "process.kill()" in start_script
    assert "/opt/nemo-gym/.venv/bin/ray start" in remote_script
    assert '--env RAY_CLI="/opt/nemo-gym/.venv/bin/ray"' in remote_script
    assert '--env R2E_GYM_CLUSTER_PYTHON="/opt/nemo-gym/.venv/bin/python"' in remote_script
    assert 'export RAY_ADDRESS="${GYM_HOST}:6381"' in remote_script
    assert (
        "exec bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/r2e-gym/start_r2e_gym_local.sh"
        in remote_script
    )
    assert 'apptainer build "${local_sif}" "docker://${docker_image}"' in prepare_script
    assert 'PREPARE_LOCK_DIR="${R2E_GYM_OUTPUT_DIR}/.prepare_r2e_gym.lockdir"' in prepare_script
    assert 'shared_partial="${sif_path}.partial.${BASHPID}.${RANDOM}"' in prepare_script
    assert "Usage: $0 <ray-head-ip> [golden|train]" in submit_script
    assert 'RAY_ADDRESS="${RAY_ADDRESS:-${R2E_RAY_HEAD}:6379}"' in submit_script
    assert 'RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://${R2E_RAY_HEAD}:8265}"' in submit_script
    assert '--working-dir="${R2E_GYM_WORKING_DIR}"' in submit_script
    assert 'R2E_GYM_START_SCRIPT="${R2E_GYM_START_SCRIPT:-recipes/r2e-gym/start_r2e_gym_local.sh}"' in submit_script
    assert "Usage: R2E_GYM_OUTPUT_DIR=/shared/path R2E_GYM_LIMIT=1 $0" in prepare_script
    assert 'R2E_GYM_OUTPUT_DIR="${R2E_GYM_OUTPUT_DIR:-/data/nemo-gym/r2e-gym}"' in prepare_script
    assert '+        elif command.mode == "eval":' in swe_agents_patch


def test_r2e_openhands_runtime_patch_preserves_baseline_and_replay_templates() -> None:
    start_script = (EXAMPLE_DIR / "recipes" / "r2e-gym" / "start_r2e_gym_local.sh").read_text()
    dockerfile = (EXAMPLE_DIR / "service" / "Dockerfile").read_text()
    runtime_patch = (EXAMPLE_DIR / "service" / "patches" / "openhands_r2e_runtime.patch").read_text()

    assert 'git tag -f swebench_baseline "$BASE"' in runtime_patch
    assert "refs/tags/swebench_baseline^{commit}" in runtime_patch
    assert "copy.deepcopy(stored_replay_events)" in runtime_patch
    assert "copy.deepcopy(stored_initial_action)" in runtime_patch
    assert "openhands_r2e_runtime.patch" in start_script
    assert "openhands_r2e_runtime.patch" in dockerfile
    assert "openhands_r2e_runtime_setup.patch" in dockerfile


def test_r2e_local_launcher_help_does_not_require_ray() -> None:
    launcher = EXAMPLE_DIR / "recipes" / "r2e-gym" / "start_r2e_gym_local.sh"

    result = subprocess.run(
        ["bash", str(launcher), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--data-dir <prepared-r2e-directory>" in result.stdout
    assert "--mode <golden|train>" in result.stdout
    assert "--proxy <http-proxy-url>" in result.stdout
    assert "--callback-proxy <http-proxy-url>" in result.stdout
    assert "--callback-timeout-s <positive-integer>" in result.stdout
    assert "RAY_ADDRESS" in result.stdout


def test_r2e_local_launcher_rejects_non_gcs_ray_addresses() -> None:
    launcher = EXAMPLE_DIR / "recipes" / "r2e-gym" / "start_r2e_gym_local.sh"

    for ray_address in ("http://127.0.0.1:8265", "ray://127.0.0.1:10001"):
        env = os.environ.copy()
        env["RAY_ADDRESS"] = ray_address
        result = subprocess.run(
            [
                "bash",
                str(launcher),
                "--data-dir",
                "/tmp/not-used",
                "--mode",
                "train",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 2
        assert "must be a Ray GCS address" in result.stderr
