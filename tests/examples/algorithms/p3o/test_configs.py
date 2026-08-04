# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Static comparability tests for the P3O A100x4 launch scripts."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "examples" / "algorithms" / "p3o"
FORMAL_SCRIPTS = {
    "p3o_on_policy": SCRIPT_DIR / "run_p3o_on_policy_a100x4.sh",
    "grpo_on_policy": SCRIPT_DIR / "run_grpo_on_policy_a100x4.sh",
    "p3o_temperature_1p2": SCRIPT_DIR / "run_p3o_temperature_1p2_a100x4.sh",
    "grpo_temperature_1p2": SCRIPT_DIR / "run_grpo_temperature_1p2_a100x4.sh",
}
LOW_TEMPERATURE_SCRIPTS = {
    "p3o_temperature_0p6": SCRIPT_DIR / "run_p3o_temperature_0p6_a100x4.sh",
    "grpo_temperature_0p6": SCRIPT_DIR / "run_grpo_temperature_0p6_a100x4.sh",
}
PERIODIC_SYNC_SCRIPTS = {
    "p3o_periodic_sync_interval_3": SCRIPT_DIR / "run_p3o_periodic_sync_interval_3_a100x4.sh",
    "grpo_periodic_sync_interval_3": SCRIPT_DIR / "run_grpo_periodic_sync_interval_3_a100x4.sh",
}


def _bash_executable() -> str:
    """Resolve a POSIX bash that can open the repository's own paths.

    A bare ``bash`` argv[0] is not safe to rely on: Windows resolves
    executables from ``System32`` before ``PATH``, and ``System32\\bash.exe``
    is the WSL launcher, which runs in a separate filesystem namespace and
    cannot open a ``D:\\...`` script path. Prefer an explicit Git-for-Windows
    bash, and skip rather than fail when no usable POSIX shell exists.
    """
    explicit_bash_dir = os.environ.get("GIT_BASH_DIR")
    candidates = []
    if explicit_bash_dir:
        candidates.append(shutil.which("bash", path=explicit_bash_dir))
    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\Git\usr\bin\bash.exe",
                r"C:\Program Files\Git\bin\bash.exe",
            ]
        )
    else:
        candidates.extend(["/bin/bash", "/usr/bin/bash", shutil.which("bash")])

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("no POSIX bash available to dry-run the launch scripts")


def _dry_run(script: Path, *extra_args: str, env_overrides: dict[str, str] | None = None) -> list[str]:
    env = os.environ.copy()
    env["P3O_DRY_RUN"] = "1"
    env["P3O_RAY_DASHBOARD"] = "http://example.invalid:8265"
    if env_overrides is not None:
        env.update(env_overrides)
    bash = _bash_executable()
    if os.name == "nt":
        env["PATH"] = f"{Path(bash).parent}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(
        [bash, str(script), *extra_args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.splitlines()


def _option_value(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def _comparable_args(args: list[str]) -> list[str]:
    ignored_with_value = {
        "--advantage-estimator",
        "--eps-clip",
        "--eps-clip-high",
        "--custom-generate-function-path",
        "--tb-experiment-name",
    }
    normalized = []
    index = 0
    while index < len(args):
        if args[index] in ignored_with_value:
            index += 2
        else:
            normalized.append(args[index])
            index += 1
    return normalized


def test_p3o_configs_freeze_required_formal_values():
    for args in map(_dry_run, FORMAL_SCRIPTS.values()):
        assert _option_value(args, "--num-rollout") == "11"
        assert _option_value(args, "--rollout-batch-size") == "12"
        assert _option_value(args, "--n-samples-per-prompt") == "4"
        assert _option_value(args, "--global-batch-size") == "48"
        assert _option_value(args, "--micro-batch-size") == "1"
        assert _option_value(args, "--rollout-max-response-len") == "4096"
        assert _option_value(args, "--rollout-temperature") == "1.0"
        assert _option_value(args, "--rollout-top-p") == "1.0"
        assert _option_value(args, "--lr") == "1e-5"
        assert _option_value(args, "--adam-beta2") == "0.95"
        assert _option_value(args, "--weight-decay") == "0.01"
        assert "--calculate-per-token-loss" in args
        assert "--use-rollout-logprobs" in args
        assert "--colocate" in args
        assert "--fully-async" not in args
        assert "--use-tis" not in args
        assert "--use-kl-loss" not in args
        assert "--eval-size" not in args
        assert (
            int(_option_value(args, "--num-rollout"))
            * int(_option_value(args, "--rollout-batch-size"))
            * int(_option_value(args, "--n-samples-per-prompt"))
            == 528
        )
        assert int(_option_value(args, "--rollout-batch-size")) % 4 == 0


def test_p3o_configs_are_comparable_except_algorithm_and_behavior():
    resolved = {name: _dry_run(script) for name, script in FORMAL_SCRIPTS.items()}
    expected = _comparable_args(resolved["p3o_on_policy"])
    for args in resolved.values():
        assert _comparable_args(args) == expected

    assert "--custom-generate-function-path" not in resolved["p3o_on_policy"]
    assert "--custom-generate-function-path" not in resolved["grpo_on_policy"]
    for name in ("p3o_temperature_1p2", "grpo_temperature_1p2"):
        assert _option_value(resolved[name], "--custom-generate-function-path") == (
            "examples.algorithms.p3o.rollout.generate"
        )

    for name in ("grpo_on_policy", "grpo_temperature_1p2"):
        assert _option_value(resolved[name], "--eps-clip") == "0.4"
        assert _option_value(resolved[name], "--eps-clip-high") == "0.4"
    for name in ("p3o_on_policy", "p3o_temperature_1p2"):
        assert "--eps-clip" not in resolved[name]
        assert "--eps-clip-high" not in resolved[name]


def test_p3o_low_temperature_configs_are_matched_and_named_from_temperature():
    resolved = {name: _dry_run(script) for name, script in LOW_TEMPERATURE_SCRIPTS.items()}

    p3o_args = resolved["p3o_temperature_0p6"]
    grpo_args = resolved["grpo_temperature_0p6"]
    assert _option_value(p3o_args, "--tb-experiment-name") == "p3o_temperature_0p6-seed-42"
    assert _option_value(grpo_args, "--tb-experiment-name") == "grpo_temperature_0p6-seed-42"
    assert _option_value(p3o_args, "--update-weights-interval") == "1"
    assert _option_value(grpo_args, "--update-weights-interval") == "1"
    assert _option_value(p3o_args, "--custom-generate-function-path") == ("examples.algorithms.p3o.rollout.generate")
    assert _comparable_args(p3o_args) == _comparable_args(grpo_args)

    smoke_args = _dry_run(SCRIPT_DIR / "run_p3o_smoke.sh", "p3o_temperature_0p6")
    assert _option_value(smoke_args, "--tb-experiment-name") == "p3o_temperature_0p6-seed-42"


def test_p3o_periodic_sync_configs_are_matched_and_parameterized():
    resolved = {name: _dry_run(script) for name, script in PERIODIC_SYNC_SCRIPTS.items()}

    p3o_args = resolved["p3o_periodic_sync_interval_3"]
    grpo_args = resolved["grpo_periodic_sync_interval_3"]
    assert _option_value(p3o_args, "--max-staleness") == "0"
    assert _option_value(grpo_args, "--max-staleness") == "0"
    assert _option_value(p3o_args, "--update-weights-interval") == "3"
    assert _option_value(grpo_args, "--update-weights-interval") == "3"
    assert _option_value(p3o_args, "--tb-experiment-name") == "p3o_periodic_sync_interval_3-seed-42"
    assert _option_value(grpo_args, "--tb-experiment-name") == "grpo_periodic_sync_interval_3-seed-42"
    assert _comparable_args(p3o_args) == _comparable_args(grpo_args)

    overridden = _dry_run(
        PERIODIC_SYNC_SCRIPTS["p3o_periodic_sync_interval_3"],
        env_overrides={"P3O_UPDATE_WEIGHTS_INTERVAL": "5"},
    )
    assert _option_value(overridden, "--max-staleness") == "0"
    assert _option_value(overridden, "--update-weights-interval") == "5"
    assert _option_value(overridden, "--tb-experiment-name") == "p3o_periodic_sync_interval_5-seed-42"


def test_p3o_smoke_uses_one_small_optimizer_step():
    args = _dry_run(SCRIPT_DIR / "run_p3o_smoke.sh", "p3o_temperature_1p2")

    assert _option_value(args, "--num-rollout") == "1"
    assert _option_value(args, "--rollout-batch-size") == "4"
    assert _option_value(args, "--n-samples-per-prompt") == "4"
    assert _option_value(args, "--global-batch-size") == "16"
    assert _option_value(args, "--micro-batch-size") == "1"
    assert _option_value(args, "--rollout-max-response-len") == "128"
    assert "--eval-prompt-data" not in args


def test_p3o_smoke_can_select_pipeline_parallel_size_two():
    args = _dry_run(
        SCRIPT_DIR / "run_p3o_smoke.sh",
        "p3o_on_policy",
        env_overrides={"P3O_PIPELINE_MODEL_PARALLEL_SIZE": "2", "P3O_NUM_ROLLOUT": "3"},
    )

    assert _option_value(args, "--pipeline-model-parallel-size") == "2"
    assert _option_value(args, "--num-rollout") == "3"
    assert _option_value(args, "--tb-experiment-name") == "p3o_on_policy_pp2-seed-42"


def test_p3o_runtime_env_allows_ray_job_driver_merge():
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()

    assert '"RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"' in common_script
    assert 'env_vars["P3O_BEHAVIOR_TEMPERATURE"] = os.environ["P3O_RUNTIME_BEHAVIOR_TEMPERATURE"]' in common_script
    assert '"NCCL_DEBUG": os.environ["P3O_RUNTIME_NCCL_DEBUG"]' in common_script
    assert '"TORCH_DISTRIBUTED_DEBUG": os.environ["P3O_RUNTIME_TORCH_DISTRIBUTED_DEBUG"]' in common_script


def test_p3o_runner_requires_explicit_ray_dashboard():
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()

    assert "127.0.0.1:8265" not in common_script
    assert "P3O_RAY_DASHBOARD must be set" in common_script


def test_p3o_runtime_env_bypasses_proxy_for_colocated_services():
    """Verify that proxy environment variables are not set in Ray runtime env.

    Per PR cleanup task: proxy clearing settings were removed as they are
    deployment-specific and should not be hardcoded in launch scripts. This
    test now verifies their absence rather than their presence.
    """
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()

    # Proxy variables should not appear in the runtime env construction
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        assert f'"{name}"' not in common_script or f'"{name}": os.environ' in common_script
    for name in ("NO_PROXY", "no_proxy"):
        assert f'"{name}"' not in common_script or f'"{name}": os.environ' in common_script


def test_p3o_runner_records_failed_job_exit_code_before_returning():
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()
    job_pipeline = '"${P3O_COMMAND[@]}" 2>&1 | tee "${P3O_RUN_DIR}/stdout_stderr.log"'
    pipeline_index = common_script.index(job_pipeline)
    capture_index = common_script.index("P3O_EXIT_CODE=${PIPESTATUS[0]}", pipeline_index)

    assert common_script.rfind("set +e", 0, pipeline_index) != -1
    assert common_script.index("set -e", capture_index) < common_script.index(
        'echo "${P3O_EXIT_CODE}" >"${P3O_RUN_DIR}/exit_code.txt"',
        capture_index,
    )


def test_p3o_runner_records_explicit_ray_job_identity_and_terminal_status():
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()

    assert "--submission-id" in common_script
    assert 'P3O_JOB_ID="${P3O_CONFIG_NAME}-seed-${P3O_SEED}-${P3O_RUN_ID}"' in common_script
    assert '"${P3O_RUN_DIR}/job_status.txt"' in common_script


def test_p3o_runner_records_git_identity():
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()

    assert 'P3O_GIT_COMMIT="$(git -C "${P3O_REPO_ROOT}" rev-parse HEAD)"' in common_script
    assert 'P3O_GIT_BRANCH="$(git -C "${P3O_REPO_ROOT}" symbolic-ref --short -q HEAD || true)"' in common_script
    assert 'echo "GIT_COMMIT=${P3O_GIT_COMMIT}"' in common_script
    assert 'echo "GIT_BRANCH=${P3O_GIT_BRANCH:-DETACHED}"' in common_script
    assert 'echo "GIT_DIRTY=${P3O_GIT_DIRTY}"' in common_script


@pytest.mark.parametrize("submit_exit_code", [0, 17])
def test_p3o_runner_executes_fake_ray_and_preserves_exit_code(tmp_path, submit_exit_code):
    """Exercise the non-dry runner without a cluster and preserve Ray's result."""
    if os.name == "nt":
        pytest.skip("non-dry launcher integration runs in POSIX CI")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ray = fake_bin / "ray"
    fake_ray.write_text(
        "#!/bin/bash\n"
        'if [[ "$1 $2" == "job submit" ]]; then\n'
        '  exit "${FAKE_RAY_SUBMIT_EXIT}"\n'
        "fi\n"
        'if [[ "$1 $2" == "job status" ]]; then\n'
        "  echo TERMINAL\n"
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fake_ray.chmod(0o755)

    model_dir = tmp_path / "model"
    megatron_dir = tmp_path / "megatron"
    model_dir.mkdir()
    megatron_dir.mkdir()
    train_data = tmp_path / "train.jsonl"
    eval_data = tmp_path / "eval.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    eval_data.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "output"

    env = os.environ.copy()
    env.update(
        {
            "FAKE_RAY_SUBMIT_EXIT": str(submit_exit_code),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "P3O_DRY_RUN": "0",
            "P3O_EVAL_DATA": str(eval_data),
            "P3O_MEGATRON_DIR": str(megatron_dir),
            "P3O_MODE": "smoke",
            "P3O_MODEL_DIR": str(model_dir),
            "P3O_OUTPUT_ROOT": str(output_root),
            "P3O_RAY_DASHBOARD": "http://example.invalid:8265",
            "P3O_RUN_ID": "integration",
            "P3O_TRAIN_DATA": str(train_data),
        }
    )
    result = subprocess.run(
        [_bash_executable(), str(SCRIPT_DIR / "run_p3o_on_policy_a100x4.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    run_dir = output_root / "p3o_on_policy" / "seed_42" / "integration"
    assert result.returncode == submit_exit_code
    assert (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() == str(submit_exit_code)
    assert (run_dir / "job_status.txt").read_text(encoding="utf-8").strip() == "TERMINAL"
