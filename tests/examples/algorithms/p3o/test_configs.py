# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Static comparability tests for the Task40 A100x4 launch scripts."""

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
FIXED_LAG_SCRIPTS = {
    "p3o_fixed_lag_2_mismatch": SCRIPT_DIR / "run_p3o_fixed_lag_2_mismatch_a100x4.sh",
    "grpo_fixed_lag_2_mismatch": SCRIPT_DIR / "run_grpo_fixed_lag_2_mismatch_a100x4.sh",
}


def _bash_executable() -> str:
    """Resolve a POSIX bash that can open the repository's own paths.

    A bare ``bash`` argv[0] is not safe to rely on: Windows resolves
    executables from ``System32`` before ``PATH``, and ``System32\\bash.exe``
    is the WSL launcher, which runs in a separate filesystem namespace and
    cannot open a ``D:\\...`` script path. Prefer an explicit Git-for-Windows
    bash, and skip rather than fail when no usable POSIX shell exists.
    """
    for candidate in (
        shutil.which("bash", path=os.environ.get("GIT_BASH_DIR")),
        r"C:\Program Files\Git\usr\bin\bash.exe",
        "/bin/bash",
        "/usr/bin/bash",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    resolved = shutil.which("bash")
    if resolved and os.name != "nt":
        return resolved
    pytest.skip("no POSIX bash available to dry-run the launch scripts")


def _dry_run(script: Path, *extra_args: str, env_overrides: dict[str, str] | None = None) -> list[str]:
    env = os.environ.copy()
    env["TASK40_DRY_RUN"] = "1"
    if env_overrides is not None:
        env.update(env_overrides)
    result = subprocess.run(
        [_bash_executable(), str(script), *extra_args],
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


def test_p3o_fixed_lag_configs_are_matched_and_parameterized():
    resolved = {name: _dry_run(script) for name, script in FIXED_LAG_SCRIPTS.items()}

    p3o_args = resolved["p3o_fixed_lag_2_mismatch"]
    grpo_args = resolved["grpo_fixed_lag_2_mismatch"]
    assert _option_value(p3o_args, "--max-staleness") == "0"
    assert _option_value(grpo_args, "--max-staleness") == "0"
    assert _option_value(p3o_args, "--update-weights-interval") == "3"
    assert _option_value(grpo_args, "--update-weights-interval") == "3"
    assert _option_value(p3o_args, "--tb-experiment-name") == "p3o_fixed_lag_2_mismatch-seed-42"
    assert _option_value(grpo_args, "--tb-experiment-name") == "grpo_fixed_lag_2_mismatch-seed-42"
    assert _comparable_args(p3o_args) == _comparable_args(grpo_args)

    overridden = _dry_run(
        FIXED_LAG_SCRIPTS["p3o_fixed_lag_2_mismatch"],
        env_overrides={"TASK40_UPDATE_WEIGHTS_INTERVAL": "4"},
    )
    assert _option_value(overridden, "--max-staleness") == "0"
    assert _option_value(overridden, "--update-weights-interval") == "4"
    assert _option_value(overridden, "--tb-experiment-name") == "p3o_fixed_lag_3_mismatch-seed-42"

    adjusted_temperature = _dry_run(
        FIXED_LAG_SCRIPTS["p3o_fixed_lag_2_mismatch"],
        env_overrides={"TASK40_UPDATE_WEIGHTS_INTERVAL": "11", "TASK40_BEHAVIOR_TEMPERATURE": "2.0"},
    )
    assert _option_value(adjusted_temperature, "--tb-experiment-name") == (
        "p3o_fixed_lag_10_temperature_2p0_mismatch-seed-42"
    )


def test_p3o_smoke_uses_one_small_optimizer_step():
    args = _dry_run(SCRIPT_DIR / "run_p3o_smoke.sh", "p3o_temperature_1p2")

    assert _option_value(args, "--num-rollout") == "1"
    assert _option_value(args, "--rollout-batch-size") == "4"
    assert _option_value(args, "--n-samples-per-prompt") == "4"
    assert _option_value(args, "--global-batch-size") == "16"
    assert _option_value(args, "--micro-batch-size") == "1"
    assert _option_value(args, "--rollout-max-response-len") == "128"
    assert "--eval-prompt-data" not in args


def test_p3o_runtime_env_allows_ray_job_driver_merge():
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()

    assert '"RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"' in common_script
    assert '"TASK40_BEHAVIOR_TEMPERATURE": os.environ["TASK40_RUNTIME_BEHAVIOR_TEMPERATURE"]' in common_script


def test_p3o_runtime_env_bypasses_proxy_for_colocated_services():
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        assert f'"{name}": ""' in common_script
    for name in ("NO_PROXY", "no_proxy"):
        assert f'"{name}": "*"' in common_script


def test_p3o_runner_records_failed_job_exit_code_before_returning():
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()
    job_pipeline = '"${TASK40_COMMAND[@]}" 2>&1 | tee "${TASK40_RUN_DIR}/stdout_stderr.log"'
    pipeline_index = common_script.index(job_pipeline)
    capture_index = common_script.index("TASK40_EXIT_CODE=${PIPESTATUS[0]}", pipeline_index)

    assert common_script.rfind("set +e", 0, pipeline_index) != -1
    assert common_script.index("set -e", capture_index) < common_script.index(
        'echo "${TASK40_EXIT_CODE}" >"${TASK40_RUN_DIR}/exit_code.txt"',
        capture_index,
    )


def test_p3o_runner_records_explicit_ray_job_identity_and_terminal_status():
    common_script = (SCRIPT_DIR / "common_a100x4.sh").read_text()

    assert "--submission-id" in common_script
    assert 'TASK40_JOB_ID="${TASK40_CONFIG_NAME}-seed-${TASK40_SEED}-${TASK40_RUN_ID}"' in common_script
    assert '"${TASK40_RUN_DIR}/job_status.txt"' in common_script
