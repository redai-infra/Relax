# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Static comparability tests for the P3O A100x4 launch scripts."""

import json
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
ALL_SCENARIO_SCRIPTS = {**FORMAL_SCRIPTS, **LOW_TEMPERATURE_SCRIPTS, **PERIODIC_SYNC_SCRIPTS}


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


def _shell_path(path: Path, bash: str) -> str:
    """Translate a Windows path for Git Bash; POSIX paths pass through."""
    if os.name != "nt":
        return str(path)
    del bash
    normalized = path.resolve().as_posix()
    return f"/{normalized[0].lower()}{normalized[2:]}"


def _dry_run(script: Path, *extra_args: str, env_overrides: dict[str, str] | None = None) -> list[str]:
    env = os.environ.copy()
    for name in (
        "P3O_ACTIVATION_RECOMPUTE",
        "P3O_CLIP_HIGH",
        "P3O_CLIP_LOW",
        "P3O_CLEAR_RUNTIME_PROXIES",
        "P3O_DETERMINISTIC_INFERENCE",
        "P3O_ESS_SCOPE",
        "P3O_EVAL_MAX_RESPONSE_LEN",
        "P3O_EVAL_NAME",
        "P3O_EVAL_N_SAMPLES",
        "P3O_EVAL_TEMPERATURE",
        "P3O_EVAL_TOP_P",
        "P3O_INPUT_KEY",
        "P3O_KL_MODE",
        "P3O_LABEL_KEY",
        "P3O_LOG_PROBS_CHUNK_SIZE",
        "P3O_MODE",
        "P3O_MODEL_CONFIG",
        "P3O_MODEL_ROTARY_BASE",
        "P3O_NUM_ROLLOUT",
        "P3O_RM_TYPE",
        "P3O_ROLLOUT_RESULT_DIR",
        "P3O_ROLLOUT_SHUFFLE",
        "P3O_ROLLOUT_BATCH_SIZE",
        "P3O_N_SAMPLES",
    ):
        env.pop(name, None)
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


def _run_fake_ray(
    tmp_path: Path,
    script: Path,
    *,
    submit_exit_code: int = 0,
    env_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
    """Run a real launcher path against a recording fake Ray executable."""
    bash = _bash_executable()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ray = fake_bin / "ray"
    fake_ray.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" >>"${FAKE_RAY_CALLS}"\n'
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
    train_data.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "output"
    ray_calls = tmp_path / "ray_calls.txt"

    env = os.environ.copy()
    for name in (
        "P3O_ACTIVATION_RECOMPUTE",
        "P3O_ALGORITHM",
        "P3O_BEHAVIOR_TEMPERATURE",
        "P3O_CLEAR_RUNTIME_PROXIES",
        "P3O_DETERMINISTIC_INFERENCE",
        "P3O_ENABLE_TEMPERATURE_OVERRIDE",
        "P3O_EVAL_MAX_RESPONSE_LEN",
        "P3O_EVAL_NAME",
        "P3O_EVAL_N_SAMPLES",
        "P3O_EVAL_TEMPERATURE",
        "P3O_EVAL_TOP_P",
        "P3O_LOG_PROBS_CHUNK_SIZE",
        "P3O_NCCL_DEBUG",
        "P3O_MODEL_CONFIG",
        "P3O_MODEL_ROTARY_BASE",
        "P3O_RM_TYPE",
        "P3O_ROLLOUT_RESULT_DIR",
        "P3O_ROLLOUT_SHUFFLE",
        "P3O_TORCH_DISTRIBUTED_DEBUG",
        "P3O_UPDATE_WEIGHTS_INTERVAL",
    ):
        env.pop(name, None)
    env.update(
        {
            "FAKE_RAY_CALLS": str(ray_calls),
            "FAKE_RAY_SUBMIT_EXIT": str(submit_exit_code),
            "P3O_DRY_RUN": "0",
            "P3O_MEGATRON_DIR": str(megatron_dir),
            "P3O_MODE": "smoke",
            "P3O_MODEL_DIR": str(model_dir),
            "P3O_OUTPUT_ROOT": str(output_root),
            "P3O_RAY_DASHBOARD": "http://example.invalid:8265",
            "P3O_RUN_ID": "integration",
            "P3O_TRAIN_DATA": str(train_data),
        }
    )
    if env_overrides is not None:
        env.update(env_overrides)
    if os.name == "nt":
        env["PATH"] = f"{Path(bash).parent}{os.pathsep}{env.get('PATH', '')}"
    for name in (
        "FAKE_RAY_CALLS",
        "P3O_EVAL_DATA",
        "P3O_MEGATRON_DIR",
        "P3O_MODEL_DIR",
        "P3O_OUTPUT_ROOT",
        "P3O_TRAIN_DATA",
    ):
        if name in env:
            env[name] = _shell_path(Path(env[name]), bash)

    result = subprocess.run(
        [
            bash,
            "-c",
            'export PATH="$1:$PATH"; exec "$2"',
            "p3o-runner",
            _shell_path(fake_bin, bash),
            _shell_path(script, bash),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    config_name = script.stem.removeprefix("run_").removesuffix("_a100x4")
    run_dir = output_root / config_name / "seed_42" / "integration"
    calls = ray_calls.read_text(encoding="utf-8").splitlines() if ray_calls.exists() else []
    return result, run_dir, calls


def _option_value(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def _comparable_args(args: list[str]) -> list[str]:
    ignored_with_value = {
        "--advantage-estimator",
        "--eps-clip",
        "--eps-clip-high",
        "--p3o-ess-scope",
        "--p3o-kl-mode",
        "--clip-low",
        "--clip-high",
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
        assert _option_value(args, "--input-key") == "problem"
        assert _option_value(args, "--label-key") == "answer"
        assert _option_value(args, "--rm-type") == "deepscaler"
        assert _option_value(args, "--num-rollout") == "30"
        assert _option_value(args, "--rollout-batch-size") == "4"
        assert _option_value(args, "--n-samples-per-prompt") == "16"
        assert _option_value(args, "--global-batch-size") == "64"
        assert _option_value(args, "--micro-batch-size") == "1"
        assert _option_value(args, "--rollout-max-response-len") == "4096"
        assert _option_value(args, "--rollout-temperature") == "1.0"
        assert _option_value(args, "--rollout-top-p") == "1.0"
        assert _option_value(args, "--lr") == "1e-5"
        assert _option_value(args, "--adam-beta2") == "0.95"
        assert _option_value(args, "--weight-decay") == "0.01"
        assert "--calculate-per-token-loss" in args
        assert "--use-rollout-logprobs" in args
        assert "--deterministic-mode" in args
        assert "--batch-invariant-mode" in args
        assert "--rollout-shuffle" in args
        assert "--colocate" in args
        assert "--fully-async" not in args
        assert "--use-tis" not in args
        assert "--use-kl-loss" not in args
        assert "--eval-size" not in args
        assert _option_value(args, "--num-layers") == "36"
        assert _option_value(args, "--hidden-size") == "2560"
        assert _option_value(args, "--rotary-base") == "5000000"
        assert (
            int(_option_value(args, "--num-rollout"))
            * int(_option_value(args, "--rollout-batch-size"))
            * int(_option_value(args, "--n-samples-per-prompt"))
            == 1920
        )
        assert int(_option_value(args, "--rollout-batch-size")) % 4 == 0
        assert int(_option_value(args, "--global-batch-size")) == (
            int(_option_value(args, "--rollout-batch-size")) * int(_option_value(args, "--n-samples-per-prompt"))
        )


def test_p3o_configs_use_active_algorithm_settings_only_for_p3o():
    p3o_args = _dry_run(FORMAL_SCRIPTS["p3o_on_policy"])
    grpo_args = _dry_run(FORMAL_SCRIPTS["grpo_on_policy"])

    assert _option_value(p3o_args, "--p3o-ess-scope") == "micro-batch"
    assert _option_value(p3o_args, "--p3o-kl-mode") == "proxy_safe"
    assert _option_value(p3o_args, "--clip-low") == "0.2"
    assert _option_value(p3o_args, "--clip-high") == "0.2"
    for option in ("--p3o-ess-scope", "--p3o-kl-mode", "--clip-low", "--clip-high"):
        assert option not in grpo_args


def test_p3o_configs_are_comparable_within_each_scenario():
    resolved = {name: _dry_run(script) for name, script in FORMAL_SCRIPTS.items()}
    assert _comparable_args(resolved["p3o_on_policy"]) == _comparable_args(resolved["grpo_on_policy"])
    assert _comparable_args(resolved["p3o_temperature_1p2"]) == _comparable_args(resolved["grpo_temperature_1p2"])

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
    assert _option_value(args, "--input-key") == "question"
    assert _option_value(args, "--rm-type") == "mopd"
    assert _option_value(args, "--num-layers") == "28"
    assert _option_value(args, "--hidden-size") == "1024"
    assert _option_value(args, "--rotary-base") == "1000000"
    assert _option_value(args, "--p3o-ess-scope") == "micro-batch"
    assert _option_value(args, "--p3o-kl-mode") == "proxy_safe"
    assert _option_value(args, "--rollout-result-dir") == "/dummy/output/rollout_results"
    assert "--eval-prompt-data" not in args


def test_p3o_dataset_keys_can_be_overridden():
    args = _dry_run(
        SCRIPT_DIR / "run_p3o_smoke.sh",
        "p3o_on_policy",
        env_overrides={"P3O_INPUT_KEY": "problem", "P3O_LABEL_KEY": "solution"},
    )

    assert _option_value(args, "--input-key") == "problem"
    assert _option_value(args, "--label-key") == "solution"


def test_p3o_reward_type_can_be_overridden_for_deepscaler_smoke():
    args = _dry_run(
        SCRIPT_DIR / "run_p3o_smoke.sh",
        "p3o_on_policy",
        env_overrides={"P3O_RM_TYPE": "deepscaler"},
    )

    assert _option_value(args, "--rm-type") == "deepscaler"


def test_p3o_rollout_result_dir_can_be_overridden():
    args = _dry_run(
        SCRIPT_DIR / "run_p3o_smoke.sh",
        "p3o_on_policy",
        env_overrides={"P3O_ROLLOUT_RESULT_DIR": "/evidence/raw_rollouts"},
    )

    assert _option_value(args, "--rollout-result-dir") == "/evidence/raw_rollouts"


def test_p3o_rollout_shuffle_can_be_disabled_for_a_fixed_prompt_schedule():
    args = _dry_run(
        SCRIPT_DIR / "run_p3o_smoke.sh",
        "p3o_on_policy",
        env_overrides={"P3O_ROLLOUT_SHUFFLE": "0"},
    )

    assert "--rollout-shuffle" not in args


def test_p3o_deterministic_inference_can_be_enabled_for_paired_sampling():
    args = _dry_run(
        SCRIPT_DIR / "run_p3o_smoke.sh",
        "p3o_on_policy",
        env_overrides={"P3O_DETERMINISTIC_INFERENCE": "1"},
    )

    assert args.count("--sglang-enable-deterministic-inference") == 1


def test_p3o_smoke_can_select_pipeline_parallel_size_two():
    args = _dry_run(
        SCRIPT_DIR / "run_p3o_smoke.sh",
        "p3o_on_policy",
        env_overrides={"P3O_PIPELINE_MODEL_PARALLEL_SIZE": "2", "P3O_NUM_ROLLOUT": "3"},
    )

    assert _option_value(args, "--pipeline-model-parallel-size") == "2"
    assert _option_value(args, "--num-rollout") == "3"
    assert _option_value(args, "--tb-experiment-name") == "p3o_on_policy_pp2-seed-42"


def test_p3o_runner_requires_explicit_ray_dashboard():
    env = os.environ.copy()
    env.pop("P3O_RAY_DASHBOARD", None)
    env["P3O_DRY_RUN"] = "1"
    bash = _bash_executable()
    if os.name == "nt":
        env["PATH"] = f"{Path(bash).parent}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(
        [bash, str(FORMAL_SCRIPTS["p3o_on_policy"])],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode != 0
    assert "P3O_RAY_DASHBOARD must be set" in result.stderr


@pytest.mark.parametrize(
    ("scenario", "expected_algorithm", "expected_interval", "expected_temperature"),
    [
        ("p3o_on_policy", "p3o", "1", None),
        ("grpo_on_policy", "grpo", "1", None),
        ("p3o_periodic_sync_interval_3", "p3o", "3", None),
        ("grpo_periodic_sync_interval_3", "grpo", "3", None),
        ("p3o_temperature_0p6", "p3o", "1", "0.6"),
        ("grpo_temperature_0p6", "grpo", "1", "0.6"),
        ("p3o_temperature_1p2", "p3o", "1", "1.2"),
        ("grpo_temperature_1p2", "grpo", "1", "1.2"),
    ],
)
def test_p3o_runner_executes_all_scenarios_with_fake_ray(
    tmp_path,
    scenario,
    expected_algorithm,
    expected_interval,
    expected_temperature,
):
    result, run_dir, ray_calls = _run_fake_ray(tmp_path, ALL_SCENARIO_SCRIPTS[scenario])

    assert result.returncode == 0, result.stderr
    resolved_args = (run_dir / "resolved_args.txt").read_text(encoding="utf-8").splitlines()
    runtime_env = json.loads(_option_value(ray_calls, "--runtime-env-json"))["env_vars"]
    identity = dict(
        line.split("=", 1)
        for line in (run_dir / "run_identity.env").read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    assert _option_value(resolved_args, "--advantage-estimator") == expected_algorithm
    assert _option_value(resolved_args, "--update-weights-interval") == expected_interval
    assert _option_value(ray_calls, "--submission-id") == f"{scenario}-seed-42-integration"
    assert runtime_env["NCCL_DEBUG"] == "WARN"
    assert runtime_env["TORCH_DISTRIBUTED_DEBUG"] == "OFF"
    assert runtime_env["RAY_OVERRIDE_JOB_RUNTIME_ENV"] == "1"
    assert runtime_env["NCCL_ALGO"] == "Ring"
    assert runtime_env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert runtime_env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    for proxy_name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        assert proxy_name not in runtime_env
    assert {"GIT_COMMIT", "GIT_BRANCH", "GIT_DIRTY", "started_utc", "ended_utc"} <= identity.keys()
    assert identity["model_config"].endswith("qwen3-0.6B.sh")
    assert identity["model_rotary_base"] == _option_value(resolved_args, "--rotary-base") == "1000000"
    assert identity["p3o_ess_scope"] == "micro-batch"
    assert identity["p3o_kl_mode"] == "proxy_safe"
    assert identity["clip_low"] == identity["clip_high"] == "0.2"
    assert identity["input_key"] == "question"
    assert identity["label_key"] == "answer"
    assert identity["rm_type"] == "mopd"
    assert identity["rollout_shuffle"] == "1"
    assert identity["clear_runtime_proxies"] == "0"
    assert identity["rollout_result_dir"].endswith(f"/{scenario}/seed_42/integration/rollout_results")
    assert _option_value(resolved_args, "--rollout-result-dir").endswith(
        f"/{scenario}/seed_42/integration/rollout_results"
    )
    assert identity["config"] == scenario
    assert identity["ray_job_id"] == f"{scenario}-seed-42-integration"
    if expected_temperature is None:
        assert "--custom-generate-function-path" not in resolved_args
        assert "P3O_BEHAVIOR_TEMPERATURE" not in runtime_env
    else:
        assert _option_value(resolved_args, "--custom-generate-function-path") == (
            "examples.algorithms.p3o.rollout.generate"
        )
        assert runtime_env["P3O_BEHAVIOR_TEMPERATURE"] == expected_temperature
    assert (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() == "0"
    assert (run_dir / "job_status.txt").read_text(encoding="utf-8").strip() == "TERMINAL"


def test_p3o_runner_can_clear_runtime_proxies_explicitly(tmp_path):
    result, run_dir, ray_calls = _run_fake_ray(
        tmp_path,
        FORMAL_SCRIPTS["p3o_on_policy"],
        env_overrides={"P3O_CLEAR_RUNTIME_PROXIES": "1"},
    )

    assert result.returncode == 0, result.stderr
    runtime_env = json.loads(_option_value(ray_calls, "--runtime-env-json"))["env_vars"]
    identity = (run_dir / "run_identity.env").read_text(encoding="utf-8")
    for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        assert runtime_env[proxy_name] == ""
    assert runtime_env["NO_PROXY"] == runtime_env["no_proxy"] == "*"
    assert "clear_runtime_proxies=1" in identity


def test_p3o_runner_preserves_debug_overrides(tmp_path):
    result, _, ray_calls = _run_fake_ray(
        tmp_path,
        FORMAL_SCRIPTS["p3o_on_policy"],
        env_overrides={"P3O_NCCL_DEBUG": "INFO", "P3O_TORCH_DISTRIBUTED_DEBUG": "DETAIL"},
    )

    assert result.returncode == 0, result.stderr
    runtime_env = json.loads(_option_value(ray_calls, "--runtime-env-json"))["env_vars"]
    assert runtime_env["NCCL_DEBUG"] == "INFO"
    assert runtime_env["TORCH_DISTRIBUTED_DEBUG"] == "DETAIL"


def test_p3o_runner_preserves_failed_ray_exit_code(tmp_path):
    result, run_dir, _ = _run_fake_ray(
        tmp_path,
        FORMAL_SCRIPTS["p3o_on_policy"],
        submit_exit_code=17,
    )

    assert result.returncode == 17
    assert (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() == "17"
    assert (run_dir / "job_status.txt").read_text(encoding="utf-8").strip() == "TERMINAL"


def test_p3o_smoke_runner_does_not_require_eval_data(tmp_path):
    result, _, _ = _run_fake_ray(tmp_path, FORMAL_SCRIPTS["p3o_on_policy"])

    assert result.returncode == 0, result.stderr


def test_p3o_formal_runner_requires_eval_data(tmp_path):
    result, _, ray_calls = _run_fake_ray(
        tmp_path,
        FORMAL_SCRIPTS["p3o_on_policy"],
        env_overrides={"P3O_MODE": "formal"},
    )

    assert result.returncode == 1
    assert "P3O_EVAL_DATA must be set in formal mode" in result.stderr
    assert ray_calls == []


def test_p3o_formal_runner_accepts_existing_eval_data(tmp_path):
    eval_data = tmp_path / "eval.jsonl"
    eval_data.write_text("{}\n", encoding="utf-8")

    result, run_dir, _ = _run_fake_ray(
        tmp_path,
        FORMAL_SCRIPTS["p3o_on_policy"],
        env_overrides={"P3O_MODE": "formal", "P3O_EVAL_DATA": str(eval_data)},
    )

    assert result.returncode == 0, result.stderr
    resolved_args = (run_dir / "resolved_args.txt").read_text(encoding="utf-8").splitlines()
    assert _option_value(resolved_args, "--eval-prompt-data") == "deepscaler"
    assert str(eval_data.name) in resolved_args[resolved_args.index("--eval-prompt-data") + 2]
    assert _option_value(resolved_args, "--n-samples-per-eval-prompt") == "16"
    assert _option_value(resolved_args, "--eval-max-response-len") == "4096"
    assert _option_value(resolved_args, "--eval-temperature") == "1.0"
    assert _option_value(resolved_args, "--eval-top-p") == "0.95"


def test_p3o_formal_runner_records_resource_adjusted_eval_contract(tmp_path):
    eval_data = tmp_path / "eval.jsonl"
    eval_data.write_text("{}\n", encoding="utf-8")

    result, run_dir, _ = _run_fake_ray(
        tmp_path,
        FORMAL_SCRIPTS["p3o_on_policy"],
        env_overrides={
            "P3O_MODE": "formal",
            "P3O_EVAL_DATA": str(eval_data),
            "P3O_EVAL_NAME": "local-deepscaler",
            "P3O_EVAL_N_SAMPLES": "1",
            "P3O_EVAL_MAX_RESPONSE_LEN": "2048",
            "P3O_EVAL_TEMPERATURE": "0.8",
            "P3O_EVAL_TOP_P": "0.9",
        },
    )

    assert result.returncode == 0, result.stderr
    resolved_args = (run_dir / "resolved_args.txt").read_text(encoding="utf-8").splitlines()
    identity = (run_dir / "run_identity.env").read_text(encoding="utf-8")
    assert _option_value(resolved_args, "--eval-prompt-data") == "local-deepscaler"
    assert _option_value(resolved_args, "--n-samples-per-eval-prompt") == "1"
    assert _option_value(resolved_args, "--eval-max-response-len") == "2048"
    assert _option_value(resolved_args, "--eval-temperature") == "0.8"
    assert _option_value(resolved_args, "--eval-top-p") == "0.9"
    assert "eval_name=local-deepscaler" in identity
    assert "eval_n_samples=1" in identity
    assert "eval_max_response_len=2048" in identity


def test_p3o_runner_records_resource_adjusted_activation_contract(tmp_path):
    result, run_dir, _ = _run_fake_ray(
        tmp_path,
        FORMAL_SCRIPTS["p3o_on_policy"],
        env_overrides={
            "P3O_ACTIVATION_RECOMPUTE": "1",
            "P3O_LOG_PROBS_CHUNK_SIZE": "128",
        },
    )

    assert result.returncode == 0, result.stderr
    resolved_args = (run_dir / "resolved_args.txt").read_text(encoding="utf-8").splitlines()
    identity = (run_dir / "run_identity.env").read_text(encoding="utf-8")
    assert _option_value(resolved_args, "--recompute-granularity") == "full"
    assert _option_value(resolved_args, "--recompute-method") == "uniform"
    assert _option_value(resolved_args, "--recompute-num-layers") == "1"
    assert _option_value(resolved_args, "--log-probs-chunk-size") == "128"
    assert "activation_recompute=1" in identity
    assert "log_probs_chunk_size=128" in identity


def test_p3o_runner_validates_megatron_directory_before_ray(tmp_path):
    missing_megatron = tmp_path / "missing-megatron"
    result, _, ray_calls = _run_fake_ray(
        tmp_path,
        FORMAL_SCRIPTS["p3o_on_policy"],
        env_overrides={"P3O_MEGATRON_DIR": str(missing_megatron)},
    )

    assert result.returncode == 2
    assert missing_megatron.name in result.stderr
    assert ray_calls == []


@pytest.mark.parametrize("raw_value", ["", "0", "0.0", "-1", "NaN", "Inf", "warm"])
def test_p3o_shell_rejects_invalid_behavior_temperature(raw_value):
    env = os.environ.copy()
    env.update(
        {
            "P3O_ALGORITHM": "p3o",
            "P3O_BEHAVIOR_TEMPERATURE": raw_value,
            "P3O_DRY_RUN": "1",
            "P3O_ENABLE_TEMPERATURE_OVERRIDE": "1",
            "P3O_RAY_DASHBOARD": "http://example.invalid:8265",
        }
    )
    bash = _bash_executable()
    if os.name == "nt":
        env["PATH"] = f"{Path(bash).parent}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(
        [bash, "-c", 'source "$1"; P3O_run', "p3o-test", str(SCRIPT_DIR / "common_a100x4.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode != 0
    assert "P3O_BEHAVIOR_TEMPERATURE" in result.stderr
