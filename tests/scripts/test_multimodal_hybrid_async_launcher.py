# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "training" / "multimodal" / "run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh"


def _argument_value(arguments: list[str], flag: str) -> str:
    if flag in arguments:
        index = arguments.index(flag)
        return arguments[index + 1]
    prefix = f"{flag}="
    for argument in arguments:
        if argument.startswith(prefix):
            return argument.removeprefix(prefix)
    raise ValueError(f"{flag!r} is not present")


def _run_launcher(
    tmp_path: Path,
    *,
    overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "ray-arguments.txt"
    tensorboard_capture_path = tmp_path / "ray-tensorboard-dir.txt"
    fake_ray = bin_dir / "ray"
    fake_ray.write_text(
        """#!/usr/bin/env bash
set -e
printf '%s\\n' "$@" > "${RAY_CAPTURE}"
printf '%s' "${TENSORBOARD_DIR:-}" > "${RAY_TENSORBOARD_CAPTURE}"
""",
        encoding="utf-8",
    )
    fake_ray.chmod(0o755)

    model_config = tmp_path / "test-model.sh"
    model_config.write_text(
        "MODEL_ARGS=(--test-model-arg test-model-value)\n",
        encoding="utf-8",
    )

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "RELAX_ENTRYPOINT_MODE": "ray-job",
        "MODEL_CONFIG_DIR": str(tmp_path),
        "MODEL_CONFIG_FILE": str(model_config),
        "RUNTIME_ENV_JSON": "{}",
        "RAY_CAPTURE": str(capture_path),
        "RAY_TENSORBOARD_CAPTURE": str(tensorboard_capture_path),
        "EXP_DIR": str(tmp_path / "exp"),
        "MODEL_DIR": str(tmp_path / "models"),
        "DATA_DIR": str(tmp_path / "data"),
        "PROJECT_NAME": "Relax/test-launcher",
        "NUM_ROLLOUT": "2",
    }
    if overrides:
        env.update(overrides)

    result = subprocess.run(
        ["bash", str(LAUNCHER), "hybrid-async"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    arguments = capture_path.read_text(encoding="utf-8").splitlines() if capture_path.exists() else []
    return result, arguments, capture_path, tensorboard_capture_path


def test_launcher_preserves_default_qwen35_recipe(tmp_path):
    result, arguments, _, tensorboard_capture = _run_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    assert _argument_value(arguments, "--resource") == '{"actor": [1, 4], "rollout": [1, 4]}'
    assert _argument_value(arguments, "--hf-checkpoint") == str(tmp_path / "models" / "Qwen3.5-9B")
    assert _argument_value(arguments, "--ref-load") == str(tmp_path / "models" / "Qwen3.5-9B")
    assert _argument_value(arguments, "--rollout-max-response-len") == "10240"
    assert _argument_value(arguments, "--rollout-max-context-len") == "12288"
    assert _argument_value(arguments, "--max-tokens-per-gpu") == "12288"
    assert _argument_value(arguments, "--rollout-num-gpus-per-engine") == "2"
    assert _argument_value(arguments, "--num-iters-per-train-update") == "2"
    assert _argument_value(arguments, "--save") == f"{tmp_path / 'exp' / 'Qwen3.5-9B_mcore_8xgpu'}/"
    assert _argument_value(arguments, "--save-interval") == "100"
    assert "--rollout-result-dir" not in arguments
    assert "--tensorboard-dir" not in arguments
    assert tensorboard_capture.read_text(encoding="utf-8") == ""
    assert _argument_value(arguments, "--test-model-arg") == "test-model-value"


def test_launcher_supports_no_save_memory_safe_smoke_configuration(tmp_path):
    result, arguments, _, tensorboard_capture = _run_launcher(
        tmp_path,
        overrides={
            "MODEL_NAME": "Qwen3-VL-8B-Instruct",
            "MODEL_RUN_NAME": "qwen3-vl-8b",
            "CHECKPOINT_SAVE": "0",
            "ROLLOUT_MAX_RESPONSE_LEN": "512",
            "ROLLOUT_MAX_PROMPT_LEN": "2048",
            "ROLLOUT_MAX_CONTEXT_LEN": "2560",
            "ACTOR_MAX_TOKENS_PER_GPU": "6144",
            "HYBRID_ROLLOUT_GPUS": "1",
            "ROLLOUT_NUM_GPUS_PER_ENGINE": "1",
            "NUM_ITERS_PER_TRAIN_UPDATE": "4",
        },
    )

    assert result.returncode == 0, result.stderr
    assert _argument_value(arguments, "--resource") == '{"actor": [1, 4], "rollout": [1, 1]}'
    assert _argument_value(arguments, "--hf-checkpoint") == str(tmp_path / "models" / "Qwen3-VL-8B-Instruct")
    assert _argument_value(arguments, "--rollout-max-response-len") == "512"
    assert _argument_value(arguments, "--rollout-max-context-len") == "2560"
    assert _argument_value(arguments, "--max-tokens-per-gpu") == "6144"
    assert _argument_value(arguments, "--rollout-num-gpus-per-engine") == "1"
    assert _argument_value(arguments, "--num-iters-per-train-update") == "4"
    assert "--save" not in arguments
    assert "--save-interval" not in arguments
    assert _argument_value(arguments, "--rollout-result-dir") == str(tmp_path / "exp" / "rollout_result")
    expected_tensorboard_dir = tmp_path / "exp" / "tensorboard_log"
    assert _argument_value(arguments, "--tensorboard-dir") == str(expected_tensorboard_dir)
    assert tensorboard_capture.read_text(encoding="utf-8") == str(expected_tensorboard_dir)
    runtime_env = json.loads(_argument_value(arguments, "--runtime-env-json"))
    assert runtime_env["env_vars"]["TENSORBOARD_DIR"] == str(expected_tensorboard_dir)
    log_names = [path.name for path in (tmp_path / "exp" / "logs").iterdir()]
    assert len(log_names) == 1
    assert log_names[0].startswith("qwen3-vl-8b-GRPO-gpu5-hybrid-async-")


def test_launcher_rejects_invalid_checkpoint_switch_before_ray(tmp_path):
    result, arguments, capture_path, _ = _run_launcher(
        tmp_path,
        overrides={"CHECKPOINT_SAVE": "2"},
    )

    assert result.returncode == 2
    assert "CHECKPOINT_SAVE must be 0 or 1" in result.stderr
    assert arguments == []
    assert not capture_path.exists()


def test_launcher_rejects_invalid_actor_chunk_count_before_ray(tmp_path):
    result, arguments, capture_path, _ = _run_launcher(
        tmp_path,
        overrides={
            "HYBRID_PIPELINE_FORWARD": "1",
            "NUM_ITERS_PER_TRAIN_UPDATE": "1",
        },
    )

    assert result.returncode == 2
    assert "requires NUM_ITERS_PER_TRAIN_UPDATE >= 2" in result.stderr
    assert arguments == []
    assert not capture_path.exists()


def test_launcher_exposes_schedule_matched_no_overlap_control(tmp_path):
    result, arguments, _, _ = _run_launcher(
        tmp_path,
        overrides={
            "HYBRID_PIPELINE_FORWARD": "1",
            "HYBRID_PIPELINE_OVERLAP": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "--hybrid-pipeline-forward" in arguments
    assert "--no-hybrid-pipeline-overlap" in arguments


def test_launcher_rejects_overlap_control_without_pipeline_forward(tmp_path):
    result, arguments, capture_path, _ = _run_launcher(
        tmp_path,
        overrides={"HYBRID_PIPELINE_OVERLAP": "0"},
    )

    assert result.returncode == 2
    assert "requires HYBRID_PIPELINE_FORWARD=1" in result.stderr
    assert arguments == []
    assert not capture_path.exists()
