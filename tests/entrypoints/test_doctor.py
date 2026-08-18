import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import ray
import torch
from ray import serve

from relax.entrypoints import doctor


MODEL_ARGS = [
    "--swiglu",
    "--num-layers",
    "2",
    "--hidden-size",
    "128",
    "--ffn-hidden-size",
    "256",
    "--num-attention-heads",
    "4",
    "--group-query-attention",
    "--num-query-groups",
    "2",
    "--use-rotary-position-embeddings",
    "--disable-bias-linear",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-6",
    "--vocab-size",
    "128",
    "--kv-channels",
    "32",
]


@pytest.fixture
def training_argv(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    prompt = tmp_path / "prompt.jsonl"
    prompt.write_text('{"prompt": "1+1", "label": "2"}\n', encoding="utf-8")
    return [
        "--skip-hf-validate",
        "--resource",
        '{"actor": [1, 1], "rollout": [1, 1]}',
        "--colocate",
        "--hf-checkpoint",
        str(model),
        "--ref-load",
        str(model),
        "--prompt-data",
        str(prompt),
        "--num-rollout",
        "2",
        "--rollout-batch-size",
        "2",
        "--n-samples-per-prompt",
        "2",
        "--global-batch-size",
        "4",
        "--rollout-max-response-len",
        "128",
        "--micro-batch-size",
        "1",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--advantage-estimator",
        "grpo",
        *MODEL_ARGS,
    ]


def _error_cases():
    path = Path(__file__).parent / "fixtures" / "doctor_errors.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _error_cases(), ids=lambda case: case["name"])
def test_error_library_uses_real_parser(monkeypatch, training_argv, case):
    from relax.backends.megatron import arguments as megatron_arguments
    from relax.utils import arguments

    argv = list(training_argv)
    if case.get("remove_prompt"):
        index = argv.index("--prompt-data")
        del argv[index : index + 2]
    argv.extend(case["args"])

    monkeypatch.setattr(megatron_arguments, "validate_args", lambda args: args)
    monkeypatch.setattr(arguments, "sglang_validate_args", lambda args: None)
    monkeypatch.setattr(arguments.sys, "argv", ["relax.entrypoints.train", *argv])

    with pytest.raises((AssertionError, FileNotFoundError, ValueError)) as error:
        parsed = arguments.parse_args(strict=True)
        arguments.validate_preflight_args(parsed)
    assert case["contains"] in str(error.value)


def test_report_uses_registry_roles_and_colocate_resources():
    args = Namespace(
        loss_type="rl",
        advantage_estimator="grpo",
        debug_rollout_only=False,
        debug_train_only=False,
        hybrid=False,
        fully_async=False,
        colocate=True,
        genrm_model_path=None,
        resource={"actor": [1, 8], "rollout": [1, 8]},
    )

    report = doctor._report(args, ["--resource", '{"actor": [1, 8], "rollout": [1, 8]}'])

    assert report["roles"] == ["actor", "rollout"]
    assert report["resources"]["total_gpus"] == 8
    assert report["resources"]["shares_actor_rollout_gpus"] is True


def test_hybrid_resources_are_not_colocated():
    args = Namespace(
        loss_type="rl",
        advantage_estimator="grpo",
        debug_rollout_only=False,
        debug_train_only=False,
        hybrid=True,
        fully_async=True,
        colocate=True,
        genrm_model_path=None,
        resource={"actor": [1, 8], "rollout": [1, 8]},
    )

    report = doctor._report(args, [])

    assert report["resources"]["total_gpus"] == 16
    assert report["resources"]["shares_actor_rollout_gpus"] is False


def test_redacts_success_and_error_output():
    argv = [
        "--wandb-key",
        "WANDB_SECRET",
        "--train-env-vars",
        '{"OPENAI_API_KEY": "TRAIN_SECRET", "SAFE": "visible"}',
        "--agent-env",
        "TOKEN=AGENT_SECRET",
    ]

    sanitized = " ".join(doctor._redact_argv(argv))
    error = doctor._redact_error("bad WANDB_SECRET TRAIN_SECRET AGENT_SECRET", argv)

    assert "WANDB_SECRET" not in sanitized + error
    assert "TRAIN_SECRET" not in sanitized + error
    assert "AGENT_SECRET" not in sanitized + error
    assert sanitized.count("<redacted>") == 3


def test_malformed_structured_secret_is_redacted_as_a_whole():
    argv = ["--train-env-vars", '{"OPENAI_API_KEY": "BROKEN_SECRET"']
    message = doctor._redact_error(f"invalid value {argv[1]}", argv)

    assert "BROKEN_SECRET" not in " ".join(doctor._redact_argv(argv))
    assert "BROKEN_SECRET" not in message


def test_normal_token_and_key_configuration_is_not_redacted():
    config = {"tokenizer_model": "/model", "max_tokens_per_gpu": 4096, "reward_key": "score"}

    assert doctor._redact(config) == config


def test_main_does_not_start_runtime(monkeypatch, training_argv, capsys):
    from relax.backends.megatron import arguments as megatron_arguments
    from relax.utils import arguments

    def forbidden(*_args, **_kwargs):
        raise AssertionError("runtime side effect attempted")

    monkeypatch.setattr(megatron_arguments, "validate_args", lambda args: args)
    monkeypatch.setattr(arguments, "sglang_validate_args", lambda args: None)
    monkeypatch.setattr(ray, "init", forbidden)
    monkeypatch.setattr(serve, "start", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    assert doctor.main(["--format", "json", "--", *training_argv]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"


def test_main_parser_output_capture_has_a_file_descriptor(monkeypatch, capsys):
    from relax.utils import arguments

    def parse_args(*, strict):
        assert strict is True
        assert hasattr(arguments.sys.stdout, "fileno")
        arguments.sys.stdout.fileno()
        raise ValueError("stop after fileno check")

    monkeypatch.setattr(arguments, "parse_args", parse_args)

    assert doctor.main(["--", "--resource", "{}"]) == 1
    assert "UnsupportedOperation" not in capsys.readouterr().err


def test_main_returns_nonzero_without_traceback(monkeypatch, capsys):
    from relax.utils import arguments

    monkeypatch.setattr(arguments, "parse_args", lambda strict: (_ for _ in ()).throw(ValueError("bad config")))

    assert doctor.main(["--", "--resource", "{}"]) == 1
    captured = capsys.readouterr()
    assert "bad config" in captured.err
    assert "Traceback" not in captured.err
