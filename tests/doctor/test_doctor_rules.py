# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.utils.doctor import runner as doctor_runner
from relax.utils.doctor.runner import render_json, render_text, run_doctor


FIXTURES = Path(__file__).parent / "fixtures" / "error_cases.json"


def _base_config() -> dict:
    return {
        "advantage_estimator": "grpo",
        "loss_type": "policy_loss",
        "resource": {"actor": [1, 8], "rollout": [1, 8]},
        "debug_rollout_only": False,
        "debug_train_only": False,
        "fully_async": False,
        "colocate": True,
        "hybrid": False,
        "true_on_policy_mode": False,
        "max_staleness": 0,
        "rollout_batch_size": 4,
        "global_batch_size": 32,
        "n_samples_per_prompt": 8,
        "num_steps_per_rollout": None,
        "num_rollout": 1,
        "num_epoch": None,
        "rollout_global_dataset": True,
        "over_sampling_batch_size": 4,
        "partial_rollout": False,
        "use_dynamic_global_batch_size": False,
        "use_dynamic_batch_size": False,
        "max_tokens_per_gpu": None,
        "context_parallel_size": 1,
        "dynamic_context_parallel": False,
        "rollout_max_context_len": 2048,
        "rollout_max_prompt_len": 1024,
        "num_data_storage_units": 1,
        "balance_data": False,
        "sglang_pp_size": 1,
        "sglang_pipeline_parallel_size": 1,
        "rollout_num_gpus_per_engine": 1,
        "sglang_data_parallel_size": 1,
        "sglang_enable_dp_attention": False,
        "prefill_num_servers": None,
        "rollout_external": False,
        "sglang_config": None,
        "eval_interval": None,
        "eval_config": None,
        "eval_prompt_data": None,
        "eval_datasets": None,
        "eval_size": None,
        "save_interval": None,
        "save": "./outputs/doctor-ok",
        "sft_predict_interval": None,
        "custom_dataset_class_path": None,
        "prompt_data": None,
        "sft_oversize_strategy": "drop",
        "sft_oversize_custom_function_path": None,
        "genrm_model_path": None,
        "genrm_num_gpus": 1,
        "actor_num_gpus_per_node": 8,
        "actor_num_nodes": 1,
        "rollout_num_gpus": 8,
        "lora_rank": 0,
        "lora_merge_mode": False,
        "lora_adapter_mode": False,
        "sglang_dp_size": 1,
        "qkv_format": "thd",
        "train_backend": "megatron",
        "rotate_ckpt": False,
        "async_save": False,
        "data_source_path": "relax.engine.rollout.data_source.RolloutDataSourceWithBuffer",
        "rollout_function_path": "relax.engine.rollout.sglang_rollout.generate_rollout",
        "eval_function_path": "relax.engine.rollout.sglang_rollout.generate_rollout",
        "custom_config_path": None,
        "custom_reward_function_path": None,
        "kl_coef": 0.0,
        "use_kl_loss": False,
        "ref_load": None,
    }


def _namespace(overrides: dict | None = None) -> SimpleNamespace:
    config = _base_config()
    if overrides:
        config.update(overrides)
    return SimpleNamespace(**config)


def _rule_ids(report) -> set[str]:
    return {item.rule_id for item in report.diagnostics}


def test_doctor_passes_for_minimal_colocate_config():
    report = run_doctor(argv=["--resource", '{"actor": [1, 8], "rollout": [1, 8]}'], args=_namespace())

    assert report.ok
    assert not report.diagnostics
    assert report.command[:3] == ["python", "-m", "relax.entrypoints.train"]
    assert report.topology["resource_summary"]["total_required_gpus"] == 8
    assert report.topology["data_system"]["sampler"] == "GRPOGroupNSampler"


def test_hybrid_resource_preview_uses_dedicated_actor_and_rollout_gpus():
    args = _namespace(
        {
            "resource": {"actor": [1, 8], "rollout": [1, 8]},
            "hybrid": True,
            "fully_async": True,
            "colocate": True,
        }
    )

    report = run_doctor(argv=[], args=args)

    assert report.ok
    assert report.topology["colocate"] is True
    assert report.topology["shares_actor_rollout_pg"] is False
    assert report.topology["resource_summary"]["shared_gpu"] == 0
    assert report.topology["resource_summary"]["total_required_gpus"] == 16
    assert {item["placement_group"] for item in report.topology["role_plan"]} == {"dedicated"}


def test_debug_train_only_topology_excludes_managed_teacher():
    args = _namespace(
        {
            "debug_train_only": True,
            "use_opd": True,
            "opd_type": "sglang",
            "teacher_hf_checkpoint": "/teacher",
            "resource": {"actor": [1, 8], "teacher": [1, 8]},
        }
    )

    report = run_doctor(argv=[], args=args)

    assert report.ok
    assert report.topology["roles"] == ["actor"]
    assert report.topology["resource_summary"]["total_required_gpus"] == 8
    assert all(spec["role"] != "teacher" for spec in report.topology["role_plan"])


def test_model_roles_reject_zero_gpu_resources():
    report = run_doctor(
        argv=[],
        args=_namespace({"resource": {"actor": [1, 0], "rollout": [1, 8]}}),
    )

    assert not report.ok
    assert "CONFIG_RESOURCE_SHAPE" in _rule_ids(report)
    diagnostic = next(item for item in report.diagnostics if item.rule_id == "CONFIG_RESOURCE_SHAPE")
    assert diagnostic.details["code"] == "gpu_required"
    assert diagnostic.details["role"] == "actor"


@pytest.mark.parametrize("parse_error", (None, "service plan rejected GPU allocation for a CPU-only role"))
def test_cpu_only_roles_reject_positive_gpu_resources(parse_error):
    report = run_doctor(
        argv=[],
        args=_namespace(
            {
                "fully_async": True,
                "colocate": False,
                "resource": {
                    "actor": [1, 4],
                    "rollout": [1, 4],
                    "advantages": [1, 1],
                    "actor_fwd": [1, 2],
                },
            }
        ),
        parse_error=parse_error,
    )

    resource_diagnostics = [item for item in report.diagnostics if item.rule_id == "CONFIG_RESOURCE_SHAPE"]
    assert len(resource_diagnostics) == 1
    assert "CPU-only role 'advantages' requires num_gpus=0" in resource_diagnostics[0].message


@pytest.mark.parametrize(
    "actor_resource",
    (
        [True, 8],
        [False, 8],
        [1, True],
        [1, False],
    ),
)
def test_partial_resource_shape_rejects_boolean_values(actor_resource):
    report = run_doctor(
        argv=[],
        args=_namespace({"resource": {"actor": actor_resource, "rollout": [1, 8]}}),
        parse_error="service plan rejected a non-integer resource value",
    )

    resource_diagnostics = [item for item in report.diagnostics if item.rule_id == "CONFIG_RESOURCE_SHAPE"]
    assert report.config_state == "partial"
    assert len(resource_diagnostics) == 1
    assert "must contain integers" in resource_diagnostics[0].message
    assert resource_diagnostics[0].details == {
        "role": "actor",
        "value": actor_resource,
    }


def test_fully_async_without_kl_does_not_require_reference():
    args = _namespace(
        {
            "resource": {
                "actor": [1, 4],
                "rollout": [1, 4],
                "advantages": [1, 0],
                "actor_fwd": [1, 2],
            },
            "fully_async": True,
            "colocate": False,
            "global_batch_size": 16,
        }
    )

    report = run_doctor(argv=[], args=args)

    assert report.ok
    assert "reference" not in report.topology["required_roles"]
    assert "reference" not in report.topology["missing_resource_roles"]
    assert "reference" not in report.topology["roles"]


def test_fully_async_with_kl_requires_reference():
    args = _namespace(
        {
            "resource": {
                "actor": [1, 4],
                "rollout": [1, 4],
                "advantages": [1, 0],
                "actor_fwd": [1, 2],
            },
            "fully_async": True,
            "colocate": False,
            "global_batch_size": 16,
            "use_kl_loss": True,
        }
    )

    report = run_doctor(argv=[], args=args)

    assert not report.ok
    assert "reference" in report.topology["required_roles"]
    assert "reference" in report.topology["missing_resource_roles"]
    assert "CONFIG_REQUIRED_ROLES" in _rule_ids(report)


def test_fully_async_keeps_explicit_optional_reference_in_plan():
    args = _namespace(
        {
            "resource": {
                "actor": [1, 4],
                "rollout": [1, 4],
                "advantages": [1, 0],
                "actor_fwd": [1, 2],
                "reference": [1, 2],
            },
            "fully_async": True,
            "colocate": False,
            "global_batch_size": 16,
        }
    )

    report = run_doctor(argv=[], args=args)

    assert report.ok
    assert "reference" not in report.topology["required_roles"]
    assert "reference" in report.topology["roles"]


def test_generalized_prompt_data_paths_are_checked_individually(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("{}\n")
    second.write_text("{}\n")
    args = _namespace({"prompt_data": f"[{first},{second}]@[0:100]"})

    report = run_doctor(argv=[], args=args)

    assert report.ok
    assert "CONFIG_PATHS" not in _rule_ids(report)


def test_generalized_prompt_data_reports_only_missing_physical_path(tmp_path):
    existing = tmp_path / "existing.jsonl"
    missing = tmp_path / "missing.jsonl"
    existing.write_text("{}\n")
    path_spec = f"[{existing},{missing}]@[1:5]"
    args = _namespace({"prompt_data": path_spec})

    report = run_doctor(argv=[], args=args)

    path_diagnostics = [item for item in report.diagnostics if item.rule_id == "CONFIG_PATHS"]
    assert not report.ok
    assert len(path_diagnostics) == 1
    assert path_diagnostics[0].details == {
        "field": "prompt_data",
        "path": str(missing),
        "path_spec": path_spec,
    }


@pytest.mark.parametrize("case", json.loads(FIXTURES.read_text()))
def test_error_case_library_maps_to_rule(case):
    report = run_doctor(argv=[], args=_namespace(case["config"]))

    assert case["expected_rule"] in _rule_ids(report), case["name"]
    assert not report.ok


def test_parse_error_is_reported_as_ci_failure():
    report = run_doctor(argv=["--bad-flag"], args=None, parse_error="argparse exited with code 2")

    assert not report.ok
    assert report.config_state == "unavailable"
    assert "CONFIG_PARSE_ERROR" in _rule_ids(report)


def test_strict_warnings_promotes_warning_to_error():
    args = _namespace({"over_sampling_batch_size": 8, "fully_async": False, "partial_rollout": False})
    report = run_doctor(argv=[], args=args, strict_warnings=True)

    assert not report.ok
    assert any(item.rule_id == "CONFIG_OVERSAMPLING" and item.severity == "error" for item in report.diagnostics)


def test_text_and_json_reports_include_required_sections():
    report = run_doctor(argv=["--num-rollout", "1"], args=_namespace())
    text_report = render_text(report)
    json_report = json.loads(render_json(report))

    assert "Expected launch command" in text_report
    assert "Role topology" in text_report
    assert "Final merged config" in text_report
    assert json_report["command"] == ["python", "-m", "relax.entrypoints.train", "--num-rollout", "1"]
    assert json_report["config_state"] == "validated"
    assert "topology" in json_report
    assert "config" in json_report


def test_report_redacts_agent_env_and_sensitive_options():
    api_value = "api-value"
    region_value = "region-value"
    sglang_value = "server-value"
    args = _namespace(
        {
            "agent_env": [f"OPENAI_API_KEY={api_value}", f"REGION={region_value}"],
            "sglang_api_key": sglang_value,
            "reward_key": "score",
        }
    )
    report = run_doctor(
        argv=[
            "--agent-env",
            f"OPENAI_API_KEY={api_value}",
            f"REGION={region_value}",
            "--sglang-api-key",
            sglang_value,
            "--reward-key",
            "score",
        ],
        args=args,
    )

    text_report = render_text(report)
    json_report = render_json(report)

    for sensitive_value in (api_value, region_value, sglang_value):
        assert sensitive_value not in text_report
        assert sensitive_value not in json_report
    assert report.config["agent_env"] == ["OPENAI_API_KEY=<redacted>", "REGION=<redacted>"]
    assert report.config["sglang_api_key"] == "<redacted>"
    assert report.config["reward_key"] == "score"
    assert report.argv == [
        "--agent-env",
        "OPENAI_API_KEY=<redacted>",
        "REGION=<redacted>",
        "--sglang-api-key",
        "<redacted>",
        "--reward-key",
        "score",
    ]
    assert report.command[-2:] == ["--reward-key", "score"]


def test_parse_error_redacts_secret_from_raw_argv():
    secret = "malformed-agent-secret"

    report = run_doctor(
        argv=["--agent-env", f"TOKEN={secret}", "--bad-flag"],
        args=None,
        parse_error=f"invalid agent env TOKEN={secret}",
    )
    output = render_json(report)

    assert not report.ok
    assert secret not in output
    assert "TOKEN=<redacted>" in output


def test_report_redacts_structured_train_env_and_wandb_key():
    train_secret = "train-value"
    wandb_secret = "wandb-value"
    args = _namespace(
        {
            "train_env_vars": {"OPENAI_API_KEY": train_secret},
            "wandb_key": wandb_secret,
        }
    )

    report = run_doctor(
        argv=[
            "--train-env-vars",
            f'{{"OPENAI_API_KEY":"{train_secret}"}}',
            "--wandb-key",
            wandb_secret,
        ],
        args=args,
    )
    output = render_json(report)

    assert train_secret not in output
    assert wandb_secret not in output
    assert report.argv == [
        "--train-env-vars",
        '{"OPENAI_API_KEY":"<redacted>"}',
        "--wandb-key",
        "<redacted>",
    ]
    assert report.command[-4:] == report.argv
    assert report.config["train_env_vars"] == {"OPENAI_API_KEY": "<redacted>"}
    assert report.config["wandb_key"] == "<redacted>"


def test_report_redacts_malformed_structured_train_env_without_config():
    secret = "malformed-train-value"
    malformed = f'{{"OPENAI_API_KEY":"{secret}"'

    report = run_doctor(
        argv=["--train-env-vars", malformed],
        args=None,
        parse_error=f"invalid train env: {malformed}",
    )

    for output in (render_json(report), render_text(report)):
        assert secret not in output
        assert malformed not in output
    assert report.argv == ["--train-env-vars", "<redacted>"]
    assert report.command[-2:] == report.argv


def test_report_redacts_non_mapping_train_env_in_partial_config():
    secret = "partial-train-value"
    raw_value = f'["{secret}"]'
    args = _namespace({"train_env_vars": [secret]})

    report = run_doctor(
        argv=["--train-env-vars", raw_value],
        args=args,
        parse_error=f"train_env_vars must be a mapping: {secret}",
    )

    for output in (render_json(report), render_text(report)):
        assert secret not in output
        assert raw_value not in output
    assert report.config_state == "partial"
    assert report.config["train_env_vars"] == "<redacted>"
    assert report.argv == ["--train-env-vars", "<redacted>"]
    assert report.command[-2:] == report.argv


def test_zero_num_steps_returns_structured_batch_diagnostic():
    report = run_doctor(argv=[], args=_namespace({"num_steps_per_rollout": 0}))

    assert not report.ok
    assert "CONFIG_BATCH_SIZE" in _rule_ids(report)
    diagnostic = next(item for item in report.diagnostics if item.rule_id == "CONFIG_BATCH_SIZE")
    assert diagnostic.details == {"num_steps_per_rollout": 0}


def test_rule_exception_is_converted_to_structured_diagnostic(monkeypatch):
    def broken_rule(_context):
        raise RuntimeError("broken rule")

    monkeypatch.setattr(
        doctor_runner,
        "get_rules",
        lambda: [
            SimpleNamespace(
                rule_id="BROKEN_RULE",
                supports_partial=True,
                check=broken_rule,
            )
        ],
    )

    report = run_doctor(argv=[], args=_namespace())

    assert not report.ok
    assert [item.rule_id for item in report.diagnostics] == ["DOCTOR_RULE_EXECUTION_ERROR"]
    assert report.diagnostics[0].details == {"failed_rule_id": "BROKEN_RULE"}
