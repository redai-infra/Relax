# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("case", json.loads(FIXTURES.read_text()))
def test_error_case_library_maps_to_rule(case):
    report = run_doctor(argv=[], args=_namespace(case["config"]))

    assert case["expected_rule"] in _rule_ids(report), case["name"]
    assert not report.ok


def test_parse_error_is_reported_as_ci_failure():
    report = run_doctor(argv=["--bad-flag"], args=None, parse_error="argparse exited with code 2")

    assert not report.ok
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
    assert "topology" in json_report
    assert "config" in json_report
