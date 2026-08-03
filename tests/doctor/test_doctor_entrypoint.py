# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

from relax.entrypoints import doctor


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator="grpo",
        loss_type="policy_loss",
        resource={"actor": [1, 8], "rollout": [1, 8]},
        debug_rollout_only=False,
        debug_train_only=False,
        fully_async=False,
        colocate=True,
        hybrid=False,
        true_on_policy_mode=False,
        max_staleness=0,
        rollout_batch_size=4,
        global_batch_size=32,
        n_samples_per_prompt=8,
        num_steps_per_rollout=None,
        num_rollout=1,
        num_epoch=None,
        rollout_global_dataset=True,
        over_sampling_batch_size=4,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        context_parallel_size=1,
        dynamic_context_parallel=False,
        rollout_max_context_len=2048,
        rollout_max_prompt_len=1024,
        num_data_storage_units=1,
        balance_data=False,
        sglang_pp_size=1,
        rollout_num_gpus_per_engine=1,
        sglang_data_parallel_size=1,
        sglang_enable_dp_attention=False,
        prefill_num_servers=None,
        rollout_external=False,
        sglang_config=None,
        eval_interval=None,
        eval_config=None,
        eval_prompt_data=None,
        eval_datasets=None,
        eval_size=None,
        save_interval=None,
        save="./outputs/doctor-ok",
        sft_predict_interval=None,
        custom_dataset_class_path=None,
        prompt_data=None,
        sft_oversize_strategy="drop",
        sft_oversize_custom_function_path=None,
        genrm_model_path=None,
        genrm_num_gpus=1,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        rollout_num_gpus=8,
        lora_rank=0,
        lora_merge_mode=False,
        lora_adapter_mode=False,
        sglang_dp_size=1,
        qkv_format="thd",
        train_backend="megatron",
        rotate_ckpt=False,
        async_save=False,
        data_source_path="relax.engine.rollout.data_source.RolloutDataSourceWithBuffer",
        rollout_function_path="relax.engine.rollout.sglang_rollout.generate_rollout",
        eval_function_path="relax.engine.rollout.sglang_rollout.generate_rollout",
        custom_config_path=None,
        custom_reward_function_path=None,
        kl_coef=0.0,
        use_kl_loss=False,
        ref_load=None,
    )


def test_entrypoint_returns_zero_for_valid_config(monkeypatch, capsys):
    captured = {}

    def fake_parse_training_args(argv):
        captured["argv"] = argv
        return _args()

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    code = doctor.main(["--format", "json", "--", "--num-rollout", "1"])
    output = capsys.readouterr().out

    assert code == 0
    assert captured["argv"] == ["--num-rollout", "1"]
    assert '"ok": true' in output


def test_entrypoint_returns_nonzero_for_parse_error(monkeypatch, capsys):
    def fake_parse_training_args(_argv, *, validate=True):
        raise ValueError("bad config")

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    code = doctor.main(["--", "--bad"])
    output = capsys.readouterr().out

    assert code == 1
    assert "CONFIG_PARSE_ERROR" in output


def test_entrypoint_falls_back_to_targeted_rules_after_validation_error(monkeypatch, capsys):
    calls = []

    def fake_parse_training_args(_argv, *, validate=True):
        calls.append(validate)
        if validate:
            raise ValueError("--fully-async and --colocate cannot be combined directly")
        args = _args()
        args.fully_async = True
        args.colocate = True
        return args

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    code = doctor.main(["--format", "json", "--", "--fully-async", "--colocate"])
    output = capsys.readouterr().out

    assert code == 1
    assert calls == [True, False]
    assert '"rule_id": "CONFIG_MODE_CONFLICT"' in output
    assert '"rule_id": "CONFIG_PARSE_ERROR"' in output
    assert '"config_state": "partial"' in output
    assert '"topology": {}' in output


def test_partial_fallback_skips_rules_that_require_normalized_config(monkeypatch, capsys):
    def fake_parse_training_args(_argv, *, validate=True):
        if validate:
            raise ValueError("validation stopped before normalization")
        args = _args()
        args.advantage_estimator = "unknown"
        return args

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    code = doctor.main(["--format", "json", "--", "--advantage-estimator", "unknown"])
    output = capsys.readouterr().out

    assert code == 1
    assert '"rule_id": "CONFIG_PARSE_ERROR"' in output
    assert "CONFIG_ALGORITHM_SUPPORTED" not in output


def test_partial_fallback_reports_zero_gpu_model_role(monkeypatch, capsys):
    def fake_parse_training_args(_argv, *, validate=True):
        if validate:
            raise ValueError("resource role 'actor' requires num_gpus > 0")
        args = _args()
        args.resource = {"actor": [1, 0], "rollout": [1, 8]}
        return args

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    code = doctor.main(["--format", "json", "--", "--resource", '{"actor": [1, 0], "rollout": [1, 8]}'])
    output = capsys.readouterr().out

    assert code == 1
    assert '"rule_id": "CONFIG_PARSE_ERROR"' in output
    assert '"rule_id": "CONFIG_RESOURCE_SHAPE"' in output
    assert "requires num_gpus > 0" in output


def test_doctor_skip_hf_validate_is_forwarded(monkeypatch):
    captured = {}

    def fake_parse_training_args(argv, *, validate=True):
        captured["argv"] = argv
        captured["validate"] = validate
        return _args()

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    assert doctor.main(["--doctor-skip-hf-validate", "--", "--num-rollout", "1"]) == 0
    assert captured["argv"] == ["--num-rollout", "1", "--skip-hf-validate"]
    assert captured["validate"] is True


def test_entrypoint_rejects_option_not_registered_by_any_parser(monkeypatch, capsys):
    def fake_parse_training_args(argv, *, validate=True):
        relax_parser = argparse.ArgumentParser(add_help=False)
        relax_parser.add_argument("--num-rollout")
        relax_parser.parse_known_args(argv)

        backend_parser = argparse.ArgumentParser(add_help=False)
        backend_parser.add_argument("--backend-option")
        backend_parser.parse_known_args(argv)
        return _args()

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    code = doctor.main(
        [
            "--format",
            "json",
            "--",
            "--num-rollout",
            "1",
            "--backend-option",
            "enabled",
            "--does-not-exist",
            "123",
        ]
    )
    output = capsys.readouterr().out

    assert code == 1
    assert '"rule_id": "CONFIG_UNKNOWN_ARGUMENT"' in output
    assert '"unknown_options": [' in output
    assert '"--does-not-exist"' in output
    assert '"--num-rollout"' not in output.split('"unknown_options":', 1)[1].split("]", 1)[0]
    assert '"--backend-option"' not in output.split('"unknown_options":', 1)[1].split("]", 1)[0]


def test_entrypoint_rejects_unknown_short_option_without_flagging_negative_value(monkeypatch, capsys):
    def fake_parse_training_args(argv, *, validate=True):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--rollout-top-k", type=int)
        parser.add_argument("-k", type=int)
        parser.parse_known_args(argv)
        return _args()

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    code = doctor.main(["--format", "json", "--", "--rollout-top-k", "-1", "-k8", "-x"])
    output = capsys.readouterr().out

    assert code == 1
    assert '"rule_id": "CONFIG_UNKNOWN_ARGUMENT"' in output
    unknown_options = output.split('"unknown_options":', 1)[1].split("]", 1)[0]
    assert '"-x"' in unknown_options
    assert '"-1"' not in unknown_options
    assert '"-k"' not in unknown_options
