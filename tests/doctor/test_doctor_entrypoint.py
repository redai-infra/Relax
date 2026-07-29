# Copyright (c) 2026 Relax Authors. All Rights Reserved.

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
    def fake_parse_training_args(_argv):
        raise ValueError("bad config")

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    code = doctor.main(["--", "--bad"])
    output = capsys.readouterr().out

    assert code == 1
    assert "CONFIG_PARSE_ERROR" in output


def test_doctor_skip_hf_validate_is_forwarded(monkeypatch):
    captured = {}

    def fake_parse_training_args(argv):
        captured["argv"] = argv
        return _args()

    monkeypatch.setattr(doctor, "parse_training_args", fake_parse_training_args)

    assert doctor.main(["--doctor-skip-hf-validate", "--", "--num-rollout", "1"]) == 0
    assert captured["argv"] == ["--num-rollout", "1", "--skip-hf-validate"]
