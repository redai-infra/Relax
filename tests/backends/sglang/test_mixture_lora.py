# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
from types import SimpleNamespace

import pytest


# The engine imports the real SGLang router and Megatron-backed checkpoint
# client. CPU CI omits these backend dependencies.
pytest.importorskip("megatron.core")
pytest.importorskip("sglang.srt.server_args")
pytest.importorskip("sglang_router")

from relax.backends.sglang.sglang_engine import (  # noqa: E402
    _compute_server_args,
    _configure_external_model_environment,
)
from relax.utils.mixture_lora import (  # noqa: E402
    MixtureLoraConfig,
    configure_mixture_lora_external_model,
    deserialize_mixture_lora_config,
)


def _config(**overrides):
    values = {
        "rank": 16,
        "num_experts": 4,
        "top_k": 2,
        "temperature": 0.8,
        "aux_loss_coef": 0.01,
        "alpha": 32.0,
        "target_modules": ("linear_qkv", "linear_proj"),
    }
    values.update(overrides)
    return MixtureLoraConfig(**values)


def test_mixture_lora_configures_external_package_before_spawn(monkeypatch):
    monkeypatch.delenv("RELAX_MIXTURE_LORA_CONFIG", raising=False)

    package = configure_mixture_lora_external_model(_config(), None)

    assert package == "relax.models.qwen3_mixture_lora.sglang"
    config = deserialize_mixture_lora_config(os.environ["RELAX_MIXTURE_LORA_CONFIG"])
    assert config.num_experts == 4
    assert config.rank == 16
    assert config.top_k == 2
    assert config.target_modules == ("linear_qkv", "linear_proj")


def test_single_lora_does_not_enable_external_mixture_model(monkeypatch):
    monkeypatch.setenv("RELAX_MIXTURE_LORA_CONFIG", "stale-policy-config")

    package = configure_mixture_lora_external_model(
        None,
        "custom.single_lora.package",
    )

    assert package == "custom.single_lora.package"
    assert "RELAX_MIXTURE_LORA_CONFIG" not in os.environ


def test_sglang_engine_enables_mixture_external_model_only_for_policy_rollout(monkeypatch):
    from unittest.mock import MagicMock

    from relax.backends.sglang import sglang_engine
    from relax.backends.sglang.sglang_engine import SGLangEngine

    args = SimpleNamespace(optimize_routing_replay=False, warm_hf_checkpoint_page_cache=False)
    policy = SGLangEngine(args, rank=0, enable_mixture_lora_external_model=True)
    auxiliary = SGLangEngine(args, rank=0)
    for engine in (policy, auxiliary):
        engine.server_host = "127.0.0.1"
        engine.server_port = 30000
        engine.worker_type = "regular"
        engine._skip_router_registration = True

    config = _config()
    build_config = MagicMock(return_value=config)
    configure_external = MagicMock(return_value=None)
    monkeypatch.setattr(sglang_engine, "build_mixture_lora_config", build_config)
    monkeypatch.setattr(sglang_engine, "configure_mixture_lora_external_model", configure_external)
    monkeypatch.setattr(sglang_engine, "_apply_sglang_policy_load_plan", lambda server_args, _args: server_args)
    monkeypatch.setattr(sglang_engine, "ServerArgs", lambda **kwargs: kwargs)
    process = MagicMock()
    process.is_alive.return_value = False
    monkeypatch.setattr(sglang_engine, "launch_server_process", lambda _server_args: process)

    policy._init_normal({})
    auxiliary._init_normal({})

    build_config.assert_called_once_with(args)
    assert configure_external.call_args_list[0].args == (config, None)
    assert configure_external.call_args_list[1].args == (None, None)


def test_mixture_lora_rejects_conflicting_external_package():
    with pytest.raises(ValueError, match="requires the Qwen3 external model package"):
        configure_mixture_lora_external_model(_config(), "custom.other.package")


def test_text_external_model_clears_multimodal_environment(monkeypatch):
    monkeypatch.setenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", "stale.processor")
    monkeypatch.setenv("SGLANG_EXTERNAL_MM_MODEL_ARCH", "StaleArchitecture")

    arch = _configure_external_model_environment(
        "relax.models.qwen3_mixture_lora.sglang",
        text_only=True,
    )

    assert arch is None
    assert os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "relax.models.qwen3_mixture_lora.sglang"
    assert "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE" not in os.environ
    assert "SGLANG_EXTERNAL_MM_MODEL_ARCH" not in os.environ


def test_mixture_lora_keeps_base_cpu_backup_for_colocate_sleep():
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        colocate=True,
        hf_checkpoint="/models/Qwen3-4B",
        seed=42,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=True,
        lora_rank=16,
        lora_num_experts=4,
        lora_adapter_mode=False,
    )

    kwargs, _ = _compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=1235,
        host="127.0.0.1",
        port=30000,
        base_gpu_id=0,
    )

    assert kwargs["enable_weights_cpu_backup"] is True
