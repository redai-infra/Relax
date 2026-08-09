# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os

import pytest

from relax.utils.mixture_lora import (
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
    monkeypatch.delenv("RELAX_MIXTURE_LORA_CONFIG", raising=False)

    package = configure_mixture_lora_external_model(
        None,
        "custom.single_lora.package",
    )

    assert package == "custom.single_lora.package"
    assert "RELAX_MIXTURE_LORA_CONFIG" not in os.environ


def test_mixture_lora_rejects_conflicting_external_package():
    with pytest.raises(ValueError, match="requires the Qwen3 external model package"):
        configure_mixture_lora_external_model(_config(), "custom.other.package")
