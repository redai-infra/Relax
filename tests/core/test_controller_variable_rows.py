# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest


try:
    import relax.core.controller as controller_module
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.core.controller requires the Relax runtime image: {_exc}", allow_module_level=True)


class _Sampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _config(**overrides):
    values = {
        "loss_type": "grpo",
        "rollout_batch_size": 2,
        "over_sampling_batch_size": 2,
        "n_samples_per_prompt": 3,
        "partial_rollout": False,
        "use_dynamic_global_batch_size": False,
        "max_staleness": 1,
        "global_batch_size": 8,
        "fully_async": False,
        "hybrid": False,
        "use_dynamic_batch_size": True,
        "group_rm": True,
        "agentic_custom_advantage_path": "examples.graphgpo.advantage.compute",
        "balance_data": False,
        "polling_mode": "default",
        "num_data_storage_units": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def _initialize(monkeypatch, config):
    captured = {}

    def _init(*, conf):
        captured["conf"] = conf
        return None

    monkeypatch.setattr(controller_module, "resolve_sft_algo_key", lambda _config: "grpo")
    monkeypatch.setattr(controller_module, "compute_dp_size", lambda _config: 2)
    monkeypatch.setattr(controller_module.tq, "init", _init)
    monkeypatch.setattr(controller_module, "SeqlenBalancedSampler", _Sampler)
    monkeypatch.setattr(controller_module, "GRPOGroupNSampler", _Sampler)
    controller = object.__new__(controller_module.Controller)
    controller.config = config
    controller._initialize_data_system()
    return captured["conf"]


def test_variable_row_controller_uses_row_sampler_and_padded_capacity(monkeypatch):
    monkeypatch.setenv(controller_module.AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV, "5")

    conf = _initialize(monkeypatch, _config())

    sampler = conf.controller.sampler
    assert isinstance(sampler, _Sampler)
    assert sampler.kwargs == {"n_samples_per_prompt": 1, "dp_size": 2}
    # ceil((2 prompt groups * 3 trajectories * 5 rows) / GBS 8) * 8 * (1 + staleness 1)
    assert conf.backend.SimpleStorage.total_storage_size == 64


def test_variable_row_controller_rejects_hybrid_mode(monkeypatch):
    monkeypatch.setenv(controller_module.AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV, "1")

    with pytest.raises(ValueError, match="synchronous training only"):
        _initialize(monkeypatch, _config(fully_async=True, hybrid=True))


def test_variable_row_controller_rejects_pure_fully_async_mode(monkeypatch):
    monkeypatch.setenv(controller_module.AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV, "1")

    with pytest.raises(ValueError, match="synchronous training only"):
        _initialize(monkeypatch, _config(fully_async=True, hybrid=False))


@pytest.mark.parametrize("raw_value", ["0", "-1", "+1", "1_0", "1.5", "many"])
def test_variable_row_controller_rejects_invalid_capacity_env(monkeypatch, raw_value):
    monkeypatch.setenv(controller_module.AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV, raw_value)

    with pytest.raises(ValueError, match=controller_module.AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV):
        _initialize(monkeypatch, _config())


@pytest.mark.parametrize("raw_value", [None, ""])
def test_variable_row_controller_is_explicitly_opt_in(monkeypatch, raw_value):
    if raw_value is None:
        monkeypatch.delenv(controller_module.AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV, raising=False)
    else:
        monkeypatch.setenv(controller_module.AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV, raw_value)

    conf = _initialize(monkeypatch, _config())

    assert conf.controller.sampler.kwargs == {"n_samples_per_prompt": 3}
    assert conf.backend.SimpleStorage.total_storage_size == 12


def test_default_controller_path_preserves_grouped_sampler_and_capacity(monkeypatch):
    monkeypatch.delenv(controller_module.AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV, raising=False)

    conf = _initialize(
        monkeypatch,
        _config(
            group_rm=False,
            agentic_custom_advantage_path=None,
        ),
    )

    assert conf.controller.sampler.kwargs == {"n_samples_per_prompt": 3}
    assert conf.backend.SimpleStorage.total_storage_size == 12
