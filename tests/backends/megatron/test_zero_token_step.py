# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Zero-token no-signal step handling for the shared Megatron trainer."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")


@pytest.fixture()
def model_module():
    from relax.backends.megatron import model as model_module

    return model_module


def _two_microbatch_losses(zero: bool) -> list[dict[str, object]]:
    """Two microbatches; ``values[0]`` is the (CP-local) token count."""
    return (
        [
            {"values": torch.tensor([0.0, 1.0, 2.0])},
            {"values": torch.tensor([0.0, 3.0])},
        ]
        if zero
        else [
            {"values": torch.tensor([4.0, 1.0, 2.0])},
            {"values": torch.tensor([6.0, 3.0])},
        ]
    )


def test_is_global_zero_token_step_true(model_module, monkeypatch):
    mpu = model_module.mpu
    dist = model_module.torch.distributed
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda ignore_virtual=False: True)
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda with_context_parallel=False: object())
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_group", lambda: object())
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group=None: None)
    monkeypatch.setattr(dist, "broadcast", lambda tensor, src=0, group=None: None)

    assert model_module._is_global_zero_token_step(_two_microbatch_losses(zero=True)) is True


def test_is_global_zero_token_step_false(model_module, monkeypatch):
    mpu = model_module.mpu
    dist = model_module.torch.distributed
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda ignore_virtual=False: True)
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda with_context_parallel=False: object())
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_group", lambda: object())
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group=None: None)
    monkeypatch.setattr(dist, "broadcast", lambda tensor, src=0, group=None: None)

    assert model_module._is_global_zero_token_step(_two_microbatch_losses(zero=False)) is False


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        custom_megatron_before_train_step_hook_path=None,
        ci_test=False,
        enable_mtp_training=False,
        check_for_nan_in_loss_and_grad=True,
        global_batch_size=32,
        dynamic_context_parallel=False,
        use_dynamic_batch_size=False,
        fully_async=False,
        seq_length=512,
        micro_batch_size=1,
        decoder_seq_length=512,
        attention_backend="flash",
    )


def test_train_one_step_skips_optimizer_and_scheduler_on_zero_token_step(model_module, monkeypatch):
    """A global zero-token step must not update parameters or the LR
    scheduler."""
    from megatron.core import mpu

    calls = {"optimizer_step": 0, "scheduler_step": 0}

    optimizer = SimpleNamespace(
        step=lambda: calls.__setitem__("optimizer_step", calls["optimizer_step"] + 1) or (True, 0.0, 0),
        zero_grad=lambda: None,
        get_loss_scale=lambda: SimpleNamespace(item=lambda: 1.0),
        param_groups=[],
    )
    scheduler = SimpleNamespace(
        step=lambda increment=None: calls.__setitem__("scheduler_step", calls["scheduler_step"] + 1)
    )
    model = [SimpleNamespace(zero_grad_buffer=lambda: None)]

    losses_reduced = _two_microbatch_losses(zero=True)

    monkeypatch.setattr(model_module, "_is_global_zero_token_step", lambda losses: True)
    monkeypatch.setattr(model_module, "get_args", lambda: _make_args())
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: lambda **_kwargs: losses_reduced)
    monkeypatch.setattr(model_module, "maybe_verify_critic_value_head_movement", lambda *a, **k: None)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda ignore_virtual=False: False)
    monkeypatch.setattr(mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)

    loss_reduced, grad_norm = model_module.train_one_step(
        args=_make_args(),
        rollout_id=0,
        step_id=3,
        data_iterator=[[]],
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        num_microbatches=1,
    )

    assert calls["optimizer_step"] == 0
    assert calls["scheduler_step"] == 0
    assert grad_norm == 0.0
    assert loss_reduced == {}
