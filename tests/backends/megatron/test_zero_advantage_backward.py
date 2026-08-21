# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.training.arguments")

from relax.backends.megatron import arguments as megatron_arguments  # noqa: E402
from relax.backends.megatron import loss as loss_module  # noqa: E402
from relax.backends.megatron.data import (  # noqa: E402
    _ZERO_EFFECTIVE_ADVANTAGE_KEY,
    DataIterator,
    prepare_zero_advantage_backward_metadata,
)


def test_zero_advantage_metadata_respects_loss_mask():
    rollout_data = {
        "advantages": [torch.tensor([0.0, 2.0]), torch.zeros(1)],
        "loss_masks": [torch.tensor([1.0, 0.0]), torch.ones(1)],
    }

    prepare_zero_advantage_backward_metadata(rollout_data)
    assert rollout_data[_ZERO_EFFECTIVE_ADVANTAGE_KEY] == [True, True]

    rollout_data["loss_masks"][0][1] = 1.0
    prepare_zero_advantage_backward_metadata(rollout_data)
    assert rollout_data[_ZERO_EFFECTIVE_ADVANTAGE_KEY] == [False, True]


def test_zero_advantage_metadata_is_conservative_and_follows_iterator_schedule():
    rollout_data = {
        "advantages": [torch.zeros(1), torch.tensor([float("nan")]), torch.ones(1)],
        "loss_masks": [torch.ones(1), torch.ones(1), torch.zeros(1)],
        "total_lengths": [1, 1, 1],
    }

    prepare_zero_advantage_backward_metadata(rollout_data)
    fixed_iterator = DataIterator(rollout_data, micro_batch_size=1)
    iterator = DataIterator(rollout_data, micro_batch_indices=[[2, 0], [1]])

    assert fixed_iterator.get_next([_ZERO_EFFECTIVE_ADVANTAGE_KEY])[_ZERO_EFFECTIVE_ADVANTAGE_KEY] == [True]
    assert fixed_iterator.get_next([_ZERO_EFFECTIVE_ADVANTAGE_KEY])[_ZERO_EFFECTIVE_ADVANTAGE_KEY] == [False]
    assert iterator.get_next([_ZERO_EFFECTIVE_ADVANTAGE_KEY])[_ZERO_EFFECTIVE_ADVANTAGE_KEY] == [True, True]
    assert iterator.get_next([_ZERO_EFFECTIVE_ADVANTAGE_KEY])[_ZERO_EFFECTIVE_ADVANTAGE_KEY] == [False]
    assert iterator.reset() is iterator
    assert iterator.get_next([_ZERO_EFFECTIVE_ADVANTAGE_KEY])[_ZERO_EFFECTIVE_ADVANTAGE_KEY] == [True, True]


def test_zero_advantage_metadata_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes must match"):
        prepare_zero_advantage_backward_metadata({"advantages": [torch.zeros(2)], "loss_masks": [torch.ones(1)]})


def _supported_args(**overrides):
    values = {
        "skip_zero_advantage_backward": True,
        "colocate": True,
        "fully_async": False,
        "hybrid": False,
        "advantage_estimator": "grpo",
        "use_critic": False,
        "critic_train_only": False,
        "loss_type": "policy_loss",
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "dynamic_context_parallel": False,
        "expert_model_parallel_size": 1,
        "num_experts": None,
        "is_vl_model": False,
        "lora_rank": 0,
        "enable_mtp_training": False,
        "overlap_grad_reduce": False,
        "entropy_coef": 0.0,
        "kl_loss_coef": 0.0,
        "use_kl_loss": False,
        "use_opd": False,
        "custom_loss_function_path": None,
        "custom_tis_function_path": None,
        "custom_pg_loss_reducer_function_path": None,
        "custom_megatron_before_train_step_hook_path": None,
        "recompute_loss_function": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_zero_advantage_backward_validation_accepts_supported_scope():
    megatron_arguments._validate_skip_zero_advantage_backward(_supported_args())


def test_hf_validation_rejects_multimodal_model_before_actor_init():
    args = SimpleNamespace(
        apply_rope_fusion=False,
        is_vl_model=False,
        skip_zero_advantage_backward=True,
    )
    hf_config = SimpleNamespace(text_config=SimpleNamespace())

    with pytest.raises(AssertionError, match="supports text-only models"):
        megatron_arguments._hf_validate_args(args, hf_config)

    assert not args.is_vl_model


@pytest.mark.parametrize(
    ("override", "requirement"),
    [
        ({"pipeline_model_parallel_size": 2}, "pipeline parallel size 1"),
        ({"advantage_estimator": "ppo"}, "GRPO advantage estimator"),
        ({"use_critic": True}, "critic-free training"),
        ({"critic_train_only": True}, "actor training"),
        ({"hybrid": True}, "synchronous colocate mode"),
        ({"expert_model_parallel_size": 2}, "a dense model"),
        ({"is_vl_model": True}, "a text model"),
        ({"lora_rank": 8}, "non-LoRA training"),
        ({"overlap_grad_reduce": True}, "gradient-reduction overlap disabled"),
        ({"entropy_coef": 0.01}, "zero entropy coefficient"),
        ({"use_kl_loss": True}, "KL loss disabled"),
        ({"custom_pg_loss_reducer_function_path": "custom.reducer"}, "built-in policy-loss reducer"),
    ],
)
def test_zero_advantage_backward_validation_rejects_unsafe_scope(override, requirement):
    with pytest.raises(ValueError, match=requirement):
        megatron_arguments._validate_skip_zero_advantage_backward(_supported_args(**override))


def test_skipped_backward_preserves_loss_and_does_not_reach_logits(monkeypatch):
    logits = torch.ones(2, requires_grad=True)
    zero_loss = logits.sum() * 0

    monkeypatch.setattr(
        loss_module,
        "policy_loss_function",
        lambda *_args, **_kwargs: (zero_loss, {"loss": zero_loss.detach()}),
    )
    monkeypatch.setattr(loss_module, "get_cp_local_num_tokens", lambda *_args, **_kwargs: torch.tensor(2))
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *_args, **_kwargs: object())

    args = Namespace(
        loss_type="policy_loss",
        recompute_loss_function=False,
        skip_zero_advantage_backward=True,
        calculate_per_token_loss=True,
        qkv_format="thd",
        allgather_cp=False,
        global_batch_size=1,
    )
    batch = {
        "total_lengths": [2],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        loss_module._SKIP_ZERO_ADVANTAGE_BACKWARD_KEY: True,
    }

    loss, _, log = loss_module.loss_function(args, batch, num_microbatches=1, logits=logits)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is None
    assert log["keys"] == ["loss", "zero_advantage_backward_fraction"]
    assert log["values"].tolist() == [2.0, 0.0, 2.0]


def test_internal_skip_metadata_is_ignored_when_flag_is_disabled(monkeypatch):
    logits = torch.ones(1, requires_grad=True)
    monkeypatch.setattr(
        loss_module, "policy_loss_function", lambda *_args, **_kwargs: (logits.sum(), {"loss": logits.detach().sum()})
    )
    monkeypatch.setattr(loss_module, "get_cp_local_num_tokens", lambda *_args, **_kwargs: torch.tensor(1))
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *_args, **_kwargs: object())
    args = Namespace(
        loss_type="policy_loss",
        recompute_loss_function=False,
        skip_zero_advantage_backward=False,
        calculate_per_token_loss=True,
        qkv_format="thd",
        allgather_cp=False,
        global_batch_size=1,
    )
    batch = {
        "total_lengths": [1],
        "response_lengths": [1],
        "loss_masks": [torch.ones(1)],
        loss_module._SKIP_ZERO_ADVANTAGE_BACKWARD_KEY: True,
    }

    loss, _, log = loss_module.loss_function(args, batch, num_microbatches=1, logits=logits)
    loss.backward()

    assert torch.equal(logits.grad, torch.ones_like(logits))
    assert "zero_advantage_backward_fraction" not in log["keys"]


def test_nonzero_advantage_path_preserves_model_gradient(monkeypatch):
    logits = torch.ones(2, requires_grad=True)
    nonzero_loss = logits.sum()

    monkeypatch.setattr(
        loss_module,
        "policy_loss_function",
        lambda *_args, **_kwargs: (nonzero_loss, {"loss": nonzero_loss.detach()}),
    )
    monkeypatch.setattr(loss_module, "get_cp_local_num_tokens", lambda *_args, **_kwargs: torch.tensor(2))
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *_args, **_kwargs: object())

    args = Namespace(
        loss_type="policy_loss",
        recompute_loss_function=False,
        skip_zero_advantage_backward=True,
        calculate_per_token_loss=True,
        qkv_format="thd",
        allgather_cp=False,
        global_batch_size=1,
    )
    batch = {
        "total_lengths": [2],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        loss_module._SKIP_ZERO_ADVANTAGE_BACKWARD_KEY: False,
    }

    loss, _, log = loss_module.loss_function(args, batch, num_microbatches=1, logits=logits)
    loss.backward()

    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.ones_like(logits))
    assert log["keys"] == ["loss", "zero_advantage_backward_fraction"]
    assert log["values"].tolist() == [2.0, 2.0, 0.0]
