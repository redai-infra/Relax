# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest


def test_zero_kl_loss_is_disabled_before_reference_setup(arguments_module):
    args = SimpleNamespace(use_kl_loss=True, kl_loss_coef=0.0, kl_coef=0.0)

    changed = arguments_module._normalize_zero_kl_loss_args(args)

    assert changed is True
    assert args.use_kl_loss is False


@pytest.mark.parametrize(
    ("use_kl_loss", "kl_loss_coef"),
    [(False, 0.0), (True, 0.01)],
)
def test_nonzero_or_disabled_kl_loss_is_not_normalized(arguments_module, use_kl_loss, kl_loss_coef):
    args = SimpleNamespace(use_kl_loss=use_kl_loss, kl_loss_coef=kl_loss_coef, kl_coef=0.0)

    changed = arguments_module._normalize_zero_kl_loss_args(args)

    assert changed is False
    assert args.use_kl_loss is use_kl_loss


def test_unused_fully_async_reference_resource_is_removed(arguments_module):
    args = SimpleNamespace(
        use_kl_loss=False,
        kl_loss_coef=0.0,
        kl_coef=0.0,
        resource={"actor": [1, 1], "rollout": [1, 1], "reference": [1, 1]},
    )

    changed = arguments_module._drop_unused_reference_resource(args)

    assert changed is True
    assert args.resource == {"actor": [1, 1], "rollout": [1, 1]}


def test_reward_kl_keeps_reference_resource(arguments_module):
    args = SimpleNamespace(
        use_kl_loss=False,
        kl_loss_coef=0.0,
        kl_coef=0.01,
        resource={"actor": [1, 1], "rollout": [1, 1], "reference": [1, 1]},
    )

    changed = arguments_module._drop_unused_reference_resource(args)

    assert changed is False
    assert "reference" in args.resource


def test_interval_two_requires_behavior_policy_correction(arguments_module):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        use_tis=False,
        use_rollout_logprobs=False,
        keep_old_actor=False,
    )

    with pytest.raises(ValueError, match="requires --use-tis"):
        arguments_module._validate_hybrid_weight_publication_args(args)


@pytest.mark.parametrize("correction", ["use_tis", "use_rollout_logprobs", "keep_old_actor"])
def test_interval_two_accepts_explicit_behavior_policy_correction(arguments_module, correction):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        use_tis=False,
        use_rollout_logprobs=False,
        keep_old_actor=False,
    )
    setattr(args, correction, True)

    arguments_module._validate_hybrid_weight_publication_args(args)
