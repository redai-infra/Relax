# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from tests.utils.test_arguments_helpers import _arguments_module  # noqa: F401


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
        true_on_policy_mode=False,
        use_tis=False,
        use_rollout_logprobs=False,
        keep_old_actor=False,
        max_staleness=0,
    )

    with pytest.raises(ValueError, match="old log-probs require correction"):
        arguments_module._validate_hybrid_weight_publication_args(args)


@pytest.mark.parametrize("correction", ["use_tis", "use_rollout_logprobs", "keep_old_actor"])
def test_interval_two_accepts_explicit_behavior_policy_correction(arguments_module, correction):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=False,
        use_tis=False,
        use_rollout_logprobs=False,
        keep_old_actor=False,
        max_staleness=0,
    )
    setattr(args, correction, True)

    arguments_module._validate_hybrid_weight_publication_args(args)


@pytest.mark.parametrize("correction", ["keep_old_actor", "use_rollout_logprobs"])
def test_true_on_policy_interval_requires_tis(arguments_module, correction):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=True,
        use_tis=False,
        use_rollout_logprobs=False,
        keep_old_actor=False,
        max_staleness=0,
    )
    setattr(args, correction, True)

    with pytest.raises(ValueError, match="Enable --use-tis"):
        arguments_module._validate_hybrid_weight_publication_args(args)


def test_stale_interval_rejects_single_old_actor_snapshot(arguments_module):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=False,
        use_tis=False,
        use_rollout_logprobs=False,
        keep_old_actor=True,
        max_staleness=2,
    )

    with pytest.raises(ValueError, match="Enable --use-tis or --use-rollout-logprobs"):
        arguments_module._validate_hybrid_weight_publication_args(args)


@pytest.mark.parametrize("correction", ["use_tis", "use_rollout_logprobs"])
def test_stale_interval_accepts_staleness_aware_correction(arguments_module, correction):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=False,
        use_tis=False,
        use_rollout_logprobs=False,
        keep_old_actor=False,
        max_staleness=2,
    )
    setattr(args, correction, True)

    arguments_module._validate_hybrid_weight_publication_args(args)
