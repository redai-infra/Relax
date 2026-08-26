# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from tests.utils.test_arguments_helpers import _arguments_module  # noqa: F401


def test_zero_kl_loss_is_disabled_before_reference_setup(arguments_module):
    args = SimpleNamespace(use_kl_loss=True, kl_loss_coef=0.0, kl_coef=0.0)

    changed = arguments_module._normalize_zero_kl_loss_args(args)

    assert changed is True
    assert args.use_kl_loss is False


def test_zero_kl_without_ref_load_preserves_valid_raw_actor_checkpoint(arguments_module, tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
    args = SimpleNamespace(
        use_kl_loss=True,
        kl_loss_coef=0.0,
        kl_coef=0.0,
        load=str(tmp_path),
        ref_load=None,
        hf_checkpoint="/hf-checkpoint",
        megatron_to_hf_mode="raw",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        ref_ckpt_step=None,
        start_rollout_id=None,
    )

    arguments_module._normalize_zero_kl_loss_args(args)
    arguments_module._resolve_checkpoint_load_args(args)

    assert args.use_kl_loss is False
    assert args.load == str(tmp_path)
    assert args.no_load_optim is False
    assert args.no_load_rng is False
    assert args.finetune is False


def test_raw_mode_without_load_or_ref_load_fails_before_checkpoint_loading(arguments_module):
    args = SimpleNamespace(
        load=None,
        ref_load=None,
        hf_checkpoint="/hf-checkpoint",
        megatron_to_hf_mode="raw",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        ref_ckpt_step=None,
        start_rollout_id=None,
    )

    with pytest.raises(ValueError, match="requires an existing Megatron checkpoint"):
        arguments_module._resolve_checkpoint_load_args(args)


def test_raw_mode_with_invalid_load_falls_back_to_ref_load(arguments_module, tmp_path):
    ref_load = tmp_path / "reference"
    args = SimpleNamespace(
        load=str(tmp_path / "missing-actor"),
        ref_load=str(ref_load),
        hf_checkpoint="/hf-checkpoint",
        megatron_to_hf_mode="raw",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        ref_ckpt_step=7,
        ckpt_step=None,
        start_rollout_id=None,
    )

    arguments_module._resolve_checkpoint_load_args(args)

    assert args.load == str(ref_load)
    assert args.no_load_optim is True
    assert args.no_load_rng is True
    assert args.finetune is True
    assert args.ckpt_step == 7
    assert args.start_rollout_id == 0


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


@pytest.mark.parametrize("correction", ["use_tis", "keep_old_actor", "use_rollout_logprobs"])
def test_hybrid_interval_publication_accepts_behavior_policy_correction(arguments_module, correction):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=False,
        use_tis=False,
        keep_old_actor=False,
        use_rollout_logprobs=False,
        max_staleness=0,
    )
    setattr(args, correction, True)

    arguments_module._validate_hybrid_weight_publication_args(args)


def test_hybrid_interval_publication_rejects_uncorrected_old_log_probs(arguments_module):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=False,
        use_tis=False,
        keep_old_actor=False,
        use_rollout_logprobs=False,
        max_staleness=0,
    )

    with pytest.raises(ValueError, match="old log-probs require correction"):
        arguments_module._validate_hybrid_weight_publication_args(args)


@pytest.mark.parametrize("correction", ["keep_old_actor", "use_rollout_logprobs"])
def test_true_on_policy_interval_requires_tis(arguments_module, correction):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=True,
        use_tis=False,
        keep_old_actor=False,
        use_rollout_logprobs=False,
        max_staleness=0,
    )
    setattr(args, correction, True)

    with pytest.raises(ValueError, match="Enable --use-tis"):
        arguments_module._validate_hybrid_weight_publication_args(args)


@pytest.mark.parametrize(
    ("hybrid", "update_weights_interval"),
    [(False, 2), (True, 1)],
)
def test_interval_publication_without_skipped_hybrid_updates_needs_no_correction(
    arguments_module, hybrid, update_weights_interval
):
    args = SimpleNamespace(
        hybrid=hybrid,
        update_weights_interval=update_weights_interval,
        true_on_policy_mode=False,
        use_tis=False,
        keep_old_actor=False,
        use_rollout_logprobs=False,
        max_staleness=0,
    )

    arguments_module._validate_hybrid_weight_publication_args(args)


def test_true_on_policy_interval_accepts_tis(arguments_module):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=True,
        use_tis=True,
        keep_old_actor=False,
        use_rollout_logprobs=False,
        max_staleness=0,
    )

    arguments_module._validate_hybrid_weight_publication_args(args)


def test_stale_hybrid_interval_rejects_single_old_actor_snapshot(arguments_module):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=False,
        use_tis=False,
        keep_old_actor=True,
        use_rollout_logprobs=False,
        max_staleness=2,
    )

    with pytest.raises(ValueError, match="Enable --use-tis or --use-rollout-logprobs"):
        arguments_module._validate_hybrid_weight_publication_args(args)


@pytest.mark.parametrize("correction", ["use_tis", "use_rollout_logprobs"])
def test_stale_hybrid_interval_accepts_staleness_aware_correction(arguments_module, correction):
    args = SimpleNamespace(
        hybrid=True,
        update_weights_interval=2,
        true_on_policy_mode=False,
        use_tis=False,
        keep_old_actor=False,
        use_rollout_logprobs=False,
        max_staleness=2,
    )
    setattr(args, correction, True)

    arguments_module._validate_hybrid_weight_publication_args(args)
