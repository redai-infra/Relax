# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""train_actor must trigger SFT eval every eval_interval steps when
configured."""

from argparse import Namespace

import pytest


# Importing relax.backends.megatron.actor pulls in CUDA-only deps. Skip the
# whole module on CPU-only envs — matches the pattern used in
# tests/backends/megatron/test_sft_train_data_fields.py.
try:
    from relax.backends.megatron.actor import _should_run_sft_eval  # noqa: F401
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.megatron.actor unavailable: {_exc}", allow_module_level=True)


def _mk_actor_args():
    return Namespace(
        loss_type="sft",
        compute_advantages_and_returns=False,
        eval_prompt_data=["eval", "/dev/null"],
        eval_size=None,
        eval_interval=10,
        advantage_estimator="grpo",
        save=None,
        save_interval=None,
        rotate_ckpt=False,
        offload_train=False,
        offload_rollout=False,
        num_rollout=20,
    )


def test_should_run_sft_eval_at_interval_boundary():
    args = _mk_actor_args()
    assert _should_run_sft_eval(args, rollout_id=9) is True
    assert _should_run_sft_eval(args, rollout_id=19) is True
    assert _should_run_sft_eval(args, rollout_id=4) is False
    assert _should_run_sft_eval(args, rollout_id=0) is False


def test_should_run_sft_eval_disabled_when_no_interval():
    args = _mk_actor_args()
    args.eval_interval = None
    assert _should_run_sft_eval(args, rollout_id=9) is False


def test_should_run_sft_eval_disabled_when_no_eval_source():
    args = _mk_actor_args()
    args.eval_prompt_data = None
    assert _should_run_sft_eval(args, rollout_id=9) is False


def test_should_run_sft_eval_disabled_for_non_sft():
    args = _mk_actor_args()
    args.loss_type = "policy_loss"
    assert _should_run_sft_eval(args, rollout_id=9) is False
