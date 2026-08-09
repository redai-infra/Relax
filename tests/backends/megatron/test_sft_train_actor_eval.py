# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""train_actor must trigger SFT eval every eval_interval steps when
configured."""

from argparse import Namespace

from relax.engine.sft.runtime import should_run_sft_eval, should_run_sft_predict


def _mk_actor_args():
    return Namespace(
        loss_type="sft",
        sft_objective="causal_lm",
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
    assert should_run_sft_eval(args, completed_steps=10) is True
    assert should_run_sft_eval(args, completed_steps=20) is True
    assert should_run_sft_eval(args, completed_steps=5) is False
    assert should_run_sft_eval(args, completed_steps=0) is False


def test_preference_eval_includes_true_baseline_and_final_independent_of_interval():
    args = _mk_actor_args()
    args.sft_objective = "reward_model"
    args.eval_interval = 200
    assert should_run_sft_eval(args, completed_steps=0) is True
    assert should_run_sft_eval(args, completed_steps=20) is True
    assert should_run_sft_eval(args, completed_steps=1) is False


def test_should_run_sft_eval_disabled_when_no_interval():
    args = _mk_actor_args()
    args.eval_interval = None
    assert should_run_sft_eval(args, completed_steps=10) is False


def test_should_run_sft_eval_disabled_when_no_eval_source():
    args = _mk_actor_args()
    args.eval_prompt_data = None
    assert should_run_sft_eval(args, completed_steps=10) is False


def test_should_run_sft_eval_disabled_for_non_sft():
    args = _mk_actor_args()
    args.loss_type = "policy_loss"
    assert should_run_sft_eval(args, completed_steps=10) is False


def test_should_run_sft_predict_uses_completed_step_boundaries():
    args = _mk_actor_args()
    args.sft_predict_interval = 10

    assert should_run_sft_predict(args, completed_steps=0) is False
    assert should_run_sft_predict(args, completed_steps=9) is False
    assert should_run_sft_predict(args, completed_steps=10) is True
    assert should_run_sft_predict(args, completed_steps=19) is False
    assert should_run_sft_predict(args, completed_steps=20) is True
