# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


try:
    import torch

    from relax.backends.megatron import actor as actor_module
    from relax.backends.megatron import loss as loss_module

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing Megatron/Ray dependencies")


def _hybrid_fast_path_args(**overrides):
    values = {
        "attention_dropout": 0.0,
        "compute_advantages_and_returns": True,
        "custom_megatron_before_log_prob_hook_path": None,
        "get_mismatch_metrics": False,
        "hidden_dropout": 0.0,
        "keep_old_actor": False,
        "kl_coef": 0.0,
        "lora_dropout": 0.0,
        "max_staleness": 2,
        "true_on_policy_mode": True,
        "use_kl_loss": False,
        "use_opd": False,
        "use_rollout_logprobs": False,
        "use_rollout_routing_replay": False,
        "use_routing_replay": False,
        "use_tis": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _rollout_plan(num_rollout_minis=1):
    return actor_module.RolloutMiniBatchPlan(
        num_rollout_minis=num_rollout_minis,
        mini_rollout_batch_size=32 // num_rollout_minis,
        fixed_n_samples_per_prompt=8,
        mini_global_samples=256 // num_rollout_minis,
        mini_local_sample_request=256 // num_rollout_minis,
    )


@pytest.mark.parametrize(
    (
        "use_routing_replay",
        "use_rollout_routing_replay",
        "expected_schedule",
        "expected_stage",
        "expect_fill",
        "expect_clear",
    ),
    [
        (False, False, "logprob", None, False, False),
        (True, False, "training", "record", False, False),
        (True, True, "training", "replay_forward", True, True),
    ],
)
def test_hybrid_forward_subbatch_wires_actor_logprob_schedule(
    monkeypatch,
    use_routing_replay,
    use_rollout_routing_replay,
    expected_schedule,
    expected_stage,
    expect_fill,
    expect_clear,
):
    args = SimpleNamespace(
        compute_advantages_and_returns=True,
        get_mismatch_metrics=False,
        keep_old_actor=False,
        log_probs_max_tokens_per_gpu=20480,
        max_tokens_per_gpu=12288,
        use_dynamic_batch_size=True,
        use_rollout_logprobs=False,
        use_rollout_routing_replay=use_rollout_routing_replay,
        use_routing_replay=use_routing_replay,
    )
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = args
    actor._active_model_tag = "actor"
    actor.model = object()
    actor.weights_backuper = SimpleNamespace(backup_tags=set())

    training_schedule = ([object()], [4])
    logprob_schedule = ([object()], [2])
    records = {"clear_calls": 0, "fill_calls": 0, "forward": []}

    def fake_get_data_iterator(_args, _model, sub_batch, max_tokens_per_gpu=None):
        assert sub_batch is rollout_data
        if max_tokens_per_gpu is None:
            return training_schedule
        assert max_tokens_per_gpu == args.log_probs_max_tokens_per_gpu
        return logprob_schedule

    def fake_compute_log_prob(data_iterator, num_microbatches, store_prefix):
        schedule = "training" if data_iterator is training_schedule[0] else "logprob"
        records["forward"].append(
            (schedule, num_microbatches, actor_module.os.environ.get("ROUTING_REPLAY_STAGE"), store_prefix)
        )
        return {"log_probs": [0.0]}

    def fake_fill_routing_replay(data_iterator, num_microbatches, sub_batch):
        assert data_iterator is training_schedule[0]
        assert num_microbatches is training_schedule[1]
        assert sub_batch is rollout_data
        records["fill_calls"] += 1

    def fake_clear_all_forward():
        records["clear_calls"] += 1

    actor.compute_log_prob = fake_compute_log_prob
    actor.fill_routing_replay = fake_fill_routing_replay
    actor._switch_model = lambda target_tag: setattr(actor, "_active_model_tag", target_tag)
    monkeypatch.setattr(actor_module, "get_data_iterator", fake_get_data_iterator)
    monkeypatch.setattr(actor_module.RoutingReplay, "clear_all_forward", fake_clear_all_forward)
    monkeypatch.delenv("ROUTING_REPLAY_STAGE", raising=False)
    rollout_data = {"total_lengths": [1]}

    model_switches = actor._hybrid_forward_subbatch(rollout_data)

    assert model_switches == 1
    assert records["forward"] == [
        (
            expected_schedule,
            training_schedule[1] if expected_schedule == "training" else logprob_schedule[1],
            expected_stage,
            "",
        )
    ]
    assert records["fill_calls"] == int(expect_fill)
    assert records["clear_calls"] == int(expect_clear)


def test_hybrid_true_on_policy_skips_redundant_actor_forward(monkeypatch):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = _hybrid_fast_path_args()
    actor._active_model_tag = "actor"
    actor.model = object()
    actor.weights_backuper = SimpleNamespace(backup_tags=set())

    def unexpected_data_iterator(*_args, **_kwargs):
        raise AssertionError("the skipped actor forward must not build a data iterator")

    monkeypatch.setattr(actor_module, "get_data_iterator", unexpected_data_iterator)

    assert actor._should_reuse_hybrid_train_forward_log_probs(_rollout_plan())
    assert (
        actor._hybrid_forward_subbatch(
            {"total_lengths": [1]},
            reuse_train_forward_log_probs=True,
        )
        == 0
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"use_tis": False}, "requires --use-tis"),
        ({"hidden_dropout": 0.1}, "non-zero dropout"),
        ({"keep_old_actor": True}, "keep-old-actor"),
        ({"kl_coef": 0.01}, "reward KL"),
        ({"use_routing_replay": True}, "routing replay"),
        ({"use_rollout_routing_replay": True}, "routing replay"),
        ({"use_rollout_logprobs": True}, "different old-policy sources"),
        ({"use_kl_loss": True}, "KL loss"),
        ({"use_opd": True}, "OPD"),
        ({"custom_megatron_before_log_prob_hook_path": "hook.py"}, "before-log-prob hook"),
    ],
)
def test_hybrid_true_on_policy_rejects_non_equivalent_modes(overrides, match):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = _hybrid_fast_path_args(**overrides)

    with pytest.raises(ValueError, match=match):
        actor._should_reuse_hybrid_train_forward_log_probs(_rollout_plan())


def test_hybrid_true_on_policy_requires_one_optimizer_mini():
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = _hybrid_fast_path_args()

    with pytest.raises(ValueError, match="one optimizer mini"):
        actor._should_reuse_hybrid_train_forward_log_probs(_rollout_plan(num_rollout_minis=2))


def test_hybrid_true_on_policy_rejects_nonlocal_rollout_log_probs_under_cp(monkeypatch):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        dynamic_context_parallel=False,
        is_vl_model=True,
        qkv_format="thd",
        uses_unsplit_forward=False,
    )
    monkeypatch.setattr(actor_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(actor_module.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2)

    with pytest.raises(RuntimeError, match="CP-aligned rollout_log_probs"):
        actor._validate_hybrid_rollout_log_probs(
            {
                "rollout_log_probs": [torch.zeros(10)],
                "response_lengths": [10],
                "total_lengths": [20],
            }
        )


@pytest.mark.parametrize("reuse_train_forward_log_probs", [False, True])
def test_hybrid_stream_forward_merges_chunks_before_one_advantage_and_train_call(
    monkeypatch, reuse_train_forward_log_probs
):
    args = SimpleNamespace(
        attention_dropout=0.0,
        compute_advantages_and_returns=True,
        custom_megatron_before_log_prob_hook_path=None,
        debug_train_only=True,
        dynamic_context_parallel=False,
        get_mismatch_metrics=False,
        global_batch_size=256,
        hidden_dropout=0.0,
        hybrid_stream_forward=True,
        is_vl_model=True,
        keep_old_actor=False,
        kl_coef=0.0,
        lora_dropout=0.0,
        max_staleness=2,
        multimodal_keys=None,
        n_samples_per_prompt=8,
        num_iters_per_train_update=2,
        num_rollout=1,
        num_steps_per_rollout=None,
        ref_update_interval=None,
        rollout_batch_size=32,
        rotate_ckpt=False,
        save=None,
        save_interval=None,
        qkv_format="thd",
        true_on_policy_mode=reuse_train_forward_log_probs,
        use_kl_loss=False,
        use_opd=False,
        use_rollout_logprobs=False,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        use_tis=True,
        uses_unsplit_forward=False,
    )
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = args
    actor._active_model_tag = "actor"
    actor.flops_counter = None
    actor.model = object()
    actor.optimizer = object()
    actor.opt_param_scheduler = object()
    actor.prof = SimpleNamespace(step=lambda rollout_id: None)
    actor.rollout_data_postprocess = None
    actor.tokenizer = None
    actor.weights_backuper = SimpleNamespace(backup_tags=set(), backup=lambda tag: None)

    events = {"advantage_calls": 0, "forward_chunk_sizes": [], "train_calls": 0}
    timer_state = SimpleNamespace(seq_lens=None, images_seqlens=None, audio_seqlens=None)

    def fake_forward(
        sub_batch,
        skip_redundant_model_switch=False,
        reuse_train_forward_log_probs=False,
    ):
        assert skip_redundant_model_switch
        assert reuse_train_forward_log_probs == args.true_on_policy_mode
        events["forward_chunk_sizes"].append(len(sub_batch["total_lengths"]))
        if not reuse_train_forward_log_probs:
            sub_batch["log_probs"] = [0.0] * len(sub_batch["total_lengths"])
        return 0

    def fake_advantages(_args, rollout_data):
        events["advantage_calls"] += 1
        assert len(rollout_data["total_lengths"]) == 256
        assert rollout_data[actor_module.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY] == [256]
        rollout_data["advantages"] = [0.0] * 256

    def fake_get_data_iterator(_args, _model, rollout_data):
        assert len(rollout_data["advantages"]) == 256
        assert rollout_data[actor_module.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY] == [256]
        return [object()], [1]

    def fake_train(_rollout_id, _model, _optimizer, _scheduler, _iterator, _num_microbatches):
        assert _num_microbatches == [1]
        events["train_calls"] += 1

    def fake_all_gather_object(output, value, group=None):
        output[0] = value

    actor._hybrid_forward_subbatch = fake_forward
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", fake_advantages)
    monkeypatch.setattr(actor_module, "compute_rollout_step", lambda _args, rollout_id: rollout_id)
    monkeypatch.setattr(actor_module, "get_data_iterator", fake_get_data_iterator)
    monkeypatch.setattr(
        actor_module,
        "get_debug_data",
        lambda _args, _rollout_id, batch_size, dp_rank: {
            "tokens": [[1] * 20] * batch_size,
            "total_lengths": [20] * batch_size,
            "response_lengths": [10] * batch_size,
            "rollout_log_probs": [torch.zeros(1)] * batch_size,
        },
    )
    monkeypatch.setattr(actor_module, "inverse_timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "log_perf_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "log_rollout_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "post_process_rollout_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "Timer", lambda: timer_state)
    monkeypatch.setattr(actor_module, "train", fake_train)
    monkeypatch.setattr(actor_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(
        actor_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 1,
    )
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(actor_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(actor_module.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        actor_module.mpu,
        "get_data_parallel_group",
        lambda with_context_parallel=False: object(),
    )
    monkeypatch.setattr(actor_module.tracking_utils, "flush_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module.train_dump_utils, "save_debug_train_data", lambda *_args, **_kwargs: None)

    actor.train_hybrid(0)

    assert events == {
        "advantage_calls": 1,
        "forward_chunk_sizes": [128, 128],
        "train_calls": 1,
    }


def test_true_on_policy_advantages_use_rollout_log_probs_only_as_zero_kl_template(monkeypatch):
    args = SimpleNamespace(
        advantage_estimator="grpo",
        is_vl_model=False,
        kl_coef=0.0,
        normalize_advantages=False,
        qkv_format="thd",
        true_on_policy_mode=True,
        use_opd=False,
        use_rollout_logprobs=False,
        uses_unsplit_forward=False,
    )
    rollout_data = {
        "loss_masks": [torch.ones(2), torch.ones(1)],
        "response_lengths": [2, 1],
        "rewards": [1.5, -0.5],
        "rollout_log_probs": [torch.tensor([-0.2, -0.4]), torch.tensor([-0.1])],
        "total_lengths": [2, 1],
    }
    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True)

    loss_module.compute_advantages_and_returns(args, rollout_data)

    assert "log_probs" not in rollout_data
    assert torch.equal(rollout_data["advantages"][0], torch.tensor([1.5, 1.5]))
    assert torch.equal(rollout_data["advantages"][1], torch.tensor([-0.5]))
    assert all(torch.equal(a, r) for a, r in zip(rollout_data["advantages"], rollout_data["returns"], strict=True))


def test_reward_kl_rejects_missing_actor_log_probs(monkeypatch):
    args = SimpleNamespace(
        advantage_estimator="grpo",
        is_vl_model=False,
        kl_coef=0.1,
        normalize_advantages=False,
        qkv_format="thd",
        true_on_policy_mode=True,
        use_opd=False,
        use_rollout_logprobs=False,
        uses_unsplit_forward=False,
    )
    rollout_data = {
        "loss_masks": [torch.ones(2)],
        "response_lengths": [2],
        "rewards": [1.0],
        "rollout_log_probs": [torch.tensor([-0.2, -0.4])],
        "total_lengths": [2],
    }
    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True)

    with pytest.raises(RuntimeError, match="Reward KL requires actor log_probs"):
        loss_module.compute_advantages_and_returns(args, rollout_data)


def test_true_on_policy_train_forward_matches_explicit_actor_forward_with_tis(monkeypatch):
    def fake_get_log_probs_and_entropy(logits, **_kwargs):
        return torch.empty(0), {"entropy": [torch.zeros_like(logits)], "log_probs": [logits]}

    def reducer(values):
        return values.mean()

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *_args, **_kwargs: reducer)

    common_args = {
        "advantage_estimator": "grpo",
        "calculate_per_token_loss": True,
        "custom_pg_loss_reducer_function_path": None,
        "custom_tis_function_path": None,
        "entropy_coef": 0.0,
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
        "get_mismatch_metrics": False,
        "opd_loss_coef": 0.0,
        "opd_token_selection": None,
        "qkv_format": "thd",
        "tis_clip": 10.0,
        "tis_clip_low": 0.1,
        "use_kl_loss": False,
        "use_opsm": False,
        "use_rollout_logprobs": False,
        "use_tis": True,
    }
    initial_log_probs = torch.tensor([-0.2, -0.4])
    rollout_log_probs = [torch.tensor([-0.5, -0.7])]
    common_batch = {
        "advantages": [torch.tensor([1.0, -0.5])],
        "loss_masks": [torch.ones(2)],
        "response_lengths": [2],
        "rollout_log_probs": rollout_log_probs,
        "total_lengths": [2],
        "unconcat_tokens": [torch.tensor([1, 2])],
    }

    explicit_logits = initial_log_probs.clone().requires_grad_(True)
    explicit_batch = dict(common_batch, log_probs=[initial_log_probs.clone()])
    explicit_loss, explicit_metrics = loss_module.policy_loss_function(
        SimpleNamespace(**common_args, true_on_policy_mode=False),
        explicit_batch,
        explicit_logits,
        reducer,
    )
    explicit_loss.backward()

    inline_logits = initial_log_probs.clone().requires_grad_(True)
    inline_loss, inline_metrics = loss_module.policy_loss_function(
        SimpleNamespace(**common_args, true_on_policy_mode=True),
        common_batch,
        inline_logits,
        reducer,
    )
    inline_loss.backward()

    assert torch.allclose(inline_loss, explicit_loss)
    assert torch.allclose(inline_logits.grad, explicit_logits.grad)
    for metric in ("pg_loss", "ppo_kl", "tis", "mismatch_kl"):
        assert torch.allclose(inline_metrics[metric], explicit_metrics[metric])
