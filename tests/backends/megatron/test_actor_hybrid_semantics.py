# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


try:
    from relax.backends.megatron import actor as actor_module

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing Megatron/Ray dependencies")


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


def test_hybrid_stream_forward_merges_chunks_before_one_advantage_and_train_call(monkeypatch):
    args = SimpleNamespace(
        compute_advantages_and_returns=True,
        debug_train_only=True,
        global_batch_size=256,
        hybrid_stream_forward=True,
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
        use_opd=False,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
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

    def fake_forward(sub_batch, skip_redundant_model_switch=False):
        assert skip_redundant_model_switch
        events["forward_chunk_sizes"].append(len(sub_batch["total_lengths"]))
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
            "tokens": [[1]] * batch_size,
            "total_lengths": [1] * batch_size,
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
