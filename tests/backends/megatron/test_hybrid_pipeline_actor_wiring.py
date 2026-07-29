# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch


try:
    actor_module = importlib.import_module("relax.backends.megatron.actor")
except ImportError:
    actor_module = None

pytestmark = pytest.mark.skipif(
    actor_module is None,
    reason="Megatron/Ray/TransferQueue runtime dependencies are required for the actor wiring test",
)


def test_hybrid_pipeline_runtime_rechecks_supported_parallel_topology(monkeypatch):
    assert actor_module is not None
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        num_iters_per_train_update=2,
        n_samples_per_prompt=2,
        expert_tensor_parallel_size=1,
        offload_train=False,
        offload_rollout=False,
        compute_advantages_and_returns=True,
        use_dynamic_batch_size=True,
    )
    actor.weights_backuper = SimpleNamespace(backup_tags={"actor"})
    rollout_plan = SimpleNamespace(
        num_rollout_minis=1,
        mini_global_samples=4,
        mini_local_sample_request=4,
        fixed_n_samples_per_prompt=2,
    )
    monkeypatch.setattr(actor_module.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(actor_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(actor_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(actor_module.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(actor_module.mpu, "get_expert_model_parallel_world_size", lambda: 1)

    chunk_plan = actor._validate_hybrid_pipeline_runtime(rollout_plan, dp_size=1)

    assert chunk_plan.chunks_per_mini == 2
    assert chunk_plan.chunk_local_samples == 2

    actor.args.use_dynamic_batch_size = False
    with pytest.raises(RuntimeError, match="requires use_dynamic_batch_size=True"):
        actor._validate_hybrid_pipeline_runtime(rollout_plan, dp_size=1)
    actor.args.use_dynamic_batch_size = True

    monkeypatch.setattr(actor_module.mpu, "get_context_parallel_world_size", lambda: 1)
    with pytest.raises(RuntimeError, match="requires TP=2, CP=2, EP=1, and ETP=1"):
        actor._validate_hybrid_pipeline_runtime(rollout_plan, dp_size=1)


def _rollout_batch(start: int, count: int) -> dict:
    values = list(range(start, start + count))
    return {
        "tokens": [[value] for value in values],
        "total_lengths": [1] * count,
        "response_lengths": [1] * count,
        "loss_masks": [[1] for _ in values],
        "rollout_log_probs": [[0.0] for _ in values],
        "rewards": [1.0] * count,
        "raw_reward": [1.0] * count,
    }


def test_debug_rollout_chunking_slices_every_per_sample_container():
    assert actor_module is not None
    shared = {"source": "frozen-rollout"}
    rollout_data = {
        "tokens": [[0], [1], [2], [3]],
        "total_lengths": [1, 1, 1, 1],
        "tensor_field": torch.arange(8).reshape(4, 2),
        "array_field": np.arange(12).reshape(4, 3),
        "tuple_field": ("a", "b", "c", "d"),
        "shared": shared,
    }

    chunks = actor_module.MegatronTrainRayActor._split_rollout_batch(rollout_data, 2)

    assert len(chunks) == 2
    assert chunks[0]["tokens"] == [[0], [1]]
    assert chunks[1]["tokens"] == [[2], [3]]
    assert torch.equal(chunks[0]["tensor_field"], torch.tensor([[0, 1], [2, 3]]))
    assert torch.equal(chunks[1]["tensor_field"], torch.tensor([[4, 5], [6, 7]]))
    np.testing.assert_array_equal(chunks[0]["array_field"], np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(chunks[1]["array_field"], np.arange(6, 12).reshape(2, 3))
    assert chunks[0]["tuple_field"] == ("a", "b")
    assert chunks[1]["tuple_field"] == ("c", "d")
    assert chunks[0]["shared"] is shared
    assert chunks[1]["shared"] is shared


@pytest.mark.parametrize(
    ("pipeline_enabled", "expected_fetch_sizes", "expected_forward_count"),
    [
        (False, [4], 1),
        (True, [2, 2], 2),
    ],
)
def test_train_hybrid_wires_one_update_around_flagged_actor_chunks(
    monkeypatch,
    pipeline_enabled,
    expected_fetch_sizes,
    expected_forward_count,
):
    assert actor_module is not None
    events = []
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        global_batch_size=4,
        num_steps_per_rollout=None,
        num_iters_per_train_update=2,
        hybrid_pipeline_forward=pipeline_enabled,
        hybrid_pipeline_trace_dir=None,
        hybrid_pipeline_fetch_timeout_s=1.0,
        use_rollout_routing_replay=False,
        multimodal_keys={"image": "image"},
        use_opd=False,
        debug_train_only=False,
        compute_advantages_and_returns=True,
        use_routing_replay=False,
        ref_update_interval=None,
        num_rollout=1,
        save=None,
    )
    actor._hybrid_pipeline_chunk_plan = SimpleNamespace(
        chunks_per_mini=2,
        chunk_local_samples=2,
        transfer_queue_batch_index=lambda mini_index, chunk_index: mini_index * 2 + chunk_index,
    )
    actor._active_model_tag = "actor"
    actor.rollout_data_postprocess = None
    actor.model = actor.optimizer = actor.opt_param_scheduler = None
    actor.tokenizer = actor.flops_counter = None
    actor.prof = SimpleNamespace(step=lambda **_kwargs: events.append("profile"))
    actor.weights_backuper = SimpleNamespace(
        backup_tags={"actor"},
        backup=lambda tag: events.append(("backup", tag)),
    )

    def get_data(_task, _rollout_id, _fields, expected_samples, batch_index):
        events.append(("fetch", expected_samples, batch_index))
        start = batch_index * expected_samples
        return (
            _rollout_batch(start, expected_samples),
            SimpleNamespace(global_indexes=list(range(start, start + expected_samples))),
        )

    actor._get_data_from_transfer_queue = get_data
    actor.all_consumed = lambda *_args, **_kwargs: False
    actor._restore_hybrid_pipeline_actor = lambda **kwargs: events.append(("restore", kwargs["chunk_index"]))

    def forward_chunk(batch, **kwargs):
        events.append(("forward", kwargs["chunk_index"]))
        return [list(range(len(batch["total_lengths"])))]

    actor._hybrid_actor_forward_without_switch = forward_chunk
    actor._hybrid_forward_subbatch = lambda _batch, **kwargs: events.append(("forward", kwargs["chunk_index"]))
    actor._switch_model = lambda tag: events.append(("switch", tag))
    actor._wait_for_previous_eval = lambda: None
    actor._check_services_health = lambda: None
    actor.update_weights = lambda: events.append("update_weights")
    actor._run_step_evaluation = lambda *_args, **_kwargs: None

    monkeypatch.setattr(
        actor_module.mpu,
        "get_data_parallel_world_size",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_group", lambda **_kwargs: None)
    monkeypatch.setattr(actor_module, "timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "inverse_timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "emit_hybrid_pipeline_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        actor_module,
        "compute_advantages_and_returns",
        lambda _args, _batch: events.append("advantages"),
    )
    monkeypatch.setattr(actor_module, "log_rollout_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda *_args, **_kwargs: ([], []))

    def train(_rollout_id, _model, _optimizer, _scheduler, data_iterator, num_microbatches):
        events.append(
            (
                "optimizer_schedule",
                data_iterator[0].micro_batch_indices if pipeline_enabled else None,
                num_microbatches,
            )
        )
        events.append("optimizer")

    monkeypatch.setattr(actor_module, "train", train)
    monkeypatch.setattr(actor_module.train_dump_utils, "save_debug_train_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "Timer", lambda: SimpleNamespace(seq_lens=None))
    monkeypatch.setattr(actor_module, "log_perf_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module.tracking_utils, "flush_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "compute_rollout_step", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    def all_gather_object(output, value, **_kwargs):
        output[:] = [value]

    monkeypatch.setattr(actor_module.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **_kwargs: None)

    actor.train_hybrid(rollout_id=0)

    assert [event[1] for event in events if isinstance(event, tuple) and event[0] == "fetch"] == (expected_fetch_sizes)
    assert sum(isinstance(event, tuple) and event[0] == "forward" for event in events) == expected_forward_count
    assert sum(isinstance(event, tuple) and event[0] == "restore" for event in events) == int(pipeline_enabled)
    assert events.count("advantages") == 1
    assert events.count("optimizer") == 1
    assert events.count("update_weights") == 1
    optimizer_schedule = next(
        event for event in events if isinstance(event, tuple) and event[0] == "optimizer_schedule"
    )
    if pipeline_enabled:
        assert optimizer_schedule[1:] == ([[0, 1], [2, 3]], [2])
    else:
        assert optimizer_schedule[1:] == (None, [])
