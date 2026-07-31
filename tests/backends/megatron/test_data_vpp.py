import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch


def _load_data_module(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
    megatron_training = types.ModuleType("megatron.training")
    global_vars = types.ModuleType("megatron.training.global_vars")
    tracking_utils = types.ModuleType("relax.utils.tracking_utils")
    data_utils = types.ModuleType("relax.utils.data.data")
    seqlen_balancing = types.ModuleType("relax.utils.data.seqlen_balancing")
    metrics = types.ModuleType("relax.utils.metrics")
    metric_utils = types.ModuleType("relax.utils.metrics.metric_utils")
    timer = types.ModuleType("relax.utils.timer")
    relax_training = types.ModuleType("relax.utils.training")
    train_metric_utils = types.ModuleType("relax.utils.training.train_metric_utils")
    flops_counter = types.ModuleType("relax.utils.training.flops_counter")
    opd = types.ModuleType("relax.utils.opd")
    opd_utils = types.ModuleType("relax.utils.opd.opd_utils")

    class _PackedSeqParams:
        pass

    core.mpu = mpu
    packed_seq_params.PackedSeqParams = _PackedSeqParams
    global_vars.get_args = lambda: None
    data_utils.get_minimum_num_micro_batch_size = lambda samples, max_tokens: 1
    seqlen_balancing.get_seqlen_balanced_partitions = lambda samples, num_parts, equal_size=False: []
    metric_utils.compute_rollout_step = lambda args, rollout_id: rollout_id
    timer.Timer = type("_Timer", (), {})
    relax_training.train_metric_utils = train_metric_utils
    flops_counter.FlopsCounter = object
    opd_utils.OPD_ROLLOUT_LOG_SKIP_FIELDS = frozenset()

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "megatron.core.packed_seq_params": packed_seq_params,
        "megatron.training": megatron_training,
        "megatron.training.global_vars": global_vars,
        "relax.utils.tracking_utils": tracking_utils,
        "relax.utils.data.data": data_utils,
        "relax.utils.data.seqlen_balancing": seqlen_balancing,
        "relax.utils.metrics": metrics,
        "relax.utils.metrics.metric_utils": metric_utils,
        "relax.utils.timer": timer,
        "relax.utils.training": relax_training,
        "relax.utils.training.train_metric_utils": train_metric_utils,
        "relax.utils.training.flops_counter": flops_counter,
        "relax.utils.opd": opd,
        "relax.utils.opd.opd_utils": opd_utils,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("relax.backends.megatron.data", None)
    return importlib.import_module("relax.backends.megatron.data")


def test_vpp_microbatch_rounding_uses_ceil_multiple(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    rounded = data_module._round_up_to_microbatch_group(torch.tensor([1, 2, 3, 5]), microbatch_group_size=4)

    assert rounded.tolist() == [4, 4, 4, 8]


def test_rollout_minibatch_plan_derives_from_global_batch(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=32,
        num_steps_per_rollout=None,
    )

    plan = data_module.build_rollout_minibatch_plan(args, dp_size=2)

    assert plan.num_rollout_minis == 2
    assert plan.mini_rollout_batch_size == 4
    assert plan.mini_global_samples == 32
    assert plan.mini_local_sample_request == 16


def test_rollout_minibatch_plan_prefers_explicit_steps(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=12,
        n_samples_per_prompt=8,
        global_batch_size=None,
        num_steps_per_rollout=3,
    )

    plan = data_module.build_rollout_minibatch_plan(args, dp_size=2)

    assert plan.num_rollout_minis == 3
    assert plan.mini_rollout_batch_size == 4
    assert plan.mini_global_samples == 32
    assert plan.mini_local_sample_request == 16


def test_rollout_minibatch_plan_rejects_non_divisible_prompt_groups(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=9,
        n_samples_per_prompt=8,
        global_batch_size=None,
        num_steps_per_rollout=3,
    )

    with pytest.raises(ValueError, match="mini_rollout_batch_size must be divisible"):
        data_module.build_rollout_minibatch_plan(args, dp_size=2)


def test_hybrid_forward_chunk_plan_keeps_optimizer_boundary(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        global_batch_size=256,
        num_steps_per_rollout=None,
        num_iters_per_train_update=2,
    )
    rollout_plan = data_module.build_rollout_minibatch_plan(args, dp_size=1)

    chunk_plan = data_module.build_hybrid_forward_chunk_plan(args, rollout_plan, dp_size=1)

    assert rollout_plan.num_rollout_minis == 1
    assert rollout_plan.mini_local_sample_request == 256
    assert chunk_plan.chunks_per_rollout_mini == 2
    assert chunk_plan.chunk_global_samples == 128
    assert chunk_plan.chunk_local_samples == 128


def test_hybrid_forward_chunk_plan_scales_local_request_with_dp(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        global_batch_size=256,
        num_steps_per_rollout=None,
        num_iters_per_train_update=2,
    )
    rollout_plan = data_module.build_rollout_minibatch_plan(args, dp_size=2)

    chunk_plan = data_module.build_hybrid_forward_chunk_plan(args, rollout_plan, dp_size=2)

    assert rollout_plan.mini_local_sample_request == 128
    assert chunk_plan.chunk_local_samples == 64


def test_hybrid_forward_chunk_plan_rejects_split_prompt_groups(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=3,
        n_samples_per_prompt=8,
        global_batch_size=24,
        num_steps_per_rollout=None,
        num_iters_per_train_update=2,
    )
    rollout_plan = data_module.build_rollout_minibatch_plan(args, dp_size=1)

    with pytest.raises(ValueError, match="preserve complete prompt groups"):
        data_module.build_hybrid_forward_chunk_plan(args, rollout_plan, dp_size=1)


def test_concat_rollout_batches_preserves_order_and_scalar_metadata(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    merged = data_module.concat_rollout_batches(
        [
            {
                "tokens": ["a", "b"],
                "total_lengths": [1, 2],
                "scores": torch.tensor([[1], [2]]),
                "weight_version": 7,
            },
            {
                "tokens": ["c"],
                "total_lengths": [3],
                "scores": torch.tensor([[3]]),
                "weight_version": 7,
            },
        ]
    )

    assert merged["tokens"] == ["a", "b", "c"]
    assert merged["total_lengths"] == [1, 2, 3]
    assert torch.equal(merged["scores"], torch.tensor([[1], [2], [3]]))
    assert merged["weight_version"] == 7


def test_hybrid_actor_logprob_schedule_uses_forward_budget_without_routing_replay(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    training_schedule = ([object()], [4])
    logprob_schedule = ([object()], [2])

    selected = data_module.select_hybrid_actor_logprob_schedule(
        Namespace(use_routing_replay=False), training_schedule, logprob_schedule
    )

    assert selected is logprob_schedule


def test_hybrid_actor_logprob_schedule_preserves_training_partition_for_routing_replay(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    training_schedule = ([object()], [4])
    logprob_schedule = ([object()], [2])

    selected = data_module.select_hybrid_actor_logprob_schedule(
        Namespace(use_routing_replay=True), training_schedule, logprob_schedule
    )

    assert selected is training_schedule


def test_hybrid_actor_logprob_schedule_preserves_training_partition_for_rollout_routing_replay(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    training_schedule = ([object()], [4])
    logprob_schedule = ([object()], [2])

    selected = data_module.select_hybrid_actor_logprob_schedule(
        Namespace(use_routing_replay=True, use_rollout_routing_replay=True),
        training_schedule,
        logprob_schedule,
    )

    assert selected is training_schedule


def test_get_data_iterator_uses_rollout_mini_boundaries_with_balance_data(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 2,
        raising=False,
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    args = Namespace(
        balance_data=True,
        global_batch_size=32,
        micro_batch_size=4,
        use_dynamic_batch_size=False,
    )
    rollout_data = {
        "total_lengths": list(range(32)),
        data_module.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY: [16, 16],
    }

    data_iterators, num_microbatches = data_module.get_data_iterator(args, object(), rollout_data)

    assert num_microbatches == [4, 4]
    iterator = data_iterators[0]
    first_step = [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(4)]
    second_step = [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(4)]
    assert first_step[0] == [0, 1, 2, 3]
    assert first_step[-1] == [12, 13, 14, 15]
    assert second_step[0] == [16, 17, 18, 19]
    assert second_step[-1] == [28, 29, 30, 31]


def test_get_data_iterator_balance_data_without_boundaries_uses_regular_steps(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 2,
        raising=False,
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    args = Namespace(
        balance_data=True,
        global_batch_size=16,
        micro_batch_size=4,
        use_dynamic_batch_size=False,
    )
    rollout_data = {"total_lengths": list(range(16))}

    _, num_microbatches = data_module.get_data_iterator(args, object(), rollout_data)

    assert num_microbatches == [2, 2]
