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
    training = types.ModuleType("megatron.training")
    global_vars = types.ModuleType("megatron.training.global_vars")
    tracking_utils = types.ModuleType("relax.utils.tracking_utils")
    ray = types.ModuleType("ray")

    class _PackedSeqParams:
        pass

    core.mpu = mpu
    packed_seq_params.PackedSeqParams = _PackedSeqParams
    global_vars.get_args = lambda: None

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "megatron.core.packed_seq_params": packed_seq_params,
        "megatron.training": training,
        "megatron.training.global_vars": global_vars,
        "relax.utils.tracking_utils": tracking_utils,
        "ray": ray,
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


def test_hybrid_forward_chunk_plan_matches_producer_granularity(monkeypatch):
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

    assert chunk_plan.chunks_per_mini == 2
    assert chunk_plan.chunk_global_samples == 128
    assert chunk_plan.chunk_local_samples == 128
    assert [
        chunk_plan.transfer_queue_batch_index(mini_index, chunk_index)
        for mini_index in range(3)
        for chunk_index in range(chunk_plan.chunks_per_mini)
    ] == [0, 1, 2, 3, 4, 5]


def test_hybrid_forward_chunk_plan_supports_four_prompt_aligned_stages(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        global_batch_size=256,
        num_steps_per_rollout=None,
        num_iters_per_train_update=4,
    )
    rollout_plan = data_module.build_rollout_minibatch_plan(args, dp_size=1)

    chunk_plan = data_module.build_hybrid_forward_chunk_plan(args, rollout_plan, dp_size=1)

    assert chunk_plan.chunks_per_mini == 4
    assert chunk_plan.chunk_global_samples == 64
    assert chunk_plan.chunk_local_samples == 64
    assert [
        chunk_plan.transfer_queue_batch_index(mini_index, chunk_index)
        for mini_index in range(2)
        for chunk_index in range(chunk_plan.chunks_per_mini)
    ] == list(range(8))


@pytest.mark.parametrize(
    ("num_iters", "global_batch_size", "n_samples_per_prompt", "error"),
    [
        (1, 256, 8, "num_iters_per_train_update >= 2"),
        (3, 256, 8, "mini_global_samples must be divisible"),
        (4, 24, 8, "must preserve complete prompt groups"),
    ],
)
def test_hybrid_forward_chunk_plan_rejects_unsafe_boundaries(
    monkeypatch,
    num_iters,
    global_batch_size,
    n_samples_per_prompt,
    error,
):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=global_batch_size // n_samples_per_prompt,
        n_samples_per_prompt=n_samples_per_prompt,
        global_batch_size=global_batch_size,
        num_steps_per_rollout=None,
        num_iters_per_train_update=num_iters,
    )
    rollout_plan = data_module.build_rollout_minibatch_plan(args, dp_size=1)

    with pytest.raises(ValueError, match=error):
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


@pytest.mark.parametrize(
    ("second_batch", "error"),
    [
        (
            {"tokens": ["c"], "total_lengths": [3]},
            "schema mismatch",
        ),
        (
            {
                "tokens": ["c", "unexpected"],
                "total_lengths": [3],
                "weight_version": 7,
            },
            "Per-sample rollout field",
        ),
    ],
)
def test_concat_rollout_batches_rejects_incomplete_chunks(monkeypatch, second_batch, error):
    data_module = _load_data_module(monkeypatch)

    with pytest.raises(ValueError, match=error):
        data_module.concat_rollout_batches(
            [
                {
                    "tokens": ["a", "b"],
                    "total_lengths": [1, 2],
                    "weight_version": 7,
                },
                second_batch,
            ]
        )


def test_canonicalize_rollout_chunks_reorders_every_sample_field(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    chunks = [
        (
            {
                "tokens": ["token-12", "token-10"],
                "total_lengths": [12, 10],
                "multimodal_train_inputs": [{"image": "12"}, {"image": "10"}],
                "scores": torch.tensor([[12], [10]]),
                "array": data_module.np.array([[12], [10]]),
                "weight_version": 7,
            },
            [12, 10],
        ),
        (
            {
                "tokens": ["token-13", "token-11"],
                "total_lengths": [13, 11],
                "multimodal_train_inputs": [{"image": "13"}, {"image": "11"}],
                "scores": torch.tensor([[13], [11]]),
                "array": data_module.np.array([[13], [11]]),
                "weight_version": 7,
            },
            [13, 11],
        ),
    ]

    merged, global_indexes = data_module.canonicalize_rollout_chunks(chunks, expected_sample_count=4)

    assert global_indexes == [10, 11, 12, 13]
    assert merged["tokens"] == ["token-10", "token-11", "token-12", "token-13"]
    assert merged["total_lengths"] == [10, 11, 12, 13]
    assert merged["multimodal_train_inputs"] == [
        {"image": "10"},
        {"image": "11"},
        {"image": "12"},
        {"image": "13"},
    ]
    assert torch.equal(merged["scores"], torch.tensor([[10], [11], [12], [13]]))
    assert merged["array"].tolist() == [[10], [11], [12], [13]]
    assert merged["weight_version"] == 7


@pytest.mark.parametrize(
    ("chunks", "expected_count", "error"),
    [
        (
            [
                ({"total_lengths": [1, 1]}, [10, 11]),
                ({"total_lengths": [1, 1]}, [11, 12]),
            ],
            4,
            "contain duplicates",
        ),
        (
            [({"total_lengths": [1, 1]}, [10])],
            2,
            "length mismatch",
        ),
        (
            [({"total_lengths": [1]}, [10])],
            2,
            "sample count mismatch",
        ),
    ],
)
def test_canonicalize_rollout_chunks_rejects_invalid_metadata(
    monkeypatch,
    chunks,
    expected_count,
    error,
):
    data_module = _load_data_module(monkeypatch)

    with pytest.raises((ValueError, TypeError), match=error):
        data_module.canonicalize_rollout_chunks(chunks, expected_count)


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
