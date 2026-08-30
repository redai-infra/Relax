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


def test_opd_sample_mask_gates_loss_masks_per_sample(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    batch = {
        "loss_masks": [torch.ones(2), torch.ones(1), torch.ones(3), torch.ones(2)],
        data_module.OPD_SAMPLE_MASK: [True, False, True, False],
    }

    data_module._apply_opd_sample_mask(batch)

    assert torch.equal(batch["loss_masks"][0], torch.ones(2))
    assert torch.equal(batch["loss_masks"][1], torch.zeros(1))
    assert torch.equal(batch["loss_masks"][2], torch.ones(3))
    assert torch.equal(batch["loss_masks"][3], torch.zeros(2))


def test_opd_sample_mask_gates_loss_masks_after_microbatch_permutation(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    rollout_data = {
        "total_lengths": [4, 3, 6, 5],
        "loss_masks": [torch.ones(2), torch.ones(1), torch.ones(3), torch.ones(2)],
        data_module.OPD_SAMPLE_MASK: [True, False, True, False],
    }
    iterator = data_module.DataIterator(rollout_data, micro_batch_indices=[[1], [0], [3], [2]])

    for _ in range(4):
        batch = iterator.get_next(["total_lengths", "loss_masks", data_module.OPD_SAMPLE_MASK])
        data_module._apply_opd_sample_mask(batch)
        for loss_mask, sample_mask in zip(batch["loss_masks"], batch[data_module.OPD_SAMPLE_MASK], strict=True):
            assert torch.equal(
                loss_mask, torch.zeros(loss_mask.numel()) if not sample_mask else torch.ones_like(loss_mask)
            )


def test_opd_sample_mask_leaves_ordinary_batch_untouched(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    batch = {"loss_masks": [torch.ones(2), torch.ones(1)]}

    data_module._apply_opd_sample_mask(batch)

    assert torch.equal(batch["loss_masks"][0], torch.ones(2))
    assert torch.equal(batch["loss_masks"][1], torch.ones(1))


def test_opd_sample_mask_rejects_length_mismatch(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    batch = {
        "loss_masks": [torch.ones(2), torch.ones(1)],
        data_module.OPD_SAMPLE_MASK: [True],
    }

    with pytest.raises(ValueError, match="one scalar per sample"):
        data_module._apply_opd_sample_mask(batch)
