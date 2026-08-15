# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
import types
from argparse import Namespace

import pytest


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


def _assert_window_invariants(window, dp_size, micro_batch_size):
    assert window.stop_row - window.start_row == window.actual_global_rows
    assert sum(window.actual_rows_per_dp) == window.actual_global_rows
    assert sum(window.padded_rows_per_dp) == window.padded_global_rows
    assert sum(window.padding_rows_per_dp) == window.padded_global_rows - window.actual_global_rows
    assert len(window.actual_rows_per_dp) == dp_size
    assert len(window.padded_rows_per_dp) == dp_size
    assert len(window.padding_rows_per_dp) == dp_size
    assert all(rows % micro_batch_size == 0 for rows in window.padded_rows_per_dp)
    assert all(
        actual + padding == padded
        for actual, padding, padded in zip(
            window.actual_rows_per_dp,
            window.padding_rows_per_dp,
            window.padded_rows_per_dp,
            strict=True,
        )
    )


def test_variable_row_plan_exact_batches_have_no_padding(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    plan = data_module.build_variable_row_minibatch_plan(
        actual_global_rows=512,
        global_batch_size=256,
        dp_size=8,
        micro_batch_size=32,
    )

    assert [(window.start_row, window.stop_row) for window in plan.windows] == [(0, 256), (256, 512)]
    assert [window.actual_global_rows for window in plan.windows] == [256, 256]
    assert plan.total_padded_global_rows == 512
    assert plan.total_padding_rows == 0
    for window in plan.windows:
        assert window.actual_rows_per_dp == (32,) * 8
        assert window.padding_rows_per_dp == (0,) * 8
        _assert_window_invariants(window, dp_size=8, micro_batch_size=32)


def test_variable_row_plan_pads_only_final_partial_batch(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    plan = data_module.build_variable_row_minibatch_plan(
        actual_global_rows=513,
        global_batch_size=256,
        dp_size=8,
        micro_batch_size=32,
    )

    assert [(window.start_row, window.stop_row) for window in plan.windows] == [
        (0, 256),
        (256, 512),
        (512, 513),
    ]
    assert [window.actual_global_rows for window in plan.windows] == [256, 256, 1]
    assert [window.padded_global_rows for window in plan.windows] == [256, 256, 256]
    assert plan.total_padded_global_rows == 768
    assert plan.total_padding_rows == 255

    final = plan.windows[-1]
    assert final.actual_rows_per_dp == (1, 0, 0, 0, 0, 0, 0, 0)
    assert final.padded_rows_per_dp == (32,) * 8
    assert final.padding_rows_per_dp == (31, 32, 32, 32, 32, 32, 32, 32)
    for window in plan.windows:
        _assert_window_invariants(window, dp_size=8, micro_batch_size=32)


def test_variable_row_plan_balances_real_rows_across_dp_ranks(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    plan = data_module.build_variable_row_minibatch_plan(
        actual_global_rows=5,
        global_batch_size=8,
        dp_size=2,
        micro_batch_size=2,
    )

    window = plan.windows[0]
    assert window.actual_rows_per_dp == (3, 2)
    assert window.padded_rows_per_dp == (4, 4)
    assert window.padding_rows_per_dp == (1, 2)
    assert max(window.actual_rows_per_dp) - min(window.actual_rows_per_dp) == 1
    assert plan.total_padding_rows == 3
    _assert_window_invariants(window, dp_size=2, micro_batch_size=2)


def test_variable_row_plan_supports_fewer_real_rows_than_dp_ranks(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    plan = data_module.build_variable_row_minibatch_plan(
        actual_global_rows=3,
        global_batch_size=16,
        dp_size=4,
        micro_batch_size=2,
    )

    window = plan.windows[0]
    assert window.actual_rows_per_dp == (1, 1, 1, 0)
    assert window.padded_rows_per_dp == (4, 4, 4, 4)
    assert window.padding_rows_per_dp == (3, 3, 3, 4)
    assert plan.total_padding_rows == 13
    _assert_window_invariants(window, dp_size=4, micro_batch_size=2)


def test_variable_row_plan_grid_preserves_all_rows_and_contiguous_boundaries(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    for dp_size in (1, 2, 4):
        for micro_batch_size in (1, 2, 4):
            for local_microbatches in (1, 2, 3):
                global_batch_size = dp_size * micro_batch_size * local_microbatches
                for actual_global_rows in range(1, 2 * global_batch_size + 3):
                    plan = data_module.build_variable_row_minibatch_plan(
                        actual_global_rows=actual_global_rows,
                        global_batch_size=global_batch_size,
                        dp_size=dp_size,
                        micro_batch_size=micro_batch_size,
                    )

                    assert plan.windows[0].start_row == 0
                    assert plan.windows[-1].stop_row == actual_global_rows
                    assert all(
                        left.stop_row == right.start_row
                        for left, right in zip(plan.windows, plan.windows[1:], strict=False)
                    )
                    assert sum(window.actual_global_rows for window in plan.windows) == actual_global_rows
                    assert sum(window.padded_global_rows for window in plan.windows) == plan.total_padded_global_rows
                    assert plan.total_padding_rows == plan.total_padded_global_rows - actual_global_rows
                    assert all(window.actual_global_rows == global_batch_size for window in plan.windows[:-1])
                    for window in plan.windows:
                        _assert_window_invariants(window, dp_size, micro_batch_size)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        (
            "actual_global_rows",
            {"actual_global_rows": 0, "global_batch_size": 8, "dp_size": 2, "micro_batch_size": 2},
        ),
        (
            "global_batch_size",
            {"actual_global_rows": 1, "global_batch_size": 0, "dp_size": 2, "micro_batch_size": 2},
        ),
        (
            "dp_size",
            {"actual_global_rows": 1, "global_batch_size": 8, "dp_size": 0, "micro_batch_size": 2},
        ),
        (
            "micro_batch_size",
            {"actual_global_rows": 1, "global_batch_size": 8, "dp_size": 2, "micro_batch_size": -1},
        ),
    ],
)
def test_variable_row_plan_rejects_non_positive_inputs(monkeypatch, name, kwargs):
    data_module = _load_data_module(monkeypatch)

    with pytest.raises(ValueError, match=rf"{name} must be positive"):
        data_module.build_variable_row_minibatch_plan(**kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("actual_global_rows", True),
        ("actual_global_rows", 1.0),
        ("global_batch_size", "8"),
        ("dp_size", False),
        ("micro_batch_size", 2.5),
    ],
)
def test_variable_row_plan_rejects_non_integer_inputs(monkeypatch, name, value):
    data_module = _load_data_module(monkeypatch)
    kwargs = {
        "actual_global_rows": 1,
        "global_batch_size": 8,
        "dp_size": 2,
        "micro_batch_size": 2,
    }
    kwargs[name] = value

    with pytest.raises(TypeError, match=rf"{name} must be an int"):
        data_module.build_variable_row_minibatch_plan(**kwargs)


def test_variable_row_plan_rejects_non_divisible_fixed_global_batch(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    with pytest.raises(ValueError, match="global_batch_size must be divisible by dp_size \\* micro_batch_size"):
        data_module.build_variable_row_minibatch_plan(
            actual_global_rows=7,
            global_batch_size=12,
            dp_size=2,
            micro_batch_size=4,
        )


def test_variable_row_helper_does_not_change_fixed_n_plan(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    fixed_plan = data_module.build_rollout_minibatch_plan(
        Namespace(
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            global_batch_size=32,
            num_steps_per_rollout=None,
        ),
        dp_size=2,
    )

    assert fixed_plan == data_module.RolloutMiniBatchPlan(
        num_rollout_minis=2,
        mini_rollout_batch_size=4,
        fixed_n_samples_per_prompt=8,
        mini_global_samples=32,
        mini_local_sample_request=16,
    )
