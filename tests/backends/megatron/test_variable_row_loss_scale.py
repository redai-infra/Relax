# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_COUNTS_KEY = "rollout_mini_global_sample_counts"
VALID_ROWS_KEY = "agentic_variable_row_valid_rows"


def _load_data_definitions():
    path = REPO_ROOT / "relax" / "backends" / "megatron" / "data.py"
    names = {
        "DataIterator",
        "_build_rollout_mini_loss_scales",
        "get_rollout_valid_row_flags",
        "_rollout_metrics_view",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *definitions,
        ],
        type_ignores=[],
    )

    class _Torch:
        class Tensor:
            pass

    namespace = {
        "ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY": GLOBAL_COUNTS_KEY,
        "ROLLOUT_VALID_ROW_FLAGS_KEY": VALID_ROWS_KEY,
        "RolloutBatch": dict,
        "torch": _Torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


DATA = _load_data_definitions()


def test_variable_window_loss_scale_matches_full_gbs_and_actual_tail():
    scales = DATA._build_rollout_mini_loss_scales(
        num_microbatches=[2, 2],
        actual_global_sample_counts=[8, 5],
        global_batch_size=8,
        dp_size=2,
    )
    assert scales == [0.5, 0.5, 0.8, 0.8]
    assert (
        DATA._build_rollout_mini_loss_scales(
            num_microbatches=[2],
            actual_global_sample_counts=None,
            global_batch_size=8,
            dp_size=2,
        )
        is None
    )


def test_data_iterator_injects_scale_per_microbatch_and_reset_replays_schedule():
    rollout_data = {
        "total_lengths": [4] * 8,
        "loss_masks": [[1], [1], [1], [1], [1], [1], [1], [0]],
    }
    iterator = DATA.DataIterator(
        rollout_data,
        micro_batch_size=2,
        micro_batch_loss_scales=[0.5, 0.5, 0.8, 0.8],
    )

    batches = [iterator.get_next(["loss_masks"]) for _ in range(4)]
    assert [batch["__loss_scale__"] for batch in batches] == [0.5, 0.5, 0.8, 0.8]
    assert batches[-1]["loss_masks"] == [[1], [0]]
    assert batches[-1]["loss_masks"][-1][0] * batches[-1]["__loss_scale__"] == 0

    iterator.reset()
    assert iterator.get_next(["loss_masks"])["__loss_scale__"] == 0.5


def test_variable_row_validity_is_persisted_sliced_and_removed_from_metric_view():
    rollout_data = {
        "total_lengths": [7, 8, 3, 3],
        "response_lengths": [3, 4, 1, 1],
        "raw_reward": [1.0, 0.0, 0.0, 0.0],
        VALID_ROWS_KEY: [True, True, False, False],
    }
    assert DATA.get_rollout_valid_row_flags(rollout_data) == [True, True, False, False]

    iterator = DATA.DataIterator(rollout_data, micro_batch_size=2)
    assert iterator.get_next([VALID_ROWS_KEY])[VALID_ROWS_KEY] == [True, True]
    assert iterator.get_next([VALID_ROWS_KEY])[VALID_ROWS_KEY] == [False, False]

    metric_view, metric_rows = DATA._rollout_metrics_view(rollout_data)
    assert metric_rows == 2
    assert metric_view["total_lengths"] == [7, 8]
    assert metric_view["raw_reward"] == [1.0, 0.0]
    assert VALID_ROWS_KEY not in metric_view


def test_variable_row_validity_is_strict_and_legacy_view_is_unchanged():
    legacy = {"total_lengths": [7], "response_lengths": [3]}
    metric_view, metric_rows = DATA._rollout_metrics_view(legacy)
    assert metric_view is legacy
    assert metric_rows is None

    with pytest.raises(RuntimeError, match="one flag per row"):
        DATA.get_rollout_valid_row_flags({"total_lengths": [7, 8], VALID_ROWS_KEY: [True]})
    with pytest.raises(RuntimeError, match="only bool"):
        DATA.get_rollout_valid_row_flags({"total_lengths": [7], VALID_ROWS_KEY: [1]})


def test_vpp_iterators_have_independent_scale_offsets():
    rollout_data = {"total_lengths": [4] * 4, "tokens": [[1]] * 4}
    first = DATA.DataIterator(
        rollout_data,
        micro_batch_size=2,
        micro_batch_loss_scales=[0.5, 0.8],
    )
    second = DATA.DataIterator(
        rollout_data,
        micro_batch_size=2,
        micro_batch_loss_scales=[0.5, 0.8],
    )
    assert first.get_next(["tokens"])["__loss_scale__"] == 0.5
    assert first.get_next(["tokens"])["__loss_scale__"] == 0.8
    assert second.get_next(["tokens"])["__loss_scale__"] == 0.5


def test_normal_iterator_does_not_inject_explicit_loss_scale():
    iterator = DATA.DataIterator(
        {"total_lengths": [4, 4], "tokens": [[1], [2]]},
        micro_batch_size=1,
    )
    assert "__loss_scale__" not in iterator.get_next(["tokens"])


@pytest.mark.parametrize(
    ("num_microbatches", "actual_counts"),
    [
        ([2], [0]),
        ([2], [9]),
        ([2], [True]),
        ([2, 2], [8]),
    ],
)
def test_variable_window_loss_scale_rejects_invalid_counts(num_microbatches, actual_counts):
    with pytest.raises(ValueError):
        DATA._build_rollout_mini_loss_scales(
            num_microbatches=num_microbatches,
            actual_global_sample_counts=actual_counts,
            global_batch_size=8,
            dp_size=2,
        )
