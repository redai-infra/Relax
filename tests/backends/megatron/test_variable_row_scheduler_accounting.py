# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_ROWS_ENV = "RELAX_AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE"


def _load_model_helpers():
    path = REPO_ROOT / "relax" / "backends" / "megatron" / "model.py"
    names = {
        "_normalize_scheduler_increments",
        "_initial_variable_row_tracking_step",
        "_advance_variable_row_tracking_step",
        "_restore_agentic_variable_row_resume_cursor",
        "_optimizer_scheduler_training_plan",
        "get_optimizer_param_scheduler",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *definitions,
        ],
        type_ignores=[],
    )

    class _Scheduler:
        def __init__(self, optimizer, **kwargs):
            self.optimizer = optimizer
            self.kwargs = kwargs

    namespace = {
        "AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV": MAX_ROWS_ENV,
        "Namespace": Namespace,
        "OptimizerParamScheduler": _Scheduler,
        "Sequence": list,
        "os": os,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


HELPERS = _load_model_helpers()


def test_variable_row_scheduler_uses_each_window_actual_rows():
    assert HELPERS._normalize_scheduler_increments([8, 5, 3], 3) == (8, 5, 3)
    assert HELPERS._normalize_scheduler_increments(None, 3) is None

    with pytest.raises(ValueError, match="length must match"):
        HELPERS._normalize_scheduler_increments([8, 5], 3)
    with pytest.raises(ValueError, match="positive integers"):
        HELPERS._normalize_scheduler_increments([8, 0, 3], 3)


def test_variable_row_tracking_cursor_is_monotonic_across_different_window_counts():
    args = Namespace()
    scheduler = SimpleNamespace(num_steps=100)
    cursor = HELPERS._initial_variable_row_tracking_step(args, scheduler)
    observed_steps = []

    # Rollout A has three windows, rollout B only one.  The old
    # rollout_id * current_window_count formula would collide at step 1.
    for increment in (8, 5, 3, 7):
        observed_steps.append(cursor)
        cursor = HELPERS._advance_variable_row_tracking_step(args, cursor, increment)

    assert observed_steps == [100, 108, 113, 116]
    assert len(observed_steps) == len(set(observed_steps))
    assert observed_steps == sorted(observed_steps)
    assert args._agentic_variable_row_tracking_samples == 123

    # Re-entry uses the actor-persisted cursor instead of the scheduler fallback.
    assert HELPERS._initial_variable_row_tracking_step(args, SimpleNamespace(num_steps=0)) == 123


def test_variable_row_checkpoint_resume_restores_scheduler_cursor(monkeypatch):
    monkeypatch.setenv(MAX_ROWS_ENV, "5")
    args = _scheduler_args()
    scheduler = SimpleNamespace(num_steps=321)

    assert not HELPERS._restore_agentic_variable_row_resume_cursor(args, scheduler, iteration=0)
    assert HELPERS._restore_agentic_variable_row_resume_cursor(args, scheduler, iteration=3)
    assert args._agentic_variable_row_tracking_samples == 321


def test_variable_row_checkpoint_resume_fails_closed_on_missing_or_conflicting_cursor(monkeypatch):
    monkeypatch.setenv(MAX_ROWS_ENV, "5")

    with pytest.raises(RuntimeError, match="positive checkpoint scheduler num_steps"):
        HELPERS._restore_agentic_variable_row_resume_cursor(
            _scheduler_args(), SimpleNamespace(num_steps=0), iteration=3
        )
    with pytest.raises(RuntimeError, match="requires a restored optimizer scheduler"):
        HELPERS._restore_agentic_variable_row_resume_cursor(_scheduler_args(), None, iteration=3)

    args = _scheduler_args()
    args._agentic_variable_row_tracking_samples = 320
    with pytest.raises(RuntimeError, match="cursor conflicts"):
        HELPERS._restore_agentic_variable_row_resume_cursor(args, SimpleNamespace(num_steps=321), iteration=3)


def test_checkpoint_resume_preserves_legacy_paths(monkeypatch):
    args = _scheduler_args()
    scheduler = SimpleNamespace(num_steps=321)

    monkeypatch.delenv(MAX_ROWS_ENV, raising=False)
    assert not HELPERS._restore_agentic_variable_row_resume_cursor(args, scheduler, iteration=3)

    monkeypatch.setenv(MAX_ROWS_ENV, "5")
    assert not HELPERS._restore_agentic_variable_row_resume_cursor(
        _scheduler_args(group_rm=False), scheduler, iteration=3
    )


def _scheduler_args(**overrides):
    values = {
        "num_rollout": 3,
        "rollout_batch_size": 2,
        "n_samples_per_prompt": 4,
        "global_batch_size": 8,
        "group_rm": True,
        "agentic_custom_advantage_path": "examples.graphgpo.custom_advantage.compute_custom_advantage",
        "use_dynamic_batch_size": True,
        "fully_async": False,
        "hybrid": False,
        "lr_decay_iters": None,
        "lr_wsd_decay_iters": None,
        "lr_warmup_fraction": 0.1,
        "lr_warmup_iters": 0,
        "lr_warmup_init": 0.0,
        "lr": 1e-6,
        "min_lr": 0.0,
        "lr_decay_style": "constant",
        "start_weight_decay": 0.1,
        "end_weight_decay": 0.1,
        "weight_decay_incr_style": "constant",
        "use_checkpoint_opt_param_scheduler": False,
        "override_opt_param_scheduler": False,
        "lr_wsd_decay_style": "constant",
    }
    values.update(overrides)
    return Namespace(**values)


def test_variable_row_scheduler_horizon_uses_max_real_turn_rows(monkeypatch):
    monkeypatch.setenv(MAX_ROWS_ENV, "5")
    args = _scheduler_args()

    # Per rollout: 2 * 4 trajectories * 5 rows = 40 rows = 5 windows.
    assert HELPERS._optimizer_scheduler_training_plan(args) == (15, 120)
    scheduler = HELPERS.get_optimizer_param_scheduler(args, optimizer=object())

    assert args.train_iters == 15
    assert args.lr_decay_iters == 15
    assert scheduler.kwargs["lr_decay_steps"] == 120
    assert scheduler.kwargs["wd_incr_steps"] == 120
    assert scheduler.kwargs["lr_warmup_steps"] == 12


def test_scheduler_plan_preserves_legacy_trajectory_units(monkeypatch):
    monkeypatch.delenv(MAX_ROWS_ENV, raising=False)
    args = _scheduler_args(group_rm=False)

    assert HELPERS._optimizer_scheduler_training_plan(args) == (3, 24)
    scheduler = HELPERS.get_optimizer_param_scheduler(args, optimizer=object())
    assert args.train_iters == 3
    assert scheduler.kwargs["lr_decay_steps"] == 24
    assert scheduler.kwargs["wd_incr_steps"] == 24


def test_scheduler_plan_rejects_invalid_row_bound_and_hybrid(monkeypatch):
    monkeypatch.setenv(MAX_ROWS_ENV, "0")
    with pytest.raises(ValueError, match="positive integer"):
        HELPERS._optimizer_scheduler_training_plan(_scheduler_args())

    monkeypatch.setenv(MAX_ROWS_ENV, "5")
    with pytest.raises(ValueError, match="synchronous training only"):
        HELPERS._optimizer_scheduler_training_plan(_scheduler_args(fully_async=True, hybrid=True))
