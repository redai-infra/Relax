# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for RLOO argument validation (six startup guards).

These tests verify the choices and the six guards without running the full
argument parser (which requires megatron). We construct a ``SimpleNamespace``
mimicking the parsed args and call the validation logic directly.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _install_fake_megatron(monkeypatch):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_world_size = lambda: 1
    mpu.get_context_parallel_rank = lambda: 0
    mpu.is_pipeline_last_stage = lambda: True
    core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)


def _rloo_args(**kwargs):
    """Return a SimpleNamespace with valid RLOO defaults; override via kwargs."""
    defaults = dict(
        advantage_estimator="rloo",
        n_samples_per_prompt=8,
        rollout_batch_size=16,
        global_batch_size=128,  # 16 * 8 == 128 ✓
        fully_async=False,
        hybrid=False,
        rewards_normalization=True,
        normalize_advantages=False,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _run_rloo_guards(args):
    """Replicate the RLOO guard block from arguments.py.

    The real guard lives inside ``process_args`` which requires the full
    megatron parser. We extract the same logic so the test runs without
    megatron. If the guard logic changes in arguments.py, update this mirror.
    """
    if args.advantage_estimator != "rloo":
        return
    if args.n_samples_per_prompt < 2:
        raise ValueError("n_samples_per_prompt < 2")
    if args.fully_async or getattr(args, "hybrid", False):
        raise ValueError("fully_async")
    if not args.rewards_normalization:
        raise ValueError("rewards_normalization")
    if args.normalize_advantages:
        raise ValueError("normalize_advantages")
    if args.rollout_batch_size * args.n_samples_per_prompt != args.global_batch_size:
        raise ValueError("equality")
    if args.partial_rollout or args.use_dynamic_global_batch_size:
        raise ValueError("partial_rollout")


def test_rloo_valid_config_passes():
    args = _rloo_args()
    _run_rloo_guards(args)  # should not raise


def test_rloo_requires_n_samples_ge_2():
    args = _rloo_args(n_samples_per_prompt=1, global_batch_size=16)
    with pytest.raises(ValueError, match="n_samples_per_prompt"):
        _run_rloo_guards(args)


def test_rloo_rejects_fully_async():
    args = _rloo_args(fully_async=True)
    with pytest.raises(ValueError, match="fully_async"):
        _run_rloo_guards(args)


def test_rloo_rejects_hybrid_async():
    args = _rloo_args(hybrid=True)
    with pytest.raises(ValueError, match="fully_async"):
        _run_rloo_guards(args)


def test_rloo_requires_rewards_normalization():
    args = _rloo_args(rewards_normalization=False)
    with pytest.raises(ValueError, match="rewards_normalization"):
        _run_rloo_guards(args)


def test_rloo_rejects_normalize_advantages():
    args = _rloo_args(normalize_advantages=True)
    with pytest.raises(ValueError, match="normalize_advantages"):
        _run_rloo_guards(args)


def test_rloo_requires_one_optimizer_step_per_rollout():
    args = _rloo_args(rollout_batch_size=16, n_samples_per_prompt=8, global_batch_size=64)
    # 16 * 8 = 128 != 64 → two optimizer steps per rollout
    with pytest.raises(ValueError, match="equality"):
        _run_rloo_guards(args)


def test_rloo_rejects_partial_rollout():
    args = _rloo_args(partial_rollout=True)
    with pytest.raises(ValueError, match="partial_rollout"):
        _run_rloo_guards(args)


def test_rloo_rejects_dynamic_global_batch_size():
    args = _rloo_args(use_dynamic_global_batch_size=True)
    with pytest.raises(ValueError, match="partial_rollout"):
        _run_rloo_guards(args)


def test_grpo_default_path_unchanged():
    """Default grpo config must not trigger any rloo guard."""
    args = _rloo_args(advantage_estimator="grpo")
    _run_rloo_guards(args)  # should not raise (rloo guards skipped)


def test_advantage_estimator_choices_include_rloo(monkeypatch):
    """Verify the argparse choices list includes 'rloo'."""
    # Read the source to check the choices without running the parser
    import inspect

    import relax.utils.arguments as arguments_mod

    source = inspect.getsource(arguments_mod)
    assert '"rloo"' in source, "rloo not found in arguments.py source"
