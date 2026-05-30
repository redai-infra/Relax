# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
import types
from argparse import Namespace

import pytest


def _import_resolve_global_batch_size(monkeypatch):
    device_module = types.ModuleType("relax.utils.device")
    device_module.get_dist_backend = lambda: "nccl"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device_module)

    server_args_module = types.ModuleType("sglang.srt.server_args")
    server_args_module.ServerArgs = type("ServerArgs", (), {})
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_module)

    launch_router_module = types.ModuleType("sglang_router.launch_router")
    launch_router_module.RouterArgs = type("RouterArgs", (), {})
    monkeypatch.setitem(sys.modules, "sglang_router", types.ModuleType("sglang_router"))
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router_module)

    sys.modules.pop("relax.utils.arguments", None)
    from relax.utils.arguments import _resolve_global_batch_size

    return _resolve_global_batch_size


def test_resolve_global_batch_size_rejects_non_divisible_num_steps_per_rollout(monkeypatch) -> None:
    resolve_global_batch_size = _import_resolve_global_batch_size(monkeypatch)
    args = Namespace(
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        num_steps_per_rollout=3,
        global_batch_size=None,
    )

    with pytest.raises(AssertionError, match="must be divisible by num_steps_per_rollout"):
        resolve_global_batch_size(args)


def test_resolve_global_batch_size_rejects_too_many_num_steps_per_rollout(monkeypatch) -> None:
    resolve_global_batch_size = _import_resolve_global_batch_size(monkeypatch)
    args = Namespace(
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        num_steps_per_rollout=9,
        global_batch_size=None,
    )

    with pytest.raises(AssertionError, match="must be divisible by num_steps_per_rollout"):
        resolve_global_batch_size(args)


def test_resolve_global_batch_size_rejects_non_positive_num_steps_per_rollout(monkeypatch) -> None:
    resolve_global_batch_size = _import_resolve_global_batch_size(monkeypatch)
    args = Namespace(
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        num_steps_per_rollout=0,
        global_batch_size=None,
    )

    with pytest.raises(AssertionError, match="num_steps_per_rollout must be positive"):
        resolve_global_batch_size(args)


def test_resolve_global_batch_size_rejects_non_positive_global_batch_size(monkeypatch) -> None:
    resolve_global_batch_size = _import_resolve_global_batch_size(monkeypatch)
    args = Namespace(
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        num_steps_per_rollout=None,
        global_batch_size=0,
    )

    with pytest.raises(AssertionError, match="global_batch_size must be positive"):
        resolve_global_batch_size(args)


def test_resolve_global_batch_size_derives_global_batch_size_from_num_steps_per_rollout(monkeypatch) -> None:
    resolve_global_batch_size = _import_resolve_global_batch_size(monkeypatch)
    args = Namespace(
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        num_steps_per_rollout=4,
        global_batch_size=None,
    )

    resolve_global_batch_size(args)

    assert args.global_batch_size == 2
