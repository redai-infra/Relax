# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for ``DeviceDirectBackend._update_rollout_engines`` recovery behavior.

These exercise the real method on a shell instance (``object.__new__``) with the
two collaborators it calls (``_healthcheck_rollout_engines`` /
``_remove_failed_engines``) stubbed, so we test the retry / prune / raise control
flow in isolation.

Regression target: previously the method raised ``RuntimeError`` unconditionally
after exhausting retries -- even when healthy engines remained -- which failed the
actor weight sync and escalated to a full global restart (progress lost). It must
now only raise when *no* healthy engine is left, prune the unhealthy ones, and
invalidate the cached topology signature so the group is rebuilt cleanly.
"""

from types import SimpleNamespace
from typing import Callable, Set

import pytest


# DeviceDirectBackend's module imports megatron.core at module level.
pytest.importorskip("megatron.core")

from relax.distributed.checkpoint_service.backends import device_direct
from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend


def _make_backend(engines: dict, health_fn: Callable[..., Set[int]]) -> DeviceDirectBackend:
    """Build a shell backend that only has the attributes the method
    touches."""
    backend = object.__new__(DeviceDirectBackend)
    backend.rollout_engines = dict(engines)
    backend.rollout_topology = {str(rank): {"rank": rank} for rank in engines}
    backend._rollout_topology_signature = "STALE_SIG"

    backend._healthcheck_rollout_engines = health_fn  # type: ignore[method-assign]

    def _remove(failed: Set[int]) -> None:
        for rank in set(failed):
            backend.rollout_engines.pop(rank, None)
            backend.rollout_topology.pop(str(rank), None)
            backend.rollout_topology.pop(rank, None)

    backend._remove_failed_engines = _remove  # type: ignore[method-assign]
    return backend


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(device_direct.time, "sleep", lambda *_a, **_k: None)


def _fixed(value: Set[int]) -> Callable[..., Set[int]]:
    return lambda *_a, **_k: set(value)


def _sequence(values):
    seq = list(values)

    def _fn(*_a, **_k):
        return set(seq.pop(0)) if seq else set()

    return _fn


def test_update_rollout_engines_all_healthy_returns_without_prune():
    backend = _make_backend({0: object(), 1: object()}, _fixed(set()))
    backend._update_rollout_engines(max_retries=3, retry_interval=0)
    assert set(backend.rollout_engines) == {0, 1}
    # No prune happened -> cached signature untouched.
    assert backend._rollout_topology_signature == "STALE_SIG"


def test_update_rollout_engines_dead_engine_seed_healthy_degrades_no_raise():
    # rank 1 (scale-out) always unhealthy, rank 0 (seed) healthy the whole time.
    backend = _make_backend({0: object(), 1: object()}, _fixed({1}))
    backend._update_rollout_engines(max_retries=3, retry_interval=0)
    # Dead engine pruned from both engines and topology, seed retained, no raise.
    assert set(backend.rollout_engines) == {0}
    assert "1" not in backend.rollout_topology
    # Signature bookkeeping is the caller's (init_process_group_for_rollout)
    # concern; this method does not touch it.
    assert backend._rollout_topology_signature == "STALE_SIG"


def test_update_rollout_engines_all_dead_raises():
    backend = _make_backend({0: object(), 1: object()}, _fixed({0, 1}))
    with pytest.raises(RuntimeError, match="No healthy rollout engines"):
        backend._update_rollout_engines(max_retries=2, retry_interval=0)
    assert backend.rollout_engines == {}


def test_update_rollout_engines_recovered_at_tail_keeps_all():
    # Unhealthy during the loop, but the final post-loop healthcheck sees it
    # recovered -> nothing pruned, nothing raised, signature untouched.
    health = _sequence([{1}, {1}, {1}, set()])
    backend = _make_backend({0: object(), 1: object()}, health)
    backend._update_rollout_engines(max_retries=3, retry_interval=0)
    assert set(backend.rollout_engines) == {0, 1}
    assert backend._rollout_topology_signature == "STALE_SIG"


def test_update_rollout_engines_transient_failure_recovers_in_loop():
    # Fails twice then recovers before retries are exhausted -> early return,
    # no prune (does not kill a merely-slow-loading engine).
    health = _sequence([{1}, {1}, set()])
    backend = _make_backend({0: object(), 1: object()}, health)
    backend._update_rollout_engines(max_retries=5, retry_interval=0)
    assert set(backend.rollout_engines) == {0, 1}
    assert backend._rollout_topology_signature == "STALE_SIG"


def test_update_rollout_engines_no_topology_raises():
    backend = _make_backend({}, _fixed(set()))
    backend.rollout_topology = {}
    with pytest.raises(RuntimeError, match="No rollout engines configured"):
        backend._update_rollout_engines(max_retries=2, retry_interval=0)


def test_topology_signature_distinguishes_pruned_from_full():
    # After a prune, ``init_process_group_for_rollout`` stores the signature of
    # the *pruned* topology (not the pre-prune full one). This asserts the two
    # signatures differ, so if the pruned engine is later re-listed the reuse
    # fast path sees a signature mismatch and rebuilds the group -- re-adding the
    # engine instead of orphaning it.
    full = {
        "0": {"ip": "host-a", "port": 100, "metadata": {"num_gpus_per_engine": 1}},
        "1": {"ip": "host-b", "port": 200, "metadata": {"num_gpus_per_engine": 1}},
    }
    pruned = {"0": full["0"]}
    sig_full = DeviceDirectBackend._rollout_topology_signature_of(full)
    sig_pruned = DeviceDirectBackend._rollout_topology_signature_of(pruned)
    assert sig_full != sig_pruned


def test_successful_group_build_records_signature_and_reuses_group(monkeypatch):
    backend = object.__new__(DeviceDirectBackend)
    backend.role_info = {"rank": 0}
    backend.args = SimpleNamespace(rollout_num_gpus_per_engine=1)
    backend.backend_type = "nccl"
    backend.rollout_engines = {}
    backend.rollout_topology = {}
    backend._rollout_topology_signature = None
    backend._model_update_groups = None

    created = []

    def _create(topology):
        created.append(dict(topology))
        backend.rollout_engines = {int(rank): object() for rank in topology}

    backend._create_rollout_engines = _create  # type: ignore[method-assign]
    backend._update_rollout_engines = lambda: None  # type: ignore[method-assign]
    backend._cleanup_rollout_engines = lambda: None  # type: ignore[method-assign]
    backend._healthcheck_rollout_engines = lambda: set()  # type: ignore[method-assign]
    backend._batch_request = lambda *_args, **_kwargs: []  # type: ignore[method-assign]
    backend._find_free_port_in_range = lambda *_args: 11000  # type: ignore[method-assign]

    monkeypatch.setattr(device_direct.mpu, "get_data_parallel_rank", lambda **_kwargs: 0)
    monkeypatch.setattr(device_direct.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(device_direct.mpu, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(device_direct.ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(device_direct.ray, "get", lambda value, **_kwargs: value)
    process_group = object()
    monkeypatch.setattr(device_direct, "init_process_group", lambda **_kwargs: process_group)

    topology = {
        "nodes": {
            "rollout": {
                "0": {"ip": "host-a", "port": 100, "metadata": {"num_gpus_per_engine": 1}},
                "1": {"ip": "host-b", "port": 200, "metadata": {"num_gpus_per_engine": 1}},
            }
        }
    }

    first = backend.init_process_group_for_rollout(topology)
    expected_signature = backend._rollout_topology_signature_of(topology["nodes"]["rollout"])
    assert first["group_reused"] is False
    assert backend._rollout_topology_signature == expected_signature
    assert len(created) == 1

    second = backend.init_process_group_for_rollout(topology)
    assert second["group_reused"] is True
    assert second["group_world_size"] == 3
    assert second["rollout_receiver_count"] == 2
    assert len(created) == 1
