# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.distributed.checkpoint_service.backends import device_direct as module  # noqa: E402


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self):
        return self._fn()


class _Lock:
    def __init__(self):
        self.locked = False
        self.releases = 0
        self.acquire = _RemoteMethod(self._acquire)
        self.release = _RemoteMethod(self._release)

    def _acquire(self):
        if self.locked:
            return False
        self.locked = True
        return True

    def _release(self):
        self.locked = False
        self.releases += 1


class _Handle:
    def __init__(self, *, fail=False):
        self._fail = fail

    def wait(self):
        if self._fail:
            raise RuntimeError("broadcast failed")


def _backend(lock):
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend.lock = lock
    backend.timeout_seconds = 300
    backend.weight_version = 7
    backend._group_name = "test"
    backend._model_update_groups = object()
    backend.rollout_topology = {"0": {}, "1": {}}
    backend._weight_sync_metrics = {
        "lock_wait_seconds": 0.0,
        "broadcast_seconds": 0.0,
        "receiver_finalize_seconds": 0.0,
        "broadcast_bucket_count": 0,
        "broadcast_tensor_count": 0,
        "broadcast_bytes": 0,
        "fanout_bytes": 0,
    }
    backend._batch_request = lambda *_args, **_kwargs: [object(), object()]
    return backend


def test_dcs_bucket_metrics_count_bytes_and_fanout(monkeypatch) -> None:
    lock = _Lock()
    backend = _backend(lock)
    monkeypatch.setattr(module.ray, "get", lambda value, **_kwargs: value)
    monkeypatch.setattr(module.dist, "broadcast", lambda *_args, **_kwargs: _Handle())

    tensors = [("a", torch.ones(4, dtype=torch.float32)), ("b", torch.ones(3, dtype=torch.float16))]
    backend._update_bucket_weights_from_distributed(tensors)

    assert lock.releases == 1
    assert not lock.locked
    assert backend._weight_sync_metrics["broadcast_bucket_count"] == 1
    assert backend._weight_sync_metrics["broadcast_tensor_count"] == 2
    assert backend._weight_sync_metrics["broadcast_bytes"] == 22
    assert backend._weight_sync_metrics["fanout_bytes"] == 44


def test_dcs_bucket_releases_lock_when_broadcast_fails(monkeypatch) -> None:
    lock = _Lock()
    backend = _backend(lock)
    monkeypatch.setattr(module.ray, "get", lambda value, **_kwargs: value)
    monkeypatch.setattr(module.dist, "broadcast", lambda *_args, **_kwargs: _Handle(fail=True))

    with pytest.raises(RuntimeError, match="broadcast failed"):
        backend._update_bucket_weights_from_distributed([("a", torch.ones(1))])

    assert lock.releases == 1
    assert not lock.locked


def test_publication_control_only_calls_router_on_rank_zero(monkeypatch) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    calls = []
    backend._router_publication_request = lambda action, payload: calls.append((action, payload)) or {"ok": True}
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda values, **_kwargs: None)
    monkeypatch.setattr(module, "get_gloo_group", lambda: object())

    result = backend._publication_control("prepare", {"target_version": 3})

    assert result == {"ok": True}
    assert calls == [("prepare", {"target_version": 3})]


def test_publication_control_nonzero_rank_uses_broadcast_result(monkeypatch) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend._router_publication_request = lambda *_args, **_kwargs: pytest.fail("rank 1 called router")
    monkeypatch.setattr(module.dist, "get_rank", lambda: 1)

    def broadcast(values, **_kwargs):
        values[0] = {"result": {"publication_id": "pub"}, "error": None}

    monkeypatch.setattr(module.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(module, "get_gloo_group", lambda: object())

    assert backend._publication_control("prepare", {}) == {"publication_id": "pub"}


def test_rollout_control_broadcasts_rank_zero_failure(monkeypatch) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend.timeout_seconds = 300
    backend._batch_request = lambda *_args, **_kwargs: [object()]
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda _value, **_kwargs: (_ for _ in ()).throw(RuntimeError("pause failed")),
    )
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda values, **_kwargs: None)

    with pytest.raises(RuntimeError, match="pause failed"):
        backend._rollout_control("/pause_generation", {"mode": "in_place"})


def test_rollout_control_uses_bounded_http_and_ray_deadlines(monkeypatch) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend.timeout_seconds = 300
    calls = {}

    def batch_request(endpoint, payload, *, timeout_seconds):
        calls["batch"] = (endpoint, payload, timeout_seconds)
        return [object()]

    def ray_get(value, *, timeout):
        calls["ray"] = (value, timeout)
        return []

    backend._batch_request = batch_request
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(module.ray, "get", ray_get)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda values, **_kwargs: None)

    backend._rollout_control("/pause_generation", {"mode": "in_place"})

    assert calls["batch"] == ("/pause_generation", {"mode": "in_place"}, 60.0)
    assert calls["ray"][1] == 65.0


def test_router_prepare_timeout_wraps_internal_retirement_deadline() -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend.timeout_seconds = 300
    backend.args = type("Args", (), {"sglang_router_ip": "127.0.0.1", "sglang_router_port": 9000})()
    calls = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class Client:
        def post(self, url, *, json, timeout):
            calls.update(url=url, json=json, timeout=timeout)
            return Response()

    backend.http_client = Client()

    result = backend._router_publication_request("prepare", {"timeout_seconds": 15.0})

    assert result == {"ok": True}
    assert calls["timeout"] == 20.0
