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
    requests = []
    backend._batch_request = lambda endpoint, payload: requests.append((endpoint, payload)) or [object(), object()]
    monkeypatch.setattr(module.ray, "get", lambda value, **_kwargs: value)
    monkeypatch.setattr(module.dist, "broadcast", lambda *_args, **_kwargs: _Handle())

    tensors = [("a", torch.ones(4, dtype=torch.float32)), ("b", torch.ones(3, dtype=torch.float16))]
    backend._update_bucket_weights_from_distributed(tensors, weight_version=8)

    assert lock.releases == 1
    assert not lock.locked
    assert backend._weight_sync_metrics["broadcast_bucket_count"] == 1
    assert backend._weight_sync_metrics["broadcast_tensor_count"] == 2
    assert backend._weight_sync_metrics["broadcast_bytes"] == 22
    assert backend._weight_sync_metrics["fanout_bytes"] == 44
    assert requests[0][0] == "/update_weights_from_distributed"
    assert requests[0][1]["weight_version"] == "8"


def test_dcs_bucket_releases_lock_when_broadcast_fails(monkeypatch) -> None:
    lock = _Lock()
    backend = _backend(lock)
    monkeypatch.setattr(module.ray, "get", lambda value, **_kwargs: value)
    monkeypatch.setattr(module.dist, "broadcast", lambda *_args, **_kwargs: _Handle(fail=True))

    with pytest.raises(RuntimeError, match="broadcast failed"):
        backend._update_bucket_weights_from_distributed([("a", torch.ones(1))], weight_version=8)

    assert lock.releases == 1
    assert not lock.locked


def test_publication_control_only_calls_router_on_rank_zero(monkeypatch) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    calls = []
    payload = {"publication_id": "pub", "target_version": 3, "target_actor_step": 1}
    backend._router_publication_request = lambda action, value: calls.append((action, value)) or value
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda values, **_kwargs: None)
    monkeypatch.setattr(module, "get_gloo_group", lambda: object())

    result = backend._publication_control("prepare", payload)

    assert result == payload
    assert calls == [("prepare", payload)]


def test_publication_control_nonzero_rank_uses_broadcast_result(monkeypatch) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend._router_publication_request = lambda *_args, **_kwargs: pytest.fail("rank 1 called router")
    monkeypatch.setattr(module.dist, "get_rank", lambda: 1)

    def broadcast(values, **_kwargs):
        values[0] = {"result": {"publication_id": "pub"}, "error": None}

    monkeypatch.setattr(module.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(module, "get_gloo_group", lambda: object())

    assert backend._publication_control("prepare", {}) == {"publication_id": "pub"}


@pytest.mark.parametrize(
    ("action", "status", "current_version"),
    [("commit", "committed", 8), ("fail", "failed", 7)],
)
def test_publication_terminal_response_loss_is_resolved_from_router_state(action, status, current_version) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    payload = {"publication_id": "pub-8", "target_version": 8, "target_actor_step": 4}
    attempts = 0

    def response_lost(_action, _payload):
        nonlocal attempts
        attempts += 1
        raise TimeoutError(f"{action} response lost")

    backend._router_publication_request = response_lost
    backend._router_publication_state_request = lambda: {
        "current_version": current_version,
        "current_actor_step": 4,
        "publication": None,
        "publication_prepared": False,
        "last_publication": {"status": status, **payload},
    }

    result = backend._router_publication_request_resolved(action, payload)

    assert attempts == 2
    assert result["status"] == status
    assert result["publication_id"] == "pub-8"


def test_publication_prepare_response_loss_recovers_only_completed_prepare() -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    payload = {"publication_id": "pub-8", "target_version": 8, "target_actor_step": 4}
    backend._router_publication_request = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        TimeoutError("prepare response lost")
    )
    backend._router_publication_state_request = lambda: {
        "current_version": 7,
        "current_actor_step": 3,
        "publication": payload,
        "publication_prepared": True,
        "last_publication": None,
    }

    result = backend._router_publication_request_resolved("prepare", payload)

    assert result == {"status": "prepared", **payload}


@pytest.mark.parametrize(
    ("action", "state"),
    [
        (
            "prepare",
            {
                "current_version": 7,
                "publication": {"publication_id": "pub-8", "target_version": 8, "target_actor_step": 4},
                "publication_prepared": False,
                "last_publication": None,
            },
        ),
        (
            "commit",
            {
                "current_version": 7,
                "publication": {"publication_id": "pub-8", "target_version": 8, "target_actor_step": 4},
                "publication_prepared": True,
                "last_publication": None,
            },
        ),
    ],
)
def test_publication_response_loss_never_guesses_an_unconfirmed_outcome(action, state) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    payload = {"publication_id": "pub-8", "target_version": 8, "target_actor_step": 4}
    backend._router_publication_request = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        TimeoutError(f"{action} response lost")
    )
    backend._router_publication_state_request = lambda: state

    with pytest.raises(RuntimeError, match=rf"publication {action} outcome is unconfirmed"):
        backend._router_publication_request_resolved(action, payload)


def test_publication_mismatched_success_response_is_not_accepted() -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    payload = {"publication_id": "pub-8", "target_version": 8, "target_actor_step": 4}
    backend._router_publication_request = lambda *_args, **_kwargs: {
        "status": "committed",
        "publication_id": "stale-publication",
        "target_version": 7,
        "target_actor_step": 3,
    }
    backend._router_publication_state_request = lambda: {
        "current_version": 7,
        "publication": payload,
        "publication_prepared": True,
        "last_publication": None,
    }

    with pytest.raises(RuntimeError, match="publication commit outcome is unconfirmed"):
        backend._router_publication_request_resolved("commit", payload)


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


def test_targeted_publication_requires_actor_step_before_mutating_backend() -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend.args = type(
        "Args",
        (),
        {
            "hybrid_dcs_weight_sync": True,
            "enable_cross_version_kv_continuation": True,
        },
    )()
    backend._active_publication = None

    with pytest.raises(ValueError, match="target_actor_step"):
        backend.update_weights_for_rollout(rollout_only=True)


def test_targeted_publication_forwards_actor_step_to_backend_impl(monkeypatch) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend.args = type(
        "Args",
        (),
        {
            "hybrid_dcs_weight_sync": True,
            "enable_cross_version_kv_continuation": True,
        },
    )()
    backend._active_publication = None
    captured = {}

    def update_impl(rollout_only, actor_fwd_only, *, target_actor_step):
        captured.update(
            rollout_only=rollout_only,
            actor_fwd_only=actor_fwd_only,
            target_actor_step=target_actor_step,
        )
        return {"target_actor_step": target_actor_step}

    monkeypatch.setattr(backend, "_update_weights_for_rollout_impl", update_impl)

    result = backend.update_weights_for_rollout(rollout_only=True, target_actor_step=4)

    assert result == {"target_actor_step": 4}
    assert captured == {"rollout_only": True, "actor_fwd_only": False, "target_actor_step": 4}


def test_targeted_publication_prepare_failure_does_not_skip_next_version(monkeypatch) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend.args = type(
        "Args",
        (),
        {
            "hybrid_dcs_weight_sync": True,
            "enable_cross_version_kv_continuation": True,
            "cross_version_kv_max_gap": 1,
            "max_staleness": 2,
            "targeted_retirement_timeout_seconds": 15.0,
        },
    )()
    backend.weight_version = 7
    backend._active_publication = None
    backend._group_name = "test"
    backend._is_pp_src_rank = False
    backend._lora_merge_mode = False
    backend._lora_adapter_mode = False
    backend._lora_adapter_full = None
    backend._lora_skip_rollout_base = False
    events = []
    backend._materialize_weight_source = lambda: [("weight", object())]
    backend._update_weight_from_distributed = lambda *_args, **_kwargs: events.append("weight_transfer") or 0
    backend._rollout_control = lambda endpoint, *_args, **_kwargs: events.append(endpoint)

    publication_calls = []
    prepare_attempts = 0

    def publication_control(action, payload):
        nonlocal prepare_attempts
        publication_calls.append((action, dict(payload)))
        events.append(action)
        if action == "prepare":
            prepare_attempts += 1
            if prepare_attempts == 1:
                raise TimeoutError("prepare response timed out")
            return {
                "publication_id": "pub-8",
                "target_version": payload["target_version"],
                "target_actor_step": payload["target_actor_step"],
            }
        return {"ok": True}

    backend._publication_control = publication_control
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(module.device_utils, "empty_cache", lambda: None)

    with pytest.raises(TimeoutError, match="prepare response timed out"):
        backend.update_weights_for_rollout(rollout_only=True, target_actor_step=4)

    assert backend.weight_version == 7

    metrics = backend.update_weights_for_rollout(rollout_only=True, target_actor_step=4)

    assert backend.weight_version == 8
    assert metrics["weight_version"] == 8
    prepare_calls = [payload for action, payload in publication_calls if action == "prepare"]
    assert [payload["target_version"] for payload in prepare_calls] == [8, 8]
    commit_calls = [payload for action, payload in publication_calls if action == "commit"]
    assert commit_calls == [
        {
            "publication_id": "pub-8",
            "target_version": 8,
            "target_actor_step": 4,
        }
    ]
    assert events[-5:] == [
        "prepare",
        "/pause_generation",
        "weight_transfer",
        "commit",
        "/continue_generation",
    ]


@pytest.mark.parametrize("failure_stage", ["pause", "commit"])
def test_targeted_publication_irreversible_failure_keeps_fence_and_does_not_resume(
    monkeypatch, failure_stage: str
) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend.args = type(
        "Args",
        (),
        {
            "hybrid_dcs_weight_sync": True,
            "enable_cross_version_kv_continuation": True,
            "cross_version_kv_max_gap": 1,
            "max_staleness": 2,
            "targeted_retirement_timeout_seconds": 15.0,
        },
    )()
    backend.weight_version = 7
    backend._active_publication = None
    backend._publication_irreversible = False
    backend._group_name = "test"
    backend._is_pp_src_rank = False
    backend._lora_merge_mode = False
    backend._lora_adapter_mode = False
    backend._lora_adapter_full = None
    backend._lora_skip_rollout_base = False
    events = []
    backend._materialize_weight_source = lambda: [("weight", object())]
    backend._update_weight_from_distributed = lambda *_args, **_kwargs: events.append("weight_transfer") or 0

    def rollout_control(endpoint, *_args, **_kwargs):
        events.append(endpoint)
        if endpoint == "/pause_generation" and failure_stage == "pause":
            raise RuntimeError("pause outcome unconfirmed")

    backend._rollout_control = rollout_control

    def publication_control(action, payload):
        events.append(action)
        if action == "prepare":
            return {
                "publication_id": payload["publication_id"],
                "target_version": payload["target_version"],
                "target_actor_step": payload["target_actor_step"],
            }
        if failure_stage == "commit":
            raise RuntimeError("commit outcome unconfirmed")
        return {"ok": True}

    backend._publication_control = publication_control
    backend._router_publication_request_resolved = lambda *_args, **_kwargs: pytest.fail(
        "irreversible publication attempted fail cleanup"
    )
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(module.device_utils, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match=f"{failure_stage} outcome unconfirmed"):
        backend.update_weights_for_rollout(rollout_only=True, target_actor_step=4)

    assert backend.weight_version == 7
    expected_events = ["prepare", "/pause_generation"]
    if failure_stage == "commit":
        expected_events.extend(["weight_transfer", "commit"])
    assert events == expected_events


def test_targeted_publication_continue_failure_preserves_committed_version(monkeypatch) -> None:
    backend = module.DeviceDirectBackend.__new__(module.DeviceDirectBackend)
    backend.args = type(
        "Args",
        (),
        {
            "hybrid_dcs_weight_sync": True,
            "enable_cross_version_kv_continuation": True,
            "cross_version_kv_max_gap": 1,
            "max_staleness": 2,
            "targeted_retirement_timeout_seconds": 15.0,
        },
    )()
    backend.weight_version = 7
    backend._active_publication = None
    backend._publication_irreversible = False
    backend._group_name = "test"
    backend._is_pp_src_rank = False
    backend._lora_merge_mode = False
    backend._lora_adapter_mode = False
    backend._lora_adapter_full = None
    backend._lora_skip_rollout_base = False
    events = []
    backend._materialize_weight_source = lambda: [("weight", object())]
    backend._update_weight_from_distributed = lambda *_args, **_kwargs: events.append("weight_transfer") or 0

    def rollout_control(endpoint, *_args, **_kwargs):
        events.append(endpoint)
        if endpoint == "/continue_generation":
            raise RuntimeError("continue failed")

    backend._rollout_control = rollout_control

    def publication_control(action, payload):
        events.append(action)
        if action == "prepare":
            return {
                "publication_id": payload["publication_id"],
                "target_version": payload["target_version"],
                "target_actor_step": payload["target_actor_step"],
            }
        return {"status": "committed", **payload}

    backend._publication_control = publication_control
    backend._router_publication_request_resolved = lambda *_args, **_kwargs: pytest.fail(
        "committed publication attempted fail cleanup"
    )
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(module.device_utils, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="continue failed"):
        backend.update_weights_for_rollout(rollout_only=True, target_actor_step=4)

    assert backend.weight_version == 8
    assert events == [
        "prepare",
        "/pause_generation",
        "weight_transfer",
        "commit",
        "/continue_generation",
    ]
