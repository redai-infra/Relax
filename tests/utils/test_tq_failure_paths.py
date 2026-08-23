# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Failure-path tests for the TransferQueue dataplane enablement.

Covers the four gaps the maintainer review called out, which
``tests/utils/tq/test_config.py`` (pure config construction/validation) and
``test_tq_dataplane_behavior.py`` (real TQ on SimpleStorage) did not:

* timeout -- controller ``get_config`` timeout and attach-handshake timeout
* disconnect -- store errors surface instead of returning corrupt data
* retry/disconnect -- a transient get recovers and a dead peer fails loudly
* automatic degradation as pytest (was a manual two-node script)
* the controller reaper / teardown helpers (now in ``relax.utils.tq.lifecycle``)

Real Mooncake TCP/RDMA byte-exact and wire-level checks live in the single
cross-node C0/C1/C2 benchmark.  This module keeps CI-safe failure semantics and
uses real TransferQueue classes only where a stubbed store is sufficient.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from relax.utils.tq import lifecycle as tq_lifecycle


def _has_real_submodule(dotted: str) -> bool:
    """True only if a REAL transfer_queue submodule is importable.

    CI installs a single-file ``transfer_queue`` stub;
    ``transfer_queue.storage`` does not exist there, so tests that touch the
    real MooncakeStoreClient skip on CPU CI and run only where real
    TransferQueue is installed.
    """
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ValueError, TypeError):
        # CI's single-file transfer_queue stub returns a dummy for ``__path__``,
        # so find_spec on a submodule raises TypeError instead of returning None.
        return False


_REAL_MOONCAKE_CLIENT = _has_real_submodule("transfer_queue.storage.clients.mooncake_client")


def _raise(error: BaseException) -> None:
    raise error


# ---------------------------------------------------------------------------
# Controller lifecycle: reaper (timeout / half-initialised / healthy)
# ---------------------------------------------------------------------------


class TestReapUnusableController:
    """reap_unusable_tq_controller: only unusable controllers get killed."""

    @staticmethod
    def _fake_ray(monkeypatch, *, actor, get_result=None, get_raises=None):
        """Stub the ray module used by tq_lifecycle; record kill calls."""
        killed: list = []
        fake = MagicMock()
        # Exception handlers require real exception classes; a bare MagicMock
        # here would make ``except ray.exceptions.RayError`` invalid at runtime.
        fake.exceptions = tq_lifecycle.ray.exceptions
        if actor is None:
            fake.get_actor.side_effect = ValueError("actor not found")
        else:
            fake.get_actor.return_value = actor
        if get_raises is not None:
            fake.get.side_effect = get_raises
        else:
            fake.get.return_value = get_result
        fake.kill.side_effect = lambda handle: killed.append(handle)
        monkeypatch.setattr(tq_lifecycle, "ray", fake)
        monkeypatch.setattr(tq_lifecycle, "kill_tq_controller_and_wait", lambda *a, **k: killed.append("killed"))
        return killed

    def test_no_controller_is_a_noop(self, monkeypatch):
        killed = self._fake_ray(monkeypatch, actor=None)
        assert tq_lifecycle.reap_unusable_tq_controller() is False
        assert killed == []

    def test_healthy_controller_fails_without_being_killed(self, monkeypatch):
        killed = self._fake_ray(monkeypatch, actor=MagicMock(), get_result={"backend": {}})
        with pytest.raises(RuntimeError, match="exclusive, clean Ray cluster"):
            tq_lifecycle.reap_unusable_tq_controller()
        assert killed == []

    def test_half_initialised_controller_is_reaped(self, monkeypatch):
        """conf is None == actor created but store_config never ran (F10)."""
        killed = self._fake_ray(monkeypatch, actor=MagicMock(), get_result=None)
        assert tq_lifecycle.reap_unusable_tq_controller() is True
        assert killed == ["killed"]

    @pytest.mark.parametrize("error_kind", ["timeout", "dead-actor"])
    def test_unusable_controller_is_reaped(self, monkeypatch, error_kind):
        """An unresponsive/dead controller must not turn tq.init into a
        hang."""
        if error_kind == "timeout":
            error = tq_lifecycle.ray.exceptions.GetTimeoutError("private timeout detail")
        else:
            error = tq_lifecycle.ray.exceptions.RayActorError(error_msg="private actor detail")
        killed = self._fake_ray(monkeypatch, actor=MagicMock(), get_raises=error)
        assert tq_lifecycle.reap_unusable_tq_controller() is True
        assert killed == ["killed"]

    def test_other_ray_error_fails_closed_without_killing_or_leaking_detail(self, monkeypatch):
        private_detail = "private control-plane endpoint and traceback"
        error = tq_lifecycle.ray.exceptions.RayError(private_detail)
        killed = self._fake_ray(monkeypatch, actor=MagicMock(), get_raises=error)

        with pytest.raises(RuntimeError, match="Failed to inspect") as excinfo:
            tq_lifecycle.reap_unusable_tq_controller()

        assert killed == []
        assert private_detail not in str(excinfo.value)

    def test_stored_config_ray_error_is_sanitized(self, monkeypatch):
        private_detail = "private GCS endpoint and traceback"
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "get_actor",
            lambda *_args, **_kwargs: _raise(tq_lifecycle.ray.exceptions.RayError(private_detail)),
        )

        with pytest.raises(RuntimeError) as excinfo:
            tq_lifecycle._get_stored_config()

        assert private_detail not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Controller lifecycle: teardown unmounts the Mooncake segment
# ---------------------------------------------------------------------------


class TestCloseTqAndUnmount:
    """close_tq_and_unmount: tq.close() first, then unmount the segment."""

    @staticmethod
    def _fake_tq(monkeypatch, store_client):
        calls: list[str] = []
        fake = MagicMock()
        manager = SimpleNamespace() if store_client is None else SimpleNamespace(storage_client=store_client)
        fake.get_client.return_value = MagicMock(storage_manager=manager)
        fake.close.side_effect = lambda: calls.append("tq.close")
        monkeypatch.setattr(tq_lifecycle, "tq", fake)
        return calls

    def test_mooncake_segment_is_unmounted_after_close(self, monkeypatch):
        store_client = MagicMock()
        calls = self._fake_tq(monkeypatch, store_client=store_client)
        store_client.close.side_effect = lambda: calls.append("store.close")
        tq_lifecycle.close_tq_and_unmount()
        # Order matters: tq.close() still needs the store alive for remove_all().
        assert calls == ["tq.close", "store.close"]

    def test_simple_storage_teardown_is_noop_beyond_close(self, monkeypatch):
        calls = self._fake_tq(monkeypatch, store_client=None)
        tq_lifecycle.close_tq_and_unmount()
        assert calls == ["tq.close"]

    def test_uninitialised_tq_does_not_raise(self, monkeypatch):
        fake = MagicMock()
        fake.get_client.side_effect = AssertionError("Please initialize the TransferQueue first")
        monkeypatch.setattr(tq_lifecycle, "tq", fake)
        tq_lifecycle.close_tq_and_unmount()  # must not raise
        fake.close.assert_called_once()


# ---------------------------------------------------------------------------
# Bounded attach (worker-side tq.init used to hang forever)
# ---------------------------------------------------------------------------


class TestBoundedAttach:
    """attach_tq_client: one deadline for get_config wait and tq.init."""

    @pytest.mark.parametrize("value", ["soon", "nan", "inf", "-inf"])
    def test_attach_timeout_env_rejects_unusable_values(self, monkeypatch, value):
        monkeypatch.setenv("RELAX_TQ_ATTACH_TIMEOUT_SECONDS", value)
        with pytest.raises(RuntimeError, match="finite positive") as excinfo:
            tq_lifecycle._resolve_attach_timeout()
        assert value not in str(excinfo.value)

    def test_attach_timeout_env_override_is_used(self, monkeypatch):
        monkeypatch.setenv("RELAX_TQ_ATTACH_TIMEOUT_SECONDS", "12.5")
        assert tq_lifecycle._resolve_attach_timeout() == 12.5

    def test_bounded_init_times_out_on_hung_tq_init(self, monkeypatch):
        import time

        monkeypatch.setattr(tq_lifecycle.tq, "init", lambda conf: time.sleep(5))
        with pytest.raises(tq_lifecycle.TqAttachTimeout, match="did not finish"):
            tq_lifecycle._bounded_tq_init({}, time.monotonic() + 0.2, role="test")

    def test_bounded_init_propagates_worker_error(self, monkeypatch):
        import time

        def boom(conf):
            raise ValueError("bad conf")

        monkeypatch.setattr(tq_lifecycle.tq, "init", boom)
        with pytest.raises(ValueError, match="bad conf"):
            tq_lifecycle._bounded_tq_init({}, time.monotonic() + 5.0, role="test")

    def test_await_controller_config_times_out_without_actor(self, monkeypatch):
        import time

        def no_actor(name, namespace=None):
            raise ValueError("actor not found")

        monkeypatch.setattr(tq_lifecycle.ray, "get_actor", no_actor)
        with pytest.raises(tq_lifecycle.TqAttachTimeout, match="attach timed out"):
            tq_lifecycle._await_controller_config(time.monotonic() + 0.3)

    def test_cluster_attach_covers_all_nodes_with_hard_affinity_and_one_shot_workers(self, monkeypatch):
        remote_options = {}
        scheduling_strategies = []
        submitted_refs = []

        class _Task:
            def options(self, **options):
                scheduling_strategies.append(options["scheduling_strategy"])
                return self

            def remote(self, *_args):
                ref = object()
                submitted_refs.append(ref)
                return ref

        def record_remote_options(**options):
            remote_options.update(options)
            return lambda _function: _Task()

        monkeypatch.setattr(tq_lifecycle.ray, "remote", record_remote_options)
        node_ids = ["a" * 56, "b" * 56]
        monkeypatch.setattr(tq_lifecycle, "_alive_node_ids", lambda: node_ids)
        monkeypatch.setattr(tq_lifecycle.ray, "wait", lambda refs, **_kwargs: (list(refs), []))
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda _ref: None)

        assert tq_lifecycle.verify_cluster_attach({}, timeout=0.1) == []
        assert remote_options["max_calls"] == 1
        assert remote_options["max_retries"] == 0
        assert len(submitted_refs) == len(node_ids)
        assert [(strategy.node_id, strategy.soft) for strategy in scheduling_strategies] == [
            (node_id, False) for node_id in node_ids
        ]

    def test_cluster_attach_with_no_alive_nodes_fails_closed_without_scheduling(self, monkeypatch):
        monkeypatch.setattr(tq_lifecycle, "_alive_node_ids", lambda: [])
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "remote",
            lambda **_kwargs: pytest.fail("zero-node validation must not create a remote worker"),
        )
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "wait",
            lambda *_args, **_kwargs: pytest.fail("zero-node validation must not call ray.wait"),
        )

        assert tq_lifecycle.verify_cluster_attach({}, timeout=0.1) == ["cluster: no alive Ray nodes discovered"]

    def test_alive_node_without_node_id_fails_closed(self, monkeypatch):
        monkeypatch.setattr(tq_lifecycle.ray, "nodes", lambda: [{"Alive": True}])
        failures = tq_lifecycle.verify_cluster_attach({}, timeout=0.1)
        assert failures == ["cluster: node discovery failed (RuntimeError)"]

    @staticmethod
    def _capture_handshake(monkeypatch):
        """Return ``(captured_args, get_worker)`` for the real nested worker.

        ``verify_cluster_attach`` defines ``_handshake`` inline, so the only
        way to exercise its body — and therefore its ``finally`` detach — is to
        grab the function Ray's decorator receives.
        """
        captured: list[tuple] = []
        worker: list = []

        class _Task:
            def options(self, **_kwargs):
                return self

            def remote(self, *args):
                captured.append(args)
                return object()

        def fake_remote(**_options):
            def decorate(function):
                worker.append(function)
                return _Task()

            return decorate

        monkeypatch.setattr(tq_lifecycle.ray, "remote", fake_remote)
        # Ray validates node IDs as 28-byte hex, so use a well-formed one.
        monkeypatch.setattr(tq_lifecycle, "_alive_node_ids", lambda: ["a" * 56])
        monkeypatch.setattr(tq_lifecycle.ray, "wait", lambda refs, **kwargs: (list(refs), []))
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda ref: None)
        return captured, worker

    def test_handshake_checks_mooncake_config_only_for_mooncake(self, monkeypatch):
        """The Mooncake flag controls the manager/config assertion.

        SimpleStorage has no Mooncake protocol configuration to verify.
        """
        captured, _worker = self._capture_handshake(monkeypatch)

        mooncake_conf = {"backend": {"storage_backend": "MooncakeStore"}}
        assert tq_lifecycle.verify_cluster_attach(mooncake_conf, timeout=0.1) == []
        assert captured[-1][1] is True
        assert captured[-1][2] == 0.1

        simple_conf = {"backend": {"storage_backend": "SimpleStorage"}}
        assert tq_lifecycle.verify_cluster_attach(simple_conf, timeout=0.1) == []
        assert captured[-1][1] is False

    def _run_worker(self, monkeypatch, *, assert_error=None):
        """Execute the real ``_handshake`` body and record its call order."""
        _captured, worker = self._capture_handshake(monkeypatch)
        tq_lifecycle.verify_cluster_attach({"backend": {"storage_backend": "MooncakeStore"}}, timeout=0.1)
        assert worker, "Ray decorator never received the handshake function"

        events: list[str] = []

        def fake_attach(conf, *, role, timeout):
            events.append(f"attach:{role}:{timeout}")
            return object()

        def fake_assert():
            events.append("assert")
            if assert_error is not None:
                raise assert_error

        monkeypatch.setattr(tq_lifecycle, "attach_tq_client", fake_attach)
        monkeypatch.setattr(tq_lifecycle, "assert_mooncake_rdma_configured", fake_assert)
        monkeypatch.setattr(tq_lifecycle, "detach_tq_client", lambda: events.append("detach"))
        return worker[0], events

    @pytest.mark.parametrize(
        ("backend", "verify_mooncake", "assert_error"),
        [
            ("MooncakeStore", True, None),
            ("MooncakeStore", True, RuntimeError("protocol=tcp")),
            ("SimpleStorage", False, None),
        ],
        ids=["mooncake", "mooncake-rejected", "simple"],
    )
    def test_handshake_worker_always_detaches(self, monkeypatch, backend, verify_mooncake, assert_error):
        """A rejected transport must not leave this node's segment
        registered."""
        handshake, events = self._run_worker(monkeypatch, assert_error=assert_error)
        conf = {"backend": {"storage_backend": backend}}
        if assert_error is None:
            handshake(conf, verify_mooncake, 0.1)
        else:
            with pytest.raises(RuntimeError, match="protocol=tcp"):
                handshake(conf, verify_mooncake, 0.1)
        expected = ["attach:attach-handshake:0.1"]
        if verify_mooncake:
            expected.append("assert")
        assert events == [*expected, "detach"]

    def test_ready_task_failure_is_sanitized(self, monkeypatch):
        self._capture_handshake(monkeypatch)
        secret = "worker endpoint and traceback path must stay private"
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda _ref: _raise(RuntimeError(secret)))

        failures = tq_lifecycle.verify_cluster_attach({}, timeout=0.1)

        assert failures[0].endswith("handshake task failed (RuntimeError)")
        assert secret not in failures[0]

    def test_partial_scheduling_failure_cancels_submitted_workers(self, monkeypatch):
        first_ref = object()
        cancellations: list[tuple[object, bool]] = []

        class _Task:
            calls = 0

            def options(self, **_kwargs):
                return self

            def remote(self, *_args):
                self.calls += 1
                if self.calls == 1:
                    return first_ref
                raise RuntimeError("private scheduling detail")

        monkeypatch.setattr(tq_lifecycle.ray, "remote", lambda **_kwargs: lambda _function: _Task())
        monkeypatch.setattr(tq_lifecycle, "_alive_node_ids", lambda: ["a" * 56, "b" * 56])
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "cancel",
            lambda ref, force: cancellations.append((ref, force)),
        )
        monkeypatch.setattr(tq_lifecycle.ray, "wait", lambda refs, **_kwargs: (list(refs), []))

        failures = tq_lifecycle.verify_cluster_attach({}, timeout=0.1)

        assert failures == ["cluster: handshake scheduling failed (RuntimeError)"]
        assert cancellations == [(first_ref, True)]

    def test_wait_failure_cancels_and_confirms_submitted_workers(self, monkeypatch):
        captured, _worker = self._capture_handshake(monkeypatch)
        wait_calls = 0
        cancellations: list[tuple[object, bool]] = []

        def fake_wait(submitted, **_kwargs):
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                raise RuntimeError("private wait detail")
            return list(submitted), []

        monkeypatch.setattr(tq_lifecycle.ray, "wait", fake_wait)
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "cancel",
            lambda ref, force: cancellations.append((ref, force)),
        )
        assert captured == []

        failures = tq_lifecycle.verify_cluster_attach({}, timeout=0.1)

        assert failures == ["cluster: handshake wait failed (RuntimeError)"]
        assert len(cancellations) == 1 and cancellations[0][1] is True

    def test_unconfirmed_pending_worker_aborts_fallback_boundary(self, monkeypatch):
        self._capture_handshake(monkeypatch)
        monkeypatch.setattr(tq_lifecycle.ray, "wait", lambda refs, **_kwargs: ([], list(refs)))
        monkeypatch.setattr(tq_lifecycle.ray, "cancel", lambda _ref, force: None)

        with pytest.raises(tq_lifecycle.TqHandshakeIsolationError, match="could not be confirmed stopped"):
            tq_lifecycle.verify_cluster_attach({}, timeout=0.1)

    def test_cancel_failure_aborts_fallback_boundary_without_leaking_detail(self, monkeypatch):
        self._capture_handshake(monkeypatch)
        private_detail = "private worker address and traceback path"
        monkeypatch.setattr(tq_lifecycle.ray, "wait", lambda refs, **_kwargs: ([], list(refs)))
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "cancel",
            lambda _ref, force: _raise(RuntimeError(private_detail)),
        )

        with pytest.raises(tq_lifecycle.TqHandshakeIsolationError) as excinfo:
            tq_lifecycle.verify_cluster_attach({}, timeout=0.1)
        assert private_detail not in str(excinfo.value)


class TestAssertMooncakeRdmaConfigured:
    """The client field proves configured intent, not negotiated transport.

    ``tq.init`` ignores the caller's conf when attaching to an existing
    controller, so the manager and configured protocol must still match the
    job-level contract.  Wire proof remains a benchmark responsibility.
    """

    class MooncakeStorageManager:
        """Name matters: the production check compares
        ``type(...).__name__``."""

        def __init__(self, storage_client):
            self.storage_client = storage_client

    def _patch_client(self, monkeypatch, manager):
        fake = MagicMock()
        fake.get_client.return_value = MagicMock(storage_manager=manager)
        monkeypatch.setattr(tq_lifecycle, "tq", fake)

    def test_accepts_mooncake_rdma(self, monkeypatch):
        self._patch_client(monkeypatch, self.MooncakeStorageManager(MagicMock(protocol="rdma")))
        tq_lifecycle.assert_mooncake_rdma_configured()

    def test_rejects_non_mooncake_manager(self, monkeypatch):
        self._patch_client(monkeypatch, MagicMock())
        with pytest.raises(RuntimeError, match="not MooncakeStorageManager"):
            tq_lifecycle.assert_mooncake_rdma_configured()

    def test_rejects_missing_storage_client(self, monkeypatch):
        self._patch_client(monkeypatch, self.MooncakeStorageManager(None))
        with pytest.raises(RuntimeError, match="no storage_client"):
            tq_lifecycle.assert_mooncake_rdma_configured()

    @pytest.mark.parametrize("protocol", ["tcp", None, ""])
    def test_rejects_non_rdma_protocol(self, monkeypatch, protocol):
        self._patch_client(monkeypatch, self.MooncakeStorageManager(MagicMock(protocol=protocol)))
        with pytest.raises(RuntimeError, match="not configured for protocol=rdma"):
            tq_lifecycle.assert_mooncake_rdma_configured()

    def test_cluster_attach_timeout_does_not_leave_process_global_state(self):
        """The one-shot worker dies before its abandoned tq.init can mutate
        state."""
        env = os.environ.copy()
        env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
        env.pop("RAY_ADDRESS", None)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # Ray places Unix-domain sockets below its temp directory.  A pytest
        # tmp_path can exceed Linux's 107-byte AF_UNIX path limit.
        with tempfile.TemporaryDirectory(prefix="tq-ray-") as probe_dir:
            result = subprocess.run(
                [sys.executable, "-m", "tests.utils._tq_handshake_timeout_probe", probe_dir],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
        assert result.returncode == 0, f"probe stdout:\n{result.stdout}\nprobe stderr:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Worker detach (attach-only inverse used by every worker teardown hook)
# ---------------------------------------------------------------------------


class TestWorkerDetach:
    """detach_tq_client and the teardown hooks that must invoke it."""

    def test_detach_closes_clients_and_resets_process_handles(self, monkeypatch):
        storage_client = MagicMock()
        client = MagicMock(storage_manager=SimpleNamespace(storage_client=storage_client))
        fake_tq = MagicMock()
        fake_tq.get_client.return_value = client
        fake_interface = SimpleNamespace(_TQ_CLIENT=client, _TQ_CONTROLLER=object())
        monkeypatch.setattr(tq_lifecycle.tq, "interface", fake_interface, raising=False)
        monkeypatch.setattr(tq_lifecycle, "tq", fake_tq)

        tq_lifecycle.detach_tq_client()

        storage_client.close.assert_called_once_with()
        client.close.assert_called_once_with()
        assert fake_interface._TQ_CLIENT is None
        assert fake_interface._TQ_CONTROLLER is None

    @pytest.mark.parametrize("has_client", [True, False], ids=["attached", "not-attached"])
    def test_component_del_detaches_only_an_attached_client(self, monkeypatch, has_client):
        from relax.components.base import Base

        calls = []
        monkeypatch.setattr(tq_lifecycle, "detach_tq_client", lambda: calls.append(True))
        component = Base()
        if has_client:
            component.data_system_client = object()
        component.__del__()
        assert calls == ([True] if has_client else [])
        if has_client:
            assert component.data_system_client is None


# ---------------------------------------------------------------------------
# Exclusive-owner initialization transaction
# ---------------------------------------------------------------------------


class TestInitializeTqWithFallback:
    @staticmethod
    def _conf(backend: str) -> dict:
        return {"controller": {}, "backend": {"storage_backend": backend}}

    @staticmethod
    def _patch_transaction(
        monkeypatch,
        *,
        init_effects: list[object],
        start_effects: list[object] | None = None,
        reap_effects: list[BaseException | None] | None = None,
    ) -> list[str]:
        events: list[str] = []
        effects = iter(init_effects)
        owners = iter(start_effects or [f"owner-{index}" for index in range(len(init_effects))])
        reaps = iter(reap_effects or [])

        def fake_reap():
            events.append("reap")
            effect = next(reaps, None)
            if effect is not None:
                raise effect

        monkeypatch.setattr(tq_lifecycle, "reap_unusable_tq_controller", fake_reap)

        def fake_start(*, timeout):
            events.append("start")
            effect = next(owners)
            if isinstance(effect, BaseException):
                raise effect
            return effect

        def fake_initialize(owner, conf, *, timeout):
            backend = conf["backend"]["storage_backend"]
            events.append(f"init:{backend}")
            effect = next(effects)
            if isinstance(effect, BaseException):
                raise effect
            return tq_lifecycle.TqInitResult(config=conf, owner=owner)

        monkeypatch.setattr(tq_lifecycle, "_start_owner", fake_start)
        monkeypatch.setattr(tq_lifecycle, "_initialize_owner", fake_initialize)
        return events

    def test_simple_path_also_runs_pre_init_reaper_and_becomes_owner(self, monkeypatch):
        conf = self._conf("SimpleStorage")
        events = self._patch_transaction(monkeypatch, init_effects=[None])
        result = tq_lifecycle.initialize_tq_with_fallback(conf, mode="off")
        assert result.owner == "owner-0"
        assert events == ["reap", "start", "init:SimpleStorage"]

    @pytest.mark.parametrize("mode", ["off", "auto", "required"])
    def test_healthy_existing_controller_fails_exclusive_without_starting_owner(self, monkeypatch, mode):
        requested = self._conf("SimpleStorage")
        mismatch = RuntimeError("exclusive cluster is not clean")
        events = self._patch_transaction(monkeypatch, init_effects=[], reap_effects=[mismatch])

        with pytest.raises(RuntimeError, match="exclusive cluster"):
            tq_lifecycle.initialize_tq_with_fallback(
                requested,
                mode=mode,
                fallback_conf=self._conf("SimpleStorage"),
            )

        assert events == ["reap"]

    @pytest.mark.parametrize(
        ("primary_error", "fallback_reason"),
        [
            (
                tq_lifecycle.TqInitializationError("master unavailable"),
                "mooncake_init_failed:TqInitializationError",
            ),
            (tq_lifecycle.TqInitializationTimeout("timed out"), "mooncake_init_failed:TqInitializationTimeout"),
        ],
        ids=["init-error", "init-timeout"],
    )
    def test_auto_cleans_failed_mooncake_then_retries_simple_once(self, monkeypatch, primary_error, fallback_reason):
        primary = self._conf("MooncakeStore")
        fallback = self._conf("SimpleStorage")
        events = self._patch_transaction(monkeypatch, init_effects=[primary_error, None])
        result = tq_lifecycle.initialize_tq_with_fallback(primary, mode="auto", fallback_conf=fallback)
        assert result.config["backend"]["storage_backend"] == "SimpleStorage"
        assert result.fallback_reason == fallback_reason
        assert events == [
            "reap",
            "start",
            "init:MooncakeStore",
            "reap",
            "start",
            "init:SimpleStorage",
        ]

    def test_required_cleans_failed_init_without_fallback(self, monkeypatch):
        primary = self._conf("MooncakeStore")
        fallback = self._conf("SimpleStorage")
        events = self._patch_transaction(
            monkeypatch,
            init_effects=[tq_lifecycle.TqInitializationError("master unavailable")],
        )
        with pytest.raises(tq_lifecycle.TqInitializationError, match="master unavailable"):
            tq_lifecycle.initialize_tq_with_fallback(primary, mode="required", fallback_conf=fallback)
        assert events == ["reap", "start", "init:MooncakeStore"]

    def test_cleanup_failure_aborts_auto_before_second_gate(self, monkeypatch):
        events = self._patch_transaction(
            monkeypatch,
            init_effects=[tq_lifecycle.TqCleanupTimeout("owner still running")],
        )

        with pytest.raises(tq_lifecycle.TqCleanupTimeout, match="still running"):
            tq_lifecycle.initialize_tq_with_fallback(
                self._conf("MooncakeStore"),
                mode="auto",
                fallback_conf=self._conf("SimpleStorage"),
            )

        assert events == ["reap", "start", "init:MooncakeStore"]

    def test_second_gate_failure_aborts_before_fallback_owner(self, monkeypatch):
        gate_error = RuntimeError("residual controller state is unknown")
        events = self._patch_transaction(
            monkeypatch,
            init_effects=[tq_lifecycle.TqInitializationError("master unavailable")],
            reap_effects=[None, gate_error],
        )

        with pytest.raises(RuntimeError, match="fallback initialization failed"):
            tq_lifecycle.initialize_tq_with_fallback(
                self._conf("MooncakeStore"),
                mode="auto",
                fallback_conf=self._conf("SimpleStorage"),
            )

        assert events == ["reap", "start", "init:MooncakeStore", "reap"]

    def test_owner_creation_failure_is_not_a_candidate_fallback(self, monkeypatch):
        events = self._patch_transaction(
            monkeypatch,
            init_effects=[],
            start_effects=[RuntimeError("owner scheduling failed")],
        )

        with pytest.raises(RuntimeError, match="owner scheduling failed"):
            tq_lifecycle.initialize_tq_with_fallback(
                self._conf("MooncakeStore"),
                mode="auto",
                fallback_conf=self._conf("SimpleStorage"),
            )

        assert events == ["reap", "start"]


class _RemoteMethod:
    def __init__(self, value):
        self.value = value

    def remote(self, *args, **kwargs):
        return self.value


class _FakeOwner:
    def __init__(self):
        self.ready = _RemoteMethod("ready-ref")
        self.initialize = _RemoteMethod("initialize-ref")
        self.close = _RemoteMethod("close-ref")
        self.termination_probe = _RemoteMethod("termination-ref")


class TestOwnerProcessBoundary:
    def test_start_scheduling_timeout_stops_owner_without_entering_init(self, monkeypatch):
        owner = _FakeOwner()
        stopped: list[object] = []
        monkeypatch.setattr(tq_lifecycle._TransferQueueOwner, "remote", lambda: owner)

        def timed_out(ref, *, timeout):
            assert ref == "ready-ref"
            raise tq_lifecycle.ray.exceptions.GetTimeoutError("test timeout")

        monkeypatch.setattr(tq_lifecycle.ray, "get", timed_out)
        monkeypatch.setattr(tq_lifecycle, "_stop_owner_actor", lambda handle, **_kwargs: stopped.append(handle))

        with pytest.raises(RuntimeError, match="owner creation failed"):
            tq_lifecycle._start_owner(timeout=0.1)
        assert stopped == [owner]

    def test_start_success_returns_scheduled_owner(self, monkeypatch):
        owner = _FakeOwner()
        monkeypatch.setattr(tq_lifecycle._TransferQueueOwner, "remote", lambda: owner)
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda ref, timeout: None)

        assert tq_lifecycle._start_owner(timeout=1) is owner

    def test_initialize_success_returns_the_exclusive_owner(self, monkeypatch):
        owner = _FakeOwner()
        stored = TestInitializeTqWithFallback._conf("SimpleStorage")
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda ref, timeout: stored)

        result = tq_lifecycle._initialize_owner(owner, stored, timeout=1)

        assert result.config is stored
        assert result.owner is owner

    @pytest.mark.parametrize("error_kind", ["timeout", "runtime"])
    def test_initialize_failure_cleans_owner_before_propagating(self, monkeypatch, error_kind):
        owner = _FakeOwner()
        cleaned: list[object] = []
        private_detail = "private endpoint and traceback path"
        if error_kind == "timeout":
            source_error = tq_lifecycle.ray.exceptions.GetTimeoutError("timeout")
            expected_error = tq_lifecycle.TqInitializationTimeout
        else:
            source_error = RuntimeError(private_detail)
            expected_error = tq_lifecycle.TqInitializationError
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "get",
            lambda _ref, timeout: _raise(source_error),
        )
        monkeypatch.setattr(tq_lifecycle, "_cleanup_failed_owner", lambda handle: cleaned.append(handle))

        with pytest.raises(expected_error) as excinfo:
            tq_lifecycle._initialize_owner(owner, {}, timeout=0.1)

        assert cleaned == [owner]
        if error_kind == "runtime":
            assert private_detail not in str(excinfo.value)

    def test_stop_owner_requires_terminal_probe_result(self, monkeypatch):
        owner = _FakeOwner()
        killed: list[tuple[object, bool]] = []
        monkeypatch.setattr(tq_lifecycle.ray, "kill", lambda handle, no_restart: killed.append((handle, no_restart)))
        monkeypatch.setattr(tq_lifecycle.ray, "wait", lambda refs, **_kwargs: (list(refs), []))
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "get",
            lambda _ref: _raise(tq_lifecycle.ray.exceptions.RayActorError(error_msg="dead")),
        )

        tq_lifecycle._stop_owner_actor(owner, timeout=0.1)

        assert killed == [(owner, True)]

    @pytest.mark.parametrize(
        ("probe_ready", "match"),
        [(False, "remained pending"), (True, "returned normally")],
        ids=["pending", "returned"],
    )
    def test_stop_owner_requires_a_terminal_probe_failure(self, monkeypatch, probe_ready, match):
        owner = _FakeOwner()
        monkeypatch.setattr(tq_lifecycle.ray, "kill", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "wait",
            lambda refs, **_kwargs: (list(refs), []) if probe_ready else ([], list(refs)),
        )
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda _ref: None)

        with pytest.raises(tq_lifecycle.TqCleanupTimeout, match=match):
            tq_lifecycle._stop_owner_actor(owner, timeout=0.1)

    def test_failed_owner_cleanup_waits_for_owner_then_reaps_controller(self, monkeypatch):
        owner = _FakeOwner()
        events: list[str] = []
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda _ref, timeout: events.append("close"))
        monkeypatch.setattr(
            tq_lifecycle,
            "_stop_owner_actor",
            lambda handle, **_kwargs: events.append("owner-terminal"),
        )
        monkeypatch.setattr(
            tq_lifecycle,
            "kill_tq_controller_and_wait",
            lambda **_kwargs: events.append("controller-gone"),
        )

        tq_lifecycle._cleanup_failed_owner(owner)

        assert events == ["close", "owner-terminal", "controller-gone"]

    def test_unconfirmed_owner_aborts_before_controller_cleanup(self, monkeypatch):
        owner = _FakeOwner()
        controller_cleanup: list[bool] = []
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda _ref, timeout: None)
        monkeypatch.setattr(
            tq_lifecycle,
            "_stop_owner_actor",
            lambda *_args, **_kwargs: _raise(tq_lifecycle.TqCleanupTimeout("owner pending")),
        )
        monkeypatch.setattr(
            tq_lifecycle,
            "kill_tq_controller_and_wait",
            lambda **_kwargs: controller_cleanup.append(True),
        )

        with pytest.raises(tq_lifecycle.TqCleanupTimeout, match="owner pending"):
            tq_lifecycle._cleanup_failed_owner(owner)

        assert controller_cleanup == []

    def test_owner_close_failure_still_reaps_global_controller(self, monkeypatch):
        owner = _FakeOwner()
        stopped: list[object] = []
        killed: list[bool] = []
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "get",
            lambda ref, timeout: _raise(RuntimeError("close failed")),
        )
        monkeypatch.setattr(tq_lifecycle, "_stop_owner_actor", lambda handle, **_kwargs: stopped.append(handle))
        monkeypatch.setattr(tq_lifecycle, "kill_tq_controller_and_wait", lambda **_kwargs: killed.append(True))

        with pytest.raises(RuntimeError, match="owner cleanup failed"):
            tq_lifecycle.close_tq_owner(owner)
        assert stopped == [owner]
        assert killed == [True]


# ---------------------------------------------------------------------------
# Retry / disconnect on the MooncakeStore data path
# ---------------------------------------------------------------------------


class _FlakyStore:
    """Stub store whose first ``fail_times`` reads return an error code."""

    def __init__(self, fail_times: int, code: int = -800, raise_exc: Exception | None = None):
        self.fail_times = fail_times
        self.code = code
        self.raise_exc = raise_exc
        self.get_calls: list[list[str]] = []

    def _codes(self, keys):
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.fail_times > 0:
            self.fail_times -= 1
            return [self.code] * len(keys)
        return [0] * len(keys)

    def batch_get_into(self, keys, ptrs, sizes):
        self.get_calls.append(list(keys))
        return self._codes(keys)


def _client_with_store(store) -> object:
    """A MooncakeStoreClient with only ``_store``/``replica_config`` wired up.

    ``__init__`` is skipped on purpose: it would need a live mooncake master.
    """
    from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient

    client = object.__new__(MooncakeStoreClient)
    client._store = store
    client.replica_config = None
    return client


class _InlineExecutorLoop:
    """Run storage-manager sync calls inline for deterministic async tests.

    The production manager delegates KV operations to an executor.  A real
    default executor makes the failure-path test wait for worker shutdown on
    some CI kernels, so these contract tests substitute an already-completed
    Future without changing the storage-manager control flow.
    """

    def run_in_executor(self, executor, fn, *args):
        future = asyncio.get_running_loop().create_future()
        try:
            future.set_result(fn(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


@pytest.mark.skipif(
    not _REAL_MOONCAKE_CLIENT,
    reason="needs a real transfer_queue (CI uses a single-file stub); run on a host with TransferQueue installed",
)
class TestRetryAndDisconnect:
    """Behavior surface retained in Relax; retry internals belong upstream."""

    def test_transient_get_failure_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("transfer_queue.storage.clients.mooncake_client.RETRY_DELAY_SECONDS", 0)
        store = _FlakyStore(fail_times=2)
        client = _client_with_store(store)
        keys = ["0@f0", "1@f0"]
        client._batch_get_into_with_retry(keys, [1, 2], [8, 8])
        assert store.get_calls == [keys, keys, keys]

    def test_disconnect_surfaces_instead_of_returning_garbage(self):
        """A dead peer must raise, never hand back a silently short buffer."""
        exc = RuntimeError("Failed to open segment for endpoint='<ip>:16675'")
        client = _client_with_store(_FlakyStore(fail_times=0, raise_exc=exc))
        with pytest.raises(RuntimeError, match="Failed to open segment"):
            client._batch_get_into_with_retry(["0@f0"], [1], [8])


@pytest.mark.skipif(
    not _REAL_MOONCAKE_CLIENT,
    reason="needs a real transfer_queue to verify the KV manager write/notify contract",
)
class TestMooncakeProductionStatusContract:
    @staticmethod
    def _manager(storage_client):
        from transfer_queue.storage.managers.mooncake_manager import MooncakeStorageManager

        manager = object.__new__(MooncakeStorageManager)
        manager.storage_client = storage_client
        manager.notify_data_update = AsyncMock()
        manager.controller_handshake_socket = None
        manager.storage_manager_id = "capacity-contract-test"
        manager.zmq_context = MagicMock()
        return manager

    @staticmethod
    def _data_and_meta():
        from tensordict import TensorDict

        data = TensorDict({"pixel_values": torch.randn(1, 16)}, batch_size=[1])
        meta = MagicMock()
        meta.global_indexes = [7]
        meta.partition_ids = ["capacity"]
        meta._custom_backend_meta = [{}]
        meta.get_all_custom_meta.return_value = [{}]
        return data, meta

    @pytest.mark.asyncio
    async def test_capacity_write_failure_never_notifies_production_ready(self, monkeypatch):
        monkeypatch.setattr(asyncio, "get_event_loop", lambda: _InlineExecutorLoop())
        storage_client = MagicMock()
        storage_client.put.side_effect = RuntimeError("batch_upsert_from failed: capacity exhausted")
        manager = self._manager(storage_client)
        data, meta = self._data_and_meta()

        with pytest.raises(RuntimeError, match="capacity exhausted"):
            await manager.put_data(data, meta)

        manager.notify_data_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_production_ready_is_notified_only_after_storage_success(self, monkeypatch):
        monkeypatch.setattr(asyncio, "get_event_loop", lambda: _InlineExecutorLoop())
        calls: list[str] = []
        storage_client = MagicMock()
        storage_client.put.side_effect = lambda keys, values: calls.append("put") or [None] * len(keys)
        manager = self._manager(storage_client)

        async def notify(*args, **kwargs):
            calls.append("notify")

        manager.notify_data_update.side_effect = notify
        data, meta = self._data_and_meta()
        await manager.put_data(data, meta)
        assert calls == ["put", "notify"]
