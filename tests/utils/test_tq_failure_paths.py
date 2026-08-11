# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Failure-path tests for the TransferQueue dataplane enablement.

Covers the four gaps the maintainer review called out, which the existing
``test_rdma_probe.py`` (pure config/probe logic) and
``test_tq_dataplane_behavior.py`` (real TQ on SimpleStorage) did not:

* timeout -- controller ``get_config`` timeout and probe-task timeout
* disconnect -- store errors surface instead of returning corrupt data
* retry -- ``batch_get_into`` / ``batch_upsert_from`` retry-then-raise
* byte-exactness on **MooncakeStore** (SimpleStorage-only before)
* automatic degradation as pytest (was a manual two-node script)
* the controller reaper / teardown helpers (now in ``relax.utils.tq_lifecycle``)

Everything except the MooncakeStore round-trip runs with stubs, so it is
CI-safe; the round-trip skips unless a reachable mooncake master is configured.
"""

from __future__ import annotations

import importlib.util
import os
import socket
from unittest.mock import MagicMock

import pytest
import torch

from relax.utils import tq_lifecycle
from relax.utils.rdma_probe import ProbeResult, reduce_results


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probe(node: str, protocol: str | None = "rdma", device: str = "rdma0") -> ProbeResult:
    """Build a ProbeResult without running any real probe."""
    return ProbeResult(
        node=node,
        checks=(),
        effective_protocol=protocol,
        effective_device=device if protocol else "",
        gdr_eligible=protocol == "rdma",
        errors=() if protocol else ("mooncake not importable",),
    )


def _master_address() -> str:
    """Mooncake master address for the round-trip test."""
    return os.environ.get("MC_MASTER_ADDRESS", "127.0.0.1:50051")


def _master_reachable(timeout: float = 1.0) -> bool:
    """True if something accepts TCP connections on the master address."""
    host, _, port = _master_address().rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


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

    def test_healthy_controller_is_left_alone(self, monkeypatch):
        killed = self._fake_ray(monkeypatch, actor=MagicMock(), get_result={"backend": {}})
        assert tq_lifecycle.reap_unusable_tq_controller() is False
        assert killed == []

    def test_half_initialised_controller_is_reaped(self, monkeypatch):
        """conf is None == actor created but store_config never ran (F10)."""
        killed = self._fake_ray(monkeypatch, actor=MagicMock(), get_result=None)
        assert tq_lifecycle.reap_unusable_tq_controller() is True
        assert killed == ["killed"]

    def test_get_config_timeout_is_reaped(self, monkeypatch):
        """An unresponsive controller must not turn tq.init into a hang."""
        killed = self._fake_ray(monkeypatch, actor=MagicMock(), get_raises=TimeoutError("get_config timed out"))
        assert tq_lifecycle.reap_unusable_tq_controller() is True
        assert killed == ["killed"]

    def test_dead_actor_is_reaped(self, monkeypatch):
        killed = self._fake_ray(monkeypatch, actor=MagicMock(), get_raises=RuntimeError("ActorDiedError"))
        assert tq_lifecycle.reap_unusable_tq_controller() is True
        assert killed == ["killed"]


# ---------------------------------------------------------------------------
# Controller lifecycle: teardown unmounts the Mooncake segment
# ---------------------------------------------------------------------------


class TestCloseTqAndUnmount:
    """close_tq_and_unmount: tq.close() first, then unmount the segment."""

    @staticmethod
    def _fake_tq(monkeypatch, store_client):
        calls: list[str] = []
        fake = MagicMock()
        manager = MagicMock()
        if store_client is None:
            del manager.storage_client  # SimpleStorage manager has no storage_client
        else:
            manager.storage_client = store_client
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
# Retry / disconnect on the MooncakeStore data path
# ---------------------------------------------------------------------------


class _FlakyStore:
    """Stub mooncake store: the first ``fail_times`` calls return error
    codes."""

    def __init__(self, fail_times: int, code: int = -800, raise_exc: Exception | None = None):
        self.fail_times = fail_times
        self.code = code
        self.raise_exc = raise_exc
        self.get_calls: list[list[str]] = []
        self.put_calls: list[list[str]] = []

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

    def batch_upsert_from(self, keys, ptrs, sizes, config=None):
        self.put_calls.append(list(keys))
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


@pytest.mark.skipif(
    not _REAL_MOONCAKE_CLIENT,
    reason="needs a real transfer_queue (CI uses a single-file stub); run on a host with TransferQueue installed",
)
class TestRetryAndDisconnect:
    """batch_get_into / batch_upsert_from: retry, then raise loudly."""

    def test_get_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("transfer_queue.storage.clients.mooncake_client.RETRY_DELAY_SECONDS", 0)
        store = _FlakyStore(fail_times=2)
        client = _client_with_store(store)
        client._batch_get_into_with_retry(["0@f0", "1@f0"], [1, 2], [8, 8])
        assert len(store.get_calls) == 3  # initial + 2 retries
        assert store.get_calls[-1] == ["0@f0", "1@f0"]

    def test_get_raises_after_max_retries(self, monkeypatch):
        monkeypatch.setattr("transfer_queue.storage.clients.mooncake_client.RETRY_DELAY_SECONDS", 0)
        client = _client_with_store(_FlakyStore(fail_times=99))
        with pytest.raises(RuntimeError, match="batch_get_into failed"):
            client._batch_get_into_with_retry(["0@f0"], [1], [8])

    def test_put_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("transfer_queue.storage.clients.mooncake_client.RETRY_DELAY_SECONDS", 0)
        store = _FlakyStore(fail_times=1, code=-1)
        client = _client_with_store(store)
        client._batch_upsert_with_retry(["0@f0"], [1], [8])
        assert len(store.put_calls) == 2

    def test_put_raises_after_max_retries(self, monkeypatch):
        monkeypatch.setattr("transfer_queue.storage.clients.mooncake_client.RETRY_DELAY_SECONDS", 0)
        client = _client_with_store(_FlakyStore(fail_times=99, code=-1))
        with pytest.raises(RuntimeError, match="batch_upsert_from failed"):
            client._batch_upsert_with_retry(["0@f0"], [1], [8])

    def test_disconnect_surfaces_instead_of_returning_garbage(self):
        """A dead peer must raise, never hand back a silently short buffer."""
        exc = RuntimeError("Failed to open segment for endpoint='<ip>:16675'")
        client = _client_with_store(_FlakyStore(fail_times=0, raise_exc=exc))
        with pytest.raises(RuntimeError, match="Failed to open segment"):
            client._batch_get_into_with_retry(["0@f0"], [1], [8])


# ---------------------------------------------------------------------------
# Automatic degradation (was the manual two-node fault_inject_multinode.py)
# ---------------------------------------------------------------------------


class TestAutomaticDegradation:
    """AND-reduction turns any node's failure into a job-level downgrade."""

    def test_all_nodes_rdma_stays_rdma(self):
        eff = reduce_results(
            [_probe("a"), _probe("b")], requested_backend="mooncake", requested_device="", use_gdr=False
        )
        assert (eff.backend, eff.protocol, eff.fallback_reason) == ("MooncakeStore", "rdma", "")

    def test_one_node_without_mooncake_degrades_whole_job(self):
        """Mirrors the PYTHONPATH-poisoning case of the two-node script."""
        eff = reduce_results(
            [_probe("a"), _probe("b", protocol=None)],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=False,
        )
        assert eff.backend == "SimpleStorage"
        assert "mooncake_unavailable" in eff.fallback_reason and "b" in eff.fallback_reason

    def test_crashed_probe_task_degrades_whole_job(self):
        """A probe task that raises becomes a degenerate result, never
        dropped."""
        from relax.utils.rdma_probe import _degenerate_result

        degenerate = _degenerate_result("b", "probe task raised")
        assert degenerate.effective_protocol is None
        eff = reduce_results(
            [_probe("a"), degenerate], requested_backend="mooncake", requested_device="", use_gdr=False
        )
        assert eff.backend == "SimpleStorage"

    def test_one_node_tcp_only_degrades_transport_not_backend(self):
        eff = reduce_results(
            [_probe("a"), _probe("b", protocol="tcp")],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=False,
        )
        assert (eff.backend, eff.protocol) == ("MooncakeStore", "tcp")
        assert eff.fallback_reason


# ---------------------------------------------------------------------------
# Byte-exactness on MooncakeStore (was SimpleStorage-only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_master_reachable() and _REAL_MOONCAKE_CLIENT),
    reason=(
        "needs a reachable mooncake master and a real TransferQueue install "
        "(CI uses a single-file transfer_queue stub and has no RDMA/mooncake "
        "deployment), so the MooncakeStore round-trip is skipped"
    ),
)
class TestMooncakeByteExact:
    """Real MooncakeStoreClient put/get round-trip, byte-for-byte."""

    @staticmethod
    def _client(protocol: str):
        from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient

        return MooncakeStoreClient(
            {
                "protocol": protocol,
                "device_name": os.environ.get("MC_RDMA_DEVICE", ""),
                "master_server_address": _master_address(),
                "metadata_server": "P2PHANDSHAKE",
                "local_hostname": "",
                "global_segment_size": 2 * 1024**3,
                "local_buffer_size": 512 * 1024**2,
                "hard_pin": True,
                "use_gdr": False,
            }
        )

    @pytest.mark.parametrize("protocol", ["tcp", "rdma"])
    def test_multi_dtype_shape_roundtrip_is_byte_exact(self, protocol):
        tensors = {
            "f32_2d": torch.randn(64, 1176, dtype=torch.float32),
            "bf16_2d": torch.randn(32, 512).to(torch.bfloat16),
            "i64_1d": torch.arange(4096, dtype=torch.int64),
            "noncontig": torch.randn(128, 256).t(),  # transposed == non-contiguous
        }
        client = self._client(protocol)
        try:
            keys = [f"bx_{protocol}_{name}" for name in tensors]
            values = list(tensors.values())
            client.put(keys, values)
            got = client.get(
                keys,
                shapes=[tuple(v.shape) for v in values],
                dtypes=[v.dtype for v in values],
            )
            for name, want, have in zip(tensors, values, got, strict=True):
                assert have is not None, f"{name} came back empty"
                assert torch.equal(have, want.contiguous()), f"{name} is not byte-exact"
        finally:
            client.close()
