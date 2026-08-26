# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the version-gated Mooncake runtime loss guards.

Moved out of ``test_tq_failure_paths.py`` together with the patches themselves
(``relax/utils/tq_mooncake_patches.py``); everything here is deleted with that
module once the fixes land upstream.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from relax.utils.tq_mooncake_patches import (
    _install_notification_guards,
    _install_store_guards,
    _installed_transfer_queue_revision,
    _require_pinned_transfer_queue,
    _strict_notify_and_wait,
    _StrictMooncakeStoreProxy,
)


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


class _SequenceStore:
    """Return a configured result sequence from low-level Mooncake calls."""

    def __init__(self, results: list[list[int]]) -> None:
        self.results = iter(results)

    def batch_upsert_from(self, keys, ptrs, sizes, config=None):
        return next(self.results)

    def batch_get_into(self, keys, ptrs, sizes):
        return next(self.results)

    def batch_remove(self, keys, force=True):
        return next(self.results)


class _FakeNotifySocket:
    def __init__(self, connect_error: Exception | None = None) -> None:
        self.closed = False
        self.connect_error = connect_error

    def setsockopt(self, *args, **kwargs) -> None:
        pass

    def connect(self, *args, **kwargs) -> None:
        if self.connect_error is not None:
            raise self.connect_error

    async def send_multipart(self, request) -> None:
        pass

    async def recv_multipart(self, copy=False):
        return [b"ack"]

    def close(self, linger=0) -> None:
        self.closed = True


def _client_with_store(store) -> object:
    """A MooncakeStoreClient with only ``_store``/``replica_config`` wired up.

    ``__init__`` is skipped on purpose: it would need a live mooncake master.
    """
    from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient

    client = object.__new__(MooncakeStoreClient)
    client._store = store
    client.replica_config = None
    return client


class TestVersionGate:
    """Patches refuse to install on any transfer_queue they were not written
    for."""

    _PINNED_REVISION = "58054a33834aadbcf76aacd6b1e32e25c030f2c9"

    def test_unpinned_version_is_rejected(self, monkeypatch):
        import transfer_queue

        monkeypatch.setattr(transfer_queue, "__version__", "9.9.9", raising=False)
        with pytest.raises(RuntimeError, match="not covered by Relax"):
            _require_pinned_transfer_queue()

    def test_pinned_version_and_revision_are_accepted(self, monkeypatch):
        import transfer_queue

        monkeypatch.setattr(transfer_queue, "__version__", "0.1.10.dev0", raising=False)
        monkeypatch.setattr(
            "relax.utils.tq_mooncake_patches._installed_transfer_queue_revision",
            lambda module: self._PINNED_REVISION,
        )
        _require_pinned_transfer_queue()

    def test_same_version_with_different_revision_is_rejected(self, monkeypatch):
        import transfer_queue

        monkeypatch.setattr(transfer_queue, "__version__", "0.1.10.dev0", raising=False)
        monkeypatch.setattr(
            "relax.utils.tq_mooncake_patches._installed_transfer_queue_revision",
            lambda module: "0" * 40,
        )
        with pytest.raises(RuntimeError, match="revision .* is not covered"):
            _require_pinned_transfer_queue()

    def test_missing_revision_metadata_is_rejected(self, monkeypatch):
        import transfer_queue

        monkeypatch.setattr(transfer_queue, "__version__", "0.1.10.dev0", raising=False)
        monkeypatch.setattr(
            "relax.utils.tq_mooncake_patches._installed_transfer_queue_revision",
            lambda module: None,
        )
        with pytest.raises(RuntimeError, match="revision unknown is not covered"):
            _require_pinned_transfer_queue()

    @pytest.mark.parametrize(
        ("direct_url", "expected"),
        [
            (None, None),
            ("not-json", None),
            ("{}", None),
            ('{"vcs_info": {}}', None),
            ('{"vcs_info": {"vcs": "hg", "commit_id": "58054a33834aadbcf76aacd6b1e32e25c030f2c9"}}', None),
            ('{"vcs_info": {"vcs": "git", "commit_id": "abc123"}}', None),
            (
                '{"vcs_info": {"vcs": "git", "commit_id": " 58054A33834AADBCF76AACD6B1E32E25C030F2C9 "}}',
                _PINNED_REVISION,
            ),
        ],
    )
    def test_revision_metadata_is_parsed_fail_closed(self, monkeypatch, tmp_path, direct_url, expected):
        package_root = tmp_path / "installed" / "transfer_queue"

        class Distribution:
            version = "0.1.10.dev0"

            def locate_file(self, filename):
                assert filename == "transfer_queue"
                return package_root

            def read_text(self, filename):
                assert filename == "direct_url.json"
                return direct_url

        monkeypatch.setattr(
            "relax.utils.tq_mooncake_patches.metadata.distribution",
            lambda name: Distribution(),
        )
        module = SimpleNamespace(
            __version__="0.1.10.dev0",
            __file__=package_root / "__init__.py",
        )
        assert _installed_transfer_queue_revision(module) == expected

    def test_distribution_version_mismatch_is_unverifiable(self, monkeypatch, tmp_path):
        class Distribution:
            version = "0.1.10.dev1"

        monkeypatch.setattr(
            "relax.utils.tq_mooncake_patches.metadata.distribution",
            lambda name: Distribution(),
        )
        module = SimpleNamespace(
            __version__="0.1.10.dev0",
            __file__=tmp_path / "transfer_queue" / "__init__.py",
        )
        assert _installed_transfer_queue_revision(module) is None

    def test_shadowed_module_is_unverifiable(self, monkeypatch, tmp_path):
        package_root = tmp_path / "installed" / "transfer_queue"

        class Distribution:
            version = "0.1.10.dev0"

            def locate_file(self, filename):
                return package_root

            def read_text(self, filename):
                return '{"vcs_info": {"vcs": "git", "commit_id": "58054a33834aadbcf76aacd6b1e32e25c030f2c9"}}'

        monkeypatch.setattr(
            "relax.utils.tq_mooncake_patches.metadata.distribution",
            lambda name: Distribution(),
        )
        shadowed_module = SimpleNamespace(
            __version__="0.1.10.dev0",
            __file__=tmp_path / "shadow" / "transfer_queue.py",
        )
        assert _installed_transfer_queue_revision(shadowed_module) is None


class TestMooncakeCorrectnessGuardPrimitives:
    """Low-level response validation stays runnable on the CPU-only CI stub."""

    def test_upsert_short_result_is_raised(self):
        store = _StrictMooncakeStoreProxy(_SequenceStore([[0]]))
        with pytest.raises(RuntimeError, match="returned 1 results, expected 2"):
            store.batch_upsert_from(["k0", "k1"], [1, 2], [8, 8])

    def test_get_short_result_is_raised(self):
        store = _StrictMooncakeStoreProxy(_SequenceStore([[0]]))
        with pytest.raises(RuntimeError, match="returned 1 results, expected 2"):
            store.batch_get_into(["k0", "k1"], [1, 2], [8, 8])

    def test_remove_object_not_found_is_allowed(self):
        store = _StrictMooncakeStoreProxy(_SequenceStore([[0, -704]]))
        assert store.batch_remove(["k0", "k1"], force=True) == [0, -704]

    def test_remove_non_idempotent_failure_is_raised(self):
        store = _StrictMooncakeStoreProxy(_SequenceStore([[0, -1]]))
        with pytest.raises(RuntimeError, match="batch_remove failed"):
            store.batch_remove(["k0", "k1"], force=True)

    def test_missing_wrapped_store_raises_attribute_error(self):
        store = object.__new__(_StrictMooncakeStoreProxy)
        with pytest.raises(AttributeError):
            store.close

    def test_store_guard_installation_is_idempotent(self):
        raw_store = _SequenceStore([[0]])

        class Client:
            def __init__(self):
                self._store = raw_store

        _install_store_guards(Client)
        guarded_init = Client.__init__
        _install_store_guards(Client)

        client = Client()
        assert Client.__init__ is guarded_init
        assert isinstance(client._store, _StrictMooncakeStoreProxy)
        assert client._store._store is raw_store


class TestNotificationGuardPrimitives:
    @pytest.mark.asyncio
    async def test_guarded_notify_rejects_missing_controller(self):
        class Manager:
            controller_info = None

            async def notify_data_update(self):
                raise AssertionError("original notify must not run without a controller")

            async def _notify_and_wait(self, request_msg):
                pass

        _install_notification_guards(Manager)
        guarded_notify = Manager.notify_data_update
        _install_notification_guards(Manager)
        assert Manager.notify_data_update is guarded_notify
        with pytest.raises(RuntimeError, match="has no controller"):
            await Manager().notify_data_update()


@pytest.mark.skipif(
    not _REAL_MOONCAKE_CLIENT,
    reason="needs real TransferQueue storage submodules; CPU CI uses a single-file transfer_queue stub",
)
class TestMooncakeCorrectnessGuards:
    """Integration with real TransferQueue internals; no GPU/master needed."""

    def test_retry_short_result_is_never_treated_as_success(self, monkeypatch):
        monkeypatch.setattr("transfer_queue.storage.clients.mooncake_client.RETRY_DELAY_SECONDS", 0)
        store = _StrictMooncakeStoreProxy(_SequenceStore([[-1, -1], [0]]))
        client = _client_with_store(store)
        with pytest.raises(RuntimeError, match="returned 1 results, expected 2"):
            client._batch_upsert_with_retry(["k0", "k1"], [1, 2], [8, 8])

    def test_get_retry_short_result_is_never_treated_as_success(self, monkeypatch):
        monkeypatch.setattr("transfer_queue.storage.clients.mooncake_client.RETRY_DELAY_SECONDS", 0)
        store = _StrictMooncakeStoreProxy(_SequenceStore([[-1, -1], [0]]))
        client = _client_with_store(store)
        with pytest.raises(RuntimeError, match="returned 1 results, expected 2"):
            client._batch_get_into_with_retry(["k0", "k1"], [1, 2], [8, 8])

    @pytest.mark.asyncio
    async def test_negative_production_status_ack_is_raised(self, monkeypatch):
        from transfer_queue.utils import zmq_utils

        socket = _FakeNotifySocket()
        monkeypatch.setattr(zmq_utils, "create_zmq_socket", lambda **kwargs: socket)
        monkeypatch.setattr(
            zmq_utils.ZMQMessage,
            "deserialize",
            staticmethod(
                lambda messages: SimpleNamespace(
                    request_type=zmq_utils.ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    body={"success": False, "partition_id": "p0"},
                )
            ),
        )
        manager = SimpleNamespace(
            storage_manager_id="guard-test",
            zmq_context=object(),
            controller_info=SimpleNamespace(ip="redacted", to_addr=lambda name: "inproc://controller"),
        )
        with pytest.raises(RuntimeError, match="rejected the production-status update"):
            await _strict_notify_and_wait(manager, [b"request"])
        assert socket.closed is True

    @pytest.mark.asyncio
    async def test_positive_production_status_ack_returns(self, monkeypatch):
        from transfer_queue.utils import zmq_utils

        socket = _FakeNotifySocket()
        monkeypatch.setattr(zmq_utils, "create_zmq_socket", lambda **kwargs: socket)
        monkeypatch.setattr(
            zmq_utils.ZMQMessage,
            "deserialize",
            staticmethod(
                lambda messages: SimpleNamespace(
                    request_type=zmq_utils.ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    body={"success": True, "partition_id": "p0"},
                )
            ),
        )
        manager = SimpleNamespace(
            storage_manager_id="guard-test",
            zmq_context=object(),
            controller_info=SimpleNamespace(ip="redacted", to_addr=lambda name: "inproc://controller"),
        )
        await _strict_notify_and_wait(manager, [b"request"])
        assert socket.closed is True

    @pytest.mark.asyncio
    async def test_connect_failure_closes_notification_socket(self, monkeypatch):
        from transfer_queue.utils import zmq_utils

        socket = _FakeNotifySocket(connect_error=ConnectionError("controller unavailable"))
        monkeypatch.setattr(zmq_utils, "create_zmq_socket", lambda **kwargs: socket)
        manager = SimpleNamespace(
            storage_manager_id="guard-test",
            zmq_context=object(),
            controller_info=SimpleNamespace(ip="redacted", to_addr=lambda name: "inproc://controller"),
        )
        with pytest.raises(ConnectionError, match="controller unavailable"):
            await _strict_notify_and_wait(manager, [b"request"])
        assert socket.closed is True

    @pytest.mark.asyncio
    async def test_missing_production_status_ack_is_bounded(self, monkeypatch):
        from transfer_queue.storage.managers import base as tq_base
        from transfer_queue.utils import zmq_utils

        socket = _FakeNotifySocket()
        monkeypatch.setattr(zmq_utils, "create_zmq_socket", lambda **kwargs: socket)
        monkeypatch.setattr(tq_base, "TQ_DATA_UPDATE_RESPONSE_TIMEOUT", 0)
        manager = SimpleNamespace(
            storage_manager_id="guard-test",
            zmq_context=object(),
            controller_info=SimpleNamespace(ip="redacted", to_addr=lambda name: "inproc://controller"),
        )
        with pytest.raises(TimeoutError, match="production-status ACK"):
            await _strict_notify_and_wait(manager, [b"request"])
        assert socket.closed is True
