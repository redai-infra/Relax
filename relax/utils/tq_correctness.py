# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed correctness guards for TransferQueue's Mooncake backend.

The TransferQueue revision currently pinned by Relax validates the first
Mooncake batch result, but not every retry result, logs removal failures
without raising, and treats a missing/negative production-status ACK as a
successful notification.  Those behaviours can turn an explicit storage or
controller failure into silent data loss.

Keep the compatibility guards here, close to Relax's integration boundary,
until the equivalent checks are available in the pinned TransferQueue
revision.  Installation is process-local and idempotent; every process that
creates or attaches a Mooncake client installs them before ``tq.init``.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any
from uuid import uuid4

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_PATCH_MARKER = "_relax_mooncake_correctness_guards_v1"


def _validate_result_count(operation: str, keys: list[str], results: Any) -> None:
    """Require one Mooncake result code for every requested key."""
    try:
        actual = len(results)
    except TypeError as error:
        raise RuntimeError(f"{operation} returned a non-sized result, expected {len(keys)} codes") from error
    if actual != len(keys):
        raise RuntimeError(f"{operation} returned {actual} results, expected {len(keys)}")


class _StrictMooncakeStoreProxy:
    """Validate every low-level batch response, including retry calls."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def batch_upsert_from(self, keys: list[str], *args: Any, **kwargs: Any) -> Any:
        results = self._store.batch_upsert_from(keys, *args, **kwargs)
        _validate_result_count("batch_upsert_from", keys, results)
        return results

    def batch_get_into(self, keys: list[str], *args: Any, **kwargs: Any) -> Any:
        results = self._store.batch_get_into(keys, *args, **kwargs)
        _validate_result_count("batch_get_into", keys, results)
        return results

    def batch_remove(self, keys: list[str], *args: Any, **kwargs: Any) -> Any:
        results = self._store.batch_remove(keys, *args, **kwargs)
        _validate_result_count("batch_remove", keys, results)
        failures = [(key, code) for key, code in zip(keys, results, strict=True) if code != 0]
        if failures:
            detail = ", ".join(f"{key}={code}" for key, code in failures)
            raise RuntimeError(f"batch_remove failed: {detail}")
        return results


async def _strict_notify_and_wait(self: Any, request_msg: list) -> None:
    """Notify the controller and require a positive ACK within the deadline."""
    import zmq

    from transfer_queue.storage.managers import base as tq_base
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, create_zmq_socket

    identity = f"{self.storage_manager_id}-notify-{uuid4().hex[:8]}".encode()
    sock = create_zmq_socket(
        ctx=self.zmq_context,
        socket_type=zmq.DEALER,
        ip=self.controller_info.ip,
        identity=identity,
    )
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(self.controller_info.to_addr("request_handle_socket"))

    try:
        await sock.send_multipart(request_msg)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + tq_base.TQ_DATA_UPDATE_RESPONSE_TIMEOUT

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for TransferQueue production-status ACK "
                    f"after {tq_base.TQ_DATA_UPDATE_RESPONSE_TIMEOUT}s"
                )
            try:
                messages = await asyncio.wait_for(
                    sock.recv_multipart(copy=False),
                    timeout=min(tq_base.TQ_STORAGE_POLLER_TIMEOUT, remaining),
                )
            except asyncio.TimeoutError:
                continue
            except Exception as error:
                raise RuntimeError("Failed while waiting for TransferQueue production-status ACK") from error

            response = ZMQMessage.deserialize(messages)
            if response.request_type != ZMQRequestType.NOTIFY_DATA_UPDATE_ACK:
                continue
            body = response.body if isinstance(response.body, dict) else {}
            if body.get("success") is not True:
                raise RuntimeError(
                    "TransferQueue controller rejected the production-status update "
                    f"for partition={body.get('partition_id', 'unknown')}"
                )
            return
    finally:
        try:
            if not sock.closed:
                sock.close(linger=0)
        except Exception as error:  # pragma: no cover - best-effort socket cleanup
            logger.debug(f"Failed to close TransferQueue notification socket: {error}")


def _install_store_guards(client_cls: type) -> None:
    if getattr(client_cls, _PATCH_MARKER, False):
        return

    original_init = client_cls.__init__

    @wraps(original_init)
    def guarded_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        store = getattr(self, "_store", None)
        if store is not None and not isinstance(store, _StrictMooncakeStoreProxy):
            self._store = _StrictMooncakeStoreProxy(store)

    client_cls.__init__ = guarded_init
    setattr(client_cls, _PATCH_MARKER, True)


def _install_notification_guards(manager_cls: type) -> None:
    if getattr(manager_cls, _PATCH_MARKER, False):
        return

    original_notify = manager_cls.notify_data_update

    @wraps(original_notify)
    async def guarded_notify(self: Any, *args: Any, **kwargs: Any) -> None:
        if not getattr(self, "controller_info", None):
            raise RuntimeError("TransferQueue storage manager has no controller for production-status notification")
        await original_notify(self, *args, **kwargs)

    manager_cls.notify_data_update = guarded_notify
    manager_cls._notify_and_wait = _strict_notify_and_wait
    setattr(manager_cls, _PATCH_MARKER, True)


def ensure_mooncake_correctness_guards() -> None:
    """Install and validate all guards required for safe Mooncake operation."""
    try:
        from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient
        from transfer_queue.storage.managers.base import StorageManager
    except ImportError as error:
        raise RuntimeError("Installed TransferQueue has no MooncakeStore support") from error

    required_methods = ("_batch_upsert_with_retry", "_batch_get_into_with_retry")
    missing = [name for name in required_methods if not callable(getattr(MooncakeStoreClient, name, None))]
    if missing:
        raise RuntimeError("Installed TransferQueue lacks required Mooncake retry APIs: " + ", ".join(missing))

    _install_store_guards(MooncakeStoreClient)
    _install_notification_guards(StorageManager)

    if not getattr(MooncakeStoreClient, _PATCH_MARKER, False) or not getattr(StorageManager, _PATCH_MARKER, False):
        raise RuntimeError("Failed to install Mooncake silent-data-loss guards")
