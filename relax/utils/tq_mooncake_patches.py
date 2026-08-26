# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Version-gated runtime loss guards for the pinned TransferQueue revision.

The TransferQueue revision currently pinned by Relax validates the first
Mooncake batch result but not every retry result, logs removal failures
without raising, and treats a missing or negative production-status ACK as a
successful notification.  Those behaviours can turn an explicit storage or
controller failure into silent data loss.  Until the equivalent checks land
upstream, this module patches the pinned revision at runtime; installation is
process-local and idempotent.

These are monkey patches over *private* upstream internals
(``MooncakeStoreClient.__init__``, ``StorageManager._notify_and_wait``), so
they are gated on the exact pinned package version and installed VCS revision:
any other build refuses to start rather than running unvalidated patches (see
:func:`_require_pinned_transfer_queue`).

Removal condition: delete this module and its single call site in
:func:`relax.utils.tq_correctness.ensure_mooncake_correctness_guards` once the
pinned TransferQueue itself validates every batch/retry result, raises on
non-idempotent removal failure, and requires a positive production-status ACK.
"""

from __future__ import annotations

import asyncio
import json
import re
from functools import wraps
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_PATCH_MARKER = "_relax_mooncake_correctness_guards_v1"
# Mooncake's idempotent-delete result; the pinned upstream clear path accepts it.
_MOONCAKE_OBJECT_NOT_FOUND = -704

# Exact builds these patches were written and forensically validated against.
_PATCHED_TQ_BUILDS = {
    "0.1.10.dev0": frozenset({"58054a33834aadbcf76aacd6b1e32e25c030f2c9"}),
}


def _installed_transfer_queue_revision(transfer_queue_module: Any) -> str | None:
    """Read the installed VCS revision from standard PEP 610 metadata."""
    try:
        distribution = metadata.distribution("transferqueue")
        if str(distribution.version) != str(getattr(transfer_queue_module, "__version__", "unknown")):
            return None

        module_file = getattr(transfer_queue_module, "__file__", None)
        if not module_file:
            return None
        module_path = Path(module_file).resolve()
        package_root = Path(distribution.locate_file("transfer_queue")).resolve()
        module_path.relative_to(package_root)

        direct_url = distribution.read_text("direct_url.json")
    except (metadata.PackageNotFoundError, AttributeError, OSError, TypeError, UnicodeError, ValueError):
        return None
    if direct_url is None:
        return None

    try:
        provenance = json.loads(direct_url)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(provenance, dict):
        return None

    vcs_info = provenance.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        return None
    commit_id = vcs_info.get("commit_id")
    if not isinstance(commit_id, str) or not commit_id.strip():
        return None
    revision = commit_id.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        return None
    return revision


def _require_pinned_transfer_queue() -> None:
    """Refuse to patch any TransferQueue source revision not validated here."""
    import transfer_queue

    version = str(getattr(transfer_queue, "__version__", "unknown"))
    expected_revisions = _PATCHED_TQ_BUILDS.get(version)
    if expected_revisions is None:
        raise RuntimeError(
            f"transfer_queue {version} is not covered by Relax's Mooncake loss guards "
            f"(validated versions: {', '.join(_PATCHED_TQ_BUILDS)}).  These guards replace "
            "private upstream internals (MooncakeStoreClient.__init__, "
            "StorageManager._notify_and_wait); re-validate them against the new pin and "
            "extend _PATCHED_TQ_BUILDS, or delete relax/utils/tq_mooncake_patches.py "
            "entirely if the fixes have landed upstream."
        )

    revision = _installed_transfer_queue_revision(transfer_queue)
    if revision not in expected_revisions:
        actual_revision = revision or "unknown"
        raise RuntimeError(
            f"transfer_queue {version} revision {actual_revision} is not covered by Relax's Mooncake loss guards "
            f"(validated revisions: {', '.join(sorted(expected_revisions))}).  The exact source revision is required "
            "because these guards replace private upstream internals; install the validated pin or re-validate "
            "the guards before extending _PATCHED_TQ_BUILDS."
        )


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
        if name == "_store":
            raise AttributeError(name)
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
        failures = [
            (key, code) for key, code in zip(keys, results, strict=True) if code not in (0, _MOONCAKE_OBJECT_NOT_FOUND)
        ]
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
    sock = None
    try:
        sock = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.DEALER,
            ip=self.controller_info.ip,
            identity=identity,
        )
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.controller_info.to_addr("request_handle_socket"))

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
            if sock is not None and not sock.closed:
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


def install_mooncake_loss_guards() -> None:
    """Install and verify all runtime guards (idempotent, process-local)."""
    _require_pinned_transfer_queue()
    from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient
    from transfer_queue.storage.managers.base import StorageManager

    _install_store_guards(MooncakeStoreClient)
    _install_notification_guards(StorageManager)

    if not getattr(MooncakeStoreClient, _PATCH_MARKER, False) or not getattr(StorageManager, _PATCH_MARKER, False):
        raise RuntimeError("Failed to install Mooncake silent-data-loss guards")
