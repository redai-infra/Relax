# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed capability validation for TransferQueue's Mooncake backend.

Relax validates the capabilities and environment it can inspect without
modifying TransferQueue at runtime.  Per-retry result-length validation and a
strict production-status ACK must be fixed in upstream TransferQueue and then
consumed through an updated, capability-marked pin; method-name checks alone do
not prove those semantics.

The pinned mooncake 0.3.10 additionally corrupts TCP-protocol transfers
through its auto-enabled memcpy fast path, so that path is force-disabled
here and an explicit enable is rejected (see :func:`_enforce_safe_memcpy`).
"""

from __future__ import annotations

import os


def _enforce_safe_memcpy() -> None:
    """Force-disable mooncake's memcpy fast path; reject attempts to enable it.

    mooncake 0.3.10 auto-enables ``MC_STORE_MEMCPY`` in TCP-only environments
    (``transfer_task.cpp`` "auto-detected: TCP-only environment, memcpy
    enabled") and that path silently truncates cross-node gets: roughly half of
    fresh-session first transfers returned rows whose tails were zero bytes
    from a 64 KiB-aligned offset onward while every batch code reported success
    (two-node forensic probes, 2026-08; 12/12 sessions clean with
    ``MC_STORE_MEMCPY=0`` vs ~50% corrupt without).  The same code path
    SIGSEGVs on single-node loopback.  Because the corruption is confirmed on
    the pinned mooncake build, this guard fails closed: an explicit
    ``MC_STORE_MEMCPY=1`` is rejected at startup instead of honoured.  Re-gate
    on the mooncake version once the pin moves to a release with the fix.
    """
    override = os.environ.get("MC_STORE_MEMCPY", "").strip()
    if override not in ("", "0"):
        raise RuntimeError(
            "MC_STORE_MEMCPY explicitly enables an unsafe value: the pinned mooncake 0.3.10 "
            "memcpy fast path silently truncates TCP transfers and can SIGSEGV. "
            "Unset MC_STORE_MEMCPY; Relax forces it to 0 on this version."
        )
    os.environ["MC_STORE_MEMCPY"] = "0"


def ensure_mooncake_correctness_guards() -> None:
    """Validate that the installed stack can run MooncakeStore safely.

    Enforces the memcpy environment contract and checks that the pinned
    TransferQueue ships the Mooncake retry APIs Relax's data plane relies on.
    It does not modify TransferQueue code or objects at runtime.
    """
    _enforce_safe_memcpy()
    try:
        from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient
    except ImportError as error:
        raise RuntimeError("Installed TransferQueue has no MooncakeStore support") from error

    required_methods = ("_batch_upsert_with_retry", "_batch_get_into_with_retry")
    missing = [name for name in required_methods if not callable(getattr(MooncakeStoreClient, name, None))]
    if missing:
        raise RuntimeError("Installed TransferQueue lacks required Mooncake retry APIs: " + ", ".join(missing))
