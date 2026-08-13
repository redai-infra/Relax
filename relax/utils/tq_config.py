# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build TransferQueue backend config dicts from Relax CLI args.

This module is the single place that maps Relax-side *intent* flags
(``--tq-storage-backend``, ``--tq-rdma-mode``, ``--tq-rdma-device``,
``--tq-use-gdr``) plus an :class:`~relax.utils.rdma_probe.EffectiveConfig`
into the OmegaConf dict that ``tq.init`` expects.

Mooncake internals (endpoint, buffer size, segment size, master address,
timeout) are intentionally *not* exposed as CLI flags — they come from
internal defaults or the deployment environment, per maintainer guidance.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

from relax.utils.logging_utils import get_logger
from relax.utils.rdma_probe import EffectiveConfig


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults (kept here rather than in config.yaml so they are visible to
# Relax contributors without reading the TQ package).
# ---------------------------------------------------------------------------

_DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024**3  # 4 GiB per client (config.yaml:52)
_DEFAULT_LOCAL_BUFFER_SIZE = 1 * 1024**3  # 1 GiB per client (config.yaml:54)
_DEFAULT_GDR_STAGING_MB = 1024  # config.yaml:103
_DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"  # config.yaml:42-43


def resolve_mooncake_master_address() -> str:
    """Return the externally managed Mooncake master endpoint."""
    return os.environ.get("MC_MASTER_ADDRESS", "localhost:50051")


def validate_mooncake_runtime_contract() -> None:
    """Fail fast unless installed TransferQueue has the loss-prevention fixes.

    The RDMA integration depends on per-key put/get result validation and on
    notifying production readiness only after storage succeeds. A version
    number alone is insufficient for development builds, so validate the
    concrete runtime capabilities before probing or creating a controller.
    """
    try:
        from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient
        from transfer_queue.storage.managers.base import KVStorageManager
    except ImportError as e:
        raise RuntimeError("Installed TransferQueue has no MooncakeStore support") from e

    required_methods = ("_batch_upsert_with_retry", "_batch_get_into_with_retry")
    missing = [name for name in required_methods if not callable(getattr(MooncakeStoreClient, name, None))]
    if missing:
        raise RuntimeError("Installed TransferQueue lacks required Mooncake failure handling: " + ", ".join(missing))

    put_source = inspect.getsource(KVStorageManager.put_data)
    storage_call = put_source.find("self.storage_client.put")
    ready_notify = put_source.find("self.notify_data_update")
    if storage_call < 0 or ready_notify < 0 or storage_call > ready_notify:
        raise RuntimeError(
            "Installed TransferQueue does not guarantee storage success before production-status notification"
        )


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_simple_storage_config(total_storage_size: int, num_data_storage_units: int) -> dict[str, Any]:
    """Build the ``backend`` dict for SimpleStorage (current default
    behavior)."""
    return {
        # ``tq.init`` selects the manager from this key alone (TQ config.yaml:22
        # defaults it to SimpleStorage); a backend section without it is ignored.
        "storage_backend": "SimpleStorage",
        "SimpleStorage": {
            "total_storage_size": total_storage_size,
            "num_data_storage_units": num_data_storage_units,
        },
    }


def build_mooncake_config(
    effective: EffectiveConfig,
    *,
    master_address: str | None = None,
    global_segment_size: int | None = None,
) -> dict[str, Any]:
    """Build the ``backend`` dict for MooncakeStore.

    Parameters
    ----------
    effective
        The job-level :class:`EffectiveConfig` after probing.
    master_address
        External master server address.  If ``None``, read from the
        ``MC_MASTER_ADDRESS`` env var; if still unset, fall back to localhost
        (single-node dev only — production must set the env var).
    global_segment_size
        Override the per-client segment size (default 4 GiB).  Benchmarks may
        pass a larger value (e.g. 8 GiB) to avoid staging-buffer pressure.
    """
    if master_address is None:
        master_address = resolve_mooncake_master_address()

    cfg: dict[str, Any] = {
        # Selects the manager inside ``tq.init`` (interface.py reads
        # ``backend.storage_backend``, TQ config.yaml:22 defaults it to
        # SimpleStorage).  Without this key the MooncakeStore section below is
        # parsed and then silently ignored -- the job still runs on ZMQ/TCP.
        "storage_backend": "MooncakeStore",
        "MooncakeStore": {
            # Transport
            "protocol": effective.protocol,  # "rdma" or "tcp"
            "device_name": effective.device,
            # Master / metadata — externally managed, never auto-init.
            "auto_init": False,
            "master_server_address": master_address,
            "metadata_server": _DEFAULT_METADATA_SERVER,
            "local_hostname": "",  # empty = auto-detect via Ray node IP
            # Memory
            "global_segment_size": global_segment_size or _DEFAULT_GLOBAL_SEGMENT_SIZE,
            "local_buffer_size": _DEFAULT_LOCAL_BUFFER_SIZE,
            # Do NOT silently evict produced-but-unconsumed data.
            "hard_pin": True,
            # GDR
            "use_gdr": effective.gdr,
            "gdr_staging_buffer_mb": _DEFAULT_GDR_STAGING_MB,
        },
    }
    return cfg


# ---------------------------------------------------------------------------
# Capacity validation
# ---------------------------------------------------------------------------


def estimate_payload_bytes(args: Any) -> int:
    """Rough estimate of per-step multimodal payload size in bytes.

    Used only for segment-capacity pre-check.  The real payload depends on
    image resolution and patch count; this is a conservative lower bound based
    on ``--multimodal-keys`` presence and ``n_samples_per_prompt``.
    """
    n_samples = args.n_samples_per_prompt
    rollout_batch = args.rollout_batch_size
    # Conservative: 8 MiB per sample when multimodal is enabled (real range
    # 7.4 MiB for a 400-token image to hundreds of MiB at max token budget).
    per_sample_mb = 8 if getattr(args, "multimodal_keys", None) is not None else 0
    return rollout_batch * n_samples * per_sample_mb * 1024 * 1024


def validate_segment_capacity(args: Any, effective: EffectiveConfig) -> str | None:
    """Return an error message if segment capacity is insufficient, else None.

    Only checked for MooncakeStore (SimpleStorage manages its own capacity via
    ``total_storage_size``).  The check is conservative: it uses the *per-
    client* segment size (``global_segment_size``) against the in-flight upper
    bound.
    """
    if effective.backend != "MooncakeStore":
        return None

    max_staleness = getattr(args, "max_staleness", 0)
    payload = estimate_payload_bytes(args)
    needed = payload * (max_staleness + 1)
    available = _DEFAULT_GLOBAL_SEGMENT_SIZE

    if needed > available:
        return (
            f"MooncakeStore segment capacity insufficient: estimated in-flight payload "
            f"{needed / 1024**3:.1f} GiB (rollout_batch={args.rollout_batch_size} × "
            f"n_samples={args.n_samples_per_prompt} × staleness+1={max_staleness + 1}) "
            f"exceeds global_segment_size {available / 1024**3:.1f} GiB. "
            f"Reduce batch size, increase global_segment_size, or reduce max_staleness."
        )
    return None


# ---------------------------------------------------------------------------
# Top-level resolver
# ---------------------------------------------------------------------------


def build_backend_config(
    args: Any,
    effective: EffectiveConfig,
    *,
    total_storage_size: int,
) -> tuple[dict[str, Any], str | None]:
    """Return ``(backend_config_dict, error_or_none)``.

    On error, ``backend_config_dict`` is a safe SimpleStorage fallback and
    ``error`` explains why MooncakeStore was rejected.
    """
    if effective.backend == "SimpleStorage":
        return build_simple_storage_config(
            total_storage_size=total_storage_size,
            num_data_storage_units=args.num_data_storage_units,
        ), None

    # MooncakeStore path.
    cap_error = validate_segment_capacity(args, effective)
    if cap_error:
        logger.error(cap_error)
        return build_simple_storage_config(
            total_storage_size=total_storage_size,
            num_data_storage_units=args.num_data_storage_units,
        ), cap_error

    return build_mooncake_config(effective), None
