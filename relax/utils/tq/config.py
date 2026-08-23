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
from relax.utils.tq.correctness import ensure_mooncake_correctness_guards


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
    """Return the externally managed Mooncake master endpoint.

    ``MC_MASTER_ADDRESS`` is required.  A loopback default would make every
    node of a multi-node job treat its own localhost as the master, so the
    reachability probe would degrade ``auto`` runs and abort ``off``/
    ``required`` runs even when a shared master is healthy elsewhere.
    """
    address = os.environ.get("MC_MASTER_ADDRESS", "").strip()
    if not address:
        raise RuntimeError(
            "MooncakeStore requires MC_MASTER_ADDRESS=<host:port> of the externally "
            "managed mooncake master on every node; Relax never assumes a loopback "
            "endpoint."
        )
    return address


def resolve_global_segment_size() -> int:
    """Per-client Mooncake segment size in bytes.

    Defaults to 4 GiB (TQ config.yaml:52).  Deployments whose worst-case in-
    flight payload exceeds that (see :func:`estimate_payload_bytes`) set
    ``RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB`` instead of editing code; capacity
    validation and the client config read the same value so the check can never
    pass a size the client does not actually mount.
    """
    raw = os.environ.get("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", "").strip()
    if not raw:
        return _DEFAULT_GLOBAL_SEGMENT_SIZE
    try:
        gib = float(raw)
    except ValueError as error:
        raise RuntimeError(f"RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB={raw!r} must be a positive number of GiB") from error
    if gib <= 0:
        raise RuntimeError(f"RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB={raw!r} must be a positive number of GiB")
    return int(gib * 1024**3)


def validate_mooncake_runtime_contract() -> None:
    """Install and validate the Mooncake loss-prevention contract.

    A version number alone is insufficient for development builds. Relax
    therefore installs process-local guards that validate every batch response,
    propagate removal failures, and require a positive production-status ACK.
    Every process calls this before creating or attaching a Mooncake client.
    """
    ensure_mooncake_correctness_guards()

    from transfer_queue.storage.managers.base import KVStorageManager

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


def build_simple_storage_config(total_storage_size: int | None, num_data_storage_units: int) -> dict[str, Any]:
    """Build the SimpleStorage backend config.

    ``total_storage_size=None`` preserves TransferQueue's unlimited-capacity
    benchmark semantics; production Relax jobs pass a concrete sample count.
    """
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
        External master server address.  If ``None``, read from the required
        ``MC_MASTER_ADDRESS`` env var (see
        :func:`resolve_mooncake_master_address`).
    global_segment_size
        Override the per-client segment size.  ``None`` resolves from
        ``RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB`` (default 4 GiB, see
        :func:`resolve_global_segment_size`).  Benchmarks may pass a larger
        value (e.g. 8 GiB) to avoid staging-buffer pressure.
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
            "global_segment_size": global_segment_size or resolve_global_segment_size(),
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


# Worst-case payload factors used by the segment-capacity pre-check.
# Vision: a ViT-style processor (Qwen-VL family: patch 14x14, spatial merge
# 2x2) maps one schedulable token to at most (14*2)^2 = 784 pixels, and
# ``pixel_values`` is float32 RGB, so vision bytes <= seq_length * 784 * 12.
_PIXELS_PER_VISION_TOKEN = 28 * 28
_BYTES_PER_PIXEL_VALUE = 3 * 4
# Text: token ids, logprobs, masks and rewards; 32 B/token rounds them up.
_TEXT_BYTES_PER_TOKEN = 32


def resolve_tq_capacity_batch_size(args: Any) -> int:
    """Return the largest batch that one TQ step may need to hold.

    Dynamic partial rollout schedules from the over-sampling pool, so its
    capacity contract is ``over_sampling_batch_size`` rather than the smaller
    nominal ``rollout_batch_size``.  Keep this resolution shared with the
    Controller's SimpleStorage sizing so both backends reserve for the same
    number of samples.
    """
    rollout_batch = getattr(args, "rollout_batch_size")
    if getattr(args, "partial_rollout", False) and getattr(args, "use_dynamic_global_batch_size", False):
        over_sampling_batch = getattr(args, "over_sampling_batch_size", None)
        if over_sampling_batch is not None:
            return int(over_sampling_batch)
    return int(rollout_batch)


def estimate_payload_bytes(args: Any) -> int:
    """Worst-case per-step payload upper bound in bytes.

    Derived from the token budget instead of a fixed per-sample constant: the
    processor cannot emit more vision tokens than ``--seq-length`` allows, so
    the pixel payload of one sample is bounded by
    ``seq_length * _PIXELS_PER_VISION_TOKEN * _BYTES_PER_PIXEL_VALUE``
    (e.g. 77 MiB at seq_length=8192) rather than the old 8 MiB guess that
    passed configurations which later failed puts mid-training.
    """
    n_samples = args.n_samples_per_prompt
    capacity_batch = resolve_tq_capacity_batch_size(args)
    seq_length = int(getattr(args, "seq_length", 0) or 0)
    if seq_length <= 0:
        raise RuntimeError(
            "MooncakeStore segment-capacity validation needs args.seq_length to bound "
            "the per-sample payload; got a missing or non-positive value."
        )
    per_sample = seq_length * _TEXT_BYTES_PER_TOKEN
    if getattr(args, "multimodal_keys", None) is not None:
        per_sample += seq_length * _PIXELS_PER_VISION_TOKEN * _BYTES_PER_PIXEL_VALUE
    return capacity_batch * n_samples * per_sample


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
    capacity_batch = resolve_tq_capacity_batch_size(args)
    payload = estimate_payload_bytes(args)
    needed = payload * (max_staleness + 1)
    available = resolve_global_segment_size()

    if needed > available:
        return (
            f"MooncakeStore segment capacity insufficient: worst-case in-flight payload "
            f"{needed / 1024**3:.1f} GiB (effective_batch={capacity_batch} × "
            f"n_samples={args.n_samples_per_prompt} × staleness+1={max_staleness + 1}) "
            f"exceeds global_segment_size {available / 1024**3:.1f} GiB. "
            f"Reduce batch size / max_staleness, or raise RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB."
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
