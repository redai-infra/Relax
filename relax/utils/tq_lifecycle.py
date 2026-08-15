# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TransferQueue controller/store lifecycle helpers.

Extracted from :mod:`relax.core.controller` so the behavior can be unit-tested
without importing the whole Controller (which drags in Ray Serve and Megatron).
The Controller keeps thin wrappers that delegate here.

Two problems these helpers exist for:

* **F10 anti-hang** -- ``tq.init`` attaches to the existing named actor and polls
  ``get_config`` forever while it returns ``None`` (TQ ``interface.py:109-118``),
  so a controller left behind by a run that died between actor creation and
  ``store_config`` turns the next start into a hang.
* **Segment leak on teardown** -- ``tq.close()`` only tears down ZMQ (TQ
  ``storage/managers/base.py:378``); it never calls ``storage_client.close()``,
  so a MooncakeStore segment stays mounted and registered in the master until
  ``client_ttl`` (30 s) expires, and puts from the restarted job fail with
  "Failed to open segment ... Connection refused".
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import ray
import transfer_queue as tq

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

CONTROLLER_NAME = "TransferQueueController"
CONTROLLER_NAMESPACE = "transfer_queue"
OWNER_TOKEN_FIELD = "relax_owner_token"
DEFAULT_TQ_INIT_TIMEOUT_SECONDS = 60.0
DEFAULT_TQ_ATTACH_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class TqInitResult:
    """Result of an owner-aware TransferQueue initialization transaction."""

    config: Any
    owner: Any | None
    fallback_reason: str = ""

    @property
    def owns_controller(self) -> bool:
        return self.owner is not None


class TqInitializationTimeout(TimeoutError):
    """Raised when ``tq.init`` does not finish within the bounded timeout."""


class TqAttachTimeout(TimeoutError):
    """Raised when a worker cannot attach to TransferQueue within the bound."""


class TqCleanupTimeout(TimeoutError):
    """Raised when a TQ controller cannot be confirmed gone after cleanup."""


class TqConfigurationMismatch(RuntimeError):
    """Raised when an existing controller uses an incompatible job config."""


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _backend_signature(conf: Any) -> tuple[Any, ...]:
    """Return the backend fields that must agree for a safe attach."""
    backend = _get_config_value(conf, "backend", {})
    storage_backend = _get_config_value(backend, "storage_backend", "SimpleStorage")
    if storage_backend == "MooncakeStore":
        mooncake = _get_config_value(backend, "MooncakeStore", {})
        return (
            storage_backend,
            _get_config_value(mooncake, "protocol", "tcp"),
            _get_config_value(mooncake, "device_name", "") or "",
            _get_config_value(mooncake, "master_server_address", ""),
            _get_config_value(mooncake, "metadata_server", ""),
            _get_config_value(mooncake, "global_segment_size"),
            _get_config_value(mooncake, "local_buffer_size"),
            bool(_get_config_value(mooncake, "hard_pin", False)),
            bool(_get_config_value(mooncake, "use_gdr", False)),
        )
    simple = _get_config_value(backend, "SimpleStorage", {})
    return (
        "SimpleStorage",
        _get_config_value(simple, "total_storage_size"),
        _get_config_value(simple, "num_data_storage_units"),
    )


def _sampler_signature(sampler: Any) -> tuple[Any, ...]:
    """Return sampler identity and immutable construction-time parameters.

    TransferQueue samplers keep mutable scheduling state in underscore-prefixed
    attributes.  Those caches legitimately differ between processes and must
    not prevent an attach; public attributes describe the sampling contract
    that workers and the existing controller must agree on.
    """
    if sampler is None:
        return (None, ())
    if isinstance(sampler, str):
        return ("string", sampler)
    if isinstance(sampler, type):
        return ("class", f"{sampler.__module__}.{sampler.__qualname__}")

    sampler_type = f"{type(sampler).__module__}.{type(sampler).__qualname__}"
    try:
        public_config = tuple(
            sorted((name, value) for name, value in vars(sampler).items() if not name.startswith("_"))
        )
    except TypeError:
        public_config = ()
    return (sampler_type, public_config)


def _configuration_signature(conf: Any) -> tuple[Any, ...]:
    """Return fields that must agree for workers to share a controller."""
    controller = _get_config_value(conf, "controller", {})
    return (
        _backend_signature(conf),
        bool(_get_config_value(controller, "polling_mode", False)),
        _sampler_signature(_get_config_value(controller, "sampler")),
    )


def _backend_description(conf: Any) -> str:
    """Describe the backend without logging endpoints or host information."""
    backend = _get_config_value(conf, "backend", {})
    storage_backend = _get_config_value(backend, "storage_backend", "SimpleStorage")
    if storage_backend != "MooncakeStore":
        return "SimpleStorage"
    mooncake = _get_config_value(backend, "MooncakeStore", {})
    return f"MooncakeStore/{_get_config_value(mooncake, 'protocol', 'tcp')}"


def _uses_mooncake(conf: Any) -> bool:
    backend = _get_config_value(conf, "backend", {})
    return _get_config_value(backend, "storage_backend", "SimpleStorage") == "MooncakeStore"


def uses_mooncake(conf: Any) -> bool:
    """True when ``conf`` selects the MooncakeStore backend.

    Public so the Controller can decide whether the cluster-wide attach
    handshake is required for the stored job-level config.
    """
    return _uses_mooncake(conf)


def _prepare_mooncake_runtime(conf: Any) -> None:
    if not _uses_mooncake(conf):
        return
    from relax.utils.tq_config import validate_mooncake_runtime_contract

    validate_mooncake_runtime_contract()


def kill_tq_controller_and_wait(timeout: float = 20.0) -> None:
    """Kill the TransferQueueController named actor, then wait for GCS
    deregistration.

    Waiting matters because ``ray.kill`` is asynchronous: the handle stays
    resolvable for a short window, and a re-init landing in that window
    attaches to a dying actor and fails with ``ActorDiedError``.
    """
    try:
        stale = ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
        ray.kill(stale)
        logger.info("[dataplane] Killed TransferQueueController actor (F10 guard).")
    except ValueError:
        return  # actor does not exist — nothing to kill or wait for.
    except Exception as e:
        raise RuntimeError(f"Failed to kill TransferQueueController: {e}") from e

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
        except ValueError:
            return
        time.sleep(0.4)
    raise TqCleanupTimeout(f"TransferQueueController still resolvable after {timeout}s")


def reap_unusable_tq_controller(get_config_timeout: float = 10.0) -> bool:
    """Kill the TransferQueueController only if it cannot serve a config.

    Returns ``True`` when a controller was reaped.  A controller that *does*
    return a config is left alone: it belongs to whoever created it, attaching
    to it is the intended behavior, and this keeps the guard within "clean up
    only what this job owns" (no broad pkill/killall).
    """
    try:
        existing = ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
    except ValueError:
        return False  # nothing there — nothing to reap.

    try:
        conf = ray.get(existing.get_config.remote(), timeout=get_config_timeout)
    except Exception as e:  # actor dead, unresponsive, or API missing
        logger.warning(f"[dataplane] Existing TransferQueueController is unusable ({e}); reaping it.")
        conf = None

    if conf is not None:
        logger.info("[dataplane] Existing TransferQueueController is healthy; tq.init will attach to it.")
        return False

    logger.warning("[dataplane] TransferQueueController has no stored config (half-initialised); reaping it.")
    kill_tq_controller_and_wait()
    return True


def _controller_exists() -> bool:
    try:
        ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
        return True
    except ValueError:
        return False


def _set_owner_token(conf: Any, token: str) -> None:
    controller = conf.controller if hasattr(conf, "controller") else conf["controller"]
    controller[OWNER_TOKEN_FIELD] = token


def _get_owner_token(conf: Any) -> str:
    if conf is None:
        return ""
    controller = conf.controller if hasattr(conf, "controller") else conf.get("controller", {})
    if hasattr(controller, "get"):
        return str(controller.get(OWNER_TOKEN_FIELD, ""))
    return ""


def _get_stored_config(timeout: float = 10.0) -> Any:
    controller = ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
    conf = ray.get(controller.get_config.remote(), timeout=timeout)
    if conf is None:
        raise RuntimeError("TransferQueueController returned no config after tq.init completed")
    return conf


def _close_local_tq_client() -> None:
    """Detach this process without deleting global TQ data or controller."""
    client = None
    store_client = None
    try:
        client = tq.get_client()
        store_client = getattr(client.storage_manager, "storage_client", None)
    except (AssertionError, AttributeError):
        pass

    if store_client is not None and hasattr(store_client, "close"):
        try:
            store_client.close()
        except Exception as e:  # pragma: no cover - best-effort local cleanup
            logger.warning(f"[dataplane] Failed to close attached MooncakeStore client: {e}")

    if client is not None and hasattr(client, "close"):
        try:
            client.close()
        except Exception as e:  # pragma: no cover - best-effort local cleanup
            logger.warning(f"[dataplane] Failed to close attached TransferQueue client: {e}")

    # TransferQueue has no public detach-only API. Reset only process-local
    # handles; never touch _TQ_STORAGE or the named controller actor.
    try:
        from transfer_queue import interface as tq_interface

        tq_interface._TQ_CLIENT = None
        tq_interface._TQ_CONTROLLER = None
    except (ImportError, AttributeError):  # pragma: no cover - version dependent
        pass


def log_tq_gdr_runtime_status(*, requested: bool, role: str) -> str:
    """Log requested GDR intent separately from the local client's status.

    ``enabled_unverified`` means the client selected its GDR staging path, but
    Relax has not proved that a transfer traversed GDR on the wire.  This
    avoids claiming job-wide GDR effectiveness from a driver-side capability
    probe.
    """
    if not requested:
        return "not_requested"

    status = "unknown"
    detail = "client introspection unavailable"
    try:
        manager = tq.get_client().storage_manager
        store_client = getattr(manager, "storage_client", None)
        if store_client is None or type(manager).__name__ != "MooncakeStorageManager":
            status = "inactive"
            detail = f"manager={type(manager).__name__}"
        elif getattr(store_client, "protocol", "") != "rdma":
            status = "inactive"
            detail = f"protocol={getattr(store_client, 'protocol', 'unknown')}"
        elif getattr(store_client, "_gdr_staging", None) is None:
            status = "host_rdma_fallback"
            detail = "GDR staging unavailable in this worker"
        else:
            status = "enabled_unverified"
            detail = "local GDR path selected; wire effectiveness is unknown"
    except Exception as e:  # pragma: no cover - environment/version dependent
        detail = str(e)

    log = logger.warning if status != "enabled_unverified" else logger.info
    log(f"[dataplane:gdr] role={role} requested=true status={status} experimental=true detail={detail}")
    return status


def _resolve_attach_timeout() -> float:
    """Attach deadline in seconds; override via
    ``RELAX_TQ_ATTACH_TIMEOUT_SECONDS``."""
    raw = os.environ.get("RELAX_TQ_ATTACH_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_TQ_ATTACH_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError(f"RELAX_TQ_ATTACH_TIMEOUT_SECONDS={raw!r} must be a positive number of seconds") from error
    if value <= 0:
        raise RuntimeError(f"RELAX_TQ_ATTACH_TIMEOUT_SECONDS={raw!r} must be a positive number of seconds")
    return value


def _await_controller_config(deadline: float) -> None:
    """Bounded wait until the named controller serves a non-``None`` config.

    ``tq.init`` polls ``get_config`` forever while it returns ``None`` (the F10
    hang), so a worker refuses to enter that loop unless a config is provably
    served before the deadline.
    """
    last_error = "TransferQueueController named actor not found"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TqAttachTimeout(f"TransferQueue attach timed out: {last_error}")
        try:
            controller = ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
        except ValueError:
            time.sleep(min(0.5, remaining))
            continue
        try:
            conf = ray.get(controller.get_config.remote(), timeout=max(min(remaining, 10.0), 0.1))
        except Exception as e:
            last_error = f"get_config failed: {e}"
            time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))
            continue
        if conf is not None:
            return
        last_error = "controller exists but has stored no config yet"
        time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))


def _bounded_tq_init(conf: Any, deadline: float, *, role: str) -> None:
    """Run ``tq.init`` under the remaining deadline.

    ``tq.init`` takes no timeout and its mooncake client setup blocks in native
    code, so it runs on a daemon watchdog thread.  On expiry the caller fails
    fast with :class:`TqAttachTimeout`; the abandoned thread dies with the
    failed worker process (Serve tears the replica down).
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TqAttachTimeout(f"TransferQueue attach for role={role} timed out before tq.init")
    error: list[BaseException] = []

    def _run() -> None:
        try:
            tq.init(conf=conf)
        except BaseException as e:  # propagated to the attaching caller below
            error.append(e)

    thread = threading.Thread(target=_run, name=f"tq-attach-{role}", daemon=True)
    thread.start()
    thread.join(remaining)
    if thread.is_alive():
        raise TqAttachTimeout(
            f"tq.init for role={role} did not finish within {remaining:.0f}s "
            "(mooncake setup or controller poll hung); failing this worker fast."
        )
    if error:
        raise error[0]


def attach_tq_client(conf: Any, *, requested_gdr: bool, role: str, timeout: float | None = None) -> Any:
    """Attach a component process within a bounded deadline and report its
    local experimental GDR state.

    The deadline covers both waiting for a served controller config and
    ``tq.init`` itself, because either phase can hang unboundedly (get_config
    polling and mooncake endpoint setup respectively).  ``None`` resolves the
    deadline from ``RELAX_TQ_ATTACH_TIMEOUT_SECONDS`` (default 60 s).
    """
    if timeout is None:
        timeout = _resolve_attach_timeout()
    deadline = time.monotonic() + timeout
    _prepare_mooncake_runtime(conf)
    _await_controller_config(deadline)
    _bounded_tq_init(conf, deadline, role=role)
    client = tq.get_client()
    log_tq_gdr_runtime_status(requested=requested_gdr, role=role)
    return client


def detach_tq_client() -> None:
    """Detach this worker's TQ client (attach-only inverse of
    :func:`attach_tq_client`).

    Closes the attached Mooncake storage client so its segment deregisters
    from the master immediately instead of lingering until ``client_ttl``
    expires — a stale endpoint breaks fast restarts with "Failed to open
    segment".  Only process-local handles are touched (never the named
    controller or globally stored data), so every worker teardown hook may
    call this unconditionally; force-killed workers still fall back to the
    master-side TTL.
    """
    _close_local_tq_client()


def _alive_node_ids() -> list[str]:
    """Every alive node: TQ endpoints (Serve replicas and 0-CPU actors) carry
    no placement binding, so any alive node may end up hosting one."""
    return [n["NodeID"] for n in ray.nodes() if n.get("Alive")]


def verify_cluster_attach(conf: Any, *, timeout: float | None = None) -> list[str]:
    """Bounded attach handshake from every alive node; returns failure
    summaries.

    Each task performs the same bounded :func:`attach_tq_client` a worker would
    perform (real stored config, real storage client) and detaches immediately,
    so it validates the *actual endpoints* instead of a ``/sys`` capability
    heuristic on a node the scheduler may never use.  An empty return means
    every alive node attached within the deadline; the driver aggregates
    failures and decides one job-level outcome.
    """
    if timeout is None:
        timeout = _resolve_attach_timeout()
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0, max_retries=0)
    def _handshake(handshake_conf: Any) -> None:
        from relax.utils.tq_lifecycle import attach_tq_client, detach_tq_client

        attach_tq_client(handshake_conf, requested_gdr=False, role="attach-handshake")
        detach_tq_client()

    refs: list[Any] = []
    id_by_ref: dict[Any, str] = {}
    for node_id in _alive_node_ids():
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ref = _handshake.options(scheduling_strategy=strategy).remote(conf)
        refs.append(ref)
        id_by_ref[ref] = node_id

    # Grace beyond the per-node attach deadline covers task scheduling and
    # worker startup on a busy cluster.
    wait_bound = timeout + 30.0
    ready, pending = ray.wait(refs, num_returns=len(refs), timeout=wait_bound)
    failures: list[str] = []
    for ref in ready:
        try:
            ray.get(ref)
        except Exception as e:
            failures.append(f"node {id_by_ref[ref][:12]}: {e}")
    for ref in pending:
        ray.cancel(ref, force=True)
        failures.append(f"node {id_by_ref[ref][:12]}: handshake did not return within {wait_bound:.0f}s")
    return failures


def close_tq_and_unmount(*, is_owner: bool) -> None:
    """Close TransferQueue and unmount the MooncakeStore segment.

    Order matters: ``tq.close()`` still needs the store alive for its
    ``remove_all()``, so the client handle is captured first and unmounted
    after. SimpleStorage has no ``storage_client``, so this is a no-op there.

    Global ``tq.close()`` is owner-only because upstream kills the named
    controller even in a process that merely attached to it.  Non-owners only
    detach their process-local client.
    """
    if not is_owner:
        logger.info("[dataplane] Detaching local TQ client; global controller is owned by another process.")
        _close_local_tq_client()
        return

    store_client = None
    try:
        store_client = getattr(tq.get_client().storage_manager, "storage_client", None)
    except (AssertionError, AttributeError):
        pass  # TQ not initialised in this process, or no KV client.

    tq.close()

    if store_client is not None and hasattr(store_client, "close"):
        try:
            store_client.close()
            logger.info("[dataplane] Unmounted MooncakeStore segment on teardown.")
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.warning(f"[dataplane] Failed to unmount MooncakeStore segment: {e}")


@ray.remote(num_cpus=0)
class _TransferQueueOwner:
    """Process boundary for first-time TQ initialization and global cleanup."""

    def __init__(self) -> None:
        self._owns_controller = False

    def initialize(self, conf: Any, owner_token: str) -> tuple[Any, bool]:
        _prepare_mooncake_runtime(conf)
        _set_owner_token(conf, owner_token)
        tq.init(conf=conf)
        stored_conf = _get_stored_config()
        self._owns_controller = _get_owner_token(stored_conf) == owner_token
        return stored_conf, self._owns_controller

    def close(self) -> None:
        close_tq_and_unmount(is_owner=self._owns_controller)

    def detach(self) -> None:
        _close_local_tq_client()


def _stop_owner_actor(owner: Any) -> None:
    try:
        ray.kill(owner)
    except Exception as e:  # pragma: no cover - actor may already be dead
        logger.debug(f"[dataplane] TQ owner actor already stopped: {e}")


def _cleanup_failed_owner(owner: Any, owner_token: str, *, timeout: float = 10.0) -> None:
    """Stop a failed initializer and remove only controller state it owns.

    A concurrent initializer may win the global named-actor race.  In that case
    this actor only attached, so neither its cleanup RPC nor this driver is
    allowed to kill the winning controller.
    """
    try:
        ray.get(owner.close.remote(), timeout=timeout)
    except Exception as e:
        logger.warning(f"[dataplane] TQ owner cleanup RPC failed; killing owner actor: {e}")
    finally:
        _stop_owner_actor(owner)

    try:
        stored_conf = _get_stored_config(timeout=timeout)
    except ValueError:
        return
    except Exception as e:
        logger.warning(f"[dataplane] Failed initializer left an unusable TQ controller ({e}); reaping it.")
        kill_tq_controller_and_wait()
        return

    stored_token = _get_owner_token(stored_conf)
    if stored_token == owner_token:
        logger.warning("[dataplane] Failed initializer left its TQ controller behind; reaping owned state.")
        kill_tq_controller_and_wait()
    else:
        logger.info(
            "[dataplane] Failed initializer had attached to a concurrently owned "
            "TQ controller; leaving global state intact."
        )


def _start_owner(conf: Any, *, timeout: float) -> TqInitResult:
    owner = _TransferQueueOwner.remote()
    owner_token = uuid.uuid4().hex
    try:
        stored_conf, owns_controller = ray.get(owner.initialize.remote(conf, owner_token), timeout=timeout)
    except ray.exceptions.GetTimeoutError as e:
        _cleanup_failed_owner(owner, owner_token)
        raise TqInitializationTimeout(f"tq.init did not finish within {timeout:.0f}s") from e
    except Exception:
        _cleanup_failed_owner(owner, owner_token)
        raise

    if owns_controller:
        return TqInitResult(config=stored_conf, owner=owner)

    # A concurrent initializer won the named-actor race. This process is only
    # attached and must never retain an actor capable of global tq.close().
    config_mismatch = _configuration_signature(stored_conf) != _configuration_signature(conf)
    try:
        ray.get(owner.detach.remote(), timeout=10.0)
    finally:
        _stop_owner_actor(owner)
    if config_mismatch:
        raise TqConfigurationMismatch(
            "A concurrent TransferQueue initializer won with a different backend config or controller sampling "
            "contract "
            f"(requested={_backend_description(conf)}, stored={_backend_description(stored_conf)}). "
            "Detached without modifying the winning controller."
        )
    return TqInitResult(config=stored_conf, owner=None)


def close_tq_owner(owner: Any | None, *, timeout: float = 30.0) -> None:
    """Ask the owner process to close global TQ state; attached sessions no-
    op."""
    if owner is None:
        return
    close_error: Exception | None = None
    try:
        ray.get(owner.close.remote(), timeout=timeout)
    except Exception as e:
        close_error = e
    finally:
        _stop_owner_actor(owner)
    # Do not proceed to a subsequent initialization until the actor name has
    # actually left GCS.
    if _controller_exists():
        kill_tq_controller_and_wait()
    if close_error is not None:
        raise RuntimeError(f"TransferQueue owner cleanup failed: {close_error}") from close_error


def initialize_tq_with_fallback(
    conf: Any,
    *,
    mode: str,
    fallback_conf: Any | None = None,
    timeout: float = DEFAULT_TQ_INIT_TIMEOUT_SECONDS,
) -> TqInitResult:
    """Initialize TQ atomically with owner tracking and one safe auto fallback.

    ``fallback_conf`` must be the equivalent SimpleStorage configuration.  It
    is used only for ``mode='auto'`` after a real Mooncake ``tq.init`` failure.
    ``required`` always cleans up and re-raises the original error.
    """

    def _attempt(attempt_conf: Any) -> TqInitResult:
        reap_unusable_tq_controller()
        if _controller_exists():
            # Attach semantics: use the controller's actual config. Upstream
            # tq.init(conf) returns the caller-provided config even when ignored.
            stored_conf = _get_stored_config()
            if _configuration_signature(stored_conf) != _configuration_signature(attempt_conf):
                raise TqConfigurationMismatch(
                    "Refusing to attach to an existing TransferQueueController with a different backend config "
                    "or controller sampling contract "
                    f"(requested={_backend_description(attempt_conf)}, stored={_backend_description(stored_conf)}). "
                    "Only the owner may close the existing controller."
                )
            return TqInitResult(config=stored_conf, owner=None)
        return _start_owner(attempt_conf, timeout=timeout)

    try:
        return _attempt(conf)
    except Exception as primary_error:
        if isinstance(primary_error, TqConfigurationMismatch) or mode != "auto" or fallback_conf is None:
            raise

        reason = f"mooncake_init_failed:{type(primary_error).__name__}"
        logger.warning(
            f"[dataplane] Mooncake tq.init failed ({primary_error}); "
            "cleaned partial state and retrying once with SimpleStorage."
        )
        try:
            result = _attempt(fallback_conf)
        except Exception as fallback_error:
            raise RuntimeError(
                "TransferQueue SimpleStorage fallback initialization failed after "
                f"Mooncake initialization error: {primary_error}"
            ) from fallback_error
        return TqInitResult(
            config=result.config,
            owner=result.owner,
            fallback_reason=reason,
        )
