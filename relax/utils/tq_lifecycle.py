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


class TqCleanupTimeout(TimeoutError):
    """Raised when a TQ controller cannot be confirmed gone after cleanup."""


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


def attach_tq_client(conf: Any, *, requested_gdr: bool, role: str) -> Any:
    """Attach a component process and report its local experimental GDR
    state."""
    tq.init(conf=conf)
    client = tq.get_client()
    log_tq_gdr_runtime_status(requested=requested_gdr, role=role)
    return client


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
    try:
        ray.get(owner.detach.remote(), timeout=10.0)
    finally:
        _stop_owner_actor(owner)
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
            return TqInitResult(config=_get_stored_config(), owner=None)
        return _start_owner(attempt_conf, timeout=timeout)

    try:
        return _attempt(conf)
    except Exception as primary_error:
        if mode != "auto" or fallback_conf is None:
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
