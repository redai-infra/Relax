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

import ray
import transfer_queue as tq

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

CONTROLLER_NAME = "TransferQueueController"
CONTROLLER_NAMESPACE = "transfer_queue"


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
    except Exception as e:  # pragma: no cover - best-effort cleanup
        logger.warning(f"[dataplane] Failed to kill TransferQueueController: {e}")
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)
        except ValueError:
            return
        time.sleep(0.4)
    logger.warning(f"[dataplane] TransferQueueController still resolvable after {timeout}s; proceeding anyway.")


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


def close_tq_and_unmount() -> None:
    """Close TransferQueue and unmount the MooncakeStore segment.

    Order matters: ``tq.close()`` still needs the store alive for its
    ``remove_all()``, so the client handle is captured first and unmounted
    after. SimpleStorage has no ``storage_client``, so this is a no-op there.
    """
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
