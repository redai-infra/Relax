# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""RDMA capability probe and graded-degradation state machine for
MooncakeStore.

This module runs **before** ``tq.init`` to decide the job-level effective
``{backend, protocol, device}`` triple.  The probe is intentionally side-effect
free: it only reads ``/sys``/``resource``/mooncake introspection and performs a
short handshake.  The result is AND-reduced across all data-plane nodes by the
driver so that every worker converges on the *same* effective config.

Key constraint (F10 in the RFC): probing must happen *before* ``tq.init``.
If the named actor ``TransferQueueController`` is created before the probe
succeeds, a subsequent ``tq.init`` retry via ``_init_from_existing`` will spin
in an unbounded ``while conf is None`` loop (``interface.py:109-118``) and hang
the job with no error message.
"""

from __future__ import annotations

import os
import resource
import socket
from dataclasses import dataclass
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single capability check."""

    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class ProbeResult:
    """Aggregated RDMA capability report for a single node.

    ``effective_protocol`` is the highest transport this node can use after
    graded degradation:

    * ``"rdma"``  – RDMA device ACTIVE + GID available + mooncake importable.
    * ``"tcp"``   – mooncake importable but no usable RDMA device.
    * ``None``    – mooncake not importable at all; must fall back to
      SimpleStorage.
    """

    node: str
    checks: tuple[CheckResult, ...]
    effective_protocol: str | None  # "rdma" | "tcp" | None
    effective_device: str
    gdr_eligible: bool
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True if this node can run MooncakeStore (tcp or rdma)."""
        return self.effective_protocol is not None

    def summary(self) -> str:
        """Return a multi-line human-readable report of this node's probe."""
        header = (
            f"[probe:{self.node}] protocol={self.effective_protocol} "
            f"device={self.effective_device} gdr={self.gdr_eligible}"
        )
        lines = [header]
        for c in self.checks:
            tag = "ok" if c.ok else "FAIL"
            lines.append(f"  [{tag}] {c.name}: {c.detail}" if c.detail else f"  [{tag}] {c.name}")
        return "\n".join(lines)


@dataclass(frozen=True)
class EffectiveConfig:
    """Job-level unique effective config after AND-reduction across nodes."""

    backend: str  # "MooncakeStore" or "SimpleStorage"
    protocol: str  # "rdma" or "tcp"
    device: str
    gdr: bool
    fallback_reason: str  # "" if no fallback occurred

    def log_line(self) -> str:
        """Return the single-line startup log string for this effective
        config."""
        gdr_str = "on" if self.gdr else "off"
        dev = self.device or "auto"
        base = f"[dataplane] backend={self.backend} protocol={self.protocol} device={dev} gdr={gdr_str}"
        if self.fallback_reason:
            return f"{base} fallback={self.fallback_reason}"
        return base


# ---------------------------------------------------------------------------
# Individual checks (pure read, no side effects beyond mooncake import)
# ---------------------------------------------------------------------------


def _check_mooncake_import() -> CheckResult:
    try:
        import mooncake  # noqa: F401  (import-time only)

        ver = getattr(mooncake, "__version__", "unknown")
        return CheckResult("mooncake_import", True, f"version={ver}")
    except Exception as e:  # pragma: no cover - environment dependent
        return CheckResult("mooncake_import", False, str(e))


def _check_rdma_devices() -> CheckResult:
    base = "/sys/class/infiniband"
    if not os.path.isdir(base):
        return CheckResult("rdma_devices", False, "no /sys/class/infiniband")
    devs = sorted(os.listdir(base))
    if not devs:
        return CheckResult("rdma_devices", False, "empty /sys/class/infiniband")
    return CheckResult("rdma_devices", True, ",".join(devs))


def _check_port_active(device: str = "", port: int = 1) -> CheckResult:
    if device:
        state_path = f"/sys/class/infiniband/{device}/ports/{port}/state"
        try:
            with open(state_path) as f:
                state = f.read().strip()
            ok = "ACTIVE" in state
            return CheckResult(f"port_active:{device}/{port}", ok, state)
        except FileNotFoundError:
            return CheckResult(f"port_active:{device}/{port}", False, "state file missing")
        except OSError as e:
            return CheckResult(f"port_active:{device}/{port}", False, str(e))
    # No device specified: check the first available one.
    base = "/sys/class/infiniband"
    if not os.path.isdir(base):
        return CheckResult("port_active", False, "no infiniband dir")
    for dev in sorted(os.listdir(base)):
        return _check_port_active(dev, port)
    return CheckResult("port_active", False, "no devices")


def _check_gid_available(device: str = "", gid_index: int = 3) -> CheckResult:
    if device:
        gid_path = f"/sys/class/infiniband/{device}/ports/1/gids/{gid_index}"
        try:
            with open(gid_path) as f:
                raw = f.read().strip()
            ok = raw.replace(":", "") != "0" * 32 and bool(raw)
            return CheckResult(f"gid:{device}/{gid_index}", ok, raw[:24] + "..." if len(raw) > 24 else raw)
        except FileNotFoundError:
            return CheckResult(f"gid:{device}/{gid_index}", False, "gid file missing")
        except OSError as e:
            return CheckResult(f"gid:{device}/{gid_index}", False, str(e))
    base = "/sys/class/infiniband"
    if not os.path.isdir(base):
        return CheckResult(f"gid:{gid_index}", False, "no infiniband dir")
    for dev in sorted(os.listdir(base)):
        return _check_gid_available(dev, gid_index)
    return CheckResult(f"gid:{gid_index}", False, "no devices")


def _check_memlock() -> CheckResult:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        # ``soft`` of RLIM_INFINITY is typically -1 on Linux.
        unlimited = soft in (-1, resource.RLIM_INFINITY) or soft > 2**30
        detail = f"soft={soft} hard={hard}"
        return CheckResult("memlock", unlimited, detail)
    except (ValueError, OSError) as e:
        return CheckResult("memlock", False, str(e))


def _check_health_check() -> CheckResult:  # pragma: no cover - retained for ad-hoc use
    """Call mooncake's native ``health_check()`` (NOT used by ``probe_node``).

    Returns 0=healthy, 1=not initialized/closed, 2=master unreachable.  Kept as
    a utility for post-init diagnostics; intentionally excluded from the pre-
    init probe because the master is not running at probe time.
    """
    try:
        import mooncake

        hc = getattr(mooncake, "health_check", None)
        if hc is None:
            # Some builds expose it on the store module instead.
            from mooncake import store  # type: ignore

            hc = getattr(store, "health_check", None)
        if hc is None:
            return CheckResult("health_check", False, "health_check() not found in mooncake")
        code = int(hc())
        ok = code == 0
        return CheckResult("health_check", ok, f"return_code={code}")
    except Exception as e:
        return CheckResult("health_check", False, str(e))


# ---------------------------------------------------------------------------
# Per-node probe
# ---------------------------------------------------------------------------

# NOTE: ``health_check()`` is intentionally NOT probed -- it queries the
# mooncake master, which is not yet running at probe time (auto_init=False),
# so it would always report failure and force a spurious SimpleStorage
# fallback.  Master reachability is authoritatively checked by ``tq.init``.
_CHECK_FUNCS_NO_DEVICE = [
    _check_mooncake_import,
    _check_rdma_devices,
    _check_memlock,
]


def probe_node(device: str = "") -> ProbeResult:
    """Run all capability checks on the current node.

    Parameters
    ----------
    device
        Explicit RDMA device name; empty = auto-detect first available.
    """
    node = socket.gethostname()
    checks: list[CheckResult] = []
    errors: list[str] = []

    for fn in _CHECK_FUNCS_NO_DEVICE:
        checks.append(fn())

    # Device-dependent checks.
    checks.append(_check_port_active(device))
    checks.append(_check_gid_available(device))

    # Determine effective protocol via graded degradation.
    mooncake_ok = any(c.name == "mooncake_import" and c.ok for c in checks)
    rdma_dev_ok = any(c.name == "rdma_devices" and c.ok for c in checks)
    port_ok = any(c.name.startswith("port_active") and c.ok for c in checks)
    gid_ok = any(c.name.startswith("gid") and c.ok for c in checks)
    memlock_ok = any(c.name == "memlock" and c.ok for c in checks)

    effective_protocol: str | None
    effective_device = device
    if not mooncake_ok:
        effective_protocol = None
        errors.append("mooncake not importable")
    elif rdma_dev_ok and port_ok and gid_ok and memlock_ok:
        effective_protocol = "rdma"
        if not effective_device:
            # Pick first available device for the summary.
            base = "/sys/class/infiniband"
            if os.path.isdir(base):
                effective_device = sorted(os.listdir(base))[0]
    else:
        # mooncake usable but RDMA incomplete -> degrade to TCP (still MooncakeStore).
        effective_protocol = "tcp"
        if not rdma_dev_ok:
            errors.append("no RDMA device")
        elif not port_ok:
            errors.append("HCA port not ACTIVE")
        elif not gid_ok:
            errors.append("GID unavailable")
        elif not memlock_ok:
            errors.append("memlock too low for RDMA MR registration")

    # GDR eligibility == RDMA transport available.  The actual CUDA-context
    # check (mooncake_client.py:87) runs in the *client* worker process at
    # runtime, NOT in this probe task -- a separate Ray task always reports
    # torch.cuda.is_initialized() == False, so probing it here would make GDR
    # permanently unreachable.  We assert transport capability only; the
    # runtime client performs the CUDA check and warns/falls back if needed.
    gdr_eligible = effective_protocol == "rdma"

    return ProbeResult(
        node=node,
        checks=tuple(checks),
        effective_protocol=effective_protocol,
        effective_device=effective_device,
        gdr_eligible=gdr_eligible,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Multi-node fan-out (driver -> every alive GPU node)
# ---------------------------------------------------------------------------


def _select_dataplane_node_ids(nodes: list[dict]) -> list[str]:
    """Return node IDs of alive nodes that advertise GPU resources.

    The TransferQueue data plane runs on Actor + Rollout worker nodes, which
    always advertise GPU resources.  Head / CPU-only nodes are excluded so a
    non-data-plane node cannot force a spurious RDMA degradation.
    """
    ids: list[str] = []
    for n in nodes:
        if not n.get("Alive"):
            continue
        resources = n.get("Resources") or {}
        if resources.get("GPU", 0) >= 1:
            ids.append(n["NodeID"])
    return ids


def _alive_gpu_nodes() -> list[str]:
    """Discover alive GPU node IDs from the current Ray cluster.

    Thin seam around ``ray.nodes()``; kept separate so
    :func:`probe_cluster_nodes` and its tests can stub discovery without
    spinning up Ray.
    """
    import ray

    return _select_dataplane_node_ids(ray.nodes())


def _degenerate_result(node: str, error: str) -> ProbeResult:
    """Build a :class:`ProbeResult` for a node whose probe failed or timed out.

    ``effective_protocol=None`` makes :func:`reduce_results` treat the node as
    mooncake-unavailable (degrade toward TCP / SimpleStorage) instead of
    silently dropping it, which would over-report cluster capability.
    """
    return ProbeResult(
        node=node,
        checks=tuple(),
        effective_protocol=None,
        effective_device="",
        gdr_eligible=False,
        errors=(error,),
    )


def probe_cluster_nodes(device: str = "", *, timeout: float = 60.0) -> list[ProbeResult]:
    """Probe every alive GPU-bearing node and return one result per node.

    The driver fans the probe out as a short-lived Ray remote task pinned to
    each node via ``NodeAffinitySchedulingStrategy(soft=False)`` so that
    ``probe_node`` reads *that node's* own ``/sys`` / mooncake state.  The
    caller then AND-reduces the returned list via :func:`reduce_results`,
    producing a single job-level effective config that every worker converges
    on — satisfying the requirement that the driver decide once for the whole
    job rather than just probing its own node.

    A node whose probe task raises or exceeds ``timeout`` seconds is recorded
    as a degenerate ``effective_protocol=None`` result so the reducer degrades
    the whole job rather than silently omitting the node.

    Returns ``[probe_node(device)]`` when no GPU workers are discoverable
    (single-node / local dev), preserving backward-compatible behavior.
    """
    node_ids = _alive_gpu_nodes()
    if not node_ids:
        logger.debug("No alive GPU nodes discovered; probing driver node only.")
        return [probe_node(device)]

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0.001)
    def _probe_on_node(dev: str) -> ProbeResult:
        from relax.utils.rdma_probe import probe_node as _probe

        return _probe(dev)

    refs: list[Any] = []
    id_by_ref: dict[Any, str] = {}
    for node_id in node_ids:
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ref = _probe_on_node.options(scheduling_strategy=strategy).remote(device)
        refs.append(ref)
        id_by_ref[ref] = node_id

    ready, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout)
    results: list[ProbeResult] = []
    for ref in ready:
        node_id = id_by_ref[ref]
        try:
            results.append(ray.get(ref))
        except Exception as e:  # pragma: no cover - depends on remote task failure
            logger.warning(f"[probe] node {node_id} probe task failed: {e}")
            results.append(_degenerate_result(node_id, f"probe_task_failed:{e}"))
    for ref in pending:
        node_id = id_by_ref[ref]
        ray.cancel(ref, force=True)
        logger.warning(f"[probe] node {node_id} probe timed out after {timeout}s")
        results.append(_degenerate_result(node_id, f"probe_timeout:{timeout}s"))

    return results


# ---------------------------------------------------------------------------
# Job-level AND reduction
# ---------------------------------------------------------------------------


def reduce_results(
    results: list[ProbeResult],
    *,
    requested_backend: str,
    requested_device: str,
    use_gdr: bool,
    fallback_backend: str = "SimpleStorage",
) -> EffectiveConfig:
    """AND-reduce per-node results into a single job-level effective config.

    Parameters
    ----------
    results
        One :class:`ProbeResult` per data-plane node.
    requested_backend
        ``--tq-storage-backend`` value (``"simple"`` or ``"mooncake"``).
    requested_device
        ``--tq-rdma-device`` value.
    use_gdr
        ``--tq-use-gdr`` value.
    fallback_backend
        Backend to degrade to when probe fails in auto mode.
    """
    # SimpleStorage short-circuits: no probing needed.
    if requested_backend == "simple":
        return EffectiveConfig(
            backend="SimpleStorage",
            protocol="tcp",
            device="",
            gdr=False,
            fallback_reason="",
        )

    if not results:
        return EffectiveConfig(
            backend="SimpleStorage",
            protocol="tcp",
            device="",
            gdr=False,
            fallback_reason="no probe results",
        )

    # AND reduction: the job can only run at the lowest common capability.
    any_no_mooncake = any(r.effective_protocol is None for r in results)
    all_rdma = all(r.effective_protocol == "rdma" for r in results)
    all_gdr = all(r.gdr_eligible for r in results)

    if any_no_mooncake:
        failed_nodes = [r.node for r in results if r.effective_protocol is None]
        return EffectiveConfig(
            backend=fallback_backend,
            protocol="tcp",
            device="",
            gdr=False,
            fallback_reason=f"mooncake_unavailable:{','.join(failed_nodes)}",
        )

    if all_rdma:
        # Device: if any node lacks the requested device, fall back to tcp.
        if requested_device:
            device_ok = all(r.effective_device == requested_device or not r.effective_device for r in results)
            if not device_ok:
                return EffectiveConfig(
                    backend="MooncakeStore",
                    protocol="tcp",
                    device=requested_device,
                    gdr=False,
                    fallback_reason=f"device_mismatch:{requested_device}",
                )
        return EffectiveConfig(
            backend="MooncakeStore",
            protocol="rdma",
            device=requested_device,
            gdr=use_gdr and all_gdr,
            fallback_reason="" if (not use_gdr or all_gdr) else "gdr_cuda_not_initialized",
        )

    # Some nodes can't do RDMA → degrade to TCP (still MooncakeStore).
    rdma_failed = [r.node for r in results if r.effective_protocol != "rdma"]
    return EffectiveConfig(
        backend="MooncakeStore",
        protocol="tcp",
        device=requested_device,
        gdr=False,
        fallback_reason=f"rdma_unavailable:{','.join(rdma_failed)}",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(args: Any) -> list[str]:
    """Return a list of error messages for invalid flag combinations.

    Called at startup *before* probing.  An empty list means the config is
    structurally valid (semantic/runtime validity is checked by the probe).
    """
    errors: list[str] = []
    backend = getattr(args, "tq_storage_backend", "simple")
    mode = getattr(args, "tq_rdma_mode", "off")
    use_gdr = getattr(args, "tq_use_gdr", False)

    if backend == "simple" and mode != "off":
        errors.append(
            f"--tq-rdma-mode={mode} is meaningless with --tq-storage-backend=simple "
            "(RDMA only applies to MooncakeStore). Set --tq-rdma-mode=off or "
            "--tq-storage-backend=mooncake."
        )
    if backend == "simple" and use_gdr:
        errors.append("--tq-use-gdr requires --tq-storage-backend=mooncake.")
    if use_gdr and mode == "off":
        errors.append(
            "--tq-use-gdr is set but --tq-rdma-mode=off; GDR requires RDMA transport. "
            "Set --tq-rdma-mode=auto or required."
        )
    return errors
