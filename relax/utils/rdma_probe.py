# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""RDMA capability probe and fallback decision for MooncakeStore.

This module runs **before** ``tq.init`` to decide the job-level effective
``{backend, protocol, device}`` triple.  The probe is intentionally side-effect
free: it only reads ``/sys``/``resource``/mooncake introspection and performs a
short handshake.  The result is AND-reduced across all data-plane nodes by the
driver so that every worker converges on the *same* effective config.

Only two outcomes exist: MooncakeStore over host RDMA, or the original
SimpleStorage.  Mooncake/TCP survives as a benchmark baseline only, so there is
no intermediate transport to degrade through.

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

# Accepted ``--tq-rdma-mode`` values; ``off`` keeps the SimpleStorage path.
TQ_RDMA_MODES = frozenset({"off", "auto", "required"})

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
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True if this node can run MooncakeStore (tcp or rdma)."""
        return self.effective_protocol is not None

    def summary(self) -> str:
        """Return a multi-line human-readable report of this node's probe."""
        header = f"[probe:{self.node}] protocol={self.effective_protocol} device={self.effective_device}"
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
    fallback_reason: str  # "" if no fallback occurred

    def log_line(self) -> str:
        """Return the single-line startup log string for this effective config.

        SimpleStorage deliberately reports no protocol: naming one would imply
        the data plane went through a Mooncake transport, and Mooncake/TCP is
        not a production path.
        """
        if self.backend != "MooncakeStore":
            base = f"[dataplane] backend={self.backend}"
        else:
            base = f"[dataplane] backend={self.backend} protocol={self.protocol} device={self.device or 'auto'}"
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


def _check_port_active(device: str, port: int = 1) -> CheckResult:
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


def _check_gid_available(device: str, gid_index: int = 3, port: int = 1) -> CheckResult:
    gid_path = f"/sys/class/infiniband/{device}/ports/{port}/gids/{gid_index}"
    try:
        with open(gid_path) as f:
            raw = f.read().strip()
        ok = raw.replace(":", "") != "0" * 32 and bool(raw)
        return CheckResult(f"gid:{device}/{gid_index}", ok, raw[:24] + "..." if len(raw) > 24 else raw)
    except FileNotFoundError:
        return CheckResult(f"gid:{device}/{gid_index}", False, "gid file missing")
    except OSError as e:
        return CheckResult(f"gid:{device}/{gid_index}", False, str(e))


def _list_numeric_entries(path: str) -> list[int]:
    """Sorted numeric directory entries (port numbers / GID indices)."""
    try:
        return sorted(int(name) for name in os.listdir(path) if name.isdigit())
    except OSError:
        return []


def _find_usable_gid(device: str, port: int) -> CheckResult:
    """Return the first usable (non-zero) GID on ``device``/``port``.

    Index 3 (conventionally the RoCE v2 / IPv4-mapped entry) is preferred to
    preserve the previous behaviour, then every other advertised index is
    scanned so a host that populates a different index is not misreported as
    GID-less.
    """
    indices = _list_numeric_entries(f"/sys/class/infiniband/{device}/ports/{port}/gids")
    if not indices:
        return CheckResult(f"gid:{device}", False, f"no GID entries on port {port}")
    ordered = ([3] if 3 in indices else []) + [i for i in indices if i != 3]
    for gid_index in ordered:
        check = _check_gid_available(device, gid_index, port=port)
        if check.ok:
            return check
    return CheckResult(f"gid:{device}", False, f"no non-zero GID on port {port}")


def _select_usable_rdma_device(device: str = "") -> tuple[str, list[CheckResult]]:
    """Pick one HCA whose ACTIVE port and usable GID both pass together.

    Scans every device under ``/sys/class/infiniband`` (or only ``device``
    when explicitly requested), every port of each device, and the GID table
    of the first ACTIVE port — instead of assuming the lexicographically
    first device with port 1 / GID index 3.  A single down HCA on a
    multi-HCA host therefore no longer degrades the whole node.

    Returns ``(selected_device, checks)``.  ``selected_device`` is ``""``
    when no device qualifies; ``checks`` then summarises one failure per
    inspected device for the probe report.
    """
    base = "/sys/class/infiniband"
    if not os.path.isdir(base):
        return "", [CheckResult("port_active", False, "no infiniband dir")]
    candidates = [device] if device else sorted(os.listdir(base))
    if not candidates:
        return "", [CheckResult("port_active", False, "no devices")]

    failures: list[str] = []
    for dev in candidates:
        ports = _list_numeric_entries(f"{base}/{dev}/ports")
        if not ports:
            failures.append(f"{dev}: no ports")
            continue
        port_check: CheckResult | None = None
        active_port: int | None = None
        for port in ports:
            check = _check_port_active(dev, port)
            if check.ok:
                port_check, active_port = check, port
                break
        if port_check is None or active_port is None:
            failures.append(f"{dev}: no ACTIVE port")
            continue
        gid_check = _find_usable_gid(dev, active_port)
        if not gid_check.ok:
            failures.append(f"{dev}/port{active_port}: {gid_check.detail}")
            continue
        return dev, [port_check, gid_check]

    detail = "; ".join(failures)
    return "", [CheckResult("port_active", False, detail), CheckResult("gid", False, detail)]


def _check_memlock() -> CheckResult:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        # ``soft`` of RLIM_INFINITY is typically -1 on Linux.
        unlimited = soft in (-1, resource.RLIM_INFINITY) or soft > 2**30
        detail = f"soft={soft} hard={hard}"
        return CheckResult("memlock", unlimited, detail)
    except (ValueError, OSError) as e:
        return CheckResult("memlock", False, str(e))


def _split_host_port(address: str) -> tuple[str, int]:
    """Parse ``host:port`` and bracketed IPv6 endpoints."""
    value = address.strip()
    if value.startswith("["):
        end = value.find("]")
        if end < 0 or end + 2 > len(value) or value[end + 1] != ":":
            raise ValueError(f"invalid bracketed endpoint: {address!r}")
        return value[1:end], int(value[end + 2 :])
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port:
        raise ValueError(f"expected host:port, got {address!r}")
    return host, int(port)


def _check_master_reachable(address: str, timeout: float = 2.0) -> CheckResult:
    """Verify that this node can establish a bounded TCP connection to
    master."""
    try:
        host, port = _split_host_port(address)
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return CheckResult("master_reachable", True, address)
    except (OSError, ValueError) as e:
        return CheckResult("master_reachable", False, f"{address}: {e}")


def _check_health_check() -> CheckResult:  # pragma: no cover - retained for ad-hoc use
    """Call mooncake's native ``health_check()`` (NOT used by ``probe_node``).

    Returns 0=healthy, 1=not initialized/closed, 2=master unreachable.  Kept as
    a utility for post-init diagnostics. The pre-init path uses a bounded TCP
    reachability check instead because the global health API is process-state
    dependent before a local Mooncake client has been initialized.
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


# NOTE: ``health_check()`` is intentionally NOT probed because it depends on
# local Mooncake client initialization. External-master reachability is checked
# directly with a bounded TCP connect, then authoritatively by ``tq.init``.
def probe_node(device: str = "", master_address: str = "") -> ProbeResult:
    """Run all capability checks on the current node.

    Parameters
    ----------
    device
        Explicit RDMA device name; empty = scan every HCA and select one
        whose ACTIVE port and usable GID both pass.
    """
    node = socket.gethostname()
    checks: list[CheckResult] = []
    errors: list[str] = []

    checks.append(_check_mooncake_import())
    checks.append(_check_rdma_devices())
    checks.append(_check_memlock())

    # Mooncake is externally managed by Relax deployments. When an endpoint is
    # supplied, it must already be reachable from every data-plane node before
    # the job creates a global TransferQueue controller.
    if master_address:
        checks.append(_check_master_reachable(master_address))

    selected_device, device_checks = _select_usable_rdma_device(device)
    checks.extend(device_checks)

    # Determine effective protocol via graded degradation.
    mooncake_ok = any(c.name == "mooncake_import" and c.ok for c in checks)
    rdma_dev_ok = any(c.name == "rdma_devices" and c.ok for c in checks)
    port_ok = any(c.name.startswith("port_active") and c.ok for c in checks)
    gid_ok = any(c.name.startswith("gid") and c.ok for c in checks)
    memlock_ok = any(c.name == "memlock" and c.ok for c in checks)
    master_ok = not master_address or any(c.name == "master_reachable" and c.ok for c in checks)

    effective_protocol: str | None
    effective_device = device
    if not mooncake_ok:
        effective_protocol = None
        errors.append("mooncake not importable")
    elif not master_ok:
        effective_protocol = None
        errors.append("master unreachable")
    elif rdma_dev_ok and port_ok and gid_ok and memlock_ok:
        effective_protocol = "rdma"
        # Report the jointly validated device (ACTIVE port + usable GID).
        effective_device = selected_device or effective_device
    else:
        # Mooncake is usable but this node cannot do RDMA.  ``tcp`` is recorded
        # so the reduction can distinguish "no RDMA here" from "no Mooncake at
        # all" in its fallback reason; Mooncake/TCP is not a production
        # transport (it survives only as a benchmark baseline).
        effective_protocol = "tcp"
        if not rdma_dev_ok:
            errors.append("no RDMA device")
        elif not port_ok:
            errors.append("HCA port not ACTIVE")
        elif not gid_ok:
            errors.append("GID unavailable")
        elif not memlock_ok:
            errors.append("memlock too low for RDMA MR registration")

    return ProbeResult(
        node=node,
        checks=tuple(checks),
        effective_protocol=effective_protocol,
        effective_device=effective_device,
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
        errors=(error,),
    )


def probe_cluster_nodes(
    device: str = "",
    master_address: str = "",
    *,
    timeout: float = 60.0,
) -> list[ProbeResult]:
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

    The driver is always included because the first ``tq.init`` creates a local
    Mooncake client there even when the Ray head is CPU-only. Returns only the
    driver result when no GPU workers are discoverable (single-node/local dev).
    """
    node_ids = _alive_gpu_nodes()
    driver_result = probe_node(device, master_address)
    if not node_ids:
        logger.debug("No alive GPU nodes discovered; probing driver node only.")
        return [driver_result]

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0.001)
    def _probe_on_node(dev: str, master: str) -> ProbeResult:
        from relax.utils.rdma_probe import probe_node as _probe

        return _probe(dev, master)

    refs: list[Any] = []
    id_by_ref: dict[Any, str] = {}
    for node_id in node_ids:
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ref = _probe_on_node.options(scheduling_strategy=strategy).remote(device, master_address)
        refs.append(ref)
        id_by_ref[ref] = node_id

    ready, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout)
    results: list[ProbeResult] = [driver_result]
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
    requested_device: str,
) -> EffectiveConfig:
    """AND-reduce per-node results into a single job-level effective config.

    The only two outcomes are MooncakeStore over host RDMA and the original
    SimpleStorage: Mooncake/TCP is a benchmark baseline, not a production
    transport, so it is never selected here and there is no intermediate rung
    to degrade through.

    Parameters
    ----------
    results
        One :class:`ProbeResult` per data-plane node.
    requested_device
        ``--tq-rdma-device`` value.
    """
    if not results:
        return EffectiveConfig(
            backend="SimpleStorage",
            protocol="tcp",
            device="",
            fallback_reason="no probe results",
        )

    # AND reduction: the job can only run at the lowest common capability.
    if all(r.effective_protocol == "rdma" for r in results):
        # Device: every node must expose the explicitly requested device.
        if requested_device and any(r.effective_device != requested_device for r in results):
            return EffectiveConfig(
                backend="SimpleStorage",
                protocol="tcp",
                device="",
                fallback_reason=f"device_mismatch:{requested_device}",
            )
        return EffectiveConfig(
            backend="MooncakeStore",
            protocol="rdma",
            device=requested_device,
            fallback_reason="",
        )

    no_mooncake_nodes = [r.node for r in results if r.effective_protocol is None]
    if no_mooncake_nodes:
        master_failed_nodes = [r.node for r in results if "master unreachable" in r.errors]
        reason = (
            f"master_unreachable:{','.join(master_failed_nodes)}"
            if master_failed_nodes
            else f"mooncake_unavailable:{','.join(no_mooncake_nodes)}"
        )
    else:
        reason = f"rdma_unavailable:{','.join(r.node for r in results if r.effective_protocol != 'rdma')}"
    return EffectiveConfig(
        backend="SimpleStorage",
        protocol="tcp",
        device="",
        fallback_reason=reason,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(args: Any) -> list[str]:
    """Return a list of error messages for an invalid RDMA configuration.

    Called at startup *before* probing.  An empty list means the config is
    structurally valid (semantic/runtime validity is checked by the probe).
    ``argparse`` already constrains the mode for CLI runs; this also covers
    configs restored from a checkpoint or built programmatically.
    """
    errors: list[str] = []
    mode = getattr(args, "tq_rdma_mode", "off")

    if mode not in TQ_RDMA_MODES:
        errors.append(f"--tq-rdma-mode={mode!r} must be one of {', '.join(sorted(TQ_RDMA_MODES))}.")
    return errors
