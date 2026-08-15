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
        gdr_status = "unknown" if self.gdr else "off"
        dev = self.device or "auto"
        base = (
            f"[dataplane] backend={self.backend} protocol={self.protocol} device={dev} "
            f"gdr_requested={str(self.gdr).lower()} gdr_status={gdr_status}"
        )
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
def probe_node(device: str = "", master_address: str = "", *, probe_rdma: bool = True) -> ProbeResult:
    """Run all capability checks on the current node.

    Parameters
    ----------
    device
        Explicit RDMA device name; empty = scan every HCA and select one
        whose ACTIVE port and usable GID both pass.
    probe_rdma
        When ``False``, validate only Mooncake importability and master
        reachability, then select TCP. Used by ``--tq-rdma-mode=off`` so an
        explicitly requested Mooncake/TCP backend never depends on RDMA
        hardware.
    """
    node = socket.gethostname()
    checks: list[CheckResult] = []
    errors: list[str] = []

    checks.append(_check_mooncake_import())
    if probe_rdma:
        checks.append(_check_rdma_devices())
        checks.append(_check_memlock())

    # Mooncake is externally managed by Relax deployments. When an endpoint is
    # supplied, it must already be reachable from every data-plane node before
    # the job creates a global TransferQueue controller.
    if master_address:
        checks.append(_check_master_reachable(master_address))

    # Device-dependent checks are irrelevant when RDMA is explicitly off.
    selected_device = ""
    if probe_rdma:
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
    effective_device = device if probe_rdma else ""
    if not mooncake_ok:
        effective_protocol = None
        errors.append("mooncake not importable")
    elif not master_ok:
        effective_protocol = None
        errors.append("master unreachable")
    elif not probe_rdma:
        effective_protocol = "tcp"
    elif rdma_dev_ok and port_ok and gid_ok and memlock_ok:
        effective_protocol = "rdma"
        # Report the jointly validated device (ACTIVE port + usable GID).
        effective_device = selected_device or effective_device
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


def probe_cluster_nodes(
    device: str = "",
    master_address: str = "",
    *,
    timeout: float = 60.0,
    probe_rdma: bool = True,
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
    driver_result = (
        probe_node(device, master_address) if probe_rdma else probe_node(device, master_address, probe_rdma=False)
    )
    if not node_ids:
        logger.debug("No alive GPU nodes discovered; probing driver node only.")
        return [driver_result]

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0.001)
    def _probe_on_node(dev: str, master: str, should_probe_rdma: bool) -> ProbeResult:
        from relax.utils.rdma_probe import probe_node as _probe

        return _probe(dev, master, probe_rdma=should_probe_rdma)

    refs: list[Any] = []
    id_by_ref: dict[Any, str] = {}
    for node_id in node_ids:
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ref = _probe_on_node.options(scheduling_strategy=strategy).remote(device, master_address, probe_rdma)
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
    requested_backend: str,
    requested_device: str,
    use_gdr: bool,
    fallback_backend: str = "SimpleStorage",
    rdma_mode: str = "auto",
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
    rdma_mode
        ``off`` selects Mooncake/TCP after validating Mooncake and master
        availability. ``auto`` and ``required`` reduce the probed RDMA
        capability normally; the caller decides whether a fallback is fatal.
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

    if any_no_mooncake:
        failed_nodes = [r.node for r in results if r.effective_protocol is None]
        master_failed_nodes = [r.node for r in results if "master unreachable" in r.errors]
        reason = (
            f"master_unreachable:{','.join(master_failed_nodes)}"
            if master_failed_nodes
            else f"mooncake_unavailable:{','.join(failed_nodes)}"
        )
        return EffectiveConfig(
            backend=fallback_backend,
            protocol="tcp",
            device="",
            gdr=False,
            fallback_reason=reason,
        )

    if rdma_mode == "off":
        return EffectiveConfig(
            backend="MooncakeStore",
            protocol="tcp",
            device="",
            gdr=False,
            fallback_reason="",
        )

    if all_rdma:
        # Device: if any node lacks the requested device, fall back to tcp.
        if requested_device:
            device_ok = all(r.effective_device == requested_device for r in results)
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
            # probe_node defines GDR eligibility as RDMA transport readiness;
            # CUDA staging is deliberately decided and logged by each worker.
            gdr=use_gdr,
            fallback_reason="",
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
