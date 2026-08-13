# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for RDMA capability probe and config resolution.

These tests are CPU-only and do NOT require a real TransferQueue or RDMA
hardware.  They mock the filesystem and mooncake import to exercise every
branch of the probe, reduction, and config validation logic.
"""

from __future__ import annotations

import argparse
from unittest import mock

import relax.utils.rdma_probe as rdma_probe
from relax.utils.rdma_probe import (
    CheckResult,
    EffectiveConfig,
    ProbeResult,
    _check_master_reachable,
    _degenerate_result,
    _select_dataplane_node_ids,
    _split_host_port,
    probe_cluster_nodes,
    probe_node,
    reduce_results,
    validate_config,
)
from relax.utils.tq_config import (
    build_mooncake_config,
    build_simple_storage_config,
    estimate_payload_bytes,
    validate_mooncake_runtime_contract,
    validate_segment_capacity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe(
    protocol: str | None = "rdma",
    device: str = "rdma0",
    gdr: bool = False,
    node: str = "node-A",
) -> ProbeResult:
    return ProbeResult(
        node=node,
        checks=(CheckResult("mooncake_import", True),),
        effective_protocol=protocol,
        effective_device=device,
        gdr_eligible=gdr,
    )


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        tq_storage_backend="mooncake",
        tq_rdma_mode="auto",
        tq_rdma_device="",
        tq_use_gdr=False,
        num_data_storage_units=1,
        max_staleness=0,
        n_samples_per_prompt=1,
        rollout_batch_size=32,
        multimodal_keys=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """validate_config: structural flag-combination checks before any probe."""

    def test_simple_backend_with_rdma_mode_rejected(self):
        args = _make_args(tq_storage_backend="simple", tq_rdma_mode="auto")
        errors = validate_config(args)
        assert len(errors) == 1
        assert "simple" in errors[0]

    def test_simple_backend_with_gdr_rejected(self):
        args = _make_args(tq_storage_backend="simple", tq_rdma_mode="off", tq_use_gdr=True)
        errors = validate_config(args)
        assert any("--tq-use-gdr" in e for e in errors)

    def test_gdr_without_rdma_rejected(self):
        args = _make_args(tq_storage_backend="mooncake", tq_rdma_mode="off", tq_use_gdr=True)
        errors = validate_config(args)
        assert any("rdma-mode=off" in e for e in errors)

    def test_valid_simple_off(self):
        args = _make_args(tq_storage_backend="simple", tq_rdma_mode="off")
        assert validate_config(args) == []

    def test_valid_mooncake_auto(self):
        args = _make_args(tq_storage_backend="mooncake", tq_rdma_mode="auto")
        assert validate_config(args) == []

    def test_valid_mooncake_required_gdr(self):
        args = _make_args(tq_storage_backend="mooncake", tq_rdma_mode="required", tq_use_gdr=True)
        assert validate_config(args) == []


# ---------------------------------------------------------------------------
# reduce_results
# ---------------------------------------------------------------------------


class TestReduceResults:
    """reduce_results: per-node ProbeResult -> job-level EffectiveConfig (AND reduction)."""

    def test_simple_backend_short_circuits(self):
        eff = reduce_results(
            [_make_probe()],
            requested_backend="simple",
            requested_device="rdma0",
            use_gdr=False,
        )
        assert eff.backend == "SimpleStorage"
        assert eff.protocol == "tcp"
        assert eff.gdr is False

    def test_all_nodes_rdma(self):
        eff = reduce_results(
            [_make_probe(protocol="rdma"), _make_probe(protocol="rdma", node="node-B")],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=False,
        )
        assert eff.backend == "MooncakeStore"
        assert eff.protocol == "rdma"
        assert eff.fallback_reason == ""

    def test_one_node_no_mooncake_falls_back(self):
        eff = reduce_results(
            [_make_probe(protocol="rdma"), _make_probe(protocol=None, node="node-B")],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=False,
        )
        assert eff.backend == "SimpleStorage"
        assert "node-B" in eff.fallback_reason

    def test_one_node_no_rdma_degrades_to_tcp(self):
        eff = reduce_results(
            [_make_probe(protocol="rdma"), _make_probe(protocol="tcp", node="node-B")],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=False,
        )
        assert eff.backend == "MooncakeStore"
        assert eff.protocol == "tcp"
        assert "node-B" in eff.fallback_reason

    def test_gdr_eligible_only_when_all_nodes(self):
        eff = reduce_results(
            [_make_probe(gdr=True), _make_probe(gdr=False, node="node-B")],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=True,
        )
        assert eff.gdr is False
        assert "gdr_cuda_not_initialized" in eff.fallback_reason

    def test_gdr_eligible_all_nodes(self):
        eff = reduce_results(
            [_make_probe(gdr=True), _make_probe(gdr=True, node="node-B")],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=True,
        )
        assert eff.gdr is True

    def test_empty_results_falls_back(self):
        eff = reduce_results(
            [],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=False,
        )
        assert eff.backend == "SimpleStorage"


# ---------------------------------------------------------------------------
# probe_node (mocked filesystem)
# ---------------------------------------------------------------------------


class TestProbeNode:
    """probe_node: per-node capability checks (mocked /sys and mooncake)."""

    def test_no_infiniband_dir_gives_tcp(self):
        """When /sys/class/infiniband doesn't exist, protocol should be tcp (if
        mooncake imports) or None (if not)."""
        with mock.patch("os.path.isdir", return_value=False):
            with mock.patch("relax.utils.rdma_probe._check_mooncake_import") as mi:
                mi.return_value = CheckResult("mooncake_import", True, "version=0.3.10")
                result = probe_node("")
        # mooncake importable but no RDMA device → tcp
        assert result.effective_protocol == "tcp"

    def test_active_rdma_device_gives_rdma(self):
        """When all checks pass, protocol should be rdma."""

        def fake_isdir(path):
            return "infiniband" in path

        with (
            mock.patch("os.path.isdir", side_effect=fake_isdir),
            mock.patch("os.listdir", return_value=["rdma0"]),
            mock.patch("builtins.open", mock.mock_open(read_data="4: ACTIVE")),
            mock.patch("relax.utils.rdma_probe._check_mooncake_import") as mi,
            mock.patch("relax.utils.rdma_probe._check_health_check") as hc,
            mock.patch("relax.utils.rdma_probe.resource.getrlimit", return_value=(-1, -1)),
        ):
            mi.return_value = CheckResult("mooncake_import", True, "ok")
            hc.return_value = CheckResult("health_check", True, "return_code=0")
            result = probe_node("")
        assert result.effective_protocol == "rdma"
        assert result.ok

    def test_unreachable_external_master_disables_mooncake(self, monkeypatch):
        monkeypatch.setattr(rdma_probe, "_check_mooncake_import", lambda: CheckResult("mooncake_import", True))
        monkeypatch.setattr(
            rdma_probe,
            "_check_master_reachable",
            lambda address: CheckResult("master_reachable", False, address),
        )
        result = probe_node("", "master.invalid:50051")
        assert result.effective_protocol is None
        assert "master unreachable" in result.errors

    def test_master_endpoint_parser(self):
        assert _split_host_port("master.example:50051") == ("master.example", 50051)
        assert _split_host_port("[2001:db8::1]:50051") == ("2001:db8::1", 50051)

    def test_master_reachability_is_bounded_failure(self):
        result = _check_master_reachable("127.0.0.1:1", timeout=0.01)
        assert result.ok is False


# ---------------------------------------------------------------------------
# probe_cluster_nodes (multi-node fan-out) + helpers
# ---------------------------------------------------------------------------


class TestProbeClusterNodes:
    """probe_cluster_nodes: multi-node fan-out helpers + degenerate-result handling."""

    def test_select_nodes_filters_dead_and_cpu_only(self):
        """Only alive nodes advertising GPU resources are data-plane nodes."""
        nodes = [
            {"NodeID": "n0", "Alive": True, "Resources": {"GPU": 8.0}},
            {"NodeID": "n1", "Alive": True, "Resources": {"GPU": 0}},
            {"NodeID": "n2", "Alive": False, "Resources": {"GPU": 8.0}},
            {"NodeID": "n3", "Alive": True, "Resources": {}},
        ]
        assert _select_dataplane_node_ids(nodes) == ["n0"]

    def test_degenerate_result_is_no_mooncake(self):
        """A failed/timed-out node reports effective_protocol=None so the AND-
        reducer degrades instead of silently dropping the node."""
        r = _degenerate_result("node-X", "probe_timeout:60s")
        assert r.node == "node-X"
        assert r.effective_protocol is None
        assert r.ok is False
        assert "probe_timeout" in r.errors[0]

    def test_cluster_falls_back_to_local_when_no_gpu_nodes(self, monkeypatch):
        """No alive GPU workers (single-node / local dev) -> probe driver only,
        never touching Ray remote scheduling."""
        monkeypatch.setattr(rdma_probe, "_alive_gpu_nodes", lambda: [])
        local = _make_probe(protocol="rdma", node="local-driver")
        monkeypatch.setattr(rdma_probe, "probe_node", lambda dev, master="": local)
        results = probe_cluster_nodes("")
        assert len(results) == 1
        assert results[0] is local

    def test_reduce_treats_degenerate_as_no_mooncake(self):
        """A probe failure on one node forces job-level fallback (not a silent
        drop that would over-report RDMA readiness)."""
        results = [
            _make_probe(protocol="rdma", node="n0"),
            _degenerate_result("n1", "probe_task_failed:boom"),
        ]
        eff = reduce_results(
            results,
            requested_backend="mooncake",
            requested_device="",
            use_gdr=False,
        )
        assert eff.backend == "SimpleStorage"
        assert "n1" in eff.fallback_reason

    def test_reduce_reports_master_unreachable_distinctly(self):
        unavailable = ProbeResult(
            node="n1",
            checks=(CheckResult("master_reachable", False),),
            effective_protocol=None,
            effective_device="",
            gdr_eligible=False,
            errors=("master unreachable",),
        )
        eff = reduce_results(
            [_make_probe(protocol="rdma", node="n0"), unavailable],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=False,
        )
        assert eff.backend == "SimpleStorage"
        assert eff.fallback_reason == "master_unreachable:n1"


# ---------------------------------------------------------------------------
# tq_config builders
# ---------------------------------------------------------------------------


class TestTqConfigBuilder:
    """tq_config builders: SimpleStorage/MooncakeStore dict + capacity
    validation."""

    def test_simple_storage_config(self):
        cfg = build_simple_storage_config(total_storage_size=1000, num_data_storage_units=2)
        assert cfg == {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {"total_storage_size": 1000, "num_data_storage_units": 2},
        }

    def test_storage_backend_key_selects_the_manager(self):
        """``tq.init`` reads ``backend.storage_backend``; omitting it silently
        keeps SimpleStorage."""
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="rdma0", gdr=False, fallback_reason="")
        assert build_mooncake_config(eff)["storage_backend"] == "MooncakeStore"
        assert (
            build_simple_storage_config(total_storage_size=1, num_data_storage_units=1)["storage_backend"]
            == "SimpleStorage"
        )

    def test_mooncake_config_has_hard_pin_true(self):
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="rdma0", gdr=False, fallback_reason="")
        cfg = build_mooncake_config(eff)
        mc = cfg["MooncakeStore"]
        assert mc["protocol"] == "rdma"
        assert mc["device_name"] == "rdma0"
        assert mc["hard_pin"] is True  # no silent eviction
        assert mc["auto_init"] is False  # external master
        assert mc["use_gdr"] is False

    def test_mooncake_config_gdr_propagated(self):
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="", gdr=True, fallback_reason="")
        cfg = build_mooncake_config(eff)
        assert cfg["MooncakeStore"]["use_gdr"] is True

    def test_installed_tq_satisfies_loss_prevention_contract(self):
        validate_mooncake_runtime_contract()

    def test_segment_capacity_text_only_passes(self):
        args = _make_args(multimodal_keys=None)
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="", gdr=False, fallback_reason="")
        assert validate_segment_capacity(args, eff) is None

    def test_segment_capacity_multimodal_large_batch_fails(self):
        args = _make_args(
            multimodal_keys=["pixel_values"], rollout_batch_size=256, n_samples_per_prompt=8, max_staleness=1
        )
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="", gdr=False, fallback_reason="")
        err = validate_segment_capacity(args, eff)
        assert err is not None
        assert "insufficient" in err.lower()

    def test_estimate_payload_text_only_is_zero(self):
        args = _make_args(multimodal_keys=None)
        assert estimate_payload_bytes(args) == 0
