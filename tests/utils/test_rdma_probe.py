# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for RDMA capability probe and config resolution.

These tests are CPU-only and do NOT require a real TransferQueue or RDMA
hardware.  They mock the filesystem and mooncake import to exercise every
branch of the probe, reduction, and config validation logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from unittest import mock

import pytest

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
    resolve_mooncake_master_address,
    resolve_tq_capacity_batch_size,
    validate_mooncake_runtime_contract,
    validate_segment_capacity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_real_tq_storage() -> bool:
    try:
        return importlib.util.find_spec("transfer_queue.storage.clients.mooncake_client") is not None
    except (ImportError, TypeError, ValueError):
        return False


_REAL_TQ_STORAGE = _has_real_tq_storage()


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
        seq_length=8192,
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

    def test_off_mode_keeps_mooncake_and_selects_tcp(self):
        eff = reduce_results(
            [_make_probe(protocol="rdma"), _make_probe(protocol="tcp", node="node-B")],
            requested_backend="mooncake",
            requested_device="rdma0",
            use_gdr=False,
            rdma_mode="off",
        )
        assert (eff.backend, eff.protocol, eff.device) == ("MooncakeStore", "tcp", "")
        assert eff.fallback_reason == ""

    def test_off_mode_reports_mooncake_unavailable(self):
        eff = reduce_results(
            [_make_probe(protocol=None)],
            requested_backend="mooncake",
            requested_device="",
            use_gdr=False,
            rdma_mode="off",
        )
        assert eff.backend == "SimpleStorage"
        assert "mooncake_unavailable" in eff.fallback_reason


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

        def fake_listdir(path):
            if path.endswith("/ports"):
                return ["1"]
            if path.endswith("/gids"):
                return ["3"]
            return ["rdma0"]

        with (
            mock.patch("os.path.isdir", side_effect=fake_isdir),
            mock.patch("os.listdir", side_effect=fake_listdir),
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
        assert result.effective_device == "rdma0"

    @staticmethod
    def _multi_hca_open(active_device: str):
        """Path-aware ``open`` mock: only ``active_device`` has an ACTIVE port
        and a non-zero GID at index 3."""

        def fake_open(path, *args, **kwargs):
            path = str(path)
            if path.endswith("/state"):
                state = "4: ACTIVE" if f"/{active_device}/" in path else "1: DOWN"
                return mock.mock_open(read_data=state)(path)
            if path.endswith("/gids/3"):
                return mock.mock_open(read_data="0000:0000:0000:0000:0000:ffff:0a00:0001")(path)
            if "/gids/" in path:
                return mock.mock_open(read_data="0000:0000:0000:0000:0000:0000:0000:0000")(path)
            raise FileNotFoundError(path)

        return fake_open

    def test_multi_hca_skips_down_first_device(self):
        """A down lexicographically-first HCA must not degrade the node when a
        later device has an ACTIVE port and a usable GID (review: probe every
        usable HCA before degrading RDMA)."""

        def fake_isdir(path):
            return "infiniband" in path

        def fake_listdir(path):
            if path.endswith("/ports"):
                return ["1"]
            if path.endswith("/gids"):
                return ["0", "3"]
            return ["mlx5_0", "mlx5_1"]

        with (
            mock.patch("os.path.isdir", side_effect=fake_isdir),
            mock.patch("os.listdir", side_effect=fake_listdir),
            mock.patch("builtins.open", side_effect=self._multi_hca_open("mlx5_1")),
            mock.patch("relax.utils.rdma_probe._check_mooncake_import") as mi,
            mock.patch("relax.utils.rdma_probe.resource.getrlimit", return_value=(-1, -1)),
        ):
            mi.return_value = CheckResult("mooncake_import", True, "ok")
            result = probe_node("")
        assert result.effective_protocol == "rdma"
        assert result.effective_device == "mlx5_1"

    def test_multi_hca_all_down_degrades_to_tcp(self):
        """When no HCA has an ACTIVE port, the node degrades to
        Mooncake/TCP."""

        def fake_isdir(path):
            return "infiniband" in path

        def fake_listdir(path):
            if path.endswith("/ports"):
                return ["1"]
            if path.endswith("/gids"):
                return ["3"]
            return ["mlx5_0", "mlx5_1"]

        with (
            mock.patch("os.path.isdir", side_effect=fake_isdir),
            mock.patch("os.listdir", side_effect=fake_listdir),
            mock.patch("builtins.open", side_effect=self._multi_hca_open("none")),
            mock.patch("relax.utils.rdma_probe._check_mooncake_import") as mi,
            mock.patch("relax.utils.rdma_probe.resource.getrlimit", return_value=(-1, -1)),
        ):
            mi.return_value = CheckResult("mooncake_import", True, "ok")
            result = probe_node("")
        assert result.effective_protocol == "tcp"
        assert "HCA port not ACTIVE" in result.errors

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

    def test_off_mode_probe_does_not_touch_rdma_hardware(self, monkeypatch):
        monkeypatch.setattr(rdma_probe, "_check_mooncake_import", lambda: CheckResult("mooncake_import", True))
        monkeypatch.setattr(
            rdma_probe,
            "_check_master_reachable",
            lambda address: CheckResult("master_reachable", True, address),
        )
        monkeypatch.setattr(
            rdma_probe,
            "_check_rdma_devices",
            lambda: (_ for _ in ()).throw(AssertionError("RDMA probe must not run")),
        )
        result = probe_node("", "master.example:50051", probe_rdma=False)
        assert result.effective_protocol == "tcp"
        assert {check.name for check in result.checks} == {"mooncake_import", "master_reachable"}


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

    def test_simple_storage_config_allows_unlimited_capacity(self):
        cfg = build_simple_storage_config(total_storage_size=None, num_data_storage_units=2)
        assert cfg["SimpleStorage"]["total_storage_size"] is None

    def test_storage_backend_key_selects_the_manager(self):
        """``tq.init`` reads ``backend.storage_backend``; omitting it silently
        keeps SimpleStorage."""
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="rdma0", gdr=False, fallback_reason="")
        assert build_mooncake_config(eff, master_address="master.example:50051")["storage_backend"] == "MooncakeStore"
        assert (
            build_simple_storage_config(total_storage_size=1, num_data_storage_units=1)["storage_backend"]
            == "SimpleStorage"
        )

    def test_mooncake_config_has_hard_pin_true(self):
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="rdma0", gdr=False, fallback_reason="")
        cfg = build_mooncake_config(eff, master_address="master.example:50051")
        mc = cfg["MooncakeStore"]
        assert mc["protocol"] == "rdma"
        assert mc["device_name"] == "rdma0"
        assert mc["hard_pin"] is True  # no silent eviction
        assert mc["auto_init"] is False  # external master
        assert mc["use_gdr"] is False

    def test_mooncake_config_gdr_propagated(self):
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="", gdr=True, fallback_reason="")
        cfg = build_mooncake_config(eff, master_address="master.example:50051")
        assert cfg["MooncakeStore"]["use_gdr"] is True

    def test_master_address_is_required(self, monkeypatch):
        # A loopback default would point every node at itself in multi-node
        # runs; missing deployment configuration must be rejected instead.
        monkeypatch.delenv("MC_MASTER_ADDRESS", raising=False)
        with pytest.raises(RuntimeError, match="MC_MASTER_ADDRESS"):
            resolve_mooncake_master_address()
        eff = EffectiveConfig(backend="MooncakeStore", protocol="tcp", device="", gdr=False, fallback_reason="")
        with pytest.raises(RuntimeError, match="MC_MASTER_ADDRESS"):
            build_mooncake_config(eff)

    def test_master_address_from_env(self, monkeypatch):
        monkeypatch.setenv("MC_MASTER_ADDRESS", "master.example:50051")
        assert resolve_mooncake_master_address() == "master.example:50051"

    @pytest.mark.skipif(
        not _REAL_TQ_STORAGE,
        reason="needs real TransferQueue storage submodules; CPU CI uses a single-file transfer_queue stub",
    )
    def test_installed_tq_satisfies_loss_prevention_contract(self):
        validate_mooncake_runtime_contract()

    @pytest.mark.skipif(
        not _REAL_TQ_STORAGE,
        reason="needs real TransferQueue storage submodules; CPU CI uses a single-file transfer_queue stub",
    )
    def test_contract_defaults_mooncake_memcpy_off(self, monkeypatch):
        # mooncake 0.3.10 memcpy fast path silently truncates TCP transfers;
        # the correctness guards must force it off when the operator is silent.
        monkeypatch.delenv("MC_STORE_MEMCPY", raising=False)
        validate_mooncake_runtime_contract()
        assert os.environ["MC_STORE_MEMCPY"] == "0"

    def test_contract_rejects_explicit_memcpy_enable(self, monkeypatch):
        # mooncake 0.3.10's memcpy path is confirmed to corrupt data, so the
        # guard fails closed instead of honouring an operator override.  The
        # rejection happens before any transfer_queue import, so this test
        # runs on CPU CI too.
        monkeypatch.setenv("MC_STORE_MEMCPY", "1")
        with pytest.raises(RuntimeError, match="MC_STORE_MEMCPY"):
            validate_mooncake_runtime_contract()

    def test_contract_accepts_explicit_memcpy_disable(self, monkeypatch):
        monkeypatch.setenv("MC_STORE_MEMCPY", "0")
        if _REAL_TQ_STORAGE:
            validate_mooncake_runtime_contract()
            assert os.environ["MC_STORE_MEMCPY"] == "0"

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

    def test_estimate_payload_text_only_is_small_but_nonzero(self):
        # Text payloads (ids/logprobs/masks) flow through the store too; the
        # bound is seq_length * 32 B per sample.
        args = _make_args(multimodal_keys=None)
        assert estimate_payload_bytes(args) == 32 * 1 * 8192 * 32

    def test_estimate_payload_multimodal_is_token_budget_bound(self):
        # One sample may not exceed seq_length vision tokens; at 784 pixels
        # per token and 12 B per pixel that is ~77 MiB for seq_length=8192.
        args = _make_args(multimodal_keys=["pixel_values"], rollout_batch_size=1, n_samples_per_prompt=1)
        per_sample = estimate_payload_bytes(args)
        assert per_sample == 8192 * (32 + 784 * 12)
        assert 70 * 1024**2 < per_sample < 80 * 1024**2

    def test_capacity_batch_uses_dynamic_partial_rollout_oversampling(self):
        args = _make_args(
            rollout_batch_size=16,
            partial_rollout=True,
            use_dynamic_global_batch_size=True,
            over_sampling_batch_size=64,
        )
        assert resolve_tq_capacity_batch_size(args) == 64
        assert estimate_payload_bytes(args) == 64 * 8192 * 32

    def test_capacity_batch_uses_nominal_rollout_without_dynamic_partial_rollout(self):
        args = _make_args(
            rollout_batch_size=16,
            partial_rollout=False,
            use_dynamic_global_batch_size=True,
            over_sampling_batch_size=64,
        )
        assert resolve_tq_capacity_batch_size(args) == 16

    def test_dynamic_partial_rollout_capacity_rejects_oversampling_peak(self):
        args = _make_args(
            multimodal_keys=["pixel_values"],
            rollout_batch_size=16,
            partial_rollout=True,
            use_dynamic_global_batch_size=True,
            over_sampling_batch_size=64,
        )
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="", gdr=False, fallback_reason="")
        err = validate_segment_capacity(args, eff)
        assert err is not None
        assert "effective_batch=64" in err

    def test_estimate_payload_requires_seq_length(self):
        args = _make_args(seq_length=None)
        with pytest.raises(RuntimeError, match="seq_length"):
            estimate_payload_bytes(args)

    def test_segment_capacity_multimodal_staleness_no_longer_passes(self):
        # Review (Codex P1): 32 in-flight samples x ~77 MiB x (staleness+1)=2
        # needs ~4.9 GiB and previously passed the 8 MiB/sample guess.
        args = _make_args(
            multimodal_keys=["pixel_values"], rollout_batch_size=32, n_samples_per_prompt=1, max_staleness=1
        )
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="", gdr=False, fallback_reason="")
        err = validate_segment_capacity(args, eff)
        assert err is not None and "RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB" in err

    def test_segment_capacity_env_override_raises_the_ceiling(self, monkeypatch):
        args = _make_args(
            multimodal_keys=["pixel_values"], rollout_batch_size=32, n_samples_per_prompt=1, max_staleness=1
        )
        eff = EffectiveConfig(backend="MooncakeStore", protocol="rdma", device="", gdr=False, fallback_reason="")
        monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", "8")
        assert validate_segment_capacity(args, eff) is None

    def test_segment_size_env_override_rejects_garbage(self, monkeypatch):
        from relax.utils.tq_config import resolve_global_segment_size

        monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", "four")
        with pytest.raises(RuntimeError, match="RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB"):
            resolve_global_segment_size()
        monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", "-1")
        with pytest.raises(RuntimeError, match="positive"):
            resolve_global_segment_size()
