# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Controller._resolve_tq_backend: the production backend decision.

``reduce_results`` is covered in ``tests/utils/test_rdma_probe.py``; these tests
pin the Controller branch that consumes it, because that is where ``off``
short-circuits, where ``auto`` must converge on SimpleStorage for *every* unmet
host-RDMA precondition, and where ``required`` must fail fast instead.

Mooncake/TCP is a benchmark baseline only, so no input may ever make this method
return a MooncakeStore config whose protocol is not ``rdma``.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from relax.utils.rdma_probe import ProbeResult
from tests.core.test_controller_s3_model_cleanup import controller
from tests.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


# ``relax.utils.arguments`` pulls in the full SGLang server args, which the
# GitHub CPU CI does not install (it ships only ``sglang-router``).  Reuse the
# stub fixture so the CLI assertions build a real parser without that
# dependency.
arguments_module = _arguments_module_fixture


def _config(**overrides) -> SimpleNamespace:
    """A config whose worst-case payload fits the default 4 GiB segment."""
    defaults = dict(
        tq_rdma_mode="auto",
        tq_rdma_device="",
        num_data_storage_units=1,
        n_samples_per_prompt=1,
        rollout_batch_size=8,
        seq_length=8192,
        multimodal_keys=None,
        max_staleness=0,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _probe(node: str, protocol: str | None = "rdma", device: str = "") -> ProbeResult:
    return ProbeResult(
        node=node,
        checks=(),
        effective_protocol=protocol,
        effective_device=device,
        errors=() if protocol else ("mooncake not importable",),
    )


class _Recorder:
    """Records which host-RDMA preconditions the Controller actually
    consulted."""

    def __init__(self, monkeypatch, *, contract_error=None, master_error=None, probes=None):
        self.calls: list[str] = []
        self._probes = probes if probes is not None else [_probe("node-A")]
        # ``build_mooncake_config`` re-resolves the endpoint from the
        # environment, so the success paths need a real value there too.
        monkeypatch.setenv("MC_MASTER_ADDRESS", "master.invalid:50051")

        def fake_contract() -> None:
            self.calls.append("contract")
            if contract_error is not None:
                raise contract_error

        def fake_master() -> str:
            self.calls.append("master")
            if master_error is not None:
                raise master_error
            return "master.invalid:50051"

        def fake_probe(device: str, master_address: str) -> list[ProbeResult]:
            self.calls.append("probe")
            return self._probes

        monkeypatch.setattr(controller, "validate_mooncake_runtime_contract", fake_contract)
        monkeypatch.setattr(controller, "resolve_mooncake_master_address", fake_master)
        monkeypatch.setattr(controller, "probe_cluster_nodes", fake_probe)


def _resolve(config) -> dict:
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    return instance._resolve_tq_backend(total_storage_size=64)


def _assert_simple_storage(backend: dict) -> None:
    assert backend["storage_backend"] == "SimpleStorage"
    assert "MooncakeStore" not in backend


class TestOffMode:
    """``off`` is the untouched SimpleStorage path: it probes nothing."""

    def test_off_short_circuits_without_any_precondition_check(self, monkeypatch):
        recorder = _Recorder(monkeypatch)
        backend = _resolve(_config(tq_rdma_mode="off"))
        _assert_simple_storage(backend)
        assert recorder.calls == []

    def test_missing_mode_attribute_defaults_to_off(self, monkeypatch):
        recorder = _Recorder(monkeypatch)
        config = _config()
        del config.tq_rdma_mode
        _assert_simple_storage(_resolve(config))
        assert recorder.calls == []


class TestAutoFallsBackForEveryUnmetPrecondition:
    """Gate A: in ``auto``, anything short of host RDMA yields
    SimpleStorage."""

    def test_contract_failure_falls_back_before_touching_master(self, monkeypatch):
        recorder = _Recorder(monkeypatch, contract_error=RuntimeError("retry guard missing"))
        _assert_simple_storage(_resolve(_config()))
        assert recorder.calls == ["contract"]

    def test_missing_master_endpoint_falls_back_before_probing(self, monkeypatch):
        recorder = _Recorder(monkeypatch, master_error=RuntimeError("MC_MASTER_ADDRESS required"))
        _assert_simple_storage(_resolve(_config()))
        assert recorder.calls == ["contract", "master"]

    def test_node_without_rdma_falls_back(self, monkeypatch):
        _Recorder(monkeypatch, probes=[_probe("node-A"), _probe("node-B", protocol="tcp")])
        _assert_simple_storage(_resolve(_config()))

    def test_node_without_mooncake_falls_back(self, monkeypatch):
        _Recorder(monkeypatch, probes=[_probe("node-A"), _probe("node-B", protocol=None)])
        _assert_simple_storage(_resolve(_config()))

    def test_device_mismatch_falls_back(self, monkeypatch):
        _Recorder(monkeypatch, probes=[_probe("node-A", device="rdma0"), _probe("node-B", device="rdma1")])
        _assert_simple_storage(_resolve(_config(tq_rdma_device="rdma0")))

    def test_insufficient_segment_capacity_falls_back(self, monkeypatch):
        """Worst-case multimodal payload far exceeds the default segment."""
        _Recorder(monkeypatch)
        config = _config(multimodal_keys=["pixel_values"], rollout_batch_size=64, n_samples_per_prompt=8)
        _assert_simple_storage(_resolve(config))

    def test_all_nodes_rdma_selects_mooncake(self, monkeypatch):
        _Recorder(monkeypatch, probes=[_probe("node-A"), _probe("node-B")])
        backend = _resolve(_config())
        assert backend["storage_backend"] == "MooncakeStore"
        assert backend["MooncakeStore"]["protocol"] == "rdma"


class TestRequiredFailsFast:
    """``required`` must never silently downgrade the same failures."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"contract_error": RuntimeError("retry guard missing")}, "correctness contract"),
            ({"master_error": RuntimeError("MC_MASTER_ADDRESS required")}, "master endpoint is not configured"),
            ({"probes": [_probe("node-A"), _probe("node-B", protocol="tcp")]}, "host RDMA is unavailable"),
            ({"probes": [_probe("node-A"), _probe("node-B", protocol=None)]}, "host RDMA is unavailable"),
        ],
    )
    def test_required_raises(self, monkeypatch, kwargs, match):
        _Recorder(monkeypatch, **kwargs)
        with pytest.raises(RuntimeError, match=match):
            _resolve(_config(tq_rdma_mode="required"))

    def test_required_raises_on_insufficient_capacity(self, monkeypatch):
        _Recorder(monkeypatch)
        config = _config(
            tq_rdma_mode="required",
            multimodal_keys=["pixel_values"],
            rollout_batch_size=64,
            n_samples_per_prompt=8,
        )
        with pytest.raises(RuntimeError, match="segment capacity insufficient"):
            _resolve(config)

    def test_required_accepts_full_rdma_cluster(self, monkeypatch):
        _Recorder(monkeypatch, probes=[_probe("node-A"), _probe("node-B")])
        backend = _resolve(_config(tq_rdma_mode="required"))
        assert backend["MooncakeStore"]["protocol"] == "rdma"


class TestProductionNeverSelectsMooncakeTcp:
    """Mooncake/TCP exists only as benchmark C1."""

    @pytest.mark.parametrize("mode", ["off", "auto"])
    @pytest.mark.parametrize(
        "probes",
        [
            [_probe("node-A", protocol="tcp")],
            [_probe("node-A"), _probe("node-B", protocol="tcp")],
            [_probe("node-A", protocol=None)],
            [],
        ],
    )
    def test_no_input_produces_a_tcp_mooncake_backend(self, monkeypatch, mode, probes):
        _Recorder(monkeypatch, probes=probes)
        backend = _resolve(_config(tq_rdma_mode=mode))
        if backend["storage_backend"] == "MooncakeStore":
            assert backend["MooncakeStore"]["protocol"] == "rdma"
        else:
            _assert_simple_storage(backend)

    def test_invalid_mode_is_rejected(self, monkeypatch):
        _Recorder(monkeypatch)
        with pytest.raises(ValueError, match="--tq-rdma-mode"):
            _resolve(_config(tq_rdma_mode="mooncake"))


def test_cli_exposes_only_mode_and_device(arguments_module):
    """The narrowed CLI keeps exactly two TransferQueue RDMA flags."""
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    tq_flags = sorted(
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--tq-") and "timeout" not in option
    )
    assert tq_flags == ["--tq-rdma-device", "--tq-rdma-mode"]

    args = parser.parse_args([])
    assert args.tq_rdma_mode == "off"
    assert args.tq_rdma_device == ""
    assert not hasattr(args, "tq_storage_backend")
    assert not hasattr(args, "tq_use_gdr")
