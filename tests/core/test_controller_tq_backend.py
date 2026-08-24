# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Controller._resolve_tq_backend: the production backend decision.

After the static ``/sys`` probe was removed, this method makes a pure
*configuration* decision: which mode was requested, whether the TransferQueue
correctness contract holds, whether a master endpoint is configured, and whether
the segment can hold the worst-case payload.  Host-RDMA capability itself is
established later by the real cluster-wide attach handshake
(``tests/utils/test_tq_failure_paths.py``).

These tests pin the branch structure: ``off`` must check nothing, ``auto`` must
converge on SimpleStorage for every unmet precondition, ``required`` must fail
fast on the same ones, a malformed mode must always raise, and no input may ever
produce a MooncakeStore config whose protocol is not ``rdma``.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from tests.core.test_controller_s3_model_cleanup import controller
from tests.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


# ``relax.utils.arguments`` pulls in the full SGLang server args, which the
# GitHub CPU CI does not install (it ships only ``sglang-router``).  Reuse the
# stub fixture so the CLI assertions build a real parser without that
# dependency.
arguments_module = _arguments_module_fixture

_MASTER = "master.invalid:50051"


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


class _Recorder:
    """Records which configuration preconditions the Controller consulted.

    A recorded ``probe`` entry would mean a static capability probe crept back
    in; the sequence assertions below are what keep that from happening
    silently.
    """

    def __init__(self, monkeypatch, *, contract_error=None, master_error=None):
        self.calls: list[str] = []

        def fake_contract() -> None:
            self.calls.append("contract")
            if contract_error is not None:
                raise contract_error

        def fake_master() -> str:
            self.calls.append("master")
            if master_error is not None:
                raise master_error
            return _MASTER

        monkeypatch.setattr(controller, "validate_mooncake_runtime_contract", fake_contract)
        monkeypatch.setattr(controller, "resolve_mooncake_master_address", fake_master)


def _resolve(config) -> dict:
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    return instance._resolve_tq_backend(total_storage_size=64)


def _assert_simple_storage(backend: dict) -> None:
    assert backend["storage_backend"] == "SimpleStorage"
    assert "MooncakeStore" not in backend


def _raise(error: BaseException) -> None:
    raise error


class TestOffMode:
    """``off`` is the untouched SimpleStorage path: it checks nothing."""

    def test_off_short_circuits_without_any_precondition_check(self, monkeypatch):
        recorder = _Recorder(monkeypatch)
        _assert_simple_storage(_resolve(_config(tq_rdma_mode="off")))
        assert recorder.calls == []

    def test_missing_mode_attribute_defaults_to_off(self, monkeypatch):
        recorder = _Recorder(monkeypatch)
        config = _config()
        del config.tq_rdma_mode
        _assert_simple_storage(_resolve(config))
        assert recorder.calls == []

    def test_healthy_existing_controller_aborts_before_legacy_init(self, monkeypatch):
        """The default path must not attach to or later close another job.

        Upstream ``tq.init`` ignores the caller's SimpleStorage config when a
        named controller already exists.  The exclusive-cluster check must
        therefore fail before setting the local ownership flag or calling it.
        """
        config = _config(
            tq_rdma_mode="off",
            fully_async=False,
            balance_data=False,
            polling_mode=False,
        )
        instance = controller.Controller.__new__(controller.Controller)
        instance.config = config
        instance._tq_owner = None
        instance._tq_legacy_init = False

        monkeypatch.setattr(controller, "resolve_sft_algo_key", lambda _config: "grpo")
        monkeypatch.setattr(controller, "resolve_tq_capacity_batch_size", lambda _config: 1)
        monkeypatch.setattr(controller, "GRPOGroupNSampler", lambda **_kwargs: object())
        monkeypatch.setattr(
            instance,
            "_resolve_tq_backend",
            lambda _total_storage_size: {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {"total_storage_size": 1, "num_data_storage_units": 1},
            },
        )
        monkeypatch.setattr(
            controller,
            "reap_unusable_tq_controller",
            lambda: _raise(RuntimeError("exclusive cluster is not clean")),
        )
        monkeypatch.setattr(
            controller.tq,
            "init",
            lambda **_kwargs: pytest.fail("existing controller must be rejected before tq.init"),
        )

        with pytest.raises(RuntimeError, match="exclusive cluster"):
            instance._initialize_data_system()

        assert instance._tq_owner is None
        assert instance._tq_legacy_init is False


class TestNoStaticCapabilityProbe:
    """The ``/sys``/GID/memlock probe is gone and must not return."""

    def test_resolver_never_probes_hardware(self, monkeypatch):
        recorder = _Recorder(monkeypatch)
        _resolve(_config())
        assert recorder.calls == ["contract", "master"]


class TestBackendPreconditions:
    """The same unmet precondition degrades ``auto`` and aborts
    ``required``."""

    @pytest.mark.parametrize("mode", ["auto", "required"])
    @pytest.mark.parametrize(
        ("recorder_kwargs", "config_overrides", "segment_override", "match"),
        [
            ({"contract_error": RuntimeError("retry guard missing")}, {}, None, "correctness contract"),
            ({"master_error": RuntimeError("MC_MASTER_ADDRESS required")}, {}, None, "master endpoint"),
            ({"master_error": RuntimeError("unusable master endpoint")}, {}, None, "master endpoint"),
            (
                {},
                {"multimodal_keys": ["pixel_values"], "rollout_batch_size": 64, "n_samples_per_prompt": 8},
                None,
                "segment capacity insufficient",
            ),
            ({}, {}, "four", "segment-capacity configuration is unusable"),
            ({}, {}, "nan", "segment-capacity configuration is unusable"),
            ({}, {}, "inf", "segment-capacity configuration is unusable"),
            ({}, {}, "-inf", "segment-capacity configuration is unusable"),
            ({}, {"seq_length": None}, None, "segment-capacity configuration is unusable"),
        ],
        ids=[
            "correctness-contract",
            "missing-master",
            "malformed-master",
            "insufficient-capacity",
            "invalid-segment-size",
            "nan-segment-size",
            "infinite-segment-size",
            "negative-infinite-segment-size",
            "missing-seq-length",
        ],
    )
    def test_unmet_precondition(
        self,
        monkeypatch,
        mode,
        recorder_kwargs,
        config_overrides,
        segment_override,
        match,
    ):
        recorder = _Recorder(monkeypatch, **recorder_kwargs)
        if segment_override is not None:
            monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", segment_override)

        config = _config(tq_rdma_mode=mode, **config_overrides)
        if mode == "auto":
            _assert_simple_storage(_resolve(config))
        else:
            with pytest.raises(RuntimeError, match=match):
                _resolve(config)
        assert recorder.calls == (["contract"] if recorder_kwargs.get("contract_error") else ["contract", "master"])

    def test_satisfied_preconditions_select_host_rdma(self, monkeypatch):
        _Recorder(monkeypatch)
        private_device = "private-device-name"
        messages: list[str] = []
        monkeypatch.setattr(controller.logger, "info", lambda message: messages.append(message))

        backend = _resolve(_config(tq_rdma_device=private_device))
        assert backend["storage_backend"] == "MooncakeStore"
        assert backend["MooncakeStore"]["protocol"] == "rdma"
        assert backend["MooncakeStore"]["device_name"] == private_device
        assert any("device_selection=explicit" in message for message in messages)
        assert all(private_device not in message for message in messages)

    def test_required_accepts_satisfied_preconditions(self, monkeypatch):
        _Recorder(monkeypatch)
        backend = _resolve(_config(tq_rdma_mode="required"))
        assert backend["MooncakeStore"]["protocol"] == "rdma"

    def test_validated_master_endpoint_reaches_the_client_config(self, monkeypatch):
        """The checked endpoint, rather than a later env value, reaches
        Mooncake."""
        _Recorder(monkeypatch)
        monkeypatch.setenv("MC_MASTER_ADDRESS", "someone.else.invalid:9999")
        backend = _resolve(_config())
        assert backend["MooncakeStore"]["master_server_address"] == _MASTER


class TestModeValidation:
    """A malformed mode is a configuration error in every mode."""

    def test_invalid_mode_always_raises_instead_of_falling_back(self, monkeypatch):
        recorder = _Recorder(monkeypatch)
        with pytest.raises(ValueError, match="--tq-rdma-mode"):
            _resolve(_config(tq_rdma_mode="mooncake"))
        assert recorder.calls == []

    @pytest.mark.parametrize("device", [None, ["rdma0"], "rdma0\nforged", "   "])
    def test_invalid_device_always_raises_before_preconditions(self, monkeypatch, device):
        recorder = _Recorder(monkeypatch)
        with pytest.raises(ValueError, match="--tq-rdma-device"):
            _resolve(_config(tq_rdma_device=device))
        assert recorder.calls == []


class TestProductionNeverSelectsMooncakeTcp:
    """Mooncake/TCP exists only as benchmark C1."""

    @pytest.mark.parametrize("mode", ["off", "auto", "required"])
    def test_selected_backend_is_rdma_or_simple(self, monkeypatch, mode):
        _Recorder(monkeypatch)
        backend = _resolve(_config(tq_rdma_mode=mode))
        if backend["storage_backend"] == "MooncakeStore":
            assert backend["MooncakeStore"]["protocol"] == "rdma"
        else:
            _assert_simple_storage(backend)


def test_cli_exposes_only_mode_and_device(arguments_module):
    """The narrowed CLI keeps exactly two TransferQueue RDMA flags."""
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    tq_flags = sorted(
        option for action in parser._actions for option in action.option_strings if option.startswith("--tq-")
    )
    assert tq_flags == ["--tq-rdma-device", "--tq-rdma-mode"]

    args = parser.parse_args([])
    assert args.tq_rdma_mode == "off"
    assert args.tq_rdma_device == ""
    assert not hasattr(args, "tq_storage_backend")
    assert not hasattr(args, "tq_use_gdr")


# ---------------------------------------------------------------------------
# _confirm_mooncake_attach: the cleanup evidence chain
# ---------------------------------------------------------------------------


class _AttachRecorder:
    """Orders the teardown steps taken after a failed attach handshake.

    The static probe used to reject unusable clusters before any Mooncake state
    existed.  Now the handshake fails *after* an owner and a named controller
    were created, so the ordering recorded here is what keeps a half-
    initialised controller from surviving into the next ``tq.init`` (F10 hang).
    """

    def __init__(
        self,
        monkeypatch,
        *,
        failures=None,
        verify_error=None,
        close_error=None,
        fallback_owner="fallback-owner",
    ):
        self.events: list[str] = []
        self.closed: list[object] = []
        self.initialized: list[object] = []
        self._fallback_owner = fallback_owner

        def fake_verify(conf, **_kwargs):
            self.events.append("handshake")
            if verify_error is not None:
                raise verify_error
            return list(failures or [])

        def fake_close(owner, **_kwargs):
            self.events.append("close")
            self.closed.append(owner)
            if close_error is not None:
                raise close_error

        def fake_initialize(conf, **_kwargs):
            self.events.append("init_simple")
            self.initialized.append(conf)
            return controller.TqInitResult(config=conf, owner=self._fallback_owner)

        monkeypatch.setattr(controller, "verify_cluster_attach", fake_verify)
        monkeypatch.setattr(controller, "close_tq_owner", fake_close)
        monkeypatch.setattr(controller, "initialize_tq_with_fallback", fake_initialize)


def _confirm(config, *, owner="mooncake-owner", fallback_config="simple-conf"):
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    init_result = controller.TqInitResult(config="mooncake-conf", owner=owner)
    return instance._confirm_mooncake_attach(init_result, fallback_config)


class TestAttachHandshakeCleanupChain:
    def test_success_keeps_mooncake_and_touches_no_cleanup(self, monkeypatch):
        recorder = _AttachRecorder(monkeypatch, failures=[])
        result = _confirm(_config())
        assert result.config == "mooncake-conf"
        assert result.fallback_reason == ""
        assert recorder.events == ["handshake"]

    def test_auto_closes_owner_before_initializing_simple_storage(self, monkeypatch):
        recorder = _AttachRecorder(monkeypatch, failures=["node: attach timed out"])
        result = _confirm(_config(), owner="attempt-owner")
        assert recorder.events == ["handshake", "close", "init_simple"]
        assert recorder.closed == ["attempt-owner"]
        assert result.config == "simple-conf"
        assert result.fallback_reason == "attach_handshake_failed:1_failures"

    def test_cleanup_failure_aborts_instead_of_falling_back(self, monkeypatch):
        recorder = _AttachRecorder(
            monkeypatch,
            failures=["node: attach timed out"],
            close_error=RuntimeError("TransferQueue owner cleanup failed"),
        )
        owner = object()
        instance = controller.Controller.__new__(controller.Controller)
        instance.config = _config()
        instance._tq_owner = owner
        init_result = controller.TqInitResult(config="mooncake-conf", owner=owner)

        with pytest.raises(RuntimeError, match="owner cleanup failed"):
            instance._confirm_mooncake_attach(init_result, "simple-conf")
        assert recorder.events == ["handshake", "close"]
        assert recorder.initialized == []
        assert instance._tq_owner is owner

    def test_required_closes_owner_then_raises(self, monkeypatch):
        recorder = _AttachRecorder(monkeypatch, failures=["node: protocol=tcp"])
        with pytest.raises(RuntimeError, match="attach handshake reported"):
            _confirm(_config(tq_rdma_mode="required"))
        assert recorder.events == ["handshake", "close"]
        assert recorder.initialized == []

    def test_unexpected_driver_exception_closes_owner_and_is_sanitized(self, monkeypatch):
        secret = "worker endpoint and traceback path must stay private"
        recorder = _AttachRecorder(monkeypatch, verify_error=RuntimeError(secret))
        with pytest.raises(RuntimeError, match="orchestration failed") as excinfo:
            _confirm(_config())
        assert recorder.events == ["handshake", "close"]
        assert recorder.initialized == []
        assert secret not in str(excinfo.value)

    def test_unconfirmed_worker_isolation_closes_owner_and_aborts(self, monkeypatch):
        recorder = _AttachRecorder(
            monkeypatch,
            verify_error=controller.TqHandshakeIsolationError("private cancellation detail"),
        )
        with pytest.raises(RuntimeError, match="could not be confirmed stopped") as excinfo:
            _confirm(_config())
        assert recorder.events == ["handshake", "close"]
        assert recorder.initialized == []
        assert "private cancellation detail" not in str(excinfo.value)

    def test_constructor_cleanup_boundary_includes_data_system_initialization(self, monkeypatch):
        """An exception after owner creation must still invoke constructor
        cleanup."""
        events: list[str] = []

        monkeypatch.setattr(controller, "resolve_sft_num_rollout", lambda _config: None)
        monkeypatch.setattr(controller, "HealthManager", lambda **_kwargs: object())

        def fail_after_owner_created(instance):
            instance._tq_owner = "mooncake-owner"
            raise RuntimeError("driver-side handshake orchestration failed")

        def record_cleanup(instance):
            events.append(instance._tq_owner)
            instance._tq_owner = None

        monkeypatch.setattr(controller.Controller, "_initialize_data_system", fail_after_owner_created)
        monkeypatch.setattr(controller.Controller, "_close_data_system", record_cleanup)

        config = SimpleNamespace(use_health_check=False, max_global_restart=3)
        with pytest.raises(RuntimeError, match="handshake orchestration failed"):
            controller.Controller(config)

        assert events == ["mooncake-owner"]

    def test_repeated_data_system_cleanup_closes_the_owner_once(self, monkeypatch):
        instance = controller.Controller.__new__(controller.Controller)
        instance._tq_legacy_init = False
        instance._tq_owner = "owner"
        closed: list[object] = []

        monkeypatch.setattr(controller, "close_tq_owner", lambda owner: closed.append(owner) if owner else None)

        instance._close_data_system()
        instance._close_data_system()

        assert closed == ["owner"]
        assert instance._tq_owner is None
