# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for TransferQueue backend config construction and validation.

CPU-only: nothing here starts Ray, TransferQueue, or a Mooncake client.  These
tests cover everything Relax decides *before* anything is initialised — the
requested mode, the config dicts handed to ``tq.init``, the master endpoint
format, and the segment-capacity pre-check.

Actual host-RDMA capability is not testable here by design: it is established by
the real cluster-wide attach handshake, covered in
``tests/utils/test_tq_failure_paths.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import pytest

from relax.utils.tq.config import (
    _split_host_port,
    build_mooncake_config,
    build_simple_storage_config,
    estimate_payload_bytes,
    resolve_mooncake_master_address,
    resolve_tq_capacity_batch_size,
    validate_config,
    validate_mooncake_runtime_contract,
    validate_segment_capacity,
)


_MASTER = "master.example:50051"


def _has_real_tq_storage() -> bool:
    try:
        return importlib.util.find_spec("transfer_queue.storage.clients.mooncake_client") is not None
    except (ImportError, TypeError, ValueError):
        return False


_REAL_TQ_STORAGE = _has_real_tq_storage()


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        tq_rdma_mode="auto",
        tq_rdma_device="",
        num_data_storage_units=1,
        max_staleness=0,
        n_samples_per_prompt=1,
        rollout_batch_size=32,
        multimodal_keys=None,
        seq_length=8192,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestValidateConfig:
    """Minimal mode/device checks left after the static probe was removed."""

    @pytest.mark.parametrize("mode", ["off", "auto", "required"])
    def test_accepts_every_supported_mode(self, mode):
        assert validate_config(_make_args(tq_rdma_mode=mode)) == []

    @pytest.mark.parametrize("mode", ["mooncake", "auto\nprivate", None, ["auto"]])
    def test_rejects_unknown_mode_without_echoing_it(self, mode):
        """Guards configs restored from a checkpoint or built without
        argparse."""
        errors = validate_config(_make_args(tq_rdma_mode=mode))
        assert len(errors) == 1
        assert "--tq-rdma-mode" in errors[0]
        assert repr(mode) not in errors[0]

    def test_missing_attribute_defaults_to_off(self):
        assert validate_config(argparse.Namespace()) == []

    @pytest.mark.parametrize("device", [None, ["rdma0"], "rdma0\nforged", "   "])
    def test_rejects_non_string_or_whitespace_device(self, device):
        errors = validate_config(_make_args(tq_rdma_device=device))
        assert len(errors) == 1
        assert "--tq-rdma-device" in errors[0]
        assert repr(device) not in errors[0]

    @pytest.mark.parametrize("device", ["", "rdma0", "mlx5_0"])
    def test_accepts_empty_or_printable_device_name(self, device):
        assert validate_config(_make_args(tq_rdma_device=device)) == []


class TestMasterEndpoint:
    """``MC_MASTER_ADDRESS`` is deployment configuration, validated by format
    only.

    No DNS lookup and no connection attempt: reachability is proven later by
    the real attach, and re-adding a network probe here would recreate the
    capability heuristic this phase deleted.
    """

    def test_required_and_returned_from_env(self, monkeypatch):
        monkeypatch.setenv("MC_MASTER_ADDRESS", _MASTER)
        assert resolve_mooncake_master_address() == _MASTER

    def test_missing_is_rejected(self, monkeypatch):
        # A loopback default would point every node at itself in multi-node
        # runs; missing deployment configuration must be rejected instead.
        monkeypatch.delenv("MC_MASTER_ADDRESS", raising=False)
        with pytest.raises(RuntimeError, match="MC_MASTER_ADDRESS"):
            resolve_mooncake_master_address()

    @pytest.mark.parametrize(
        "address",
        [
            "master.example",  # no port
            "master.example:",  # empty port
            "master.example:abc",  # non-numeric port
            ":50051",  # no host
            "master.example:0",  # port below range
            "master.example:65536",  # port above range
            "fe80::1",  # bare IPv6: rpartition would yield port=1
            "fe80::1:50051",  # bare IPv6 with port
            "[fe80::1]",  # bracketed host, no port
            "[fe80::1]50051",  # bracketed host, missing colon
            "master\nprivate:50051",  # embedded control character
            "master name:50051",  # embedded whitespace
        ],
    )
    def test_malformed_endpoints_are_rejected(self, monkeypatch, address):
        monkeypatch.setenv("MC_MASTER_ADDRESS", address)
        with pytest.raises(RuntimeError, match="not a usable endpoint"):
            resolve_mooncake_master_address()

    def test_rejection_never_echoes_the_endpoint(self, monkeypatch):
        """The Controller logs this error verbatim.

        An internal hostname or IP is deployment detail that must not reach job
        logs, so the message names the defect only.
        """
        secret = "prod-master-07.internal.corp:0"
        monkeypatch.setenv("MC_MASTER_ADDRESS", secret)
        with pytest.raises(RuntimeError) as excinfo:
            resolve_mooncake_master_address()
        message = str(excinfo.value)
        assert "prod-master-07" not in message
        assert "internal.corp" not in message
        assert "port is outside" in message

    def test_accepts_hostname_and_bracketed_ipv6(self):
        assert _split_host_port(_MASTER) == ("master.example", 50051)
        assert _split_host_port("[2001:db8::1]:50051") == ("2001:db8::1", 50051)


class TestBackendConfigDicts:
    """The dicts handed to ``tq.init``."""

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
        assert build_mooncake_config(master_address=_MASTER)["storage_backend"] == "MooncakeStore"
        assert (
            build_simple_storage_config(total_storage_size=1, num_data_storage_units=1)["storage_backend"]
            == "SimpleStorage"
        )

    def test_mooncake_defaults_to_host_rdma(self):
        """Production never names a protocol: the default is the only one it
        ships."""
        mc = build_mooncake_config(master_address=_MASTER, device="rdma0")["MooncakeStore"]
        assert mc["protocol"] == "rdma"
        assert mc["device_name"] == "rdma0"
        assert mc["hard_pin"] is True  # no silent eviction
        assert mc["auto_init"] is False  # external master
        assert mc["master_server_address"] == _MASTER

    def test_empty_device_is_left_to_mooncake(self):
        mc = build_mooncake_config(master_address=_MASTER)["MooncakeStore"]
        assert mc["device_name"] == ""

    def test_tcp_is_reachable_only_by_explicit_request(self):
        """Mooncake/TCP survives as benchmark C1, never as a production
        default."""
        mc = build_mooncake_config(master_address=_MASTER, protocol="tcp")["MooncakeStore"]
        assert mc["protocol"] == "tcp"

    def test_mooncake_config_pins_host_rdma(self):
        """This phase ships host RDMA only: ``use_gdr`` is always False and no
        GDR staging buffer is configured."""
        mc = build_mooncake_config(master_address=_MASTER)["MooncakeStore"]
        assert mc["use_gdr"] is False
        assert "gdr_staging_buffer_mb" not in mc

    def test_master_address_is_not_re_read_from_env(self, monkeypatch):
        """The validated endpoint must be the one the client receives."""
        monkeypatch.setenv("MC_MASTER_ADDRESS", "other.example:9999")
        mc = build_mooncake_config(master_address=_MASTER)["MooncakeStore"]
        assert mc["master_server_address"] == _MASTER

    @pytest.mark.parametrize("master", ["not-an-endpoint", "host:0", "fe80::1", 50051])
    def test_builder_rejects_unvalidated_master_address(self, master):
        with pytest.raises(ValueError, match="master_address"):
            build_mooncake_config(master_address=master)

    @pytest.mark.parametrize("segment_size", [0, -1, True, 1.5])
    def test_explicit_segment_size_must_be_a_positive_integer(self, segment_size):
        with pytest.raises(ValueError, match="positive integer"):
            build_mooncake_config(master_address=_MASTER, global_segment_size=segment_size)


class TestCorrectnessContract:
    """The Mooncake loss-prevention gate."""

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
        # rejection happens before any transfer_queue import, so this test runs
        # on CPU CI too.
        monkeypatch.setenv("MC_STORE_MEMCPY", "1")
        with pytest.raises(RuntimeError, match="MC_STORE_MEMCPY"):
            validate_mooncake_runtime_contract()

    def test_contract_rejection_does_not_echo_memcpy_override(self, monkeypatch):
        private_detail = "1\nprivate deployment detail"
        monkeypatch.setenv("MC_STORE_MEMCPY", private_detail)
        with pytest.raises(RuntimeError, match="MC_STORE_MEMCPY") as excinfo:
            validate_mooncake_runtime_contract()
        assert private_detail not in str(excinfo.value)

    def test_contract_accepts_explicit_memcpy_disable(self, monkeypatch):
        monkeypatch.setenv("MC_STORE_MEMCPY", "0")
        if _REAL_TQ_STORAGE:
            validate_mooncake_runtime_contract()
            assert os.environ["MC_STORE_MEMCPY"] == "0"

    @pytest.mark.skipif(
        not _REAL_TQ_STORAGE,
        reason="needs real TransferQueue storage submodules; CPU CI uses a single-file transfer_queue stub",
    )
    def test_unreadable_source_becomes_a_runtime_error(self, monkeypatch):
        """A compiled/stripped install cannot prove put/notify ordering.

        The failure must stay inside the RuntimeError boundary the Controller
        catches; a bare OSError would abort an ``auto`` run that is supposed to
        fall back.
        """
        import inspect as inspect_module

        from relax.utils.tq import config as config_module

        def raise_oserror(_obj):
            raise OSError("could not get source code")

        monkeypatch.setattr(config_module.inspect, "getsource", raise_oserror)
        assert inspect_module is not None  # the patch targets the module's own alias
        with pytest.raises(RuntimeError, match="Cannot verify TransferQueue put/notify ordering"):
            validate_mooncake_runtime_contract()


class TestSegmentCapacity:
    """Configuration-level capacity pre-check (kept: not a hardware probe)."""

    def test_text_only_passes(self):
        assert validate_segment_capacity(_make_args(multimodal_keys=None)) is None

    def test_multimodal_large_batch_fails(self):
        args = _make_args(
            multimodal_keys=["pixel_values"], rollout_batch_size=256, n_samples_per_prompt=8, max_staleness=1
        )
        err = validate_segment_capacity(args)
        assert err is not None
        assert "insufficient" in err.lower()

    def test_multimodal_staleness_no_longer_passes(self):
        # Review (Codex P1): 32 in-flight samples x ~77 MiB x (staleness+1)=2
        # needs ~4.9 GiB and previously passed the 8 MiB/sample guess.
        args = _make_args(
            multimodal_keys=["pixel_values"], rollout_batch_size=32, n_samples_per_prompt=1, max_staleness=1
        )
        err = validate_segment_capacity(args)
        assert err is not None and "RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB" in err

    def test_env_override_raises_the_ceiling(self, monkeypatch):
        args = _make_args(
            multimodal_keys=["pixel_values"], rollout_batch_size=32, n_samples_per_prompt=1, max_staleness=1
        )
        monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", "8")
        assert validate_segment_capacity(args) is None

    @pytest.mark.parametrize("value", ["four", "-1", "nan", "inf", "-inf"])
    def test_segment_size_env_override_rejects_unusable_values(self, monkeypatch, value):
        from relax.utils.tq.config import resolve_global_segment_size

        monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", value)
        with pytest.raises(RuntimeError, match="finite positive") as excinfo:
            resolve_global_segment_size()
        assert value not in str(excinfo.value)

    def test_segment_size_env_override_cannot_round_down_to_zero_bytes(self, monkeypatch):
        from relax.utils.tq.config import resolve_global_segment_size

        monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", "1e-20")
        with pytest.raises(RuntimeError, match="at least one byte"):
            resolve_global_segment_size()


class TestPayloadEstimate:
    """The token-budget bound behind the capacity check."""

    def test_text_only_is_small_but_nonzero(self):
        # Text payloads (ids/logprobs/masks) flow through the store too; the
        # bound is seq_length * 32 B per sample.
        assert estimate_payload_bytes(_make_args(multimodal_keys=None)) == 32 * 1 * 8192 * 32

    def test_multimodal_is_token_budget_bound(self):
        # One sample may not exceed seq_length vision tokens; at 784 pixels per
        # token and 12 B per pixel that is ~77 MiB for seq_length=8192.
        args = _make_args(multimodal_keys=["pixel_values"], rollout_batch_size=1, n_samples_per_prompt=1)
        per_sample = estimate_payload_bytes(args)
        assert per_sample == 8192 * (32 + 784 * 12)
        assert 70 * 1024**2 < per_sample < 80 * 1024**2

    def test_requires_seq_length(self):
        with pytest.raises(RuntimeError, match="seq_length"):
            estimate_payload_bytes(_make_args(seq_length=None))

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
        err = validate_segment_capacity(args)
        assert err is not None
        assert "effective_batch=64" in err
