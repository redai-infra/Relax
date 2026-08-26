# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts for the retained cross-node acceptance benchmark."""

from __future__ import annotations

import csv
import io
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.benchmarks import tq_cross_node_bench as bench


def _argv(protocol: str, *extra: str) -> list[str]:
    return [
        "bench",
        "--protocol",
        protocol,
        "--consumer-node-id",
        "node",
        "--tcp-device",
        "eth0",
        "--csv",
        "x",
        *extra,
    ]


def _raise(error: BaseException) -> None:
    raise error


def test_simple_and_multimodal_profiles_use_the_expected_runtime_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    import transfer_queue

    monkeypatch.setattr(transfer_queue, "GRPOGroupNSampler", lambda **_kwargs: object())
    assert bench.build_conf("simple", master="", device="", segment_gib=1).backend.storage_backend == "SimpleStorage"
    payload = bench.make_multimodal_payload(num_samples=3, total_mib=1)
    column = payload.get("multimodal_train_inputs")
    assert type(column).__name__ == "NonTensorStack"
    rows = column.tolist()
    assert all(set(row) == {"pixel_values", "image_grid_thw"} for row in rows)
    assert len({tuple(row["pixel_values"].shape) for row in rows}) > 1
    digest = bench.field_byte_digests(payload, ["multimodal_train_inputs"])
    rows[1]["pixel_values"][0, 0] += 1
    assert bench.field_byte_digests(payload, ["multimodal_train_inputs"]) != digest


@pytest.mark.parametrize(
    ("protocol", "ib_bytes", "tcp_bytes", "payload_bytes", "expected"),
    [
        ("rdma", 800, 0, 1000, True),
        ("rdma", 799, 0, 1000, False),
        ("rdma", 900, 901, 1000, True),
        ("tcp", 0, 200, 1000, True),
        ("tcp", 0, 199, 1000, False),
        ("tcp", 201, 200, 1000, True),
        ("simple", 0, 1, 1000, True),
        ("simple", 1, 0, 1000, False),
        ("rdma", 1, 0, 4 * 1024**3, False),
        ("tcp", 0, 1, 4 * 1024**3, False),
    ],
)
def test_wire_proof_matrix(protocol: str, ib_bytes: int, tcp_bytes: int, payload_bytes: int, expected: bool) -> None:
    assert bench.wire_is_proven(protocol, ib_bytes, tcp_bytes, payload_bytes) is expected


def test_counter_scope_failure_modes_and_idle_subtraction(tmp_path) -> None:
    for device, port, value in (("rdma0", 1, 11), ("rdma1", 1, 17), ("rdma1", 2, 29)):
        directory = tmp_path / "infiniband" / device / "ports" / str(port) / "counters"
        directory.mkdir(parents=True)
        (directory / "port_rcv_data").write_text(str(value))
    tcp = tmp_path / "net" / "eth0" / "statistics"
    tcp.mkdir(parents=True)
    (tcp / "rx_bytes").write_text("101")
    assert bench.read_counters("eth0", "rdma1", 2, sysfs_root=tmp_path) == {
        "ib:rdma1:2": 29 * 4,
        "tcp:eth0": 101,
    }
    with pytest.raises(ValueError, match="safe device"):
        bench.read_counters("eth0", "*", 1, sysfs_root=tmp_path)
    with pytest.raises(RuntimeError, match="TCP receive counter"):
        bench.read_counters("missing", sysfs_root=tmp_path)
    with pytest.raises(RuntimeError, match="counter set changed"):
        bench._counter_delta({"tcp:x": 1}, {}, "tcp:")
    with pytest.raises(RuntimeError, match="reset or wrapped"):
        bench._counter_delta({"tcp:x": 2}, {"tcp:x": 1}, "tcp:")
    assert bench._subtract_idle_noise(1_100, 100, 1.0, 1.0) == 1_000
    assert bench._subtract_idle_noise(50, 100, 1.0, 1.0) == 0
    with pytest.raises(RuntimeError, match="duration must be positive"):
        bench._throughput_gbs(1_000, 0.0, "put")


@pytest.mark.parametrize(
    "argv",
    [
        _argv("rdma"),
        _argv("tcp", "--master", "master.example:50051", "--device", "rdma0"),
        *[
            _argv("rdma", "--master", "master.example:50051", "--device", device, "--rdma-port", "1")
            for device in ("../rdma0", "*", "rdma*", "rdma?", "[ab]")
        ],
        _argv("tcp", "--master", "master.example:50051", "--rdma-port", "1"),
        _argv("simple", "--master", "master.example:50051"),
    ],
)
def test_cli_rejects_ambiguous_or_unsafe_counter_configuration(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="2"):
        bench.parse_args()


@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["cleanup-ok", "cleanup-fails"])
@pytest.mark.parametrize(
    ("byte_exact", "tcp_bytes", "error_type", "match", "error_kind"),
    [
        (False, 1_000, AssertionError, "byte-exact", "ByteExactMismatch"),
        (True, 0, RuntimeError, "wire proof failed", "WireProofFailed"),
    ],
    ids=["byte-exact", "wire-proof"],
)
def test_failed_gate_is_recorded_and_cleanup_does_not_mask_it(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
    byte_exact: bool,
    tcp_bytes: int,
    error_type: type[BaseException],
    match: str,
    error_kind: str,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(bench.ray, "get", lambda value: value)
    consumer = SimpleNamespace(
        sample_idle_counters=SimpleNamespace(remote=lambda _seconds: {"seconds": 1.0, "ib_bytes": 0, "tcp_bytes": 0}),
        begin_round=SimpleNamespace(remote=lambda: ({"tcp:x": 0}, 1.0)),
        fetch=SimpleNamespace(
            remote=lambda *_args: {
                "get_ms": 1.0,
                "round_seconds": 1.0,
                "ib_bytes": 0,
                "tcp_bytes": tcp_bytes,
                "byte_exact": byte_exact,
                "mismatch_fields": [] if byte_exact else ["field"],
            }
        ),
    )

    class Producer:
        def put(self, *_args: Any, **_kwargs: Any) -> None:
            events.append("put")

        def clear_partition(self, _partition: str) -> None:
            events.append("clear")
            if cleanup_fails:
                raise RuntimeError("partition cleanup failed")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=bench.CSV_COLUMNS)
    writer.writeheader()
    with pytest.raises(error_type, match=match) as excinfo:
        bench._run_round(
            producer=Producer(),
            consumer=consumer,
            payload=SimpleNamespace(batch_size=[1]),
            fields=["field"],
            expected={"field": ()},
            nbytes=1_000,
            protocol="tcp",
            profile="synthetic",
            requested_mib=1,
            run=1,
            writer=writer,
            csv_handle=output,
            provenance={"relax_sha": "a" * 40, "tq_commit": "b" * 40, "mooncake_version": "test"},
        )
    row = next(csv.DictReader(io.StringIO(output.getvalue())))
    assert (row["status"], row["error_kind"], row["byte_exact"]) == ("fail", error_kind, str(byte_exact))
    assert row["mismatch_fields"] == ("" if byte_exact else "field")
    assert isinstance(excinfo.value.__cause__, RuntimeError) is cleanup_fails
    assert events == ["put", "clear"]


def test_provenance_records_versions_and_rejects_dirty_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    results = iter([SimpleNamespace(returncode=0, stdout=""), SimpleNamespace(returncode=0, stdout="a" * 40 + "\n")])
    monkeypatch.setattr(bench.subprocess, "run", lambda *_args, **_kwargs: next(results))
    direct_url = '{"vcs_info":{"commit_id":"' + "b" * 40 + '"}}'
    monkeypatch.setattr(
        bench.importlib_metadata,
        "distribution",
        lambda _name: SimpleNamespace(read_text=lambda _filename: direct_url),
    )
    monkeypatch.setattr(bench.importlib_metadata, "version", lambda _name: "0.3.test")
    assert bench.collect_provenance(tmp_path) == {
        "relax_sha": "a" * 40,
        "tq_commit": "b" * 40,
        "mooncake_version": "0.3.test",
    }
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=" M tracked-file\n"),
    )
    with pytest.raises(RuntimeError, match="clean tracked Relax checkout"):
        bench.collect_provenance(tmp_path)


@pytest.fixture
def teardown_events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from relax.utils.tq import lifecycle

    events: list[str] = []
    monkeypatch.setattr(bench.ray, "kill", lambda *_args, **_kwargs: events.append("consumer-killed"))
    monkeypatch.setattr(lifecycle, "detach_tq_client", lambda: events.append("producer-detached"))
    monkeypatch.setattr(lifecycle, "close_tq_owner", lambda _owner: events.append("owner-closed"))
    monkeypatch.setattr(bench.ray, "shutdown", lambda: events.append("ray-shutdown"))
    return events


@pytest.mark.parametrize("failure", ["consumer", "owner", "none"])
def test_teardown_is_ordered_and_does_not_touch_unowned_state(
    monkeypatch: pytest.MonkeyPatch, teardown_events: list[str], failure: str
) -> None:
    from relax.utils.tq import lifecycle

    consumer, attached, owner = None, False, None
    expected = ["ray-shutdown"]
    error_match = None
    if failure == "consumer":
        consumer, attached, owner = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: "ref")), True, "owner"
        monkeypatch.setattr(bench.ray, "get", lambda *_args, **_kwargs: _raise(RuntimeError("consumer failed")))

        def fail_owner_cleanup(_owner: Any) -> None:
            teardown_events.append("owner-closed")
            raise RuntimeError("owner failed")

        monkeypatch.setattr(lifecycle, "close_tq_owner", fail_owner_cleanup)
        expected = ["consumer-killed", "producer-detached", "owner-closed", "ray-shutdown"]
        error_match = "consumer failed"
    elif failure == "owner":
        owner = "owner"
        monkeypatch.setattr(lifecycle, "close_tq_owner", lambda _owner: _raise(RuntimeError("owner failed")))
        error_match = "owner failed"
    context = pytest.raises(RuntimeError, match=error_match) if error_match else nullcontext()
    with context:
        bench._teardown_benchmark(consumer, producer_attached=attached, owner=owner)
    assert teardown_events == expected
