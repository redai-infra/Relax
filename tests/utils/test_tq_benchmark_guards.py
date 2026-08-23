# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts for the retained cross-node TransferQueue benchmark."""

import sys
import warnings
from types import SimpleNamespace

import pytest
import torch

from scripts.benchmarks import tq_cross_node_bench


def test_simple_config_builds_without_mooncake_runtime(monkeypatch):
    import transfer_queue

    monkeypatch.setattr(transfer_queue, "GRPOGroupNSampler", lambda **_kwargs: object())
    conf = tq_cross_node_bench.build_conf("simple", master="", device="", segment_gib=1)
    assert conf.backend.storage_backend == "SimpleStorage"


def test_digest_normalizes_dense_and_nested_rows_without_losing_shape():
    dense = {"field": torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        nested = {"field": torch.nested.nested_tensor(list(dense["field"].unbind()))}

    expected = tq_cross_node_bench.field_byte_digests(dense, ["field"])
    assert tq_cross_node_bench.field_byte_digests(nested, ["field"]) == expected
    assert expected["field"][0][:2] == ("torch.int16", (2,))

    changed = {"field": dense["field"].clone()}
    changed["field"][1, 1] += 1
    assert tq_cross_node_bench.field_byte_digests(changed, ["field"]) != expected

    same_bytes_different_shape = {"field": dense["field"].reshape(1, 2, 2)}
    one_row = {"field": dense["field"].reshape(1, 4)}
    assert tq_cross_node_bench.field_byte_digests(same_bytes_different_shape, ["field"]) != (
        tq_cross_node_bench.field_byte_digests(one_row, ["field"])
    )


def test_multimodal_profile_uses_production_non_tensor_stack_and_recursive_digest():
    payload = tq_cross_node_bench.make_multimodal_payload(num_samples=3, total_mib=1)
    multimodal = payload.get("multimodal_train_inputs")
    assert type(multimodal).__name__ == "NonTensorStack"
    rows = multimodal.tolist()
    assert all(set(row) == {"pixel_values", "image_grid_thw"} for row in rows)
    assert len({tuple(row["pixel_values"].shape) for row in rows}) > 1

    expected = tq_cross_node_bench.field_byte_digests(payload, ["multimodal_train_inputs"])
    rows[1]["pixel_values"][0, 0] += 1
    assert tq_cross_node_bench.field_byte_digests(payload, ["multimodal_train_inputs"]) != expected


@pytest.mark.parametrize(
    ("protocol", "ib_bytes", "tcp_bytes", "payload_bytes", "expected"),
    [
        ("rdma", 800, 0, 1000, True),
        ("rdma", 799, 0, 1000, False),
        ("rdma", 900, 901, 1000, False),
        ("tcp", 0, 200, 1000, True),
        ("tcp", 0, 199, 1000, False),
        ("tcp", 201, 200, 1000, False),
        ("simple", 0, 1, 1000, True),
        ("simple", 1, 0, 1000, False),
        ("rdma", 1, 0, 4 * 1024**3, False),
        ("tcp", 0, 1, 4 * 1024**3, False),
    ],
)
def test_wire_proof_is_protocol_and_volume_specific(protocol, ib_bytes, tcp_bytes, payload_bytes, expected):
    assert tq_cross_node_bench.wire_is_proven(protocol, ib_bytes, tcp_bytes, payload_bytes) is expected


def test_read_counters_scopes_rdma_to_selected_hca(tmp_path):
    for device, value in (("rdma0", 11), ("rdma1", 29)):
        path = tmp_path / "infiniband" / device / "ports" / "1" / "counters"
        path.mkdir(parents=True)
        (path / "port_rcv_data").write_text(str(value))
    tcp = tmp_path / "net" / "eth0" / "statistics"
    tcp.mkdir(parents=True)
    (tcp / "rx_bytes").write_text("101")

    counters = tq_cross_node_bench.read_counters("eth0", rdma_device="rdma1", sysfs_root=tmp_path)

    assert counters == {"ib:rdma1:1": 29 * 4, "tcp:eth0": 101}


@pytest.mark.parametrize(
    "argv",
    [
        ["bench", "--protocol", "rdma", "--consumer-node-id", "node", "--tcp-device", "eth0", "--csv", "x"],
        [
            "bench",
            "--protocol",
            "tcp",
            "--consumer-node-id",
            "node",
            "--master",
            "master.example:50051",
            "--device",
            "rdma0",
            "--tcp-device",
            "eth0",
            "--csv",
            "x",
        ],
        [
            "bench",
            "--protocol",
            "rdma",
            "--consumer-node-id",
            "node",
            "--master",
            "master.example:50051",
            "--device",
            "../rdma0",
            "--tcp-device",
            "eth0",
            "--csv",
            "x",
        ],
    ],
)
def test_cli_rejects_ambiguous_or_unsafe_counter_configuration(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="2"):
        tq_cross_node_bench.parse_args()


class _RemoteCall:
    def __init__(self, result):
        self.result = result

    def remote(self):
        return self.result


def test_teardown_keeps_cleanup_order_when_consumer_shutdown_fails(monkeypatch):
    events: list[str] = []
    consumer = SimpleNamespace(shutdown=_RemoteCall("shutdown-ref"))
    monkeypatch.setattr(
        tq_cross_node_bench.ray,
        "get",
        lambda _ref, timeout: (_ for _ in ()).throw(RuntimeError("consumer shutdown failed")),
    )
    monkeypatch.setattr(tq_cross_node_bench.ray, "kill", lambda *_args, **_kwargs: events.append("consumer-killed"))
    monkeypatch.setattr(tq_cross_node_bench, "close_tq_unmount_and_wait", lambda: events.append("owner-closed"))
    monkeypatch.setattr(tq_cross_node_bench.ray, "shutdown", lambda: events.append("ray-shutdown"))

    with pytest.raises(RuntimeError, match="consumer shutdown failed"):
        tq_cross_node_bench._teardown_benchmark(consumer, owner_attempted=True)

    assert events == ["consumer-killed", "owner-closed", "ray-shutdown"]


def test_teardown_dirty_cluster_does_not_close_unowned_controller(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(tq_cross_node_bench, "close_tq_unmount_and_wait", lambda: events.append("owner-closed"))
    monkeypatch.setattr(tq_cross_node_bench.ray, "shutdown", lambda: events.append("ray-shutdown"))

    tq_cross_node_bench._teardown_benchmark(None, owner_attempted=False)

    assert events == ["ray-shutdown"]


def test_clean_cluster_guard_never_kills_a_healthy_existing_controller(monkeypatch):
    controller = object()
    killed: list[object] = []
    monkeypatch.setattr(tq_cross_node_bench.ray, "get_actor", lambda *_args, **_kwargs: controller)
    monkeypatch.setattr(tq_cross_node_bench.ray, "kill", lambda handle, **_kwargs: killed.append(handle))

    with pytest.raises(RuntimeError, match="clean exclusive Ray cluster"):
        tq_cross_node_bench.require_clean_cluster()

    assert killed == []


def test_teardown_shutdowns_ray_when_owner_cleanup_fails(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        tq_cross_node_bench,
        "close_tq_unmount_and_wait",
        lambda: (_ for _ in ()).throw(RuntimeError("owner cleanup failed")),
    )
    monkeypatch.setattr(tq_cross_node_bench.ray, "shutdown", lambda: events.append("ray-shutdown"))

    with pytest.raises(RuntimeError, match="owner cleanup failed"):
        tq_cross_node_bench._teardown_benchmark(None, owner_attempted=True)

    assert events == ["ray-shutdown"]


def test_production_controller_cleanup_timeout_fails_closed(monkeypatch):
    from relax.utils.tq import lifecycle

    controller = object()
    killed: list[object] = []
    monkeypatch.setattr(lifecycle.ray, "get_actor", lambda *_args, **_kwargs: controller)
    monkeypatch.setattr(lifecycle.ray, "kill", lambda handle: killed.append(handle))

    with pytest.raises(lifecycle.TqCleanupTimeout, match="still resolvable"):
        lifecycle.kill_tq_controller_and_wait(timeout=0)

    assert killed == [controller]
