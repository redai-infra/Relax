#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cross-node TransferQueue C0/C1/C2 acceptance benchmark.

Invoke once per fresh process: ``simple`` is C0, benchmark-only Mooncake/TCP is
C1, and Mooncake/host-RDMA is C2.  The driver produces while one persistent
consumer is hard-pinned to another Ray node.  Dtype/shape/raw-byte digests and
receive-counter wire proof are mandatory; each measured round is flushed to CSV
before either gate can fail.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics
import struct
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ray


PROTOCOL_LABELS = {
    "simple": "C0 SimpleStorage",
    "tcp": "C1 Mooncake/TCP",
    "rdma": "C2 Mooncake/RDMA",
}
CSV_COLUMNS = "protocol profile payload_mib actual_mib run byte_exact wire_proven put_ms get_ms put_gbs get_gbs ib_mb tcp_mb".split()
_TCP_WIRE_MIN_PAYLOAD_RATIO = 0.20
_RDMA_WIRE_MIN_PAYLOAD_RATIO = 0.80


def _is_safe_device_name(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and all(character.isprintable() and not character.isspace() for character in value)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-node TQ byte-exact and wire-proof benchmark")
    parser.add_argument("--protocol", required=True, choices=sorted(PROTOCOL_LABELS))
    parser.add_argument("--consumer-node-id", required=True, help="Alive Ray NodeID distinct from the driver node")
    parser.add_argument("--master", default=os.environ.get("MC_MASTER_ADDRESS", ""), help="Mooncake master host:port")
    parser.add_argument("--device", default="", help="RDMA device required by the C2 wire-proof counter")
    parser.add_argument("--tcp-device", required=True, help="Network interface used for the TCP receive counter")
    parser.add_argument(
        "--payload-profiles", nargs="+", default=["synthetic", "multimodal"], choices=["synthetic", "multimodal"]
    )
    parser.add_argument("--payload-mib", nargs="+", type=int, default=[256, 1024, 2048, 4096])
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--segment-gib", type=int, default=16)
    parser.add_argument("--csv", required=True, help="Per-round CSV output path")
    args = parser.parse_args()
    for name in ("num_samples", "repeats", "segment_gib"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.payload_mib or any(size <= 0 for size in args.payload_mib):
        parser.error("--payload-mib values must be positive")
    if args.protocol != "simple" and not args.master:
        parser.error("--master or MC_MASTER_ADDRESS is required for Mooncake protocols")
    if not _is_safe_device_name(args.tcp_device):
        parser.error("--tcp-device must be a single printable interface name without whitespace or path separators")
    if args.protocol == "rdma":
        if not _is_safe_device_name(args.device):
            parser.error("--device is required for RDMA and must be a single printable device name")
    elif args.device:
        parser.error("--device is valid only with --protocol rdma")
    return args


def _columns_for_budget(total_bytes: int, num_samples: int, dtype: Any) -> int:
    import torch

    return max(1, total_bytes // (num_samples * torch.tensor([], dtype=dtype).element_size()))


def make_synthetic_payload(num_samples: int, total_mib: int):
    """Deterministic mixed-dtype payload with a fixed schema."""
    import torch
    from tensordict import TensorDict

    generator = torch.Generator().manual_seed(20260824)
    field_budget = total_mib * 1024**2 // 3
    fp32_cols = _columns_for_budget(field_budget, num_samples, torch.float32)
    bf16_cols = _columns_for_budget(field_budget, num_samples, torch.bfloat16)
    int64_cols = _columns_for_budget(field_budget, num_samples, torch.int64)
    return TensorDict(
        {
            "activations": torch.randn(num_samples, fp32_cols, dtype=torch.float32, generator=generator),
            "logits": torch.randn(num_samples, bf16_cols, dtype=torch.bfloat16, generator=generator),
            "tokens": torch.randint(0, 151_000, (num_samples, int64_cols), dtype=torch.int64, generator=generator),
        },
        batch_size=[num_samples],
    )


def make_multimodal_payload(num_samples: int, total_mib: int):
    """Deterministic production-shaped non-tensor vision-language payload."""
    import torch

    from relax.utils.utils import dict_to_tensordict

    target_bytes = total_mib * 1024**2
    seq_len = min(4096, max(128, target_bytes // max(1, num_samples * 128 * 1024)))
    fixed_bytes = num_samples * (seq_len * 8 + 8 + 4)
    pixel_budget = max(num_samples * 1536 * 4, target_bytes - fixed_bytes)
    base_patches = max(1, pixel_budget // (num_samples * 1536 * 4))
    generator = torch.Generator().manual_seed(20260824)
    multimodal_rows = []
    for index in range(num_samples):
        variation = (index % 3) - 1
        requested_patches = max(1, base_patches + variation * max(1, base_patches // 20))
        height = max(1, int(requested_patches**0.5))
        width = max(1, (requested_patches + height - 1) // height)
        patches = height * width
        pixel_values = torch.empty((patches, 1536), dtype=torch.float32)
        pixel_values.uniform_(-1.0, 1.0, generator=generator)
        multimodal_rows.append(
            {
                "pixel_values": pixel_values,
                "image_grid_thw": torch.tensor([[1, height, width]], dtype=torch.int64),
            }
        )
    token_row = list(range(seq_len))
    payload = dict_to_tensordict(
        {
            "tokens": [token_row for _ in range(num_samples)],
            "sample_id": list(range(num_samples)),
            "rewards": [float(index) / max(1, num_samples - 1) for index in range(num_samples)],
            "multimodal_train_inputs": multimodal_rows,
        },
        batch_size=num_samples,
    )
    if type(payload.get("multimodal_train_inputs")).__name__ != "NonTensorStack":
        raise RuntimeError("multimodal benchmark payload did not produce a NonTensorStack column")
    return payload


def _unwrap_non_tensor(value: Any) -> Any:
    if type(value).__name__ == "NonTensorStack":
        return value.tolist()
    if type(value).__name__ == "NonTensorData":
        return value.data
    return value


def _column_rows(value: Any) -> list[Any]:
    import torch

    value = _unwrap_non_tensor(value)
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            return list(value.unbind())
        if value.ndim == 0:
            return [value]
        return list(value.unbind())
    if isinstance(value, (list, tuple)):
        return [_unwrap_non_tensor(row) for row in value]
    raise TypeError(f"unsupported benchmark payload column: {type(value).__name__}")


def _leaf_digest(value: Any) -> Any:
    import numpy as np
    import torch

    value = _unwrap_non_tensor(value)
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            return ("nested", tuple(_leaf_digest(row) for row in value.unbind()))
        contiguous = value.detach().cpu().contiguous()
        flat = contiguous.reshape(-1)
        raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
        return (str(contiguous.dtype), tuple(contiguous.shape), hashlib.sha256(raw).hexdigest())
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            f"np.{contiguous.dtype}",
            tuple(contiguous.shape),
            hashlib.sha256(contiguous.tobytes()).hexdigest(),
        )
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("benchmark payload dictionaries require string keys")
        return ("dict", tuple((key, _leaf_digest(value[key])) for key in sorted(value)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_leaf_digest(item) for item in value))
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, float):
        raw = struct.pack("!d", value)
    elif isinstance(value, (bool, int)) or value is None:
        raw = repr(value).encode("utf-8")
    else:
        raise TypeError(f"unsupported benchmark payload leaf: {type(value).__name__}")
    return (f"py.{type(value).__name__}", (), hashlib.sha256(raw).hexdigest())


def field_byte_digests(payload, fields: list[str]) -> dict[str, tuple[Any, ...]]:
    """Ordered row digests preserving dtype, shape and every raw byte."""
    digests: dict[str, tuple[Any, ...]] = {}
    for field in fields:
        digests[field] = tuple(_leaf_digest(row) for row in _column_rows(payload.get(field)))
    return digests


def _value_bytes(value: Any) -> int:
    import numpy as np
    import torch

    value = _unwrap_non_tensor(value)
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            return sum(_value_bytes(row) for row in value.unbind())
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_value_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_bytes(item) for item in value)
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, float):
        return 8
    if isinstance(value, (bool, int)) or value is None:
        return len(repr(value).encode("utf-8"))
    raise TypeError(f"unsupported benchmark payload leaf: {type(value).__name__}")


def payload_bytes(payload) -> int:
    return sum(_value_bytes(row) for field in payload.keys() for row in _column_rows(payload.get(field)))


def read_counters(tcp_device: str, rdma_device: str = "", sysfs_root: Path = Path("/sys/class")) -> dict[str, int]:
    """Read receive-byte counters on a selected HCA and TCP interface."""
    counters: dict[str, int] = {}
    ib_root = sysfs_root / "infiniband"
    if ib_root.is_dir():
        pattern = (
            f"{rdma_device}/ports/*/counters/port_rcv_data" if rdma_device else "*/ports/*/counters/port_rcv_data"
        )
        for counter_path in sorted(ib_root.glob(pattern)):
            try:
                counters[f"ib:{counter_path.parents[3].name}:{counter_path.parents[1].name}"] = (
                    int(counter_path.read_text().strip()) * 4
                )
            except (OSError, ValueError):
                continue
    tcp_path = sysfs_root / "net" / tcp_device / "statistics" / "rx_bytes"
    try:
        counters[f"tcp:{tcp_device}"] = int(tcp_path.read_text().strip())
    except (OSError, ValueError):
        pass
    return counters


def _counter_delta(before: dict[str, int], after: dict[str, int], prefix: str) -> int:
    return sum(after[key] - before.get(key, after[key]) for key in after if key.startswith(prefix))


def wire_is_proven(protocol: str, ib_bytes: int, tcp_bytes: int, payload_nbytes: int) -> bool:
    if protocol == "rdma":
        return ib_bytes >= _RDMA_WIRE_MIN_PAYLOAD_RATIO * payload_nbytes and ib_bytes >= tcp_bytes
    if protocol == "tcp":
        return tcp_bytes >= _TCP_WIRE_MIN_PAYLOAD_RATIO * payload_nbytes and tcp_bytes >= ib_bytes
    # SimpleStorage may place some units on the consumer node, so C0 cannot
    # require a payload-volume ratio. It still must not look like RDMA.
    return tcp_bytes > 0 and tcp_bytes >= ib_bytes


def build_conf(protocol: str, master: str, device: str, segment_gib: int):
    from omegaconf import OmegaConf
    from transfer_queue import GRPOGroupNSampler

    from relax.utils.tq.config import (
        build_mooncake_config,
        build_simple_storage_config,
        validate_mooncake_runtime_contract,
    )

    if protocol == "simple":
        backend = build_simple_storage_config(total_storage_size=None, num_data_storage_units=2)
    else:
        validate_mooncake_runtime_contract()
        backend = build_mooncake_config(
            master_address=master,
            device=device,
            protocol=protocol,
            global_segment_size=segment_gib * 1024**3,
        )
    return OmegaConf.create(
        {
            "controller": {"sampler": GRPOGroupNSampler(n_samples_per_prompt=1), "polling_mode": True},
            "backend": backend,
        },
        flags={"allow_objects": True},
    )


def require_clean_cluster() -> None:
    try:
        ray.get_actor("TransferQueueController", namespace="transfer_queue")
    except ValueError:
        return
    raise RuntimeError("TransferQueueController already exists; benchmark requires a clean exclusive Ray cluster")


def close_tq_unmount_and_wait() -> None:
    import transfer_queue as tq

    from relax.utils.tq.lifecycle import kill_tq_controller_and_wait

    store_client = None
    try:
        store_client = getattr(tq.get_client().storage_manager, "storage_client", None)
    except (AssertionError, AttributeError):
        pass
    try:
        tq.close()
    finally:
        try:
            if store_client is not None and hasattr(store_client, "close"):
                store_client.close()
        finally:
            kill_tq_controller_and_wait()


@ray.remote(num_cpus=0.001, max_restarts=0)
class TQConsumer:
    """Persistent, hard-pinned consumer mirroring a Relax component actor."""

    def __init__(self, conf: Any, tcp_device: str, rdma_device: str, protocol: str):
        counters = read_counters(tcp_device, rdma_device if protocol == "rdma" else "")
        if f"tcp:{tcp_device}" not in counters:
            raise RuntimeError("TCP receive counter is unavailable on the selected consumer node")
        if protocol == "rdma" and not any(key.startswith("ib:") for key in counters):
            raise RuntimeError("RDMA receive counter is unavailable on the selected consumer node")
        if protocol == "tcp":
            os.environ["MC_TCP_ENABLE_CONNECTION_POOL"] = "1"
        from relax.utils.tq.lifecycle import attach_tq_client

        # This runs inside the pinned Ray worker.  The bounded helper applies
        # Mooncake correctness guards in this process before client creation.
        self.client = attach_tq_client(conf, role="benchmark-consumer")
        self.tcp_device = tcp_device
        self.rdma_device = rdma_device if protocol == "rdma" else ""

    def describe(self) -> tuple[str, str]:
        manager = self.client.storage_manager
        storage_client = getattr(manager, "storage_client", None)
        return type(manager).__name__, getattr(storage_client, "protocol", "")

    def fetch(self, fields: list[str], batch_size: int, partition: str, expected: dict) -> dict[str, Any]:
        before = read_counters(self.tcp_device, self.rdma_device)
        started = time.perf_counter()
        meta = self.client.get_meta(
            data_fields=fields,
            batch_size=batch_size,
            partition_id=partition,
            mode="fetch",
            task_name="tq-benchmark",
        )
        received = self.client.get_data(meta)
        get_ms = (time.perf_counter() - started) * 1000
        after = read_counters(self.tcp_device, self.rdma_device)
        actual = field_byte_digests(received, fields)
        if actual != expected:
            mismatches = [field for field in fields if actual.get(field) != expected.get(field)]
            raise AssertionError(f"byte-exact mismatch after TQ get: fields={mismatches}")
        return {
            "get_ms": get_ms,
            "ib_bytes": _counter_delta(before, after, "ib:"),
            "tcp_bytes": _counter_delta(before, after, "tcp:"),
        }

    def shutdown(self) -> None:
        from relax.utils.tq.lifecycle import detach_tq_client

        detach_tq_client()


def _validate_attached_backend(protocol: str, manager: str, attached_protocol: str) -> None:
    if protocol == "simple" and manager != "AsyncSimpleStorageManager":
        raise RuntimeError(f"C0 attached unexpected storage manager {manager}")
    if protocol != "simple" and (manager != "MooncakeStorageManager" or attached_protocol != protocol):
        raise RuntimeError(
            f"{PROTOCOL_LABELS[protocol]} attached manager={manager} protocol={attached_protocol or '-'}"
        )


def _teardown_benchmark(consumer: Any, owner_attempted: bool) -> None:
    """Unmount the consumer, clean owned global state, then stop local Ray."""
    try:
        if consumer is not None:
            try:
                ray.get(consumer.shutdown.remote(), timeout=10)
            finally:
                ray.kill(consumer, no_restart=True)
    finally:
        try:
            if owner_attempted:
                close_tq_unmount_and_wait()
        finally:
            ray.shutdown()


def main() -> None:
    args = parse_args()
    if args.protocol == "tcp":
        os.environ["MC_TCP_ENABLE_CONNECTION_POOL"] = "1"
    import transfer_queue as tq
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    with open(args.csv, "w", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        csv_handle.flush()
        ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
        consumer = None
        measured_runs = 0
        owner_attempted = False
        try:
            driver_node_id = ray.get_runtime_context().get_node_id()
            alive_ids = {node["NodeID"] for node in ray.nodes() if node.get("Alive")}
            if args.consumer_node_id not in alive_ids:
                raise RuntimeError("--consumer-node-id is not an alive Ray node")
            if args.consumer_node_id == driver_node_id:
                raise RuntimeError("producer and consumer must run on different Ray nodes")
            require_clean_cluster()

            conf = build_conf(args.protocol, args.master, args.device, args.segment_gib)
            owner_attempted = True
            tq.init(conf=conf)
            producer = tq.get_client()
            strategy = NodeAffinitySchedulingStrategy(node_id=args.consumer_node_id, soft=False)
            consumer = TQConsumer.options(scheduling_strategy=strategy).remote(
                conf, args.tcp_device, args.device, args.protocol
            )
            manager, attached_protocol = ray.get(consumer.describe.remote())
            _validate_attached_backend(args.protocol, manager, attached_protocol)
            print(f"[setup] {PROTOCOL_LABELS[args.protocol]} consumer=remote-node", flush=True)

            for profile in args.payload_profiles:
                for requested_mib in args.payload_mib:
                    payload = (
                        make_multimodal_payload(args.num_samples, requested_mib)
                        if profile == "multimodal"
                        else make_synthetic_payload(args.num_samples, requested_mib)
                    )
                    fields = sorted(payload.keys())
                    expected = field_byte_digests(payload, fields)
                    nbytes = payload_bytes(payload)
                    put_rates: list[float] = []
                    get_rates: list[float] = []
                    for run in range(args.repeats + 1):
                        partition = f"bench-{profile}-{requested_mib}-{run}"
                        started = time.perf_counter()
                        producer.put(payload, partition_id=partition)
                        put_ms = (time.perf_counter() - started) * 1000
                        fetched = ray.get(consumer.fetch.remote(fields, payload.batch_size[0], partition, expected))
                        producer.clear_partition(partition)
                        if run == 0:
                            continue
                        proven = wire_is_proven(args.protocol, fetched["ib_bytes"], fetched["tcp_bytes"], nbytes)
                        put_gbs = nbytes / put_ms / 1e6
                        get_gbs = nbytes / fetched["get_ms"] / 1e6
                        put_rates.append(put_gbs)
                        get_rates.append(get_gbs)
                        writer.writerow(
                            {
                                "protocol": args.protocol,
                                "profile": profile,
                                "payload_mib": requested_mib,
                                "actual_mib": round(nbytes / 1024**2, 2),
                                "run": run,
                                "byte_exact": True,
                                "wire_proven": proven,
                                "put_ms": round(put_ms, 2),
                                "get_ms": round(fetched["get_ms"], 2),
                                "put_gbs": round(put_gbs, 3),
                                "get_gbs": round(get_gbs, 3),
                                "ib_mb": round(fetched["ib_bytes"] / 1e6, 1),
                                "tcp_mb": round(fetched["tcp_bytes"] / 1e6, 1),
                            }
                        )
                        csv_handle.flush()
                        measured_runs += 1
                        if not proven:
                            raise RuntimeError(
                                f"wire proof failed for protocol={args.protocol} "
                                f"profile={profile} payload={requested_mib}MiB run={run}"
                            )
                    print(
                        f"[{profile} {requested_mib}MiB] byte_exact=PASS wire=PASS "
                        f"put={statistics.mean(put_rates):.2f}GB/s "
                        f"get_mean={statistics.mean(get_rates):.2f}GB/s "
                        f"get_median={statistics.median(get_rates):.2f}GB/s "
                        f"get_std={statistics.pstdev(get_rates):.2f}GB/s",
                        flush=True,
                    )
            print(f"[csv] wrote {measured_runs} measured rounds", flush=True)
        finally:
            _teardown_benchmark(consumer, owner_attempted)


if __name__ == "__main__":
    main()
