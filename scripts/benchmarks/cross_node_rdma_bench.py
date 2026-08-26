#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cross-node RDMA benchmark using raw MooncakeDistributedStore.

This bypasses TransferQueue to measure the raw transport layer across two
nodes, isolating TCP vs RDMA net benefit.

Setup:
  Node A (holder):  python scripts/benchmarks/cross_node_rdma_bench.py --role holder \
                       --master 0.0.0.0:50051 --segment-gb 16 --device mlx5_bond_0
  Node B (client):  python scripts/benchmarks/cross_node_rdma_bench.py --role client \
                       --master <ip>:50051 --device mlx5_bond_0 \
                       --protocol rdma --payload-mib 64 --repeats 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch


DEFAULT_DEVICE = ""


def parse_args() -> argparse.Namespace:
    """Parse cross-node benchmark CLI arguments."""
    p = argparse.ArgumentParser(description="Cross-node RDMA benchmark")
    p.add_argument(
        "--role",
        required=True,
        choices=["holder", "client"],
        help="holder=segment holder on node A, client=benchmark on node B",
    )
    p.add_argument("--master", required=True, help="Master address host:port")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="RDMA device name")
    p.add_argument("--protocol", default="rdma", choices=["tcp", "rdma"])
    p.add_argument("--segment-gb", type=int, default=16, help="Segment size in GiB (holder only)")
    p.add_argument(
        "--payload-mib",
        nargs="+",
        type=int,
        default=[8, 64, 256],
        help="Payload sizes in MiB (client only, --mode simple)",
    )
    p.add_argument(
        "--mode",
        default="simple",
        choices=["simple", "multimodal"],
        help="simple=1D tensor, multimodal=32 samples x [patch,1176] variable-length",
    )
    p.add_argument("--num-samples", type=int, default=32, help="Num samples (multimodal mode only)")
    p.add_argument("--patch-min", type=int, default=1213, help="Min patch count per sample (multimodal)")
    p.add_argument("--patch-max", type=int, default=2471, help="Max patch count per sample (multimodal)")
    p.add_argument("--hidden", type=int, default=1176, help="Hidden dim per patch (multimodal)")
    p.add_argument("--repeats", type=int, default=5, help="Repetitions (client only)")
    p.add_argument("--warmup", type=int, default=2, help="Warmup rounds (client only)")
    return p.parse_args()


def create_store(args, segment_size: int):
    """Create a MooncakeDistributedStore."""
    from mooncake.store import MooncakeDistributedStore

    local_hostname = os.environ.get("MC_TCP_BIND_ADDRESS", "")
    store = MooncakeDistributedStore()
    store.setup(
        local_hostname,  # local_hostname
        "P2PHANDSHAKE",  # metadata_server
        segment_size,  # global_segment_size (0 on client = no local segment)
        1024 * 1024 * 1024,  # local_buffer_size (1 GiB)
        args.protocol,  # protocol
        args.device,  # device_name
        args.master,  # master_server_address
    )
    return store


def run_holder(args):
    """Run as segment holder on node A — mounts a large segment and waits."""
    segment_size = args.segment_gb * 1024**3
    print(
        f"[holder] Creating MooncakeDistributedStore: segment={args.segment_gb} GiB, "
        f"protocol={args.protocol}, device={args.device}, master={args.master}"
    )
    store = create_store(args, segment_size)
    print("[holder] Segment mounted. Holding... (Ctrl+C to stop)")
    print(f"[holder] Master: {args.master}")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[holder] Shutting down...")
    finally:
        # Release the segment even on unexpected exit so the master does not
        # leak a pinned-memory segment across benchmark runs.
        try:
            store.close()
        except Exception as e:
            print(f"[holder] store.close() failed: {e}")


def run_client(args):
    """Run as client on node B — put/get payloads, measure throughput."""
    print(
        f"[client] Connecting: protocol={args.protocol}, device={args.device}, "
        f"master={args.master}, segment_size=0 (forces cross-node)"
    )
    print(f"[client] mode={args.mode}, repeats={args.repeats}")
    store = create_store(args, 0)  # segment_size=0 → all data lands on holder

    if args.mode == "multimodal":
        _run_multimodal(args, store)
    else:
        _run_simple(args, store)

    store.close()
    print("[client] Done")


def _make_multimodal_payload(args):
    """Create a realistic pixel_values payload: N samples x [patch, hidden]
    float32.

    Patch counts are uniformly spread in [patch_min, patch_max] to match real
    Qwen2-VL batches. All samples are packed into one contiguous 1D buffer
    (what TQ serializes into for transport).
    """
    import random

    rng = random.Random(42)  # deterministic
    patches = [rng.randint(args.patch_min, args.patch_max) for _ in range(args.num_samples)]
    total_elements = sum(p * args.hidden for p in patches)
    total_bytes = total_elements * 4  # float32
    data = torch.randn(total_elements, dtype=torch.float32)
    return data, total_bytes, patches


def _run_simple(args, store):
    for payload_mib in args.payload_mib:
        payload_bytes = payload_mib * 1024 * 1024
        data = torch.randn(payload_bytes // 4, dtype=torch.float32)
        key = f"bench_{payload_mib}mib"
        _bench_one(store, key, data, payload_bytes, f"{payload_mib} MiB", args)


def _run_multimodal(args, store):
    data, total_bytes, patches = _make_multimodal_payload(args)
    total_mib = total_bytes / 1024 / 1024
    print(
        f"  [multimodal] {args.num_samples} samples, patches {min(patches)}-{max(patches)}, "
        f"hidden={args.hidden}, total={total_mib:.1f} MiB ({total_bytes / 1e6:.1f} MB)"
    )
    _bench_one(store, "bench_multimodal", data, total_bytes, f"{total_mib:.0f}M mm", args)

    # Also test with n_samples_per_prompt=8 group duplication (GRPO redundancy)
    for mult in [4, 8]:
        group_data = data.repeat(mult)
        group_bytes = total_bytes * mult
        group_mib = group_bytes / 1024 / 1024
        _bench_one(store, f"bench_mm_{mult}x", group_data, group_bytes, f"{group_mib:.0f}M mm×{mult}", args)


def _bench_one(store, key, data, payload_bytes, label, args):
    """Put/get one payload, print timing."""
    # Warmup
    for _ in range(args.warmup):
        store.put_tensor(key, data)
        _ = store.get_tensor(key)

    put_times = []
    get_times = []
    for i in range(args.repeats):
        t0 = time.perf_counter()
        store.put_tensor(key, data)
        put_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        retrieved = store.get_tensor(key)
        get_ms = (time.perf_counter() - t0) * 1000

        # Correctness is a hard requirement, not an assert: ``python -O`` strips
        # asserts, and a silent None skip would inflate throughput. Fail loudly.
        if retrieved is None:
            raise RuntimeError(f"get_tensor returned None for {key} (data lost)")
        if not torch.equal(retrieved, data):
            raise RuntimeError(f"Byte mismatch for {key}")
        del retrieved

        put_gbs = payload_bytes / put_ms / 1e6 if put_ms > 0 else 0
        get_gbs = payload_bytes / get_ms / 1e6 if get_ms > 0 else 0
        put_times.append(put_ms)
        get_times.append(get_ms)
        print(
            f"  [{label:>12}] run {i + 1}/{args.repeats}: "
            f"put={put_ms:>7.1f}ms ({put_gbs:.2f} GB/s)  "
            f"get={get_ms:>7.1f}ms ({get_gbs:.2f} GB/s)"
        )

    put_med = statistics.median(put_times)
    get_med = statistics.median(get_times)
    put_gbs = payload_bytes / put_med / 1e6 if put_med > 0 else 0
    get_gbs = payload_bytes / get_med / 1e6 if get_med > 0 else 0
    print(
        f"  [{label:>12}] MEDIAN: "
        f"put={put_med:>7.1f}ms ({put_gbs:.2f} GB/s)  "
        f"get={get_med:>7.1f}ms ({get_gbs:.2f} GB/s)"
    )
    print()


def main():
    """Dispatch to holder (node A) or client (node B) per ``--role``."""
    args = parse_args()
    if args.role == "holder":
        run_holder(args)
    else:
        run_client(args)


if __name__ == "__main__":
    main()
