#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TransferQueue RDMA benchmark: SimpleStorage vs Mooncake/TCP vs
Mooncake/RDMA.

Runs put/get on synthetic payloads and reports throughput for three
configurations, isolating the RDMA net benefit (C2 - C1).

Usage (single-node loopback, no master needed for C0):
    python scripts/benchmarks/tq_rdma_bench.py \
        --payload-mib 1 16 64 256 \
        --num-samples 256 \
        --num-fields 1 8 32 \
        --repeats 3

Usage (cross-node RDMA, needs external mooncake master):
    MC_MASTER_ADDRESS=node-A:50051 \
    MC_TCP_BIND_ADDRESS=<ip> \
    python scripts/benchmarks/tq_rdma_bench.py \
        --configs C0 C1 C2 \
        --payload-mib 64 256 \
        --num-samples 64 \
        --repeats 3 \
        --device mlx5_bond_0
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    p = argparse.ArgumentParser(description="TransferQueue RDMA benchmark")
    p.add_argument(
        "--configs",
        nargs="+",
        default=["C0", "C1", "C2"],
        choices=["C0", "C1", "C2"],
        help="C0=SimpleStorage, C1=Mooncake/TCP, C2=Mooncake/RDMA",
    )
    p.add_argument(
        "--payload-mib",
        nargs="+",
        type=int,
        default=[16, 64, 256],
        help="Payload sizes in MiB (total across all fields)",
    )
    p.add_argument("--num-samples", type=int, default=256, help="Number of samples (rows)")
    p.add_argument("--num-fields", nargs="+", type=int, default=[1, 8, 32], help="Number of fields per sample")
    p.add_argument("--repeats", type=int, default=3, help="Repetitions per config (report median)")
    p.add_argument("--warmup", type=int, default=1, help="Warmup rounds (not counted)")
    p.add_argument("--device", type=str, default="", help="RDMA device name (e.g. mlx5_bond_0). Empty = auto.")
    p.add_argument(
        "--master-address",
        type=str,
        default=None,
        help="Mooncake master address. Default: env MC_MASTER_ADDRESS or localhost:50051",
    )
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--output-csv", type=str, default=None, help="Write results to CSV file")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Config builders
# --------------------------------------------------------------------------- #

CONFIG_MAP = {
    "C0": {"backend": "SimpleStorage", "protocol": "tcp"},
    "C1": {"backend": "MooncakeStore", "protocol": "tcp"},
    "C2": {"backend": "MooncakeStore", "protocol": "rdma"},
}


def build_tq_config(config_name: str, args: argparse.Namespace, num_storage_units: int = 1):
    """Build the tq.init config dict for a given benchmark config.

    Reuses :mod:`relax.utils.tq.config` builders so the benchmark cannot drift
    from the production config shape (single source of truth for
    keys/defaults).
    """
    from omegaconf import OmegaConf
    from transfer_queue import GRPOGroupNSampler

    from relax.utils.rdma_probe import EffectiveConfig
    from relax.utils.tq.config import build_mooncake_config, build_simple_storage_config

    cfg = CONFIG_MAP[config_name]
    sampler = GRPOGroupNSampler(n_samples_per_prompt=1)

    if cfg["backend"] == "SimpleStorage":
        backend_dict = build_simple_storage_config(
            total_storage_size=1024**3, num_data_storage_units=num_storage_units
        )
    else:
        master_addr = args.master_address or os.environ.get("MC_MASTER_ADDRESS", "localhost:50051")
        eff = EffectiveConfig(
            backend="MooncakeStore",
            protocol=cfg["protocol"],
            device=args.device,
            gdr=False,
            fallback_reason="",
        )
        backend_dict = build_mooncake_config(eff, master_address=master_addr, global_segment_size=8 * 1024**3)

    return OmegaConf.create(
        {
            "controller": {"sampler": sampler, "polling_mode": True},
            "backend": backend_dict,
        },
        flags={"allow_objects": True},
    )


def wait_actor_gone(name: str = "TransferQueueController", timeout: float = 20.0) -> None:
    """Wait for a named TQ actor to leave GCS, failing closed on timeout."""
    import ray

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ray.get_actor(name, namespace="transfer_queue")
        except ValueError:
            return
        time.sleep(0.4)
    raise TimeoutError(f"Ray actor {name!r} is still registered after {timeout:.1f}s")


def close_tq_and_wait(timeout: float = 20.0) -> None:
    """Close TQ, unmount the Mooncake segment, and confirm controller exit.

    Required between configs: ``tq.init`` attaches to an existing controller
    and ignores the new conf (interface.py:152), so without close+wait every
    config after the first silently reuses the first config's backend.

    ``tq.close()`` tears down only the ZMQ layer (managers/base.py:378); it never
    calls ``storage_client.close()``.  With MooncakeStore that leaves the segment
    mounted and still registered in the master, so the next config's put targets a
    dead endpoint ("Failed to open segment ... Connection refused") until the
    master's ``client_ttl`` (30 s) expires.  Unmount explicitly instead.
    """
    import transfer_queue as tq

    store_client = None
    try:
        store_client = getattr(tq.get_client().storage_manager, "storage_client", None)
    except (AssertionError, AttributeError):
        pass

    tq.close()  # runs remove_all() through the store, so unmount has to come after

    if store_client is not None and hasattr(store_client, "close"):
        store_client.close()  # unmounts the segment and deregisters from the master

    wait_actor_gone(timeout=timeout)


# --------------------------------------------------------------------------- #
# Payload generation
# --------------------------------------------------------------------------- #


def make_payload(num_samples: int, num_fields: int, total_mib: int, dtype: str):
    """Create a synthetic TensorDict of ``num_fields`` tensors.

    Each tensor has shape (num_samples, N) with the requested dtype, where N is
    chosen so total bytes ≈ total_mib * 1024^2.

    Returns a ``TensorDict`` with ``batch_size=[num_samples]`` (what TQ's
    ``client.put`` expects).
    """
    from tensordict import TensorDict

    dt = getattr(torch, dtype)
    elem_size = torch.tensor([], dtype=dt).element_size()
    total_bytes = total_mib * 1024 * 1024
    per_field_bytes = total_bytes // num_fields
    cols = max(1, per_field_bytes // (elem_size * num_samples))

    data = {}
    for f in range(num_fields):
        data[f"field_{f}"] = torch.randn(num_samples, cols, dtype=dt)
    return TensorDict(data, batch_size=[num_samples])


def payload_bytes(payload) -> int:
    """Return total bytes across all tensor fields in ``payload``."""
    return sum(payload[key].nelement() * payload[key].element_size() for key in payload.keys())


# --------------------------------------------------------------------------- #
# Benchmark core
# --------------------------------------------------------------------------- #


def run_one(config_name: str, payload: dict, args: argparse.Namespace) -> dict:
    """Run put/get once and return timing."""
    import transfer_queue as tq

    if CONFIG_MAP[config_name]["backend"] == "MooncakeStore":
        from relax.utils.tq.config import validate_mooncake_runtime_contract

        validate_mooncake_runtime_contract()

    # Close any prior controller and wait for GCS deregistration so this config
    # gets a fresh backend (tq.init otherwise attaches to the existing one).
    close_tq_and_wait()
    tq_config = build_tq_config(config_name, args)
    tq_config = tq.init(conf=tq_config) or tq_config
    client = tq.get_client()

    nbytes = payload_bytes(payload)

    # Warmup
    for _ in range(args.warmup):
        client.put(payload, partition_id="bench")
        client.clear_partition("bench")

    # Timed put
    t0 = time.perf_counter()
    client.put(payload, partition_id="bench")
    put_ms = (time.perf_counter() - t0) * 1000

    # Timed get: create a fetch meta, then get_data
    field_names = sorted(payload.keys())
    t0 = time.perf_counter()
    fetch_meta = client.get_meta(
        data_fields=field_names,
        batch_size=args.num_samples,
        partition_id="bench",
        mode="fetch",
        task_name="bench",
    )
    data = client.get_data(fetch_meta)
    get_ms = (time.perf_counter() - t0) * 1000

    # Correctness spot-check (best-effort: get_data may add non-tensor fields)
    mismatches = []
    for k in field_names:
        try:
            if not torch.equal(data[k], payload[k]):
                mismatches.append(k)
        except Exception:
            pass  # non-tensor field, skip
    if mismatches:
        raise RuntimeError(f"Byte mismatch in fields: {mismatches}")

    client.clear_partition("bench")

    return {
        "config": config_name,
        "backend": CONFIG_MAP[config_name]["backend"],
        "protocol": CONFIG_MAP[config_name]["protocol"],
        "put_ms": put_ms,
        "get_ms": get_ms,
        "nbytes": nbytes,
        "put_gbs": nbytes / put_ms / 1e6 if put_ms > 0 else 0,
        "get_gbs": nbytes / get_ms / 1e6 if get_ms > 0 else 0,
    }


def run_config(config_name: str, payload: dict, args: argparse.Namespace) -> dict:
    """Run ``args.repeats`` times, return median."""
    results = []
    for i in range(args.repeats):
        r = run_one(config_name, payload, args)
        results.append(r)
        print(
            f"  [{config_name}] run {i + 1}/{args.repeats}: put={r['put_ms']:.1f}ms "
            f"({r['put_gbs']:.2f} GB/s)  get={r['get_ms']:.1f}ms ({r['get_gbs']:.2f} GB/s)"
        )

    put_vals = sorted(r["put_ms"] for r in results)
    get_vals = sorted(r["get_ms"] for r in results)
    put_med = statistics.median(put_vals)
    get_med = statistics.median(get_vals)
    nbytes = results[0]["nbytes"]
    return {
        "config": config_name,
        "backend": CONFIG_MAP[config_name]["backend"],
        "protocol": CONFIG_MAP[config_name]["protocol"],
        "put_ms_median": put_med,
        "get_ms_median": get_med,
        "put_ms_min": min(put_vals),
        "put_ms_max": max(put_vals),
        "get_ms_min": min(get_vals),
        "get_ms_max": max(get_vals),
        "nbytes": nbytes,
        "put_gbs_median": nbytes / put_med / 1e6 if put_med > 0 else 0,
        "get_gbs_median": nbytes / get_med / 1e6 if get_med > 0 else 0,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark across all requested payload/field/config
    combinations."""
    print("=" * 80)
    print("TransferQueue RDMA Benchmark")
    print(f"  configs: {args.configs}")
    print(f"  payload sizes: {args.payload_mib} MiB")
    print(f"  samples: {args.num_samples}, fields: {args.num_fields}")
    print(f"  repeats: {args.repeats}, dtype: {args.dtype}")
    print(f"  device: {args.device or 'auto'}")
    print("=" * 80)

    all_results = []

    for total_mib in args.payload_mib:
        for nf in args.num_fields:
            payload = make_payload(args.num_samples, nf, total_mib, args.dtype)
            actual_mib = payload_bytes(payload) / 1024 / 1024
            print(f"\n--- {actual_mib:.1f} MiB / {args.num_samples} samples / {nf} fields ---")

            for cfg in args.configs:
                try:
                    result = run_config(cfg, payload, args)
                    all_results.append(result)
                    print(
                        f"  [{cfg}] MEDIAN: put={result['put_ms_median']:.1f}ms "
                        f"({result['put_gbs_median']:.2f} GB/s)  "
                        f"get={result['get_ms_median']:.1f}ms ({result['get_gbs_median']:.2f} GB/s)"
                    )
                except Exception as e:
                    print(f"  [{cfg}] FAILED: {e}")
                    all_results.append(
                        {
                            "config": cfg,
                            "backend": CONFIG_MAP[cfg]["backend"],
                            "protocol": CONFIG_MAP[cfg]["protocol"],
                            "error": str(e),
                            "payload_mib": actual_mib,
                            "num_fields": nf,
                        }
                    )

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY (median throughput)")
    print(
        f"{'Config':<6} {'Backend':<16} {'Proto':<6} {'Payload':>8} {'Fields':>7} "
        f"{'put ms':>8} {'put GB/s':>9} {'get ms':>8} {'get GB/s':>9}"
    )
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['config']:<6} {r['backend']:<16} {r['protocol']:<6} {'ERR':>8}")
            continue
        actual_mib = r["nbytes"] / 1024 / 1024
        print(
            f"{r['config']:<6} {r['backend']:<16} {r['protocol']:<6} "
            f"{actual_mib:>7.1f}M {'?':>7} "
            f"{r['put_ms_median']:>8.1f} {r['put_gbs_median']:>9.2f} "
            f"{r['get_ms_median']:>8.1f} {r['get_gbs_median']:>9.2f}"
        )

    # RDMA net benefit if both C1 and C2 present
    c1_results = [r for r in all_results if r["config"] == "C1" and "error" not in r]
    c2_results = [r for r in all_results if r["config"] == "C2" and "error" not in r]
    if c1_results and c2_results:
        print("\n--- RDMA net benefit (C2 - C1, same backend, protocol only) ---")
        for c1, c2 in zip(c1_results, c2_results):
            put_delta = c2["put_gbs_median"] - c1["put_gbs_median"]
            get_delta = c2["get_gbs_median"] - c1["get_gbs_median"]
            put_pct = (put_delta / c1["put_gbs_median"] * 100) if c1["put_gbs_median"] > 0 else 0
            get_pct = (get_delta / c1["get_gbs_median"] * 100) if c1["get_gbs_median"] > 0 else 0
            print(f"  put: {put_delta:+.2f} GB/s ({put_pct:+.0f}%)  get: {get_delta:+.2f} GB/s ({get_pct:+.0f}%)")

    # CSV output
    if args.output_csv:
        import csv

        with open(args.output_csv, "w", newline="") as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        print(f"\nCSV written to {args.output_csv}")

    print("\n[dataplane] benchmark complete")


def main() -> None:
    """Initialize Ray before touching named actors and always clean up."""
    import ray

    args = parse_args()
    ray.init(ignore_reinit_error=True)
    try:
        run_benchmark(args)
    finally:
        # Tear down the last config's controller so a subsequent benchmark run
        # starts from a clean slate, then release the local Ray runtime.
        try:
            close_tq_and_wait()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
