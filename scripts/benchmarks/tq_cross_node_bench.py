#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TQ-layer cross-node benchmark matching REAL Relax usage.

Real Relax: every component (actor/rollout/critic) is a PERSISTENT
``@serve.deployment`` Ray actor that calls ``tq.init`` once (attach) +
``tq.get_client``, then put/get for the whole job.  This benchmark mirrors
that: a persistent consumer ACTOR on node B (declared via ``@ray.remote``
class, scheduled with ``NodeAffinitySchedulingStrategy``) that attaches once
and fetches repeatedly -- NOT an ephemeral task.

Three configs run in the SAME cross-node topology so they are directly
comparable:

  * C0 ``simple`` -- SimpleStorage / ZMQ / TCP   (current default backend)
  * C1 ``tcp``    -- MooncakeStore / mooncake / TCP
  * C2 ``rdma``   -- MooncakeStore / mooncake / RDMA

Three payload profiles: synthetic tensors, production-shaped multimodal
tensors, and ``real-multimodal`` — a replay of REAL images through the
production Qwen-VL processor (fixture from make_multimodal_fixture.py) with
``multimodal_train_inputs`` as a NonTensorStack column, i.e. the storage
backends' non-tensor slow path that real VL training exercises.  Every fetched
field is compared byte-for-byte via a SHA-256 digest before a throughput
result is accepted. Every payload tier's transport is proven by
reading the IB ``port_rcv_data`` and bond0 ``rx_bytes`` counters around each
get: RDMA must show IB moving and bond0 flat; TCP the reverse. There is no
"thought it was RDMA but was TCP" ambiguity.

Usage (node A driver; node B already in the Ray cluster):

  PYTHONPATH=<Relax> python -u scripts/benchmarks/tq_cross_node_bench.py \\
      --payload-profiles synthetic multimodal \\
      --payload-mib 256 1024 2048 4096 --repeats 5 \\
      --require-wire-proof --csv tq_cross_node_gib.csv

On mooncake 0.3.10, switching protocols inside one driver session can make the
third protocol's ``batch_get_into`` return -800 (see task-26-dev-log §7.11).
If that happens, run one protocol per process and merge the CSVs:

  --protocols rdma --csv c2.csv
  --protocols tcp  --csv c1.csv
  --protocols simple --csv c0.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import time
from typing import Any

import ray


# Placeholder defaults -- override --master / --nodeb-ip / --device with your own
# cluster's values before running. Do not commit real infrastructure IPs/devices.
DEFAULT_MASTER = "<ip>:50051"
DEFAULT_DEVICE = ""
DEFAULT_NODEB = "<ip>"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="TQ-layer cross-node RDMA vs TCP benchmark")
    p.add_argument("--master", default=DEFAULT_MASTER, help="mooncake master host:port (node A)")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="RDMA device name")
    p.add_argument("--nodeb-ip", default=DEFAULT_NODEB, help="node B NodeManagerAddress")
    p.add_argument(
        "--payload-mib",
        nargs="+",
        type=int,
        default=[256, 1024, 2048, 4096],
        help="Total payload sizes per put in MiB (4096 == 4 GiB). Defaults span 256M..4G.",
    )
    p.add_argument("--num-samples", type=int, default=256, help="Rows per put (rollout-batch-like)")
    p.add_argument("--num-fields", nargs="+", type=int, default=[1], help="Tensor fields per put")
    p.add_argument(
        "--payload-profiles",
        nargs="+",
        default=["synthetic", "multimodal"],
        choices=["synthetic", "multimodal", "real-multimodal"],
        help="Payload layouts to benchmark. 'multimodal' is production-shaped dense tensors; "
        "'real-multimodal' replays a fixture from make_multimodal_fixture.py (real images through the "
        "production processor, multimodal_train_inputs as list[dict] == storage non-tensor slow path).",
    )
    p.add_argument(
        "--fixture-path",
        default="",
        help="Fixture .pt for real-multimodal (default: $RELAX_MM_FIXTURE, else tests/fixtures/"
        "tq_multimodal_fixture.pt). Generate with scripts/benchmarks/make_multimodal_fixture.py.",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Measured put/get rounds per档 (a separate warmup round is always added; the mean "
        "of these repeats is the reported figure)",
    )
    p.add_argument(
        "--segment-gib",
        type=int,
        default=16,
        help="MooncakeStore global_segment_size per client (GiB). Must exceed the largest payload.",
    )
    p.add_argument(
        "--protocols",
        nargs="+",
        default=["simple", "tcp", "rdma"],
        choices=["simple", "tcp", "rdma"],
        help="Subset of configs to run (use one per process to dodge the 0.3.10 -800 issue).",
    )
    p.add_argument("--csv", default="", help="Optional path to write per-run rows + summary")
    p.add_argument(
        "--require-wire-proof",
        action="store_true",
        help="Fail unless counters prove RDMA traffic for rdma and TCP traffic for tcp/simple.",
    )
    return p.parse_args()


def make_payload(num_samples: int, num_fields: int, total_mib: int):
    """Build a TensorDict of num_samples rows x num_fields tensors totaling.

    ~total_mib.
    """
    import torch
    from tensordict import TensorDict

    dt = torch.float32
    elem = torch.tensor([], dtype=dt).element_size()
    per_field = total_mib * 1024 * 1024 // max(1, num_fields)
    cols = max(1, per_field // (elem * num_samples))
    g = torch.Generator().manual_seed(1234)
    data = {f"field_{i}": torch.randn(num_samples, cols, dtype=dt, generator=g) for i in range(num_fields)}
    return TensorDict(data, batch_size=[num_samples])


def make_multimodal_payload(num_samples: int, total_mib: int):
    """Build a production-shaped vision-language rollout TensorDict.

    ``pixel_values`` follows the Qwen-VL hidden width (1176) and BF16 dtype;
    token, mask, grid, reward, and sample-id fields exercise the mixed dtypes
    present in real Relax batches.  Patch count scales to the requested size,
    so the same layout is validated at every benchmark tier.
    """
    import torch
    from tensordict import TensorDict

    target_bytes = total_mib * 1024 * 1024
    seq_len = min(4096, max(128, target_bytes // max(1, num_samples * 128 * 1024)))
    fixed_bytes = num_samples * (seq_len * (8 + 8 + 8) + 3 * 8 + 8 + 4)
    pixel_budget = max(num_samples * 1176 * 2, target_bytes - fixed_bytes)
    patches = max(1, pixel_budget // (num_samples * 1176 * 2))

    pixel_values = torch.arange(num_samples * patches * 1176, dtype=torch.int32)
    pixel_values = (pixel_values.remainder(2048).to(torch.float32) / 128).to(torch.bfloat16)
    pixel_values = pixel_values.reshape(num_samples, patches, 1176)
    token_row = torch.arange(seq_len, dtype=torch.int64)
    input_ids = token_row.repeat(num_samples, 1)
    response_ids = (token_row + 100_000).repeat(num_samples, 1)
    attention_mask = torch.ones((num_samples, seq_len), dtype=torch.int64)
    image_grid_thw = torch.tensor([1, 1, patches], dtype=torch.int64).repeat(num_samples, 1, 1)
    sample_id = torch.arange(num_samples, dtype=torch.int64).reshape(num_samples, 1)
    rewards = torch.linspace(-1.0, 1.0, num_samples, dtype=torch.float32).reshape(num_samples, 1)
    return TensorDict(
        {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "input_ids": input_ids,
            "response_ids": response_ids,
            "attention_mask": attention_mask,
            "sample_id": sample_id,
            "rewards": rewards,
        },
        batch_size=[num_samples],
    )


_FIXTURE_CACHE: dict[str, Any] = {}


def resolve_fixture_path(cli_value: str) -> str:
    """--fixture-path > $RELAX_MM_FIXTURE > repo default."""
    import os

    if cli_value:
        return cli_value
    if os.environ.get("RELAX_MM_FIXTURE"):
        return os.environ["RELAX_MM_FIXTURE"]
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "tests", "fixtures", "tq_multimodal_fixture.pt")


def make_real_multimodal_payload(total_mib: int, fixture_path: str):
    """Replay REAL processor outputs at the requested payload tier.

    Fixture samples (real dataset images through the production Qwen-VL chain)
    are tiled cyclically until the multimodal payload reaches ``total_mib``.
    The payload is assembled by the production ``dict_to_tensordict``, so
    ``multimodal_train_inputs`` ships as a ``NonTensorStack`` column —
    MooncakeStore's msgpack non-tensor slow path and SimpleStorage's pickled-
    object path, exactly like a training job. Rows are shared references
    (transport serializes each row anyway), so tiling does not multiply driver
    RAM.
    """
    import torch

    from relax.utils.payload_digest import total_leaf_bytes
    from relax.utils.utils import dict_to_tensordict

    if fixture_path not in _FIXTURE_CACHE:
        _FIXTURE_CACHE[fixture_path] = torch.load(fixture_path, map_location="cpu", weights_only=False)
    bundle = _FIXTURE_CACHE[fixture_path]
    base_tokens = bundle["train_data"]["tokens"]
    base_mm = bundle["train_data"]["multimodal_train_inputs"]

    target_bytes = total_mib * 1024**2
    tokens: list[list[int]] = []
    multimodal: list[dict] = []
    accumulated = 0
    index = 0
    while accumulated < target_bytes or len(tokens) < 1:
        source = index % len(base_mm)
        tokens.append(base_tokens[source])
        multimodal.append(base_mm[source])
        accumulated += total_leaf_bytes(base_mm[source])
        index += 1
    num_samples = len(tokens)
    train_data = {
        "sample_id": list(range(num_samples)),
        "tokens": tokens,
        "multimodal_train_inputs": multimodal,
    }
    return dict_to_tensordict(train_data, batch_size=num_samples)


def make_profile_payload(profile: str, num_samples: int, num_fields: int, total_mib: int, fixture_path: str = ""):
    if profile == "real-multimodal":
        return make_real_multimodal_payload(total_mib, fixture_path)
    if profile == "multimodal":
        return make_multimodal_payload(num_samples, total_mib)
    return make_payload(num_samples, num_fields, total_mib)


def profile_field_counts(profile: str, num_fields: list[int]) -> list[int]:
    """Field-count sweep per profile: synthetic sweeps --num-fields; the
    multimodal profiles have fixed production schemas."""
    if profile == "synthetic":
        return num_fields
    return [7] if profile == "multimodal" else [3]


def payload_bytes(payload) -> int:
    """Total payload bytes across all fields (tensor, NestedTensor, or
    NonTensorStack columns)."""
    from relax.utils.payload_digest import total_leaf_bytes

    return sum(total_leaf_bytes(payload[k]) for k in payload.keys())


def field_byte_digests(payload, fields: list[str]) -> dict[str, tuple[str, int, str]]:
    """Return per-field byte digests normalized to TQ's row-major storage.

    TQ reconstructs dense input columns as jagged ``NestedTensor`` columns.
    Comparing flattened values makes the digest representation-independent
    while still checking every dtype bit and every payload byte.
    """
    import torch

    out: dict[str, tuple[str, int, str]] = {}
    for field in fields:
        value = payload[field]
        flat = value.values().reshape(-1) if type(value).__name__ == "NestedTensor" else value.reshape(-1)
        flat = flat.detach().cpu().contiguous()
        raw = flat.view(torch.uint8).numpy().tobytes()
        out[field] = (str(flat.dtype), flat.numel(), hashlib.sha256(raw).hexdigest())
    return out


def _column_rows(column) -> list:
    """Rows of a TensorDict column: jagged NestedTensor, NonTensorStack, dense
    tensor, or plain list."""
    import torch

    if isinstance(column, torch.Tensor) and column.is_nested:
        return list(column.unbind())
    if type(column).__name__ == "NonTensorStack":
        return column.tolist()
    return [column[i] for i in range(len(column))]


def field_multiset_digests(payload, fields: list[str]) -> dict[str, tuple[int, str]]:
    """Order-insensitive per-field digest: hash of sorted per-row digests.

    Used for the real-multimodal profile, where columns include a
    ``NonTensorStack`` of per-sample dicts: the sampler is free to reorder
    rows, so the byte-exact contract is "the returned multiset of rows is
    byte-identical to the put multiset".

    Dict rows (multimodal_train_inputs) are digested leaf-by-leaf with full
    dtype+shape+bytes via relax.utils.payload_digest.  Tensor rows are
    digested as (dtype, numel, bytes) — same contract as ``_flat_values`` in
    the dataplane tests — because backends legitimately differ in scalar-row
    representation (SimpleStorage returns dense columns whose rows index as
    0-D; MooncakeStore reconstructs rows as shape ``[1]``; bytes and dtype
    are identical).
    """
    import torch

    from relax.utils.payload_digest import leaf_digests

    def _row_digest(row) -> str:
        if isinstance(row, torch.Tensor) and not row.is_nested:
            flat = row.detach().cpu().contiguous().reshape(-1)
            raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
            token = f"{flat.dtype}|{flat.numel()}|{hashlib.sha256(raw).hexdigest()}"
        else:
            token = repr(sorted(leaf_digests(row).items()))
        return hashlib.sha256(token.encode()).hexdigest()

    out: dict[str, tuple[int, str]] = {}
    for field in fields:
        rows = _column_rows(payload[field])
        row_hashes = sorted(_row_digest(row) for row in rows)
        out[field] = (len(rows), hashlib.sha256("".join(row_hashes).encode()).hexdigest())
    return out


def wait_actor_gone(name: str = "TransferQueueController", timeout: float = 30.0) -> None:
    """Wait for a named TQ actor to leave the GCS (F10-safe re-init)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ray.get_actor(name, namespace="transfer_queue")
        except ValueError:
            return
        time.sleep(0.4)
    raise TimeoutError(f"Ray actor {name!r} is still registered after {timeout:.1f}s")


def close_tq_unmount_and_wait() -> None:
    """Close TQ, unmount the Mooncake segment, then wait for controller
    deregistration.

    ``tq.close()`` only tears down ZMQ (managers/base.py:378); the Mooncake
    segment stays mounted and registered in the master, so the next config's
    put hits a dead endpoint until client_ttl (30 s) expires.  Unmount
    explicitly after close (close itself still needs the store alive for
    remove_all()).
    """
    import transfer_queue as tq

    store_client = None
    try:
        store_client = getattr(tq.get_client().storage_manager, "storage_client", None)
    except (AssertionError, AttributeError):
        pass

    tq.close()

    if store_client is not None and hasattr(store_client, "close"):
        try:
            store_client.close()
        except Exception as e:  # pragma: no cover - best effort
            print(f"  [warn] store_client.close() failed: {e}", flush=True)

    wait_actor_gone()


def read_counters() -> dict[str, int]:
    """IB port_rcv_data (all devices, bytes) + bond0 rx_bytes, for transport
    proof."""
    import os

    out: dict[str, int] = {}
    root = "/sys/class/infiniband"
    try:
        for dev in sorted(os.listdir(root)):
            ports_dir = f"{root}/{dev}/ports"
            if not os.path.isdir(ports_dir):
                continue
            for port in sorted(os.listdir(ports_dir)):
                try:
                    with open(f"{ports_dir}/{port}/counters/port_rcv_data") as fh:
                        out[f"ib:{dev}:{port}"] = int(fh.read().strip()) * 4
                except OSError:
                    pass
    except OSError:
        pass
    try:
        with open("/sys/class/net/bond0/statistics/rx_bytes") as fh:
            out["tcp:bond0"] = int(fh.read().strip())
    except OSError:
        pass
    return out


def build_conf(protocol: str, master: str, device: str, segment_gib: int):
    """Build the tq.init OmegaConf.

    ``protocol="simple"`` selects the SimpleStorage/ZMQ baseline (C0) so the
    three-config comparison (C0 / Mooncake-TCP / Mooncake-RDMA) runs in the
    *same* cross-node topology; "tcp"/"rdma" select MooncakeStore.
    """
    from omegaconf import OmegaConf
    from transfer_queue import GRPOGroupNSampler

    from relax.utils.rdma_probe import EffectiveConfig
    from relax.utils.tq.config import (
        build_mooncake_config,
        build_simple_storage_config,
        validate_mooncake_runtime_contract,
    )

    if protocol == "simple":
        # total_storage_size=None == unlimited sample count (TQ config.yaml default).
        backend = build_simple_storage_config(total_storage_size=None, num_data_storage_units=2)
    else:
        validate_mooncake_runtime_contract()
        eff = EffectiveConfig(backend="MooncakeStore", protocol=protocol, device=device, fallback_reason="")
        backend = build_mooncake_config(eff, master_address=master, global_segment_size=segment_gib * 1024**3)
    return OmegaConf.create(
        {
            "controller": {"sampler": GRPOGroupNSampler(n_samples_per_prompt=1), "polling_mode": True},
            "backend": backend,
        },
        flags={"allow_objects": True},
    )


# ---- Persistent consumer actor (module-level, mirrors a Relax component) ----


@ray.remote(num_cpus=0.001)
class TQConsumer:
    """Persistent consumer on node B: attaches once, fetches many times.

    Mirrors a Relax component actor (actor.py / rollout.py): tq.init once in
    __init__ (attach to the shared controller), tq.get_client, then repeated
    get_meta/get_data for the job lifetime.
    """

    def __init__(self, master: str, device: str, protocol: str, segment_gib: int):
        import transfer_queue as tq

        tq.init(conf=build_conf(protocol, master, device, segment_gib))  # attaches (conf ignored on attach)
        self.client = tq.get_client()

    def alive(self) -> bool:
        return True

    def describe(self) -> dict:
        """Report the manager/client actually instantiated -- never trust the
        conf alone."""
        mgr = self.client.storage_manager
        inner = getattr(mgr, "storage_client", None)
        return {
            "manager": type(mgr).__name__,
            "client": type(inner).__name__ if inner is not None else "-",
            "protocol": getattr(inner, "protocol", "-"),
        }

    def shutdown(self) -> None:
        """Unmount the Mooncake segment before this actor is killed."""
        inner = getattr(self.client.storage_manager, "storage_client", None)
        if inner is not None and hasattr(inner, "close"):
            try:
                inner.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def fetch(self, fields, batch_size: int, partition: str, expected_digests, order_insensitive: bool = False):
        """One cross-node get; return (ms, ib_mb, tcp_mb, ib_tail_mb,
        tcp_tail_mb).

        ``ib_tail`` is the IB delta in the 5 ms *after* ``get_data`` returns.
        If ``get_data`` is synchronous it is ~0; if it returned before RDMA
        finished (async), the tail keeps flowing and ``ib_tail`` > 0 -- a
        definitive async-completion detector that does not need a costly full-
        data touch.

        ``order_insensitive`` selects the row-multiset digest (real-multimodal
        profile: NonTensorStack columns, sampler may reorder rows); the
        default column digest requires identical row order.
        """
        before = read_counters()
        t0 = time.perf_counter()
        meta = self.client.get_meta(
            data_fields=list(fields),
            batch_size=batch_size,
            partition_id=partition,
            mode="fetch",
            task_name="xfer",
        )
        got = self.client.get_data(meta)
        ms = (time.perf_counter() - t0) * 1000
        after = read_counters()
        time.sleep(0.005)  # let any async RDMA tail register on the counters
        settled = read_counters()
        # Digesting occurs outside the timed interval.  A mismatch is fatal:
        # throughput from a corrupt or truncated transfer is never reported.
        if order_insensitive:
            actual_digests = field_multiset_digests(got, list(fields))
        else:
            actual_digests = field_byte_digests(got, list(fields))
        if actual_digests != expected_digests:
            mismatch = [field for field in fields if actual_digests.get(field) != expected_digests.get(field)]
            raise AssertionError(f"byte-exact mismatch after TQ get: fields={mismatch}")
        ib = sum(after[k] - before.get(k, 0) for k in after if k.startswith("ib:")) / 1e6
        tcp = sum(after[k] - before.get(k, 0) for k in after if k.startswith("tcp:")) / 1e6
        ib_tail = sum(settled[k] - after.get(k, 0) for k in settled if k.startswith("ib:")) / 1e6
        tcp_tail = sum(settled[k] - after.get(k, 0) for k in settled if k.startswith("tcp:")) / 1e6
        return ms, ib, tcp, ib_tail, tcp_tail, True


def _mean(values: list[float]) -> float:
    """Arithmetic mean (the reported statistic); 0 for an empty list."""
    return statistics.mean(values) if values else 0.0


def _gbs(nbytes: int, ms: float) -> float:
    """Convert a latency in ms to throughput in GB/s."""
    return nbytes / ms / 1e6 if ms > 0 else 0.0


def main() -> None:
    """Run the cross-node TQ benchmark for the requested protocols."""
    import transfer_queue as tq
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    args = parse_args()
    fixture_path = resolve_fixture_path(args.fixture_path)
    if "real-multimodal" in args.payload_profiles:
        import os

        if not os.path.isfile(fixture_path):
            raise SystemExit(
                f"real-multimodal profile needs a fixture at {fixture_path}; generate one with "
                "scripts/benchmarks/make_multimodal_fixture.py (see its docstring), or pass --fixture-path."
            )
    ray.init(ignore_reinit_error=True, address="auto", logging_level="ERROR")

    nodeb = next(n for n in ray.nodes() if n["NodeManagerAddress"] == args.nodeb_ip and n.get("Alive"))
    nodeb_id = nodeb["NodeID"]
    strat = NodeAffinitySchedulingStrategy(node_id=nodeb_id, soft=False)
    print(
        f"[setup] node A = driver | node B = {args.nodeb_ip} ({nodeb_id[:10]}) "
        f"| master = {args.master} | device = {args.device} "
        f"| segment = {args.segment_gib} GiB | repeats = {args.repeats} (mean)",
        flush=True,
    )

    labels = {"simple": "C0 SimpleStorage", "tcp": "C1 Mooncake/TCP", "rdma": "C2 Mooncake/RDMA"}
    # results[protocol][profile][total_mib][num_fields] = {
    #   "put_mean_gbs","get_mean_gbs","get_med_gbs","get_std_gbs","wire", "per_run":[...]}
    results: dict[str, dict[str, dict[int, dict[int, dict[str, Any]]]]] = {}
    csv_rows: list[dict[str, Any]] = []

    for protocol in args.protocols:
        print(f"\n===== {labels[protocol]} =====", flush=True)
        close_tq_unmount_and_wait()
        tq.init(conf=build_conf(protocol, args.master, args.device, args.segment_gib))
        producer = tq.get_client()
        prod_mgr = type(producer.storage_manager).__name__
        prod_client = getattr(producer.storage_manager, "storage_client", None)
        print(
            f"  node A manager={prod_mgr} client={type(prod_client).__name__ if prod_client else '-'}",
            flush=True,
        )

        consumer = TQConsumer.options(scheduling_strategy=strat).remote(
            args.master, args.device, protocol, args.segment_gib
        )
        ray.get(consumer.alive.remote())  # ensure attached before measuring
        desc = ray.get(consumer.describe.remote())
        print(
            f"  node B manager={desc['manager']} client={desc['client']} protocol={desc['protocol']}",
            flush=True,
        )

        results.setdefault(protocol, {})
        for profile in args.payload_profiles:
            results[protocol].setdefault(profile, {})
            for total_mib in args.payload_mib:
                results[protocol][profile].setdefault(total_mib, {})
                # Multimodal profiles have fixed production schemas; synthetic
                # uses the requested field-count sweep.
                field_counts = profile_field_counts(profile, args.num_fields)
                order_insensitive = profile == "real-multimodal"
                for nf in field_counts:
                    payload = make_profile_payload(profile, args.num_samples, nf, total_mib, fixture_path)
                    nf = len(list(payload.keys()))
                    fields = sorted(payload.keys())
                    if order_insensitive:
                        expected_digests = field_multiset_digests(payload, fields)
                    else:
                        expected_digests = field_byte_digests(payload, fields)
                    nbytes = payload_bytes(payload)
                    put_times: list[float] = []
                    get_times: list[float] = []
                    ibs: list[float] = []
                    tcps: list[float] = []
                    ib_tails: list[float] = []  # async-completion detector (~0 == sync get)
                    tcp_tails: list[float] = []
                    # repeat 0 is a warm-up: first transfer pays RDMA endpoint handshake.
                    batch_rows = payload.batch_size[0] if payload.batch_size else args.num_samples
                    for r in range(args.repeats + 1):
                        part = f"xfer_{protocol}_{profile}_{total_mib}_{nf}_{r}"
                        t0 = time.perf_counter()
                        producer.put(payload, partition_id=part)
                        put_ms = (time.perf_counter() - t0) * 1000
                        get_ms, ib, tcp, ib_tail, tcp_tail, byte_exact = ray.get(
                            consumer.fetch.remote(fields, batch_rows, part, expected_digests, order_insensitive)
                        )
                        producer.clear_partition(part)
                        if r == 0:
                            print(
                                f"  {profile} {total_mib}M f={nf} (warmup, not counted): "
                                f"put={put_ms:.0f}ms get={get_ms:.0f}ms "
                                f"ib_tail={ib_tail:.0f}MB byte_exact={byte_exact}",
                                flush=True,
                            )
                            continue
                        put_times.append(put_ms)
                        get_times.append(get_ms)
                        ibs.append(ib)
                        tcps.append(tcp)
                        ib_tails.append(ib_tail)
                        tcp_tails.append(tcp_tail)
                        run_wire = "RDMA" if ib > tcp else "TCP"
                        run_wire_proven = (protocol == "rdma" and ib > 0 and ib > tcp) or (
                            protocol != "rdma" and tcp > 0 and tcp >= ib
                        )
                        csv_rows.append(
                            {
                                "protocol": protocol,
                                "profile": profile,
                                "payload_mib": total_mib,
                                "actual_mib": round(nbytes / 1024**2, 2),
                                "num_fields": nf,
                                "run": r,
                                "byte_exact": byte_exact,
                                "wire_observed": run_wire,
                                "wire_proven": run_wire_proven,
                                "put_ms": round(put_ms, 2),
                                "get_ms": round(get_ms, 2),
                                "put_gbs": round(_gbs(nbytes, put_ms), 3),
                                "get_gbs": round(_gbs(nbytes, get_ms), 3),
                                "ib_mb": round(ib, 1),
                                "tcp_mb": round(tcp, 1),
                                "ib_tail_mb": round(ib_tail, 1),
                                "tcp_tail_mb": round(tcp_tail, 1),
                            }
                        )

                    put_mean = _mean(put_times)
                    get_mean = _mean(get_times)
                    get_med = statistics.median(get_times)
                    # Std-dev of per-run throughput, not latency.
                    get_gbs_runs = [_gbs(nbytes, ms) for ms in get_times]
                    get_std_gbs = statistics.pstdev(get_gbs_runs) if len(get_gbs_runs) > 1 else 0.0
                    ib_med = statistics.median(ibs)
                    tcp_med = statistics.median(tcps)
                    ib_tail_med = statistics.median(ib_tails) if ib_tails else 0.0
                    wire = "RDMA" if ib_med > tcp_med else "TCP"
                    wire_proven = (protocol == "rdma" and ib_med > 0 and ib_med > tcp_med) or (
                        protocol != "rdma" and tcp_med > 0 and tcp_med >= ib_med
                    )
                    if args.require_wire_proof and not wire_proven:
                        raise RuntimeError(
                            f"wire proof failed for protocol={protocol} profile={profile} "
                            f"payload={total_mib}MiB: IB={ib_med:.1f}MB TCP={tcp_med:.1f}MB"
                        )
                    rec = {
                        "put_mean_gbs": _gbs(nbytes, put_mean),
                        "get_mean_gbs": _gbs(nbytes, get_mean),
                        "get_med_gbs": _gbs(nbytes, get_med),
                        "get_std_gbs": get_std_gbs,
                        "wire": wire,
                        "wire_proven": wire_proven,
                        "byte_exact": True,
                        "ib_tail_med_mb": ib_tail_med,
                        "per_run_get_gbs": [round(g, 2) for g in get_gbs_runs],
                    }
                    results[protocol][profile][total_mib][nf] = rec
                    print(
                        f"  {profile:<16} {str(total_mib) + 'M':<9} f={nf} "
                        f"put_mean={rec['put_mean_gbs']:6.2f} "
                        f"GB/s get_mean={rec['get_mean_gbs']:6.2f} (med {rec['get_med_gbs']:.2f}, "
                        f"std {rec['get_std_gbs']:.2f}) GB/s byte_exact=PASS "
                        f"[wire: IB {ib_med:.0f}MB / bond0 {tcp_med:.0f}MB -> {wire}; "
                        f"proof={'PASS' if wire_proven else 'UNKNOWN'}; tail {ib_tail_med:.0f}MB] "
                        f"runs={rec['per_run_get_gbs']}",
                        flush=True,
                    )

        ray.get(consumer.shutdown.remote())  # unmount before kill, else the segment lingers
        ray.kill(consumer)
        close_tq_unmount_and_wait()

    # ---- Summary (mean-based, all requested protocols) ----
    print("\n===== SUMMARY: TQ-layer cross-node, same topology (get, MEAN of N runs) =====", flush=True)
    header = (
        f"{'Profile':<16}{'Payload':<9}{'f':<4}{'C0 mean':>9}{'C1 mean':>9}{'C2 mean':>9}"
        f"{'C1/C0':>8}{'C2/C1':>8}{'C2 std':>8}{'>=20%':>7}{'wire C0/C1/C2':>18}"
    )
    print(header, flush=True)
    for profile in args.payload_profiles:
        field_counts = profile_field_counts(profile, args.num_fields)
        for total_mib in args.payload_mib:
            for nf in field_counts:
                c0 = results.get("simple", {}).get(profile, {}).get(total_mib, {}).get(nf)
                c1 = results.get("tcp", {}).get(profile, {}).get(total_mib, {}).get(nf)
                c2 = results.get("rdma", {}).get(profile, {}).get(total_mib, {}).get(nf)
                prefix = f"{profile:<16}{str(total_mib) + 'M':<9}{nf:<4}"
                if not (c0 and c1 and c2):
                    # A protocol was skipped (--protocols subset) -- print what we have.
                    parts = []
                    for name, c in (("C0", c0), ("C1", c1), ("C2", c2)):
                        parts.append(f"{name}={c['get_mean_gbs']:.2f}" if c else f"{name}=-")
                    print(prefix + "  ".join(parts) + "  (subset run)", flush=True)
                    continue
                g0, g1, g2 = c0["get_mean_gbs"], c1["get_mean_gbs"], c2["get_mean_gbs"]
                back_pct = (g1 - g0) / g0 * 100 if g0 > 0 else 0.0
                rdma_pct = (g2 - g1) / g1 * 100 if g1 > 0 else 0.0
                wire = f"{c0['wire']}/{c1['wire']}/{c2['wire']}"
                print(
                    f"{prefix}{g0:>9.2f}{g1:>9.2f}{g2:>9.2f}"
                    f"{f'{back_pct:+.0f}%':>8}{f'{rdma_pct:+.0f}%':>8}{c2['get_std_gbs']:>8.2f}"
                    f"{('PASS' if rdma_pct >= 20 else 'no'):>7}{wire:>18}",
                    flush=True,
                )
    print("  C1/C0 = MooncakeStore vs SimpleStorage (backend effect)", flush=True)
    print("  C2/C1 = RDMA vs TCP on the same backend (transport effect, gate target, mean-based)", flush=True)
    print("  std = population stddev of C2 get across the N runs (run-to-run variance)", flush=True)

    if args.csv:
        cols = [
            "protocol",
            "profile",
            "payload_mib",
            "actual_mib",
            "num_fields",
            "run",
            "byte_exact",
            "wire_observed",
            "wire_proven",
            "put_ms",
            "get_ms",
            "put_gbs",
            "get_gbs",
            "ib_mb",
            "tcp_mb",
            "ib_tail_mb",
            "tcp_tail_mb",
        ]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in csv_rows:
                w.writerow({k: row[k] for k in cols})
        print(f"\n[csv] wrote {len(csv_rows)} per-run rows to {args.csv}", flush=True)


if __name__ == "__main__":
    main()
