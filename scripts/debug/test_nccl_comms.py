# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""NCCL collectives + P2P smoke test on an existing Ray cluster.

Spawns ``world_size`` GPU actors (1 GPU each), initializes ``torch.distributed``
with the NCCL backend, builds TP / PP / DP subgroups following Megatron's rank
layout (``rank = pp * tp * dp + dp * tp + tp_idx``), and runs a battery of
collective + P2P ops with per-op timeout detection.

The primary use case is diagnosing a training hang whose py-spy stack points
at ``ncclGroupEndInternal (group.cc:694) / asyncJobLaunch`` — i.e. NCCL is
stuck setting up P2P channels. This test isolates whether NCCL init + basic
collectives + PP-style ``batch_isend_irecv`` work on the same cluster, without
the model / dataloader in the way.

Launch (via ray-job entrypoint)::

    bash scripts/entrypoint/ray-job.sh \\
        scripts/debug/run-test-nccl-comms.sh \\
        --world-size 16 --tp 4 --pp 2

Or direct (env already set up by ray-job.sh)::

    ray job submit --address="http://127.0.0.1:8265" \\
        --runtime-env-json="${RUNTIME_ENV_JSON}" \\
        -- python3 scripts/debug/test_nccl_comms.py --world-size 16 --tp 4 --pp 2
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from datetime import timedelta
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote
class NCCLTestActor:
    """A single-GPU worker that participates in torch.distributed
    collectives."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        tp: int,
        pp: int,
        master_port: int,
        init_timeout_s: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.tp = tp
        self.pp = pp
        assert world_size % (tp * pp) == 0, f"world_size {world_size} not divisible by tp*pp={tp * pp}"
        self.dp = world_size // (tp * pp)
        self.master_port = master_port
        self.init_timeout_s = init_timeout_s
        self.master_addr: str | None = None  # set by set_master() before init()

        # Derived rank indices (Megatron layout: outermost PP, middle DP, innermost TP).
        self.tp_idx = rank % tp
        self.dp_idx = (rank // tp) % self.dp
        self.pp_idx = rank // (tp * self.dp)

        self.tp_group: dist.ProcessGroup | None = None
        self.pp_group: dist.ProcessGroup | None = None
        self.dp_group: dist.ProcessGroup | None = None
        self.tp_ranks: list[int] = []
        self.pp_ranks: list[int] = []
        self.dp_ranks: list[int] = []

    # ---------- placement / diagnostics ----------
    def info(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "tp_idx": self.tp_idx,
            "dp_idx": self.dp_idx,
            "pp_idx": self.pp_idx,
            "host": socket.gethostname(),
            "ip": ray._private.services.get_node_ip_address(),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "pid": os.getpid(),
        }

    # ---------- torch.distributed init ----------
    def set_master(self, master_addr: str) -> None:
        """Called by driver after rank 0's actor IP is known."""
        self.master_addr = master_addr

    def init(self) -> dict[str, Any]:
        assert self.master_addr is not None, "call set_master() before init()"
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = "0"  # Ray gives each actor exactly 1 GPU visible as :0

        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            world_size=self.world_size,
            rank=self.rank,
            timeout=timedelta(seconds=self.init_timeout_s),
        )
        self._build_groups()
        # Warm up NCCL by touching a small tensor on WORLD.
        t = torch.ones(4, dtype=torch.float32, device="cuda")
        dist.all_reduce(t)
        torch.cuda.synchronize()
        return {
            "rank": self.rank,
            "tp_ranks": self.tp_ranks,
            "pp_ranks": self.pp_ranks,
            "dp_ranks": self.dp_ranks,
        }

    def _build_groups(self) -> None:
        tp, pp, dp = self.tp, self.pp, self.dp

        # TP groups: fix (pp_idx, dp_idx), vary tp_idx.
        for p in range(pp):
            for d in range(dp):
                ranks = [p * tp * dp + d * tp + t for t in range(tp)]
                grp = dist.new_group(ranks=ranks, backend="nccl")
                if self.rank in ranks:
                    self.tp_group = grp
                    self.tp_ranks = ranks

        # PP groups: fix (tp_idx, dp_idx), vary pp_idx.
        for d in range(dp):
            for t in range(tp):
                ranks = [p * tp * dp + d * tp + t for p in range(pp)]
                grp = dist.new_group(ranks=ranks, backend="nccl")
                if self.rank in ranks:
                    self.pp_group = grp
                    self.pp_ranks = ranks

        # DP groups: fix (tp_idx, pp_idx), vary dp_idx.
        for p in range(pp):
            for t in range(tp):
                ranks = [p * tp * dp + d * tp + t for d in range(dp)]
                grp = dist.new_group(ranks=ranks, backend="nccl")
                if self.rank in ranks:
                    self.dp_group = grp
                    self.dp_ranks = ranks

    # ---------- collectives ----------
    def _resolve_group(self, name: str) -> tuple[dist.ProcessGroup | None, list[int]]:
        if name == "world":
            return None, list(range(self.world_size))
        if name == "tp":
            return self.tp_group, self.tp_ranks
        if name == "pp":
            return self.pp_group, self.pp_ranks
        if name == "dp":
            return self.dp_group, self.dp_ranks
        raise ValueError(f"unknown group: {name}")

    def all_reduce(self, group_name: str, tensor_mb: int) -> dict[str, Any]:
        group, ranks = self._resolve_group(group_name)
        if group is not None and len(ranks) < 2:
            return {"rank": self.rank, "group": group_name, "ranks": ranks, "skipped": "size<2"}
        n = max(1, tensor_mb * 1024 * 1024 // 4)
        t = torch.full((n,), float(self.rank), dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        expected = float(sum(ranks))
        first = t[0].item()
        ok = abs(first - expected) < 1e-3
        return {
            "rank": self.rank,
            "group": group_name,
            "ranks": ranks,
            "tensor_mb": tensor_mb,
            "sec": elapsed,
            "first": first,
            "expected": expected,
            "ok": ok,
        }

    def all_gather(self, group_name: str, tensor_mb: int) -> dict[str, Any]:
        group, ranks = self._resolve_group(group_name)
        if group is not None and len(ranks) < 2:
            return {"rank": self.rank, "group": group_name, "skipped": "size<2"}
        n = max(1, tensor_mb * 1024 * 1024 // 4)
        t = torch.full((n,), float(self.rank), dtype=torch.float32, device="cuda")
        out = [torch.empty_like(t) for _ in ranks]
        torch.cuda.synchronize()
        t0 = time.time()
        dist.all_gather(out, t, group=group)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        got = [o[0].item() for o in out]
        ok = got == [float(r) for r in ranks]
        return {
            "rank": self.rank,
            "group": group_name,
            "ranks": ranks,
            "tensor_mb": tensor_mb,
            "sec": elapsed,
            "got_first": got,
            "ok": ok,
        }

    def pp_batched_isend_irecv(self, tensor_mb: int) -> dict[str, Any]:
        """Reproduce the Megatron ``_batched_p2p_ops`` pattern inside a PP
        subgroup.

        Each rank sends to its next PP neighbour and receives from its previous
        PP neighbour, all wrapped inside a single ``batch_isend_irecv``. This
        is the exact call site whose ``ncclGroupEnd`` was hanging in the
        reported training run.
        """
        if self.pp_group is None or len(self.pp_ranks) < 2:
            return {"rank": self.rank, "skipped": "pp<2"}
        local_pp = self.pp_ranks.index(self.rank)
        next_rank = self.pp_ranks[local_pp + 1] if local_pp + 1 < len(self.pp_ranks) else None
        prev_rank = self.pp_ranks[local_pp - 1] if local_pp - 1 >= 0 else None

        n = max(1, tensor_mb * 1024 * 1024 // 4)
        ops: list[dist.P2POp] = []
        recv_buf = None
        send_val = float(self.rank)
        # Pass the pp_group explicitly so batch_isend_irecv uses the same NCCL
        # comm Megatron does — otherwise the WORLD group is used, which exercises
        # a different set of NCCL channels than the hanging training run.
        if next_rank is not None:
            send_buf = torch.full((n,), send_val, dtype=torch.float32, device="cuda")
            ops.append(dist.P2POp(dist.isend, send_buf, next_rank, group=self.pp_group))
        if prev_rank is not None:
            recv_buf = torch.empty(n, dtype=torch.float32, device="cuda")
            ops.append(dist.P2POp(dist.irecv, recv_buf, prev_rank, group=self.pp_group))

        torch.cuda.synchronize()
        t0 = time.time()
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        got = recv_buf[0].item() if recv_buf is not None else None
        expected = float(prev_rank) if prev_rank is not None else None
        ok = (expected is None) or (abs(got - expected) < 1e-3)
        return {
            "rank": self.rank,
            "pp_ranks": self.pp_ranks,
            "next": next_rank,
            "prev": prev_rank,
            "tensor_mb": tensor_mb,
            "sec": elapsed,
            "got": got,
            "expected": expected,
            "ok": ok,
        }

    def barrier(self) -> int:
        dist.barrier()
        return self.rank

    def pp_1f1b_forward_only(
        self,
        tensor_mb: int,
        num_microbatches: int,
        with_tp_allgather: bool,
    ) -> dict[str, Any]:
        """Reproduce Megatron's 1F1B forward-only pipeline pattern.

        For ``num_microbatches`` iterations each rank does:
          recv_forward (from prev PP peer) → optional TP all-gather → +1
          → send_forward (to next PP peer)

        With ``with_tp_allgather=True`` this interleaves PP P2P with TP
        collectives on the same stream — the shape that actually happens in
        Megatron's forward step (sequence-parallel gather inside every layer).
        This exercises multiple concurrent NCCL comms and is much closer to
        the real hang than a bare ``batch_isend_irecv``.
        """
        if self.pp_group is None or len(self.pp_ranks) < 2:
            return {"rank": self.rank, "skipped": "pp<2"}
        n = max(1, tensor_mb * 1024 * 1024 // 4)
        local_pp = self.pp_ranks.index(self.rank)
        is_first_pp = local_pp == 0
        is_last_pp = local_pp == len(self.pp_ranks) - 1
        prev_rank = None if is_first_pp else self.pp_ranks[local_pp - 1]
        next_rank = None if is_last_pp else self.pp_ranks[local_pp + 1]

        tp_out: list[torch.Tensor] | None = None
        if with_tp_allgather and self.tp_group is not None and len(self.tp_ranks) > 1:
            tp_out = [torch.empty(n, dtype=torch.float32, device="cuda") for _ in self.tp_ranks]

        torch.cuda.synchronize()
        t0 = time.time()
        completed = 0
        for mb in range(num_microbatches):
            # ── PP recv_forward (mimics p2p_communication.recv_forward) ──
            if prev_rank is not None:
                input_t = torch.empty(n, dtype=torch.float32, device="cuda")
                reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, input_t, prev_rank, group=self.pp_group)])
                for r in reqs:
                    r.wait()
            else:
                input_t = torch.full((n,), float(mb + self.rank), dtype=torch.float32, device="cuda")

            # ── TP all-gather (mimics sequence-parallel gather inside forward) ──
            if tp_out is not None:
                dist.all_gather(tp_out, input_t, group=self.tp_group)

            # ── trivial "compute" — just add 1 so the tensor is used ──
            output_t = input_t + 1.0

            # ── PP send_forward ──
            if next_rank is not None:
                reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, output_t, next_rank, group=self.pp_group)])
                for r in reqs:
                    r.wait()
            completed = mb + 1

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        return {
            "rank": self.rank,
            "pp_ranks": self.pp_ranks,
            "num_microbatches": num_microbatches,
            "completed": completed,
            "with_tp_allgather": with_tp_allgather,
            "tensor_mb": tensor_mb,
            "sec": elapsed,
            "sec_per_mb": elapsed / max(1, completed),
        }

    def pp_p2p_stress(
        self,
        tensor_mb: int,
        iters: int,
        vary_size: bool,
    ) -> dict[str, Any]:
        """Hammer PP batched_isend_irecv ``iters`` times to catch intermittent
        hangs. When ``vary_size`` is True, tensor size cycles through.

        {0.5x, 1x, 2x} of base — exercises different NCCL chunking paths.
        """
        if self.pp_group is None or len(self.pp_ranks) < 2:
            return {"rank": self.rank, "skipped": "pp<2"}
        local_pp = self.pp_ranks.index(self.rank)
        next_rank = self.pp_ranks[local_pp + 1] if local_pp + 1 < len(self.pp_ranks) else None
        prev_rank = self.pp_ranks[local_pp - 1] if local_pp - 1 >= 0 else None
        multipliers = [1, 2, 1, 0] if vary_size else [1] * 4  # 0 handled below

        torch.cuda.synchronize()
        t0 = time.time()
        max_iter_sec = 0.0
        completed = 0
        for it in range(iters):
            mult = multipliers[it % len(multipliers)] if vary_size else 1
            eff_mb = max(1, tensor_mb * max(1, mult) // (2 if mult == 0 else 1))
            n = eff_mb * 1024 * 1024 // 4
            ops: list[dist.P2POp] = []
            recv_buf = None
            if next_rank is not None:
                send_buf = torch.full((n,), float(self.rank + it), dtype=torch.float32, device="cuda")
                ops.append(dist.P2POp(dist.isend, send_buf, next_rank, group=self.pp_group))
            if prev_rank is not None:
                recv_buf = torch.empty(n, dtype=torch.float32, device="cuda")
                ops.append(dist.P2POp(dist.irecv, recv_buf, prev_rank, group=self.pp_group))
            it0 = time.time()
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
            torch.cuda.synchronize()
            iter_sec = time.time() - it0
            max_iter_sec = max(max_iter_sec, iter_sec)
            completed = it + 1
        elapsed = time.time() - t0
        return {
            "rank": self.rank,
            "iters": iters,
            "completed": completed,
            "vary_size": vary_size,
            "base_mb": tensor_mb,
            "sec": elapsed,
            "sec_per_iter_avg": elapsed / max(1, completed),
            "sec_per_iter_max": max_iter_sec,
        }

    def teardown(self) -> bool:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        return True


def _pick_master_addr() -> str:
    """Choose master addr = first alive GPU node (prefer head)."""
    gpu_nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]
    if not gpu_nodes:
        raise RuntimeError("no alive GPU nodes in the Ray cluster")
    gpu_nodes.sort(key=lambda n: (not n.get("IsHeadNode", False), n["NodeManagerAddress"]))
    return gpu_nodes[0]["NodeManagerAddress"]


def _detect_nodes() -> int:
    return sum(1 for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0)


def _wait_or_timeout(name: str, futures: list, timeout_s: int) -> list | None:
    """Wait on ray futures, distinguishing HANG vs completion."""
    print(f"\n== {name} (timeout={timeout_s}s) ==", flush=True)
    t0 = time.time()
    try:
        results = ray.get(futures, timeout=timeout_s)
    except ray.exceptions.GetTimeoutError:
        elapsed = time.time() - t0
        # figure out which futures completed vs hung
        ready, unready = ray.wait(futures, num_returns=len(futures), timeout=0)
        done = ray.get(ready) if ready else []
        done_ranks = sorted(r.get("rank", "?") for r in done if isinstance(r, dict))
        hung_count = len(unready)
        print(
            f"  ❌ TIMEOUT after {elapsed:.1f}s — {len(done)}/{len(futures)} returned, {hung_count} hung", flush=True
        )
        if done_ranks:
            print(f"     completed ranks: {done_ranks}", flush=True)
        print("     ➜ HANG signature: py-spy the hung actors to check NCCL state", flush=True)
        return None
    elapsed = time.time() - t0
    max_sec = max((r.get("sec", 0.0) for r in results if isinstance(r, dict)), default=0.0)
    all_ok = all(r.get("ok", True) for r in results if isinstance(r, dict))
    mark = "✅" if all_ok else "⚠️ VALUE MISMATCH"
    print(f"  {mark} finished in {elapsed:.1f}s (per-rank max {max_sec:.3f}s)", flush=True)
    for r in results:
        print(f"    {r}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world-size", type=int, required=True, help="total number of GPU actors")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (innermost)")
    ap.add_argument("--pp", type=int, default=1, help="pipeline-parallel size (outermost)")
    ap.add_argument("--nodes", type=int, default=0, help="node count; 0 = auto-detect from ray cluster")
    ap.add_argument("--master-port", type=int, default=29511, help="TCP init port for torch.distributed")
    ap.add_argument("--init-timeout-s", type=int, default=180)
    ap.add_argument("--tensor-mb", type=int, default=8, help="payload size in MiB for each op")
    ap.add_argument("--big-tensor-mb", type=int, default=256, help="payload size for large-allreduce test")
    ap.add_argument("--num-microbatches", type=int, default=32, help="microbatches for pp_1f1b test")
    ap.add_argument("--stress-iters", type=int, default=200, help="iterations for pp_p2p_stress test")
    ap.add_argument("--per-op-timeout", type=int, default=60, help="seconds before an op is declared hung")
    ap.add_argument(
        "--tests",
        nargs="+",
        default=[
            "allreduce_world",
            "allreduce_tp",
            "allreduce_dp",
            "allgather_pp",
            "p2p_pp",
            "pp_1f1b_bare",
            "pp_1f1b_with_tp",
            "pp_p2p_stress",
            "allreduce_world_big",
        ],
        help="which tests to run in order",
    )
    ap.add_argument(
        "--skip-placement",
        action="store_true",
        help="skip STRICT_SPREAD placement (default forces one bundle per node)",
    )
    args = ap.parse_args()

    ray.init(address="auto", log_to_driver=True)

    nodes = args.nodes or _detect_nodes()
    if args.world_size % nodes != 0 and not args.skip_placement:
        print(
            f"WARN: world_size {args.world_size} not divisible by nodes {nodes}; falling back to --skip-placement.",
            file=sys.stderr,
        )
        args.skip_placement = True
    gpus_per_node = args.world_size // max(nodes, 1)

    master_addr_hint = _pick_master_addr()
    print(f"== master_addr hint={master_addr_hint} (actual master resolved after placement)", flush=True)
    print(
        f"== world={args.world_size} tp={args.tp} pp={args.pp} dp={args.world_size // (args.tp * args.pp)}", flush=True
    )
    print(
        f"== detected {nodes} GPU node(s), gpus_per_node={gpus_per_node} placement={'STRICT_SPREAD' if not args.skip_placement else 'default'}",
        flush=True,
    )

    scheduling_strategies: list[Any]
    if args.skip_placement:
        scheduling_strategies = [None] * args.world_size
        pg = None
    else:
        # One bundle per node with `gpus_per_node` GPUs, STRICT_SPREAD forces
        # each bundle onto a distinct node — mimics Megatron's PP=nodes layout
        # so PP peers land on different physical nodes (exercising cross-node IB).
        # Bundle must reserve CPU=`gpus_per_node` too so all `gpus_per_node`
        # actors (each taking 1 CPU by ray default) fit inside one bundle.
        bundles = [{"GPU": gpus_per_node, "CPU": gpus_per_node} for _ in range(nodes)]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready(), timeout=120)
        scheduling_strategies = [
            PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=(r // gpus_per_node))
            for r in range(args.world_size)
        ]

    actors = []
    for rank in range(args.world_size):
        opts = {"num_gpus": 1}
        if scheduling_strategies[rank] is not None:
            opts["scheduling_strategy"] = scheduling_strategies[rank]
        actor = NCCLTestActor.options(**opts).remote(
            rank=rank,
            world_size=args.world_size,
            tp=args.tp,
            pp=args.pp,
            master_port=args.master_port,
            init_timeout_s=args.init_timeout_s,
        )
        actors.append(actor)

    print("\n== placement ==", flush=True)
    infos = ray.get([a.info.remote() for a in actors], timeout=120)
    for info in infos:
        print(f"  {info}", flush=True)

    # Rank 0's TCPStore server binds to *its own* node's IP; every other rank
    # must reach it. STRICT_SPREAD does not pin bundle 0 to a specific node, so
    # we can only resolve the master addr AFTER placement is known.
    master_addr = infos[0]["ip"]
    print(f"\n== master_addr={master_addr} master_port={args.master_port} (from rank 0's placement)", flush=True)
    ray.get([a.set_master.remote(master_addr) for a in actors], timeout=30)

    print("\n== init_process_group + build TP/PP/DP subgroups ==", flush=True)
    init_res = _wait_or_timeout("init", [a.init.remote() for a in actors], args.init_timeout_s)
    if init_res is None:
        print("\ninit hang — bailing out. Check MASTER_ADDR reachability + NCCL_SOCKET_IFNAME.", file=sys.stderr)
        sys.exit(2)

    def run(name: str, per_actor_fn):
        return _wait_or_timeout(name, [per_actor_fn(a) for a in actors], args.per_op_timeout)

    dp = args.world_size // (args.tp * args.pp)

    if "allreduce_world" in args.tests:
        run(f"AllReduce WORLD ({args.tensor_mb} MB)", lambda a: a.all_reduce.remote("world", args.tensor_mb))
    if "allreduce_tp" in args.tests and args.tp > 1:
        run(
            f"AllReduce TP subgroup size={args.tp} ({args.tensor_mb} MB)",
            lambda a: a.all_reduce.remote("tp", args.tensor_mb),
        )
    if "allreduce_dp" in args.tests and dp > 1:
        run(
            f"AllReduce DP subgroup size={dp} ({args.tensor_mb} MB)",
            lambda a: a.all_reduce.remote("dp", args.tensor_mb),
        )
    if "allgather_pp" in args.tests and args.pp > 1:
        run(
            f"AllGather PP subgroup size={args.pp} ({args.tensor_mb} MB)",
            lambda a: a.all_gather.remote("pp", args.tensor_mb),
        )
    if "p2p_pp" in args.tests and args.pp > 1:
        run(
            f"PP batch_isend_irecv (send→next / recv←prev, {args.tensor_mb} MB) ★ hang-repro shape",
            lambda a: a.pp_batched_isend_irecv.remote(args.tensor_mb),
        )
    if "pp_1f1b_bare" in args.tests and args.pp > 1:
        run(
            f"1F1B forward, {args.num_microbatches} mb × {args.tensor_mb} MB, no TP interleave",
            lambda a: a.pp_1f1b_forward_only.remote(args.tensor_mb, args.num_microbatches, False),
        )
    if "pp_1f1b_with_tp" in args.tests and args.pp > 1:
        run(
            f"1F1B forward + TP all-gather, {args.num_microbatches} mb × {args.tensor_mb} MB ★ Megatron-shape",
            lambda a: a.pp_1f1b_forward_only.remote(args.tensor_mb, args.num_microbatches, args.tp > 1),
        )
    if "pp_p2p_stress" in args.tests and args.pp > 1:
        run(
            f"PP P2P stress ×{args.stress_iters} iters, variable size (0.5/1/2×)",
            lambda a: a.pp_p2p_stress.remote(args.tensor_mb, args.stress_iters, True),
        )
    if "allreduce_world_big" in args.tests:
        run(f"AllReduce WORLD ({args.big_tensor_mb} MB)", lambda a: a.all_reduce.remote("world", args.big_tensor_mb))

    print("\n== teardown ==", flush=True)
    try:
        ray.get([a.teardown.remote() for a in actors], timeout=60)
    except ray.exceptions.GetTimeoutError:
        print("  teardown timed out (best-effort, ignoring)", file=sys.stderr)

    if pg is not None:
        from ray.util.placement_group import remove_placement_group

        remove_placement_group(pg)

    print("\n== done ==", flush=True)


if __name__ == "__main__":
    main()
