# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Cross-image numerical probe for the GDN context-parallel paths.

RFC Task 32 acceptance item 3.4-2: the FLA 0.4.1 -> 0.4.2 upgrade and the MCore
backport must not move the numbers on any path that already existed, i.e. CP=1,
MCore headwise CP, and Relax's all-gather fallback. That cannot be a normal unit
test, because it compares *two images* -- it needs the same tensors produced
under the old dependency set and the new one.

So this module is a runner, not a pytest file (hence the non-``test_`` name):

    # inside the OLD image
    python -m tests.backends.megatron.gdn_cp_numeric_probe dump --mode headwise --out /out/old
    # inside the NEW image
    python -m tests.backends.megatron.gdn_cp_numeric_probe dump --mode headwise --out /out/new
    # anywhere
    python -m tests.backends.megatron.gdn_cp_numeric_probe compare --ref /out/old --cand /out/new

Everything that could drift between images is pinned by hand: every parameter is
overwritten with a tensor drawn from a name-seeded CPU generator, and the inputs
come from a fixed-seed CPU generator too. So a difference in the report is a
difference in the *kernels*, not in initialisation order or RNG plumbing.

Modes:
  cp1        1 GPU, no CP at all.
  headwise   2 GPUs, MCore's native cp2hp all-to-all path.
  all_gather 2 GPUs, Relax's `_dcp_gdn_forward` fallback (head geometry chosen so
             the dispatcher cannot use headwise), which also exercises the FLA
             conv/scan kernels on the full gathered sequence.
  chunkwise  2 GPUs, the newly backported path (candidate image only).
"""

from __future__ import annotations

import argparse
import json
import os
import zlib
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp


MODE_WORLD_SIZE = {"cp1": 1, "headwise": 2, "all_gather": 2, "chunkwise": 2}
# all_gather is only reachable when the heads do NOT divide tp * cp.
MODE_HEADS = {
    "cp1": (4, 8),
    "headwise": (4, 8),
    "chunkwise": (4, 8),
    "all_gather": (1, 2),
}
SEQ_LENS = [256, 128]
HIDDEN_SIZE = 512


def _deterministic_fill_(module) -> None:
    """Overwrite every parameter from a name-seeded generator.

    Makes the dump independent of Megatron's initialisation code, which is one
    of the things the patch touches.
    """
    with torch.no_grad():
        for name, p in sorted(module.named_parameters()):
            gen = torch.Generator(device="cpu").manual_seed(zlib.crc32(name.encode()) & 0x7FFFFFFF)
            values = torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.05
            p.copy_(values.to(device=p.device, dtype=p.dtype))


def _build(mode, cp_size):
    import torch.nn.functional as F
    from megatron.core import parallel_state
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    model_parallel_cuda_manual_seed(123)
    num_key_heads, num_value_heads = MODE_HEADS[mode]

    if mode == "all_gather":
        # Relax relaxes Megatron's headwise `% (tp*cp)` config gate down to `% tp` so a
        # non-divisible geometry can even be constructed. v1 keeps this helper working
        # untouched, and using it in both images keeps the A/B identical. (The backported
        # `linear_cp_mode="all_gather"` is the eventual replacement; it is covered by
        # test_gdn_chunkwise_cp_layout.py instead, so it cannot skew this comparison.)
        from relax.backends.megatron.model import _relax_gdn_cp_config_assert

        _relax_gdn_cp_config_assert()

    config = TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        num_attention_heads=8,
        num_query_groups=2,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        activation_func=F.silu,
        bf16=True,
        tensor_model_parallel_size=1,
        context_parallel_size=cp_size,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=num_value_heads,
        transformer_impl="transformer_engine",
        # The CP algorithm is static config, resolved once at launch. Only chunkwise needs
        # to be declared here, and only the candidate image has the field at all -- for
        # cp1 / headwise / all_gather both images must run literally the same code, which
        # is the whole point of this probe.
        **(
            {"linear_cp_mode": "chunkwise"}
            if mode == "chunkwise" and "linear_cp_mode" in TransformerConfig.__dataclass_fields__
            else {}
        ),
    )
    pg_collection = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    gdn = (
        GatedDeltaNet(
            config,
            submodules=get_experimental_attention_variant_module_spec(config=config).submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        )
        .cuda()
        .bfloat16()
    )
    _deterministic_fill_(gdn)
    return gdn, config


def _dump_worker(rank, mode, out_dir):
    world_size = MODE_WORLD_SIZE[mode]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    torch.cuda.set_device(rank)
    import torch.distributed as dist

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=world_size,
    )
    cp_size = world_size
    gdn, config = _build(mode, cp_size)

    if mode == "all_gather":
        # Install Relax's runtime GDN wrapper and satisfy its full-recompute gate.
        from relax.backends.megatron import model as relax_model

        relax_model.get_args = lambda: SimpleNamespace(recompute_granularity="full")
        relax_model._patch_gdn_for_dynamic_cp()

    device = torch.device("cuda", rank)
    total = sum(SEQ_LENS)
    cu = torch.tensor([0, SEQ_LENS[0], total], device=device, dtype=torch.int32)
    local_total = total // cp_size

    gen = torch.Generator(device="cpu").manual_seed(20260805)
    hidden_full = torch.randn(total, 1, HIDDEN_SIZE, generator=gen, dtype=torch.float32)
    grad_full = torch.randn(total, 1, HIDDEN_SIZE, generator=gen, dtype=torch.float32)

    if cp_size == 1:
        hidden_local = hidden_full
        grad_local = grad_full
    else:
        from relax.backends.megatron.cp_utils import gdn_cp_slice

        cu_list = [0, SEQ_LENS[0], total]
        hidden_local = gdn_cp_slice(hidden_full, cu_list, cp_size, rank)
        grad_local = gdn_cp_slice(grad_full, cu_list, cp_size, rank)
    assert hidden_local.shape[0] == local_total

    h = hidden_local.to(device=device, dtype=torch.bfloat16).clone().requires_grad_(True)
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(SEQ_LENS),
        max_seqlen_kv=max(SEQ_LENS),
        cp_group=parallel_state.get_context_parallel_group(),
        local_cp_size=cp_size,
    )
    out, _ = gdn(h, None, packed_seq_params=psp)
    (out.float() * grad_local.to(device)).sum().backward()

    payload = {"out": out.detach().float().cpu(), "grad_hidden": h.grad.detach().float().cpu()}
    for name, p in sorted(gdn.named_parameters()):
        payload[f"grad::{name}"] = p.grad.detach().float().cpu()

    os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, os.path.join(out_dir, f"{mode}.rank{rank}.pt"))
    if rank == 0:
        import importlib.metadata as md

        meta = {
            "mode": mode,
            "world_size": world_size,
            "fla": md.version("flash-linear-attention"),
            "fla_core": md.version("fla-core"),
            "torch": torch.__version__,
            "heads": MODE_HEADS[mode],
            "seq_lens": SEQ_LENS,
        }
        with open(os.path.join(out_dir, f"{mode}.meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(json.dumps(meta))

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _cmp(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rms = ((a - b).square().mean().sqrt() / (b.square().mean().sqrt() + 1e-12)).item()
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / denom).max().item(),
        "rms_ratio": rms,
        "cosine": cos,
        "bitwise_equal": bool(torch.equal(a, b)),
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--mode", choices=sorted(MODE_WORLD_SIZE), required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--port", default="29601")

    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cand", required=True)
    c.add_argument("--mode", choices=sorted(MODE_WORLD_SIZE), required=True)
    c.add_argument("--atol", type=float, default=2e-4)
    c.add_argument("--rtol", type=float, default=2e-3)
    c.add_argument("--cos", type=float, default=0.99999)

    args = ap.parse_args()
    if args.cmd == "dump":
        os.environ["MASTER_PORT"] = args.port
        world_size = MODE_WORLD_SIZE[args.mode]
        mp.spawn(_dump_worker, args=(args.mode, args.out), nprocs=world_size, join=True)
        return

    world_size = MODE_WORLD_SIZE[args.mode]
    failures, rows = [], []
    for rank in range(world_size):
        ref = torch.load(os.path.join(args.ref, f"{args.mode}.rank{rank}.pt"), weights_only=True)
        cand = torch.load(os.path.join(args.cand, f"{args.mode}.rank{rank}.pt"), weights_only=True)
        assert set(ref) == set(cand), f"tensor sets differ: {set(ref) ^ set(cand)}"
        for key in sorted(ref):
            stats = _cmp(cand[key], ref[key])
            rows.append({"rank": rank, "tensor": key, **stats})
            # RFC section 5, "MCore GDN, CP=1: candidate image vs old image".
            ok = stats["max_abs"] <= args.atol + args.rtol * abs(ref[key]).max().item()
            ok = ok and stats["cosine"] >= args.cos
            if not ok:
                failures.append(rows[-1])

    print(json.dumps(rows, indent=2))
    print(
        f"\n{args.mode}: {len(rows)} tensors compared, "
        f"{sum(r['bitwise_equal'] for r in rows)} bitwise identical, {len(failures)} outside tolerance"
    )
    if failures:
        print("FAIL")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
