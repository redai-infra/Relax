# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Distributed (TP/PP/CP) + packed-sequence variant of
``compare_sglang_megatron_dotsocr.py``, run as **three independent stages** so
SGLang's child-process model doesn't fight with ``torchrun``'s env vars.

  * **Two samples packed into one sequence** (one multimodal: boxes.png +
    "Describe this image."; one text-only: arbitrary prompt).
  * **TP / PP / CP all > 1** by default (2 / 2 / 2 on 8 GPUs), mirroring the
    bridge VL+CP path in ``relax/backends/megatron/data.py:185-216``.

Stages (each one is a separate process invocation — see
``scripts/debug/run-compare-dotsocr-packed.sh`` for the orchestrator)::

    # stage 1 — single process, single GPU
    python  compare_sglang_megatron_dotsocr_packed.py --side front --dump-dir D
        produces:  D/hf.sample{A,B}.json     (skipped with --skip-hf)
                   D/sglang.sample{A,B}.json (skipped with --skip-sglang)

    # stage 2 — torchrun, 8 GPUs
    torchrun --nproc-per-node=8 compare_sglang_megatron_dotsocr_packed.py \\
        --side megatron --dump-dir D --tp-size 2 --pp-size 2 --cp-size 2
        produces:  D/megatron.sample{A,B}.json
        path:      dist.init_process_group + mpu.initialize_model_parallel
                   → bridge.AutoBridge.load_hf_weights (PP-aware scatter)
                   → get_forward_backward_func(forward_only=True), packed input
                     mirrors relax/backends/megatron/data.py:185-216
                   → loss_func: fused_vocab_parallel_cross_entropy with
                     shift-by-1 targets, then CP-zigzag-gather per-sample
                   → broadcast per-sample logprob to global rank 0 → dump

    # stage 3 — single process, no GPU needed
    python  compare_sglang_megatron_dotsocr_packed.py --side compare --dump-dir D
        reads whatever dumps are present and pairwise-compares them.

Reusable plumbing (chat-template build, HF/SGLang run, compare summary)
is imported from ``compare_sglang_megatron_dotsocr`` so this file stays
focused on the packing + distributed parts. Each stage is independently
re-runnable — useful when only the Megatron side is being iterated on.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist


# Reuse single-GPU helpers verbatim.
sys.path.insert(0, str(Path(__file__).parent))
from compare_sglang_megatron_dotsocr import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    DTYPE_MAP,
    DumpRecord,
    _build_messages,
    _load_image_any,
    compare,
    run_hf,
    run_sglang,
)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-checkpoint", required=True, help="Path to dots.mocr HF checkpoint dir.")
    p.add_argument(
        "--image",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/boxes.png",
        help="Image path or URL for sample A (multimodal).",
    )
    p.add_argument(
        "--mm-prompt",
        default="Describe this image.",
        help="User text accompanying the image in sample A.",
    )
    p.add_argument(
        "--text-prompt",
        default="What is the capital of France? Answer in one sentence.",
        help="User text for sample B (text-only).",
    )
    p.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt applied to both samples.",
    )
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument(
        "--side",
        required=True,
        choices=["front", "hf", "sglang", "megatron", "compare"],
        help=(
            "Pick ONE stage per invocation. "
            "'front' = HF + SGLang sequentially (single process). "
            "'hf' / 'sglang' = just that one side (single process). "
            "'megatron' = distributed packed forward (REQUIRES torchrun). "
            "'compare' = read dumps + print pairwise stats (single process)."
        ),
    )
    p.add_argument("--dump-dir", default="/tmp/relax_dotsocr_debug_packed")
    p.add_argument("--mem-fraction-static", type=float, default=0.5)
    p.add_argument("--top-n-worst", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # Parallelism (assert tp*pp*cp == WORLD_SIZE at startup).
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--pp-size", type=int, default=2)
    p.add_argument("--cp-size", type=int, default=2)
    # Sequence parallel adds another moving piece; default off so the debug
    # script stays as close as possible to the single-GPU baseline. Flip on
    # only when you specifically want to reproduce a prod-style SP run.
    p.add_argument("--sequence-parallel", action="store_true", default=False)
    p.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HF ground truth (avoids flash_attn dependency from dots vision module).",
    )
    p.add_argument(
        "--skip-sglang",
        action="store_true",
        help=(
            "Skip SGLang ground truth (avoid version-skew failures from dots_vlm "
            "processor when the env has a newer transformers than SGLang expects)."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rank 0 front-end: HF + SGLang for two samples
# ---------------------------------------------------------------------------


def _per_sample_args(base: argparse.Namespace, sample: str) -> argparse.Namespace:
    """Project the multi-sample debug args onto the single-sample shape that
    ``compare_sglang_megatron_dotsocr.run_hf`` / ``run_sglang`` expect."""
    ns = argparse.Namespace(**vars(base))
    if sample == "A":
        ns.image = base.image
        ns.prompt = base.mm_prompt
    elif sample == "B":
        ns.image = None
        ns.prompt = base.text_prompt
    else:
        raise ValueError(sample)
    # Bypass the single-GPU script's intermediate-tensor capture machinery
    # (extra disk I/O + memory). It needs an inspect_layers attribute even
    # though we don't use the dumps here.
    ns.inspect_layers = ""
    ns.check_weights = False
    ns.no_attention_mask_megatron = False
    ns.bypass_dots_wrapper = False
    ns.megatron_attention_backend = "flash"
    return ns


def _front_dump_path(dump_dir: Path, side: str, sample: str) -> Path:
    return dump_dir / f"{side}.sample{sample}.json"


def _save_per_sample(rec: DumpRecord, dump_dir: Path, sample: str) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = _front_dump_path(dump_dir, rec.side, sample)
    with path.open("w") as f:
        json.dump(asdict(rec), f)
    print(f"[rank0] dumped {rec.side} sample {sample}: {len(rec.input_ids)} tokens -> {path}")


def _load_per_sample(dump_dir: Path, side: str, sample: str) -> Optional[DumpRecord]:
    path = _front_dump_path(dump_dir, side, sample)
    if not path.exists():
        return None
    with path.open("r") as f:
        data = json.load(f)
    return DumpRecord(**data)


def run_hf_two_samples(args: argparse.Namespace, dump_dir: Path) -> None:
    """Single-process: run HF for both samples and dump per-sample logprobs."""
    for sample in ("A", "B"):
        sub = _per_sample_args(args, sample)
        rec = run_hf(sub)
        rec.side = "hf"
        _save_per_sample(rec, dump_dir, sample)
        gc.collect()
        torch.cuda.empty_cache()


def run_sglang_two_samples(args: argparse.Namespace, dump_dir: Path) -> None:
    """Single-process: launch one SGLang Engine per sample (separate Engine
    instances are simpler than reusing one — the engine subprocess is fully
    torn down via ``engine.shutdown()`` inside ``run_sglang`` between samples,
    so GPU memory is reclaimed cleanly)."""
    for sample in ("A", "B"):
        sub = _per_sample_args(args, sample)
        rec = run_sglang(sub)
        rec.side = "sglang"
        _save_per_sample(rec, dump_dir, sample)
        gc.collect()
        torch.cuda.empty_cache()


def run_front(args: argparse.Namespace, dump_dir: Path) -> None:
    """Convenience: run HF then SGLang back-to-back in the same process.
    Each side is best-effort — missing deps just print a warning and we keep
    going so the rest of the pipeline still has data to compare."""
    if not args.skip_hf:
        try:
            run_hf_two_samples(args, dump_dir)
        except (ImportError, ModuleNotFoundError) as e:
            print(f"[front] HF unavailable ({e}); continuing with SGLang only", file=sys.stderr)
    if not args.skip_sglang:
        try:
            run_sglang_two_samples(args, dump_dir)
        except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as e:
            print(f"[front] SGLang unavailable ({e!r}); continuing", file=sys.stderr)


# ---------------------------------------------------------------------------
# Distributed init + provider construction
# ---------------------------------------------------------------------------


def _init_distributed(args: argparse.Namespace) -> None:
    """Bring up NCCL + Megatron MPU from the torchrun-provided RANK/LOCAL_RANK/
    WORLD_SIZE env.

    Asserts ``tp*pp*cp == WORLD_SIZE``.
    """
    from megatron.core import mpu, tensor_parallel

    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        raise RuntimeError(
            "--side megatron must be launched under torchrun (RANK/WORLD_SIZE missing). "
            "Use scripts/debug/run-compare-dotsocr-packed.sh to orchestrate all three stages."
        )
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    expected = args.tp_size * args.pp_size * args.cp_size
    assert world_size == expected, (
        f"WORLD_SIZE={world_size} but tp*pp*cp={expected} (tp={args.tp_size} pp={args.pp_size} cp={args.cp_size})"
    )

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", world_size=world_size, rank=rank)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=args.tp_size,
            pipeline_model_parallel_size=args.pp_size,
            context_parallel_size=args.cp_size,
            expert_model_parallel_size=1,
            order="tp-cp-ep-dp-pp",
        )
        tensor_parallel.model_parallel_cuda_manual_seed(args.seed)


def _apply_provider_overrides(provider, args: argparse.Namespace, dtype: torch.dtype) -> None:
    """Mirror ``relax.backends.megatron.model_provider`` CLI overrides but keep
    things minimal: parallel_output=True (vocab-parallel logits for the fused
    CE path), variable_seq_lengths=True (packed THD), no rope fusion
    (multimodal requirement)."""
    provider.tensor_model_parallel_size = args.tp_size
    provider.pipeline_model_parallel_size = args.pp_size
    provider.context_parallel_size = args.cp_size
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = bool(args.sequence_parallel)
    provider.variable_seq_lengths = True
    # Megatron's MCoreTransformerConfig.__post_init__ rejects the default
    # "allgather" MoE dispatcher whenever variable_seq_lengths is True — even
    # for non-MoE models, since the check fires before num_moe_experts is
    # inspected. "alltoall" is a no-op for non-MoE DotsOCR.
    provider.moe_token_dispatcher_type = "alltoall"
    provider.apply_rope_fusion = False
    provider.attention_softmax_in_fp32 = True
    provider.fp16 = dtype == torch.float16
    provider.bf16 = dtype == torch.bfloat16
    provider.params_dtype = dtype


# ---------------------------------------------------------------------------
# Two-sample packed-input construction
# ---------------------------------------------------------------------------


def _process_sample(args: argparse.Namespace, sample: str) -> dict:
    """Run the HF AutoProcessor (or tokenizer fallback) for one sample and
    return the canonical ``input_ids`` + optional pixel_values/grid_thw.

    Mirrors ``_build_processor_inputs`` from the single-GPU script but inlined
    so we can call it for both samples without going through
    ``_build_messages``'s argparse coupling.
    """
    from transformers import AutoProcessor, AutoTokenizer

    sub = _per_sample_args(args, sample)
    messages = _build_messages(sub)
    if sub.image:
        processor = AutoProcessor.from_pretrained(sub.hf_checkpoint, trust_remote_code=True)
        image = _load_image_any(sub.image)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_out = processor(text=[text], images=[image], padding=False, return_tensors="pt")
        return {
            "text": text,
            "input_ids": proc_out["input_ids"][0],  # [L]
            "pixel_values": proc_out["pixel_values"],  # [N_patches, C*ph*pw]
            "image_grid_thw": proc_out["image_grid_thw"],  # [num_images, 3]
        }
    tokenizer = AutoTokenizer.from_pretrained(sub.hf_checkpoint, trust_remote_code=True)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return {
        "text": text,
        "input_ids": enc["input_ids"][0],
        "pixel_values": None,
        "image_grid_thw": None,
    }


def _build_packed_inputs(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> dict:
    """Build the Megatron forward kwargs for our two samples, picking the
    right path based on CP size — exactly mirroring ``relax/backends/megatron/
    data.py``:

    * ``cp_size > 1`` (VL path, lines 185-216 in data.py): hand the bridge
      *unsplit* BSHD-padded tokens + attention_mask + a thd
      ``PackedSeqParams`` keyed on tp*cp*2-aligned cu_seqlens; the wrapper's
      ``preprocess_packed_seqs`` does THD-pack + zigzag-CP-split internally.
    * ``cp_size == 1`` (lines 240-266 in data.py): pre-pack ourselves by
      concatenating samples into a single ``[1, T_total_padded]`` THD stream
      and pass the matching ``PackedSeqParams``; attention_mask is None so
      the wrapper's ``unsplit_mode`` stays off."""
    from megatron.core.packed_seq_params import PackedSeqParams

    sample_A = _process_sample(args, "A")
    sample_B = _process_sample(args, "B")

    ids_list = [sample_A["input_ids"].to(torch.long), sample_B["input_ids"].to(torch.long)]
    align_size = args.tp_size * args.cp_size * 2

    seqlens = torch.tensor([t.size(0) for t in ids_list], dtype=torch.int32, device=device)
    seqlens_padded = (seqlens + align_size - 1) // align_size * align_size
    cu_seqlens_padded = torch.zeros(len(ids_list) + 1, dtype=torch.int32, device=device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_padded, dim=0)
    max_seqlen_padded = int(seqlens_padded.max().item())
    padded_lens = seqlens_padded.tolist()

    pad_token_id = 0
    common = {
        "samples": [sample_A, sample_B],
        "ids_list": ids_list,
        "lens": [int(t.size(0)) for t in ids_list],
        "padded_lens": padded_lens,
    }
    if sample_A["pixel_values"] is not None:
        common["pixel_values"] = sample_A["pixel_values"].to(device=device, dtype=dtype)
        common["image_grid_thw"] = sample_A["image_grid_thw"].to(device=device, dtype=torch.long)
    else:
        common["pixel_values"] = None
        common["image_grid_thw"] = None

    if args.cp_size > 1:
        # Unsplit BSHD path — wrapper packs + zigzag-CP-splits internally.
        T_max = max_seqlen_padded
        unsplit_tokens = torch.full((len(ids_list), T_max), pad_token_id, dtype=torch.long, device=device)
        unsplit_attention_mask = torch.zeros((len(ids_list), T_max), dtype=torch.bool, device=device)
        for i, ids in enumerate(ids_list):
            unsplit_tokens[i, : ids.size(0)] = ids.to(device)
            unsplit_attention_mask[i, : ids.size(0)] = True
        vlm_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            max_seqlen_q=max_seqlen_padded,
            max_seqlen_kv=max_seqlen_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )
        common.update(
            mode="unsplit",
            unsplit_tokens=unsplit_tokens,
            unsplit_attention_mask=unsplit_attention_mask,
            vlm_packed_seq_params=vlm_packed_seq_params,
        )
        return common

    # CP == 1: pre-pack ourselves. tokens = concat(pad-right(sample_i, padded_i))
    # then unsqueeze(0) → [1, T_total]. cu_seqlens is the cumulative padded
    # boundary so THD attention treats each segment as an independent sample.
    packed_segments = []
    for ids, padded in zip(ids_list, padded_lens):
        seg = torch.full((padded,), pad_token_id, dtype=torch.long, device=device)
        seg[: ids.size(0)] = ids.to(device)
        packed_segments.append(seg)
    packed_tokens = torch.cat(packed_segments).unsqueeze(0)  # [1, T_total_padded]
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_padded,
        max_seqlen_kv=max_seqlen_padded,
    )
    common.update(
        mode="packed",
        packed_tokens=packed_tokens,
        packed_seq_params=packed_seq_params,
    )
    return common


# ---------------------------------------------------------------------------
# CP zigzag helpers for targets + gather
# ---------------------------------------------------------------------------


def _zigzag_local_slice(sample: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """Replicate ``relax/backends/megatron/cp_utils.slice_with_cp`` for a 1-D
    tensor that is already padded to ``2*cp_size`` boundary.

    Used to derive the per-rank target tokens that match the bridge's internal
    THD-packed CP-zigzag-split of hidden states.
    """
    assert sample.size(0) % (2 * cp_size) == 0, sample.shape
    chunk_size = sample.size(0) // (2 * cp_size)
    s1, e1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    s2, e2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([sample[s1:e1], sample[s2:e2]])


def _cp_gather_zigzag_to_full(local: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Inverse of zigzag split: all-gather the per-rank ``[2*chunk_size]``
    pieces and reassemble the original padded sample of length
    ``2*cp_size*chunk_size``.

    Every rank in the CP group ends up with the same full tensor (we only
    consume it on the PP-last + TP-rank-0 path afterwards, but the all-gather
    is cheap).
    """
    from megatron.core import mpu

    cp_group = mpu.get_context_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return local
    assert local.size(0) == 2 * chunk_size, (local.shape, chunk_size)

    gathered = [torch.empty_like(local) for _ in range(cp_size)]
    dist.all_gather(gathered, local.contiguous(), group=cp_group)

    out = torch.empty(2 * cp_size * chunk_size, dtype=local.dtype, device=local.device)
    for r in range(cp_size):
        out[r * chunk_size : (r + 1) * chunk_size] = gathered[r][:chunk_size]
        s2 = (2 * cp_size - r - 1) * chunk_size
        out[s2 : s2 + chunk_size] = gathered[r][chunk_size:]
    return out


# ---------------------------------------------------------------------------
# Distributed Megatron forward (packed)
# ---------------------------------------------------------------------------


def _build_megatron_model(args: argparse.Namespace, dtype: torch.dtype):
    from megatron.bridge import AutoBridge
    from megatron.core import mpu
    from megatron.core.enums import ModelType

    import relax.models  # noqa: F401  side-effect: register DotsOCRBridge

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    _apply_provider_overrides(provider, args, dtype)
    provider.finalize()

    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()
    model = provider.provide(pre_process=pre_process, post_process=post_process)
    # Megatron's forward_backward schedules call ``get_attr_wrapped_model(model,
    # "model_type")`` to decide PP send/recv semantics. The standard
    # ``get_model`` helper sets this on the wrapped module — we build the model
    # by hand so we have to set it ourselves. DotsOCR is a decoder-only LM:
    # encoder_or_decoder is the correct ModelType.
    model.model_type = ModelType.encoder_or_decoder
    model = model.cuda().to(dtype).eval()
    bridge.load_hf_weights([model])
    return model, bridge


def _compute_local_per_pos_logprob(logits: torch.Tensor, local_targets: torch.Tensor) -> torch.Tensor:
    """``logits``: ``[s_local, 1, v/tp]`` from GPTModel parallel_output=True.
    ``local_targets``: ``[s_local]`` already CP-zigzag-split. Returns
    ``[s_local]`` per-position logprob = log P(local_targets[t] | context).

    The TP all-reduce inside ``fused_vocab_parallel_cross_entropy`` makes the
    output identical across TP ranks."""
    from megatron.core import mpu
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    targets = local_targets.unsqueeze(1)  # [s_local, 1]
    ce = fused_vocab_parallel_cross_entropy(logits, targets, mpu.get_tensor_model_parallel_group())
    return -ce.squeeze(1)  # [s_local]


def _per_sample_gathered_logprobs(logits: torch.Tensor, packed: dict, args: argparse.Namespace) -> list[torch.Tensor]:
    """End-to-end last-PP-stage logprob computation:

    1. Build shifted targets per sample (target[t] = sample[t+1]), pad-right.
    2. CP-zigzag-split each sample → concat → local_targets [s_local_total].
    3. Run TP-fused CE → per-position logprob [s_local_total].
    4. Split back into per-sample chunks, CP-gather each → full padded.

    Returns a list of length-``padded_lens[i]`` CPU float tensors, one per
    sample. After this every rank in the CP group has the same data; outside
    the last PP stage this function is not called.
    """
    cp_size = args.cp_size
    from megatron.core import mpu

    cp_rank = mpu.get_context_parallel_rank()

    # Step 1+2: shifted targets per sample, then CP-zigzag-split & concat.
    local_targets_per_sample = []
    chunk_sizes = []
    for ids, padded_len in zip(packed["ids_list"], packed["padded_lens"]):
        sample = torch.zeros(padded_len, dtype=torch.long, device=logits.device)
        sample[: ids.size(0)] = ids.to(logits.device)
        # Shift left by 1 so logit at position t predicts sample[t+1].
        # The last position has no successor → leave as 0 (we mask the
        # trailing positions out after the gather).
        shifted = torch.zeros_like(sample)
        shifted[:-1] = sample[1:]
        local_target = _zigzag_local_slice(shifted, cp_rank, cp_size)
        local_targets_per_sample.append(local_target)
        chunk_sizes.append(padded_len // (2 * cp_size))
    local_targets = torch.cat(local_targets_per_sample)  # [s_local_total]

    # GPTModel output layout depends on the path: in this VL+CP+thd unsplit
    # mode it comes back as BSH ``[1, s_local_total, v/tp]`` on the last PP
    # stage (the bridge wrapper's preprocess_packed_seqs trailing transpose
    # doesn't propagate through PP send/recv). fused_vocab_parallel_cross_entropy
    # wants SBH ``[s, b, v]``. Detect and transpose if needed instead of
    # hard-coding either layout.
    if logits.dim() == 3 and logits.size(0) == 1 and logits.size(1) == local_targets.size(0):
        logits = logits.transpose(0, 1).contiguous()
    assert logits.dim() == 3 and logits.size(0) == local_targets.size(0) and logits.size(1) == 1, (
        f"unexpected logits shape {tuple(logits.shape)}; "
        f"expected ({local_targets.size(0)}, 1, vocab/tp). "
        f"packed lens={packed['padded_lens']}, cp_size={cp_size}, chunk_sizes={chunk_sizes}"
    )

    per_pos = _compute_local_per_pos_logprob(logits, local_targets)  # [s_local_total]

    # Split back per-sample (in zigzag-local order: 2*chunk_size per sample).
    splits = [2 * cs for cs in chunk_sizes]
    per_sample_local = per_pos.split(splits)

    out: list[torch.Tensor] = []
    for local_chunk, cs in zip(per_sample_local, chunk_sizes):
        full_padded = _cp_gather_zigzag_to_full(local_chunk, cs)
        out.append(full_padded.detach().to(torch.float32).cpu())
    return out


def run_megatron_packed(args: argparse.Namespace, dump_dir: Path) -> None:
    """All-rank Megatron packed forward + per-sample logprob gather.

    Rank 0 writes the per-sample dumps.
    """
    from megatron.core import mpu
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    dtype = DTYPE_MAP[args.dtype]
    device = torch.device("cuda")
    rank = dist.get_rank()
    is_last_pp = mpu.is_pipeline_last_stage()

    model, _bridge = _build_megatron_model(args, dtype)

    # Every rank builds the same packed inputs (deterministic processor output);
    # the bridge only consumes them on the first PP stage but we keep the
    # ids_list everywhere because the loss_func on the last PP stage needs
    # them too. Cheap (≤ a few KB of tensors).
    packed = _build_packed_inputs(args, dtype, device)

    if rank == 0:
        print(
            f"[rank0] packed: mode={packed['mode']} lens={packed['lens']} "
            f"padded_lens={packed['padded_lens']} "
            f"image_grid_thw={(packed['image_grid_thw'].tolist() if packed['image_grid_thw'] is not None else None)}"
        )

    def forward_step(_data_iter, model_chunk, return_schedule_plan: bool = False):
        assert not return_schedule_plan
        if packed["mode"] == "unsplit":
            # CP>1 VL path: wrapper internally THD-packs + zigzag-CP-splits.
            kwargs = dict(
                input_ids=packed["unsplit_tokens"],
                position_ids=None,
                attention_mask=packed["unsplit_attention_mask"],
                labels=None,
                packed_seq_params=packed["vlm_packed_seq_params"],
                loss_mask=None,
            )
        else:
            # CP=1: tokens already THD-concatenated to [1, T_total_padded];
            # attention_mask=None keeps wrapper's unsplit_mode off so the
            # internal THD repack is skipped.
            kwargs = dict(
                input_ids=packed["packed_tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=packed["packed_seq_params"],
                loss_mask=None,
            )
        if packed["pixel_values"] is not None:
            kwargs["pixel_values"] = packed["pixel_values"]
            kwargs["image_grid_thw"] = packed["image_grid_thw"]
        output = model_chunk(**kwargs)

        def loss_func(logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
            # ``logits`` shape: [s_local, 1, v/tp]. We don't care about a real
            # loss — return 0 to satisfy Megatron's interface, and stash the
            # per-sample gathered logprobs in the dict so we can pull them
            # out of forward_data_store on the last PP stage.
            try:
                from megatron.core import mpu as _mpu

                print(
                    f"[rank{dist.get_rank()}] loss_func entered: "
                    f"logits.shape={tuple(logits.shape)} "
                    f"tp_rank={_mpu.get_tensor_model_parallel_rank()} "
                    f"cp_rank={_mpu.get_context_parallel_rank()}",
                    flush=True,
                )
                per_sample = _per_sample_gathered_logprobs(logits, packed, args)
                print(
                    f"[rank{dist.get_rank()}] loss_func gathered ok: sample_lens={[t.shape[0] for t in per_sample]}",
                    flush=True,
                )
            except Exception as e:
                import traceback

                print(
                    f"[rank{dist.get_rank()}] loss_func RAISED: {type(e).__name__}: {e}\n" + traceback.format_exc(),
                    flush=True,
                )
                raise
            zero_loss = logits.new_zeros(())
            return zero_loss, {"per_sample_logprobs": per_sample}

        return output, loss_func

    # ``seq_length`` / ``micro_batch_size`` are hints Megatron uses to size
    # the PP send/recv buffers (with variable_seq_lengths=True the actual
    # shape is dynamic). Match the actual forward input shape to keep the
    # hint accurate even in single-stage runs.
    if packed["mode"] == "unsplit":
        sl_hint = int(packed["unsplit_tokens"].shape[1])
        mbs_hint = int(packed["unsplit_tokens"].shape[0])
    else:
        sl_hint = int(packed["packed_tokens"].shape[1])
        mbs_hint = 1

    forward_backward_func = get_forward_backward_func()
    try:
        forward_data_store = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=iter([None]),  # single fake step; forward_step ignores it
            model=[model],
            num_microbatches=1,
            seq_length=sl_hint,
            micro_batch_size=mbs_hint,
            forward_only=True,
        )
    except Exception as e:
        import traceback

        print(
            f"[rank{rank}] forward_backward_func RAISED: {type(e).__name__}: {e}\n" + traceback.format_exc(),
            flush=True,
        )
        raise
    print(f"[rank{rank}] forward_backward_func returned (is_last_pp={is_last_pp})", flush=True)

    # Hard sync before broadcast: if any rank failed in loss_func / forward
    # pipelining, this barrier exposes it instead of deadlocking inside the
    # broadcast collective below. WORLD barrier fails fast on any laggard.
    dist.barrier()
    print(f"[rank{rank}] post-forward barrier passed", flush=True)

    # Only the last PP stage has populated forward_data_store. Pick one rank
    # there (tp=0 + cp=0) — by construction it has the same gathered logprobs
    # as every other (tp, cp) rank on that stage after the CP all-gather +
    # TP-fused-CE all-reduce inside the loss_func.
    #
    # We use plain ``dist.broadcast`` over NCCL with pre-sized float tensors
    # rather than ``broadcast_object_list`` — the object_list path silently
    # spins up a temporary gloo group under NCCL backends and hangs on
    # certain build combos. All ranks already know ``padded_lens`` from
    # ``_build_packed_inputs`` so no metadata round-trip is needed.
    src_rank = (args.pp_size - 1) * args.tp_size * args.cp_size  # order tp-cp-ep-dp-pp
    is_src = is_last_pp and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0
    if is_src:
        local_per_sample = forward_data_store[0]["per_sample_logprobs"]
        print(f"[rank{rank}] src gather complete: lens={[t.shape[0] for t in local_per_sample]}")

    per_sample_full: list[torch.Tensor] = []
    for i, padded_len in enumerate(packed["padded_lens"]):
        buf = torch.zeros(padded_len, dtype=torch.float32, device="cuda")
        if is_src:
            buf.copy_(local_per_sample[i].to(device="cuda", dtype=torch.float32))
        dist.broadcast(buf, src=src_rank)
        per_sample_full.append(buf.cpu())

    if rank == 0:
        for sample_name, ids, padded_logprob, sample_dict in zip(
            ("A", "B"),
            packed["ids_list"],
            per_sample_full,
            packed["samples"],
        ):
            ids_list = ids.tolist()
            # logit at position t-1 predicts token at position t
            # padded_logprob is computed with target = sample[t+1] at logit
            # position t, so padded_logprob[t-1] = log P(ids[t] | ids[:t]).
            per_token: list[Optional[float]] = [None]
            for t in range(1, len(ids_list)):
                per_token.append(float(padded_logprob[t - 1].item()))
            rec = DumpRecord(
                side="megatron",
                input_ids=ids_list,
                logprobs=per_token,
                meta={
                    "dtype": args.dtype,
                    "tp": args.tp_size,
                    "pp": args.pp_size,
                    "cp": args.cp_size,
                    "sample": sample_name,
                    "padded_len": packed["padded_lens"][0 if sample_name == "A" else 1],
                    "raw_len": len(ids_list),
                    "prompt_text_first_300": sample_dict["text"][:300],
                },
            )
            _save_per_sample(rec, dump_dir, sample_name)


# ---------------------------------------------------------------------------
# Compare (per sample, pairwise across sides)
# ---------------------------------------------------------------------------


def run_compare(args: argparse.Namespace, dump_dir: Path) -> None:
    for sample in ("A", "B"):
        print(f"\n========== sample {sample} ==========")
        records: dict[str, DumpRecord] = {}
        for side in ("megatron", "hf", "sglang"):
            rec = _load_per_sample(dump_dir, side, sample)
            if rec is not None:
                records[side] = rec
            else:
                print(f"[compare] sample {sample}: missing {side} dump")
        if len(records) >= 2:
            compare(records, top_n_worst=args.top_n_worst)


# ---------------------------------------------------------------------------
# Entry point — torchrun-aware
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    if args.side == "front":
        run_front(args, dump_dir)
    elif args.side == "hf":
        run_hf_two_samples(args, dump_dir)
    elif args.side == "sglang":
        run_sglang_two_samples(args, dump_dir)
    elif args.side == "megatron":
        _init_distributed(args)
        # No try/finally barrier here — if any rank raises, let it propagate
        # and exit; torchrun reaps the others within seconds. Wrapping with a
        # barrier in a finally block silently converts a single-rank crash
        # into an 8-way deadlock that's only diagnosable via py-spy.
        run_megatron_packed(args, dump_dir)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    elif args.side == "compare":
        run_compare(args, dump_dir)
    else:
        raise AssertionError(args.side)
    return 0


if __name__ == "__main__":
    sys.exit(main())
