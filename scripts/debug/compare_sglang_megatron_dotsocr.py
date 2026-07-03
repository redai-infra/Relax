# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare per-input-token log-probabilities between SGLang and Megatron for
DotsOCR2 on the same multimodal input (text + image).

Single-GPU debugging utility (tp=pp=cp=ep=1). The goal is to localize the
sglang ↔ megatron mismatch users hit during RL training by replaying the
*same* loading paths that production uses:

  * Megatron side: ``AutoBridge.from_hf_pretrained`` + ``to_megatron_provider``
    + ``load_hf_weights`` — identical to ``relax.backends.megatron.model_provider``
    when ``--megatron-to-hf-mode bridge`` is used.
  * SGLang side: offline ``sglang.Engine`` started with the
    ``SGLANG_EXTERNAL_MODEL_PACKAGE=relax.models.dots_ocr.sglang`` env var,
    so ``DotsOCRForCausalLM`` from this repo is the actual implementation
    loaded — identical to what the rollout engine instantiates at runtime.

Run modes (``--side``):
  * ``megatron``: build the megatron model, forward, dump logprobs to disk.
  * ``sglang``:   launch the sglang engine, generate with ``return_logprob``,
                  dump logprobs to disk.
  * ``compare``:  load both dumps and print summary stats + worst positions.
  * ``all`` (default): run all three sequentially in one process.

The two sides do NOT share GPU memory simultaneously in ``all`` mode — we
free the megatron model before launching SGLang's worker subprocess.

Example::

    python scripts/debug/compare_sglang_megatron_dotsocr.py \\
        --hf-checkpoint /data/rednote-hilab/dots.mocr/ \\
        --image /data/sample.png \\
        --prompt "Describe this image."

Use ``--side megatron`` / ``--side sglang`` for incremental debugging when
one side fails and you don't want to re-run the other.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Set by run_megatron() right before provider construction so
# _apply_single_gpu_provider_overrides() picks the user's chosen backend.
_CURRENT_ATTENTION_BACKEND = "flash"


@dataclass
class DumpRecord:
    """Per-token logprob dump persisted to disk for cross-side comparison."""

    side: str  # "megatron" | "sglang"
    input_ids: list[int]  # post-image-expansion token sequence (length T)
    # logprobs[i] = log P(input_ids[i] | input_ids[:i]), length T;
    # entry 0 is None (no conditioning available).
    logprobs: list[Optional[float]]
    meta: dict


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process "
    "here </think><answer> answer here </answer>"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-checkpoint", required=True, help="Path to dots.mocr HF checkpoint dir.")
    p.add_argument(
        "--image",
        default=None,
        help=(
            "Image file path or http(s) URL (jpg/png/webp). "
            "Omit for text-only mode (skips vision tower on all sides; useful "
            "to isolate whether a mismatch is multimodal-specific or hits the "
            "language backbone too)."
        ),
    )
    p.add_argument("--prompt", default="Describe this image.", help="User text prompt.")
    p.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt (defaults to the GRPO system prompt used in run-dotsocr2-8xgpu.sh).",
    )
    p.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Skip system prompt entirely (debug only).",
    )
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP), help="Model dtype.")
    p.add_argument(
        "--side",
        default="all",
        choices=["megatron", "sglang", "hf", "compare", "all"],
        help=(
            "Which side to run; 'all' does megatron→hf→sglang→compare in one shot. "
            "'hf' uses transformers.AutoModelForCausalLM as a third ground-truth "
            "reference so you can tell which engine is the broken one."
        ),
    )
    p.add_argument(
        "--dump-dir",
        default="/tmp/relax_dotsocr_debug",
        help="Directory for per-side logprob dumps and comparison report.",
    )
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.65,
        help="SGLang static memory fraction (lower this when sharing GPU with megatron).",
    )
    p.add_argument(
        "--top-n-worst",
        type=int,
        default=20,
        help="Report the N positions with the largest |Δ logprob|.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for ``model_parallel_cuda_manual_seed`` (matches relax default).",
    )
    p.add_argument(
        "--check-weights",
        action="store_true",
        default=True,
        help=(
            "After bridge.load_hf_weights(), round-trip weights back to HF format "
            "via bridge.export_hf_weights() and diff against the original safetensors. "
            "Tells you whether load is silently dropping/mangling weights."
        ),
    )
    p.add_argument("--no-check-weights", dest="check_weights", action="store_false")
    p.add_argument(
        "--no-attention-mask-megatron",
        action="store_true",
        default=False,
        help=(
            "Pass attention_mask=None to Megatron forward. "
            "Megatron's TE core_attention expects [B,1,T,T] boolean (or None → "
            "auto-causal); the [B,T] all-ones mask we send by default may be "
            "silently mis-interpreted as bidirectional. If this flag fixes the "
            "divergence, that's the bug."
        ),
    )
    p.add_argument(
        "--bypass-dots-wrapper",
        action="store_true",
        default=False,
        help=(
            "Call model.language_model.forward(input_ids=...) directly, bypassing "
            "DotsOCRModel.forward's embedding-clone + decoder_input data prep + "
            "custom _position_ids. If this aligns megatron to HF but normal path "
            "doesn't, the bug is in relax/models/dots_ocr/megatron/model.py "
            "wrapper rather than in megatron-bridge's Qwen2 attention itself."
        ),
    )
    p.add_argument(
        "--megatron-attention-backend",
        default="flash",
        choices=["flash", "fused", "unfused", "auto"],
        help=(
            "Megatron-TE attention backend. 'flash' = TE FusedAttention (default, "
            "matches RL training). 'unfused' = pure pytorch eager matmul attention "
            "(most HF-like — use to test whether the Megatron-vs-HF divergence is "
            "caused by TE flash attention itself)."
        ),
    )
    p.add_argument(
        "--inspect-layers",
        default="0",
        help=(
            "Comma-separated layer indices to install fine-grained sub-module "
            "hooks on (self_attention output + mlp output). Splits a "
            "'layer N diverges' finding into 'attention diverges' vs "
            "'mlp diverges'. Default: '0'."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared input prep — both sides must agree on the (text, image, input_ids)
# tuple, otherwise the comparison is meaningless.
# ---------------------------------------------------------------------------


def _build_messages(args: argparse.Namespace) -> list[dict]:
    """Mirror the rollout-side chat template (apply-chat-template + system
    prompt + multimodal user turn) so we exercise the same prefill SGLang would
    see during RL training.

    When ``--image`` is omitted, the user turn is a plain text string and no
    vision tokens are inserted.
    """
    messages = []
    if not args.no_system_prompt and args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    if args.image:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": args.image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": args.prompt})
    return messages


def _load_image_any(src: str):
    """Open a PIL image from a local path or an http(s) URL."""
    from io import BytesIO

    from PIL import Image

    if src.startswith(("http://", "https://")):
        import requests

        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(src).convert("RGB")


def _build_processor_inputs(args: argparse.Namespace):
    """Run the HF processor *once* to get the canonical post-expansion
    ``input_ids`` (+ ``pixel_values`` / ``image_grid_thw`` when an image is
    provided).

    Both sides will use these exact ``input_ids``; the megatron forward
    consumes ``pixel_values`` + ``image_grid_thw`` directly, while sglang re-
    tokenizes from text + image_data (and we assert its expanded ids match).
    When ``--image`` is omitted, this falls back to the raw tokenizer path so
    we exercise the language backbone only.
    """
    from transformers import AutoProcessor, AutoTokenizer

    messages = _build_messages(args)
    if args.image:
        processor = AutoProcessor.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        image = _load_image_any(args.image)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_out = processor(text=[text], images=[image], padding=False, return_tensors="pt")
        return processor, text, image, proc_out

    # Text-only path: the multimodal AutoProcessor for dots.mocr requires an
    # image; fall back to the tokenizer + a dict that mimics processor output.
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    proc_out = {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask")}
    return tokenizer, text, None, proc_out


# ---------------------------------------------------------------------------
# Megatron side
# ---------------------------------------------------------------------------


def _init_single_gpu_distributed(seed: int) -> None:
    """Bring up tp=pp=cp=ep=1 mpu state.

    Crucially seeds the model-parallel
    RNG via ``tensor_parallel.model_parallel_cuda_manual_seed`` — skipping
    this triggers ``cuda rng state model-parallel-rng is not added`` the
    first time a TP layer forwards (mirrors ``relax/backends/megatron/
    initialize.py:32``).
    """
    import torch.distributed as dist
    from megatron.core import mpu, tensor_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29503")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", world_size=1, rank=0)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        tensor_parallel.model_parallel_cuda_manual_seed(seed)


def _apply_single_gpu_provider_overrides(provider, dtype: torch.dtype) -> None:
    """Force the bridge provider into a 1-GPU configuration that mirrors
    relax's CLI overrides in ``relax.backends.megatron.model_provider``."""
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.variable_seq_lengths = False
    provider.apply_rope_fusion = False
    provider.attention_softmax_in_fp32 = True
    if hasattr(provider, "attention_backend"):
        provider.attention_backend = _CURRENT_ATTENTION_BACKEND
    provider.fp16 = dtype == torch.float16
    provider.bf16 = dtype == torch.bfloat16
    provider.params_dtype = dtype


# ---------------------------------------------------------------------------
# Layer-by-layer intermediate capture. We register forward hooks on the
# embedding, every transformer layer, the final norm, and the lm_head so we
# can localize the first point of divergence between HF and Megatron. All
# captured tensors are normalized to BSH/BSV (batch-first) and saved as
# bf16 CPU tensors to keep dump size manageable.
# ---------------------------------------------------------------------------


def _to_bsh(t: torch.Tensor, kind: str) -> torch.Tensor:
    """Megatron decoder layers + final_layernorm + output_layer produce
    SBH/SBV; HF produces BSH/BSV.

    Normalize both to BSH/BSV using ``kind`` as the hint of which side this
    tensor came from.
    """
    if kind == "megatron" and t.dim() == 3:
        return t.transpose(0, 1).contiguous()
    return t


def _capture_hook(store: dict, name: str, kind: str):
    def hook(_mod, _inp, out):
        # Many megatron layers return (hidden, residual_or_None) or
        # (logits, bias). Take the first tensor of any tuple/list.
        if isinstance(out, (tuple, list)):
            out = next((o for o in out if isinstance(o, torch.Tensor)), None)
            if out is None:
                return
        if not isinstance(out, torch.Tensor):
            return
        store[name] = _to_bsh(out, kind).detach().to(torch.bfloat16).cpu()

    return hook


def _capture_rope_hook(store: dict, name: str):
    """Special hook for rotary embedding modules.

    Megatron's RotaryEmbedding returns a freqs tensor (shape ``[seq, 1, 1,
    dim]``); HF's Qwen2RotaryEmbedding returns a ``(cos, sin)`` tuple. Both
    stored as float32 CPU tensors so the compare step can do the cos/sin math
    consistently.
    """

    def hook(_mod, _inp, out):
        if isinstance(out, (tuple, list)):
            for i, t in enumerate(out):
                if isinstance(t, torch.Tensor):
                    store[f"{name}.{i}"] = t.detach().float().cpu()
        elif isinstance(out, torch.Tensor):
            store[name] = out.detach().float().cpu()

    return hook


def _capture_input_hook(store: dict, name: str, kind: str):
    """Pre-hook variant: capture the *input* of a module, not its output.

    Used to grab HF Attention's pre-o_proj tensor (which is post-RoPE + post-
    softmax + post-V-matmul) so we can compare against Megatron's
    ``core_attention`` *output* (same logical position).
    """

    def pre_hook(_mod, args):
        if not args:
            return
        t = args[0]
        if not isinstance(t, torch.Tensor):
            return
        store[name] = _to_bsh(t, kind).detach().to(torch.bfloat16).cpu()

    return pre_hook


def _parse_inspect_layers(spec: str) -> set[int]:
    return {int(x) for x in spec.split(",") if x.strip()}


def _install_megatron_hooks(model, inspect_layers: set[int]) -> tuple[dict, list]:
    """Register hooks on relax DotsOCRModel → returns (captured_dict, handles).

    For layers in ``inspect_layers`` we additionally hook ``self_attention`` +
    its three sub-modules (linear_qkv, core_attention, linear_proj) so we can
    localize *inside* attention: linear_qkv tests fused-RMSNorm+QKV projection,
    core_attention tests RoPE+softmax+matmul, linear_proj tests the o_proj
    output.
    """
    captured: dict = {}
    handles: list = []
    lm = model.language_model
    handles.append(lm.embedding.register_forward_hook(_capture_hook(captured, "embed", "megatron")))
    # Scan for every rotary-like submodule and hook them all under unique
    # names. Different Megatron versions / TE versions place rotary at
    # GPTModel level (lm.rotary_pos_emb) OR per-layer
    # (decoder.layers.N.self_attention.rotary_pos_emb) OR inside core_attention.
    rope_targets = []
    for n, m in model.named_modules():
        nl = n.lower()
        if ("rotary" in nl or "rope" in nl) and isinstance(m, torch.nn.Module):
            rope_targets.append((n, m))
    if rope_targets:
        for n, m in rope_targets:
            key = f"rotary_freqs::{n}"
            handles.append(m.register_forward_hook(_capture_rope_hook(captured, key)))
        print(f"[megatron-hooks] hooked {len(rope_targets)} rotary-like modules:")
        for n, m in rope_targets[:10]:
            print(f"    - {n} ({type(m).__name__})")
        if len(rope_targets) > 10:
            print(f"    ... +{len(rope_targets) - 10} more")
    else:
        print("[megatron-hooks] WARNING no rotary-like submodules found at all — rotary may be a free function")
    for i, layer in enumerate(lm.decoder.layers):
        handles.append(layer.register_forward_hook(_capture_hook(captured, f"layer.{i:02d}", "megatron")))
        if i in inspect_layers:
            if hasattr(layer, "self_attention"):
                attn = layer.self_attention
                handles.append(
                    attn.register_forward_hook(_capture_hook(captured, f"layer.{i:02d}.attn_out", "megatron"))
                )
                for sub in ("linear_qkv", "core_attention", "linear_proj"):
                    if hasattr(attn, sub):
                        handles.append(
                            getattr(attn, sub).register_forward_hook(
                                _capture_hook(captured, f"layer.{i:02d}.attn.{sub}", "megatron")
                            )
                        )
            if hasattr(layer, "mlp"):
                handles.append(
                    layer.mlp.register_forward_hook(_capture_hook(captured, f"layer.{i:02d}.mlp_out", "megatron"))
                )
    if getattr(lm.decoder, "final_layernorm", None) is not None:
        handles.append(
            lm.decoder.final_layernorm.register_forward_hook(_capture_hook(captured, "final_norm", "megatron"))
        )
    handles.append(lm.output_layer.register_forward_hook(_capture_hook(captured, "lm_head", "megatron")))
    return captured, handles


def _install_hf_hooks(model, inspect_layers: set[int]) -> tuple[dict, list]:
    """Register hooks on HF DotsOCRForCausalLM (Qwen2-like layout)."""
    captured: dict = {}
    handles: list = []
    inner = model.model  # Qwen2Model
    handles.append(inner.embed_tokens.register_forward_hook(_capture_hook(captured, "embed", "hf")))
    rotary_mod = getattr(inner, "rotary_emb", None)
    if rotary_mod is not None:
        handles.append(rotary_mod.register_forward_hook(_capture_rope_hook(captured, "rotary_cos_sin")))
        print(f"[hf-hooks] rotary hook on model.model.rotary_emb ({type(rotary_mod).__name__})")
    else:
        # transformers 5.x may have lifted rotary into per-layer modules.
        cand = []
        for n, m in model.named_modules():
            if "rotary" in n.lower() or "rope" in n.lower():
                cand.append((n, type(m).__name__))
        print(f"[hf-hooks] WARNING model.model.rotary_emb is None. Rotary-like modules found: {cand[:10]}")
    for i, layer in enumerate(inner.layers):
        handles.append(layer.register_forward_hook(_capture_hook(captured, f"layer.{i:02d}", "hf")))
        if i in inspect_layers:
            if hasattr(layer, "self_attn"):
                sattn = layer.self_attn
                handles.append(sattn.register_forward_hook(_capture_hook(captured, f"layer.{i:02d}.attn_out", "hf")))
                # Q/K/V projections and o_proj — direct counterparts to
                # Megatron's linear_qkv split and linear_proj.
                for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    if hasattr(sattn, sub):
                        handles.append(
                            getattr(sattn, sub).register_forward_hook(
                                _capture_hook(captured, f"layer.{i:02d}.attn.{sub}", "hf")
                            )
                        )
                # Pre-hook on o_proj captures the attention compute result
                # (post-RoPE/softmax/V-matmul, pre-output-projection). Same
                # logical position as Megatron's ``core_attention`` output —
                # we use the same key so compare_intermediates pairs them up.
                if hasattr(sattn, "o_proj"):
                    handles.append(
                        sattn.o_proj.register_forward_pre_hook(
                            _capture_input_hook(captured, f"layer.{i:02d}.attn.core_attention", "hf")
                        )
                    )
            if hasattr(layer, "mlp"):
                handles.append(
                    layer.mlp.register_forward_hook(_capture_hook(captured, f"layer.{i:02d}.mlp_out", "hf"))
                )
    final_norm = getattr(inner, "norm", None) or getattr(inner, "final_layernorm", None)
    if final_norm is not None:
        handles.append(final_norm.register_forward_hook(_capture_hook(captured, "final_norm", "hf")))
    handles.append(model.lm_head.register_forward_hook(_capture_hook(captured, "lm_head", "hf")))
    return captured, handles


def _intermediates_path(dump_dir: Path, side: str) -> Path:
    return dump_dir / f"{side}.intermediates.pt"


def _load_hf_safetensors_to_cpu(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Read the canonical HF weights straight from disk so we don't depend on
    AutoModel building the live model.

    Returns ``{hf_name: cpu_tensor}``.
    """
    import glob

    from safetensors import safe_open

    state: dict[str, torch.Tensor] = {}
    shards = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not shards:
        raise RuntimeError(f"no .safetensors files under {checkpoint_dir}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as st:
            for name in st.keys():
                state[name] = st.get_tensor(name)
    return state


def _print_provider_vs_hf_config(provider, hf_config) -> None:
    """Side-by-side dump of attention-/RoPE-/norm-relevant fields.

    When the forward output diverges but weights round-trip cleanly, the bug is
    almost always a provider attribute that didn't get propagated from the HF
    config (rope_theta, attention_bias, head_dim, rope_scaling…). This table
    makes any such mismatch immediately visible.
    """
    # Show the *raw* config storage so it's obvious where HF stashed rope_theta.
    # dots.mocr is non-standard: rope_theta lives in rope_scaling['rope_theta'],
    # not at top level. transformers 5.x auto-migrates this into rope_parameters,
    # which is what megatron-bridge's rope_theta_from_hf actually reads.
    print("\n========== RAW HF config rope/attention fields ==========")
    for k in (
        "rope_theta",
        "rope_scaling",
        "rope_parameters",
        "partial_rotary_factor",
        "max_window_layers",
        "use_sliding_window",
        "sliding_window",
        "attention_bias",
        "_attn_implementation",
        "head_dim",
    ):
        if hasattr(hf_config, k):
            print(f"  hf_config.{k} = {getattr(hf_config, k)!r}")
    print("  hf_config.auto_map = ", getattr(hf_config, "auto_map", None))
    print("  hf_config.architectures = ", getattr(hf_config, "architectures", None))

    rope_scaling_hf = getattr(hf_config, "rope_scaling", None)
    rope_scaling_pv = getattr(provider, "rope_scaling", None) or getattr(provider, "use_rope_scaling", None)
    rope_interleaved_pv = getattr(provider, "rotary_interleaved", None)
    # HF Qwen2 family is non-interleaved (Llama "halves" RoPE). The provider
    # default in TransformerConfig is False (matches HF). We report both so
    # any deviation is loud.
    rows = [
        ("rope_theta", getattr(hf_config, "rope_theta", None), getattr(provider, "rotary_base", None)),
        ("rope_scaling", rope_scaling_hf, rope_scaling_pv),
        ("rotary_percent", "1.0 (HF Qwen2 default)", getattr(provider, "rotary_percent", None)),
        ("rotary_interleaved", "False (HF Qwen2 = Llama halves)", rope_interleaved_pv),
        ("apply_rope_fusion", "N/A", getattr(provider, "apply_rope_fusion", None)),
        ("rms_norm_eps", getattr(hf_config, "rms_norm_eps", None), getattr(provider, "layernorm_epsilon", None)),
        ("hidden_size", getattr(hf_config, "hidden_size", None), getattr(provider, "hidden_size", None)),
        ("ffn_hidden_size", getattr(hf_config, "intermediate_size", None), getattr(provider, "ffn_hidden_size", None)),
        (
            "num_attention_heads",
            getattr(hf_config, "num_attention_heads", None),
            getattr(provider, "num_attention_heads", None),
        ),
        (
            "num_key_value_heads",
            getattr(hf_config, "num_key_value_heads", None),
            getattr(provider, "num_query_groups", None),
        ),
        (
            "head_dim",
            getattr(hf_config, "head_dim", None)
            or (getattr(hf_config, "hidden_size", 0) // max(getattr(hf_config, "num_attention_heads", 1), 1)),
            getattr(provider, "kv_channels", None),
        ),
        ("attention_bias", getattr(hf_config, "attention_bias", None), getattr(provider, "add_qkv_bias", None)),
        ("qk_layernorm", getattr(hf_config, "qk_layernorm", False), getattr(provider, "qk_layernorm", None)),
        ("hidden_act", getattr(hf_config, "hidden_act", None), getattr(provider, "activation_func", None)),
        (
            "max_position_embeddings",
            getattr(hf_config, "max_position_embeddings", None),
            getattr(provider, "seq_length", None),
        ),
        (
            "tie_word_embeddings",
            getattr(hf_config, "tie_word_embeddings", None),
            getattr(provider, "share_embeddings_and_output_weights", None),
        ),
        ("vocab_size", getattr(hf_config, "vocab_size", None), getattr(provider, "vocab_size", None)),
        (
            "attention_softmax_in_fp32",
            "N/A (HF eager uses fp32)",
            getattr(provider, "attention_softmax_in_fp32", None),
        ),
    ]

    def _norm(v):
        # Make values structurally comparable: strings stay; callables print as their name.
        if callable(v):
            return getattr(v, "__name__", repr(v))
        if isinstance(v, str) and v.startswith(("N/A", "1.0 (", "False (")):
            return v  # already an annotation, don't compare
        return v

    print("\n========== forward-path config: HF config vs Megatron provider ==========")
    print(f"  {'field':<28} {'HF config':<35} {'Megatron provider':<35} match")
    for name, hf_v, pv_v in rows:
        hf_n, pv_n = _norm(hf_v), _norm(pv_v)
        annot = isinstance(hf_n, str) and (hf_n.startswith("N/A") or "(" in hf_n)
        ok = "✓" if annot else ("✓" if hf_n == pv_n else "✗ MISMATCH")
        print(f"  {name:<28} {str(hf_n):<35} {str(pv_n):<35} {ok}")
    print()


def _check_megatron_weights_vs_hf(args: argparse.Namespace, bridge, model) -> None:
    """Round-trip megatron weights back to HF format and compare against the
    on-disk safetensors.

    The bridge's ``export_hf_weights`` un-fuses linear_qkv into q/k/v and
    linear_fc1 into gate/up using the same mapping that ``load_hf_weights``
    consumed in reverse — so a clean round-trip means the bridge mapping is OK
    and the loaded weights match what the HF model started with. Any diff here
    is either a missing AutoMapping or numerical corruption during load.
    """
    print("\n========== weight check: megatron (post-load) vs HF safetensors ==========")
    print(f"  checkpoint = {args.hf_checkpoint}")

    print("  reading on-disk HF safetensors ...")
    hf_disk = _load_hf_safetensors_to_cpu(args.hf_checkpoint)
    print(f"  HF on disk: {len(hf_disk)} tensors")

    print("  round-tripping megatron -> HF via bridge.export_hf_weights ...")
    exported: dict[str, torch.Tensor] = {}
    for item in bridge.export_hf_weights([model], cpu=True, show_progress=False):
        # HFWeightTuple may be a 2- or 3-tuple depending on bridge version.
        name = item[0]
        tensor = item[1]
        exported[name] = tensor.detach().cpu()
    print(f"  Megatron round-trip: {len(exported)} tensors")

    missing_in_mg = [k for k in hf_disk.keys() if k not in exported]
    extra_in_mg = [k for k in exported.keys() if k not in hf_disk]
    print(f"  missing from megatron: {len(missing_in_mg)}  extra in megatron: {len(extra_in_mg)}")
    if missing_in_mg:
        print(f"    first 10 missing: {missing_in_mg[:10]}")
    if extra_in_mg:
        print(f"    first 10 extra:   {extra_in_mg[:10]}")

    rows: list[tuple[str, tuple, tuple, float, float, float]] = []
    for name, ref in hf_disk.items():
        if name not in exported:
            continue
        mg = exported[name]
        if ref.shape != mg.shape:
            rows.append((name, tuple(mg.shape), tuple(ref.shape), float("nan"), float("nan"), float("nan")))
            continue
        a = mg.float().reshape(-1)
        b = ref.float().reshape(-1)
        diff = (a - b).abs()
        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
        rows.append((name, tuple(mg.shape), tuple(ref.shape), cos, diff.mean().item(), diff.max().item()))

    # Highlight layer 0 first (smoking gun for "first-layer-diverges"), then
    # show overall worst-N across the whole model.
    layer0 = [r for r in rows if r[0].startswith("model.layers.0.")]
    print(f"\n  --- layer 0 weights ({len(layer0)} tensors) ---")
    print(f"  {'name':<55} {'shape_mg':<20} {'shape_hf':<20} {'cos':>10} {'|Δ|mean':>10} {'|Δ|max':>10}")
    for name, smg, shf, cos, dm, dx in sorted(layer0, key=lambda r: r[0]):
        print(f"  {name:<55} {str(smg):<20} {str(shf):<20} {cos:>10.6f} {dm:>10.6f} {dx:>10.6f}")

    bad = [r for r in rows if (not (r[3] != r[3])) and (r[3] < 0.999 or r[5] > 0.01)]
    bad.sort(key=lambda r: (r[3], -r[5]))
    print(f"\n  --- worst {min(30, len(bad))} weights across full model (cos<0.999 or |Δ|max>0.01) ---")
    for name, smg, shf, cos, dm, dx in bad[:30]:
        print(f"  {name:<55} {str(smg):<20} {str(shf):<20} {cos:>10.6f} {dm:>10.6f} {dx:>10.6f}")
    if not bad:
        print("  (none — all loaded weights round-trip cleanly to HF format)")
    print()


@torch.no_grad()
def run_megatron(args: argparse.Namespace) -> DumpRecord:
    """Build the Megatron DotsOCR model the same way relax does for the actor,
    forward a single (text, image) sample, return per-token logprobs."""
    import torch.nn.functional as F
    from megatron.bridge import AutoBridge

    # Side-effect import: relax/models/__init__.py runs the
    # @MegatronModelBridge.register_bridge(source="DotsOCRForCausalLM", ...)
    # decorator on DotsOCRBridge. Without this AutoBridge.from_hf_pretrained
    # raises "Model architecture 'DotsOCRForCausalLM' is not yet supported".
    import relax.models  # noqa: F401

    dtype = DTYPE_MAP[args.dtype]
    _init_single_gpu_distributed(args.seed)

    global _CURRENT_ATTENTION_BACKEND
    _CURRENT_ATTENTION_BACKEND = args.megatron_attention_backend
    print(f"[megatron] using attention_backend={_CURRENT_ATTENTION_BACKEND}")

    processor, text, image, proc_out = _build_processor_inputs(args)
    print(f"[megatron] chat_template text (first 300 chars): {text[:300]!r}")

    print(f"[megatron] loading AutoBridge from {args.hf_checkpoint}")
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    _apply_single_gpu_provider_overrides(provider, dtype)
    provider.finalize()

    print("[megatron] building model via provider.provide()")
    model = provider.provide(pre_process=True, post_process=True)
    model = model.cuda().to(dtype).eval()

    print("[megatron] loading HF weights -> Megatron via bridge.load_hf_weights")
    bridge.load_hf_weights([model])

    if args.check_weights:
        _check_megatron_weights_vs_hf(args, bridge, model)
        try:
            hf_config = bridge.hf_pretrained.config
            _print_provider_vs_hf_config(provider, hf_config)
        except Exception as e:
            print(f"[megatron] provider/config side-by-side dump skipped: {e}")

    # Dump every rotary-like submodule's inv_freq buffer (and key scalar
    # attrs) so we can compare to HF's inv_freq even when Megatron's rotary
    # is buried behind a free-function apply_rotary path that's not hookable.
    print("\n[megatron] scanning for rotary submodules + inv_freq buffers...")
    found_any = False
    for n, m in model.named_modules():
        if not ("rotary" in n.lower() or "rope" in n.lower()):
            continue
        found_any = True
        inv = getattr(m, "inv_freq", None)
        base = getattr(m, "rotary_base", None) or getattr(m, "base", None)
        interleaved = getattr(m, "rotary_interleaved", None)
        kvc = getattr(m, "kv_channels", None) or getattr(m, "dim", None)
        print(f"  module: {n} ({type(m).__name__})")
        print(f"    rotary_base={base!r} rotary_interleaved={interleaved!r} dim={kvc!r}")
        if isinstance(inv, torch.Tensor):
            print(f"    inv_freq: shape={tuple(inv.shape)} dtype={inv.dtype} device={inv.device}")
            print(f"    inv_freq[:8]={inv[:8].tolist()}")
            # Persist for the compare step.
            torch.save(
                {
                    "inv_freq": inv.detach().float().cpu(),
                    "rotary_base": base,
                    "rotary_interleaved": interleaved,
                    "module": n,
                },
                Path(args.dump_dir) / "megatron.rope_invfreq.pt",
            )
        else:
            print(f"    inv_freq: not a tensor ({type(inv).__name__})")
    if not found_any:
        print("  no rotary-like submodules found at all")

    device = torch.device("cuda")
    input_ids = proc_out["input_ids"].to(device=device, dtype=torch.long)  # [1, T]
    attention_mask = proc_out.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    if args.no_attention_mask_megatron:
        print("[megatron] forcing attention_mask=None (rely on TE auto-causal mask)")
        attention_mask = None

    pixel_values = None
    image_grid_thw = None
    if args.image:
        pixel_values = proc_out["pixel_values"].to(device=device, dtype=dtype)
        image_grid_thw = proc_out["image_grid_thw"].to(device=device, dtype=torch.long)
        # Sanity: the vision tower derives cu_seqlens from grid_thw.device, and
        # flash_attn requires cu_seqlens on CUDA. Bail loudly if anything is CPU.
        for name, t in [
            ("input_ids", input_ids),
            ("pixel_values", pixel_values),
            ("image_grid_thw", image_grid_thw),
        ]:
            assert t.is_cuda, f"{name} ended up on {t.device}; expected CUDA"
        print(
            f"[megatron] forward input_ids={tuple(input_ids.shape)}@{input_ids.dtype} "
            f"pixel_values={tuple(pixel_values.shape)}@{pixel_values.dtype} "
            f"grid_thw={image_grid_thw.tolist()}"
        )
    else:
        print(f"[megatron] text-only forward input_ids={tuple(input_ids.shape)}@{input_ids.dtype}")

    inspect_layers = _parse_inspect_layers(args.inspect_layers)
    captured, handles = _install_megatron_hooks(model, inspect_layers)
    try:
        if args.bypass_dots_wrapper:
            # Bypass relax DotsOCRModel.forward — call the inner GPTModel
            # directly with the same inputs HF gets. Isolates whether the
            # divergence comes from our wrapper's data prep (embedding clone
            # + decoder_input + custom position_ids) vs the megatron-bridge
            # attention path itself.
            assert pixel_values is None, "bypass mode is text-only (no vision injection)"
            print("[megatron] BYPASS: calling language_model directly")
            logits = model.language_model(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=attention_mask,
            )
        else:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
    finally:
        for h in handles:
            h.remove()
    # GPTModel returns either (s, b, v) or (b, s, v) depending on parallel_output;
    # with tp=1, sp=False, parallel_output=True the layout is (s, b, v).
    if logits.dim() == 3 and logits.shape[0] == input_ids.shape[1]:
        logits = logits.transpose(0, 1).contiguous()  # -> (1, T, V)
    assert logits.shape[:2] == (1, input_ids.shape[1]), (
        f"unexpected logits shape {tuple(logits.shape)} for input_ids {tuple(input_ids.shape)}"
    )

    intermediates_path = _intermediates_path(Path(args.dump_dir), "megatron")
    intermediates_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, intermediates_path)
    print(f"[megatron] dumped {len(captured)} intermediate tensors -> {intermediates_path}")

    logprobs_all = F.log_softmax(logits.float(), dim=-1)  # (1, T, V)
    ids = input_ids[0].tolist()
    per_token: list[Optional[float]] = [None]
    for t in range(1, len(ids)):
        per_token.append(float(logprobs_all[0, t - 1, ids[t]].item()))

    rec = DumpRecord(
        side="megatron",
        input_ids=ids,
        logprobs=per_token,
        meta={
            "dtype": args.dtype,
            "vocab_size": int(logits.shape[-1]),
            "prompt_text_first_300": text[:300],
            "pixel_values_shape": (list(pixel_values.shape) if pixel_values is not None else None),
            "image_grid_thw": (image_grid_thw.tolist() if image_grid_thw is not None else None),
            "image": args.image,
        },
    )

    # Free GPU memory before sglang spins up its worker subprocess.
    del logits, logprobs_all, model, bridge
    gc.collect()
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------------------
# HuggingFace reference side — independent third opinion. Loaded via
# ``AutoModelForCausalLM`` + ``trust_remote_code`` so we get whatever the
# checkpoint shipped as its canonical implementation. When megatron and
# sglang disagree, this side tells you which one matches the reference.
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_hf(args: argparse.Namespace) -> DumpRecord:
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    dtype = DTYPE_MAP[args.dtype]
    device = torch.device("cuda")

    _, text, _, proc_out = _build_processor_inputs(args)
    print(f"[hf] chat_template text (first 300 chars): {text[:300]!r}")

    print(f"[hf] loading AutoModelForCausalLM from {args.hf_checkpoint} (trust_remote_code)")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.hf_checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )

    # Dump every rotary-like submodule's inv_freq buffer for direct comparison
    # to Megatron's inv_freq. Same idea as the megatron-side dump.
    print("\n[hf] scanning for rotary submodules + inv_freq buffers...")
    found_any = False
    for n, m in model.named_modules():
        if not ("rotary" in n.lower() or "rope" in n.lower()):
            continue
        found_any = True
        inv = getattr(m, "inv_freq", None)
        print(f"  module: {n} ({type(m).__name__})")
        if isinstance(inv, torch.Tensor):
            print(f"    inv_freq: shape={tuple(inv.shape)} dtype={inv.dtype}")
            print(f"    inv_freq[:8]={inv[:8].tolist()}")
            torch.save(
                {"inv_freq": inv.detach().float().cpu(), "module": n},
                Path(args.dump_dir) / "hf.rope_invfreq.pt",
            )
        else:
            print(f"    inv_freq: not a tensor ({type(inv).__name__})")
    if not found_any:
        print("  no rotary-like submodules found at all")

    input_ids = proc_out["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = proc_out.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    fwd_kwargs = dict(input_ids=input_ids, use_cache=False)
    if args.image:
        pixel_values = proc_out["pixel_values"].to(device=device, dtype=dtype)
        image_grid_thw = proc_out["image_grid_thw"].to(device=device, dtype=torch.long)
        fwd_kwargs["pixel_values"] = pixel_values
        fwd_kwargs["image_grid_thw"] = image_grid_thw
        print(
            f"[hf] forward input_ids={tuple(input_ids.shape)} "
            f"pixel_values={tuple(pixel_values.shape)} grid_thw={image_grid_thw.tolist()}"
        )
    else:
        pixel_values = None
        image_grid_thw = None
        print(f"[hf] text-only forward input_ids={tuple(input_ids.shape)}")
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask
    inspect_layers = _parse_inspect_layers(args.inspect_layers)
    captured, handles = _install_hf_hooks(model, inspect_layers)
    try:
        outputs = model(**fwd_kwargs)
    finally:
        for h in handles:
            h.remove()
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    assert logits.dim() == 3 and logits.shape[:2] == (1, input_ids.shape[1]), (
        f"unexpected HF logits shape {tuple(logits.shape)} for input_ids {tuple(input_ids.shape)}"
    )

    intermediates_path = _intermediates_path(Path(args.dump_dir), "hf")
    intermediates_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, intermediates_path)
    print(f"[hf] dumped {len(captured)} intermediate tensors -> {intermediates_path}")

    logprobs_all = F.log_softmax(logits.float(), dim=-1)
    ids = input_ids[0].tolist()
    per_token: list[Optional[float]] = [None]
    for t in range(1, len(ids)):
        per_token.append(float(logprobs_all[0, t - 1, ids[t]].item()))

    rec = DumpRecord(
        side="hf",
        input_ids=ids,
        logprobs=per_token,
        meta={
            "dtype": args.dtype,
            "vocab_size": int(logits.shape[-1]),
            "prompt_text_first_300": text[:300],
            "pixel_values_shape": (list(pixel_values.shape) if pixel_values is not None else None),
            "image_grid_thw": (image_grid_thw.tolist() if image_grid_thw is not None else None),
            "image": args.image,
        },
    )

    del logits, logprobs_all, model, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------------------
# SGLang side
# ---------------------------------------------------------------------------


def _resolve_external_pkg() -> str:
    """Matches the value passed via ``--sglang-external-model-package`` in
    ``run-dotsocr2-8xgpu.sh``."""
    return "relax.models.dots_ocr.sglang"


@torch.no_grad()
def run_sglang(args: argparse.Namespace) -> DumpRecord:
    """Launch the offline ``sglang.Engine`` with the relax external model
    package, fire one generate() call with ``return_logprob=True``, and pull
    per-input-token logprobs out of meta_info."""
    # MUST be set *before* importing sglang so registry picks up the external
    # model package (same as relax/backends/sglang/sglang_engine.py:_init_normal).
    external_pkg = _resolve_external_pkg()
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = external_pkg
    print(f"[sglang] SGLANG_EXTERNAL_MODEL_PACKAGE={external_pkg}")

    from sglang import Engine

    _, text, _, proc_out = _build_processor_inputs(args)
    expected_ids = proc_out["input_ids"][0].tolist()
    print(f"[sglang] chat_template text (first 300 chars): {text[:300]!r}")

    sglang_dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.dtype]
    engine = Engine(
        model_path=args.hf_checkpoint,
        tp_size=1,
        dtype=sglang_dtype,
        trust_remote_code=True,
        mem_fraction_static=args.mem_fraction_static,
        # Greedy & deterministic — sampling decisions don't affect input-token
        # logprobs but max_new_tokens=1 keeps the call cheap.
        random_seed=args.seed,
        # Disable cuda graphs for fair comparison with megatron forward path;
        # they shouldn't change numerics but rule them out.
        disable_cuda_graph=True,
    )

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    generate_kwargs = dict(
        prompt=text,
        sampling_params=sampling_params,
        return_logprob=True,
        # logprob_start_len=0 → return logprobs for *every* input token; the
        # first position is always None (no preceding context).
        logprob_start_len=0,
    )
    if args.image:
        generate_kwargs["image_data"] = [args.image]
    out = engine.generate(**generate_kwargs)
    engine.shutdown()

    # SGLang's input_token_logprobs entries are (logprob, token_id, token_text)
    # tuples; the first one is (None, first_token_id, ...).
    meta = out["meta_info"]
    raw = meta["input_token_logprobs"]
    sglang_ids = [int(item[1]) for item in raw]
    per_token: list[Optional[float]] = [(None if item[0] is None else float(item[0])) for item in raw]

    if sglang_ids != expected_ids:
        # Not fatal — image-token expansion can differ depending on processor
        # registration. Surface it loudly so the user knows the comparison is
        # not strictly token-aligned.
        n_match = sum(1 for a, b in zip(sglang_ids, expected_ids) if a == b)
        print(
            f"[sglang] WARNING: input_ids differ from HF-processor expectation "
            f"len(sglang)={len(sglang_ids)} len(expected)={len(expected_ids)} "
            f"matching_prefix_count={n_match}",
            file=sys.stderr,
        )

    return DumpRecord(
        side="sglang",
        input_ids=sglang_ids,
        logprobs=per_token,
        meta={
            "dtype": args.dtype,
            "prompt_text_first_300": text[:300],
            "expected_input_ids_first_50": expected_ids[:50],
        },
    )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def _dump_path(dump_dir: Path, side: str) -> Path:
    return dump_dir / f"{side}.json"


def save_record(rec: DumpRecord, dump_dir: Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = _dump_path(dump_dir, rec.side)
    with path.open("w") as f:
        json.dump(
            {
                "side": rec.side,
                "input_ids": rec.input_ids,
                "logprobs": rec.logprobs,
                "meta": rec.meta,
            },
            f,
        )
    print(f"[{rec.side}] dumped {len(rec.input_ids)} tokens -> {path}")


def load_record(dump_dir: Path, side: str) -> DumpRecord:
    path = _dump_path(dump_dir, side)
    with path.open("r") as f:
        data = json.load(f)
    return DumpRecord(side=data["side"], input_ids=data["input_ids"], logprobs=data["logprobs"], meta=data["meta"])


def _compare_pair(a: DumpRecord, b: DumpRecord, top_n_worst: int) -> None:
    """Print id-prefix alignment + |Δ| stats + cosine + worst positions for one
    (a, b) pair.

    Each pair gets its own alignment because mm-image-pad sentinels may differ
    across engines and the divergence point is pair- specific.
    """
    label = f"{a.side:>8} vs {b.side:<8}"
    print(f"\n---------- {label} ----------")
    print(f"len({a.side})={len(a.input_ids)}  len({b.side})={len(b.input_ids)}")

    n = min(len(a.input_ids), len(b.input_ids))
    aligned_n = 0
    for i in range(n):
        if a.input_ids[i] != b.input_ids[i]:
            break
        aligned_n = i + 1
    print(f"aligned id-prefix = {aligned_n} / {n}")
    if aligned_n < n:
        i = aligned_n
        print(f"first id mismatch at pos={i}: {a.side}={a.input_ids[i]} {b.side}={b.input_ids[i]} (truncating)")

    diffs: list[tuple[int, int, float, float, float]] = []
    for i in range(1, aligned_n):  # skip position 0 (no preceding context)
        x = a.logprobs[i]
        y = b.logprobs[i]
        if x is None or y is None:
            continue
        diffs.append((i, a.input_ids[i], x, y, x - y))

    if not diffs:
        print("no comparable positions (both sides have None or no overlap)")
        return

    abs_diffs_t = torch.tensor([abs(d[4]) for d in diffs])
    print(f"compared {len(diffs)} positions")
    print(
        f"|Δ| mean={abs_diffs_t.mean().item():.6f} "
        f"max={abs_diffs_t.max().item():.6f} "
        f"p50={abs_diffs_t.median().item():.6f} "
        f"p90={abs_diffs_t.quantile(0.9).item():.6f} "
        f"p99={abs_diffs_t.quantile(0.99).item():.6f}"
    )

    a_lp = torch.tensor([d[2] for d in diffs], dtype=torch.float64)
    b_lp = torch.tensor([d[3] for d in diffs], dtype=torch.float64)
    cos_lp = torch.nn.functional.cosine_similarity(a_lp, b_lp, dim=0).item()
    cos_p = torch.nn.functional.cosine_similarity(a_lp.exp(), b_lp.exp(), dim=0).item()
    print(f"cosine(logprob) = {cos_lp:.8f}")
    print(f"cosine(prob)    = {cos_p:.8f}")

    diffs.sort(key=lambda d: abs(d[4]), reverse=True)
    n_show = min(top_n_worst, len(diffs))
    print(f"\ntop {n_show} worst positions ({a.side} - {b.side}):")
    print(f"{'pos':>6} {'token_id':>10} {a.side:>12} {b.side:>12} {'delta':>12}")
    for pos, tok, x, y, d in diffs[:n_show]:
        print(f"{pos:>6d} {tok:>10d} {x:>12.4f} {y:>12.4f} {d:>+12.4f}")


def compare(records: dict[str, DumpRecord], top_n_worst: int) -> None:
    """Pairwise-compare every available side.

    With megatron+sglang+hf this prints 3 blocks; with only 2 sides present it
    prints 1 block. The pairwise view is what lets you triangulate which engine
    is the broken one — e.g. if megatron-vs-hf and megatron-vs-sglang are both
    bad but sglang-vs-hf is near 1.0, megatron is the culprit.
    """
    print("\n========== compare summary ==========")
    available = sorted(records.keys())
    print(f"sides present: {available}")
    pairs = [(a, b) for i, a in enumerate(available) for b in available[i + 1 :]]
    if not pairs:
        print("nothing to compare — need at least 2 sides")
        return
    for a_name, b_name in pairs:
        _compare_pair(records[a_name], records[b_name], top_n_worst)


def _stats_one_tensor(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-tensor diagnostics: flatten cosine + |Δ|/relative-error stats.

    Both inputs are upcast to float32 for stable accumulation; the relative
    error denominator uses |b| with a small floor to avoid div-by-zero at near-
    zero entries (typical in attention masks and post-RMSNorm tails).
    """
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    n = min(a32.numel(), b32.numel())
    a32, b32 = a32[:n], b32[:n]
    cos = torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0), dim=1).item()
    diff = (a32 - b32).abs()
    return {
        "n": n,
        "shape_a": tuple(a.shape),
        "shape_b": tuple(b.shape),
        "cos": cos,
        "abs_mean": diff.mean().item(),
        "abs_max": diff.max().item(),
        "rel_mean": (diff / (b32.abs() + 1e-6)).mean().item(),
        "norm_a": a32.norm().item(),
        "norm_b": b32.norm().item(),
    }


def _trim_to_aligned_prefix(a: torch.Tensor, b: torch.Tensor, aligned_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For BSH/BSV tensors, restrict to the first ``aligned_n`` positions so we
    don't include image-pad sentinels that differ between engines."""
    if a.dim() >= 2 and b.dim() >= 2:
        n = min(a.shape[1], b.shape[1], aligned_n)
        return a[:, :n], b[:, :n]
    return a, b


def _compare_rope_freqs(mg_dict: dict, hf_dict: dict) -> None:
    """Compare RoPE cos/sin between Megatron and HF for the first few
    positions. Megatron stores ``rotary_freqs`` (raw freqs tensor → cos/sin
    computed via .cos()/.sin()); HF stores ``rotary_cos_sin.0`` and
    ``rotary_cos_sin.1`` directly.

    For RoPE to match, both sides must agree on:
      1. inv_freq construction (rope_theta, head_dim)
      2. position application (positions used)
      3. duplicate-along-last-dim layout (cat([f,f]) vs interleave)
    """
    # mg may have hooked multiple rotary modules; pick the first one whose
    # output looks like a [seq, ..., dim] tensor (skip empty / per-layer
    # placeholders if any).
    mg_rotary_keys = sorted(k for k in mg_dict if k.startswith("rotary_freqs"))
    mg_freqs = None
    mg_freqs_src = None
    for k in mg_rotary_keys:
        t = mg_dict[k]
        if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[-1] >= 32:
            mg_freqs = t
            mg_freqs_src = k
            break
    hf_cos = hf_dict.get("rotary_cos_sin.0")
    hf_sin = hf_dict.get("rotary_cos_sin.1")
    if mg_freqs is None or hf_cos is None or hf_sin is None:
        print(
            f"\n[rope] RoPE freq capture incomplete: "
            f"mg_freqs={'present' if mg_freqs is not None else 'MISSING'} "
            f"hf_cos={'present' if hf_cos is not None else 'MISSING'} "
            f"hf_sin={'present' if hf_sin is not None else 'MISSING'}. "
            "Check the [megatron-hooks]/[hf-hooks] WARNING lines printed during model build "
            "for where rotary modules actually live in this version."
        )
        mg_keys = [k for k in mg_dict.keys() if "rotary" in k.lower() or "rope" in k.lower()]
        hf_keys = [k for k in hf_dict.keys() if "rotary" in k.lower() or "rope" in k.lower()]
        print(f"  mg captured keys (rotary-like): {mg_keys}")
        print(f"  hf captured keys (rotary-like): {hf_keys}")
        return

    # Megatron freqs shape: [seq, 1, 1, dim] typically. Reduce to [seq, dim].
    mg_freqs_2d = mg_freqs
    while mg_freqs_2d.dim() > 2:
        mg_freqs_2d = mg_freqs_2d.squeeze(1)
    if mg_freqs_2d.dim() != 2:
        print(f"\n[rope] unexpected mg freqs shape {tuple(mg_freqs.shape)}")
        return
    mg_cos = mg_freqs_2d.cos()
    mg_sin = mg_freqs_2d.sin()

    # HF cos/sin shape: [batch, seq, dim] typically. Drop batch.
    if hf_cos.dim() == 3:
        hf_cos = hf_cos[0]
        hf_sin = hf_sin[0]

    print("\n========== RoPE freq check (megatron vs hf) ==========")
    print(f"  megatron freqs source:    {mg_freqs_src}")
    print(f"  megatron freqs raw shape: {tuple(mg_freqs.shape)} -> reduced to {tuple(mg_freqs_2d.shape)}")
    print(f"  hf cos shape:             {tuple(hf_cos.shape)}")
    print(f"  hf sin shape:             {tuple(hf_sin.shape)}")
    n = min(mg_cos.shape[0], hf_cos.shape[0])
    d = min(mg_cos.shape[-1], hf_cos.shape[-1])
    mg_cos, mg_sin = mg_cos[:n, :d], mg_sin[:n, :d]
    hf_cos, hf_sin = hf_cos[:n, :d], hf_sin[:n, :d]
    cos_diff = (mg_cos - hf_cos).abs()
    sin_diff = (mg_sin - hf_sin).abs()
    cos_cos = torch.nn.functional.cosine_similarity(
        mg_cos.reshape(-1).unsqueeze(0), hf_cos.reshape(-1).unsqueeze(0), dim=1
    ).item()
    sin_cos = torch.nn.functional.cosine_similarity(
        mg_sin.reshape(-1).unsqueeze(0), hf_sin.reshape(-1).unsqueeze(0), dim=1
    ).item()
    print(f"  cos: |Δ|mean={cos_diff.mean():.6f} max={cos_diff.max():.6f} cosine_sim={cos_cos:.8f}")
    print(f"  sin: |Δ|mean={sin_diff.mean():.6f} max={sin_diff.max():.6f} cosine_sim={sin_cos:.8f}")

    # Per-position diagnostic for the first few positions and the first 8 + last 8
    # dims of each. If RoPE storage is duplicate-cat (first half == second half),
    # mg_cos[pos, 0] should equal mg_cos[pos, dim//2].
    half = d // 2
    print(f"\n  per-position sample (first 4 positions, dim={d}, half={half}):")
    print(
        f"  {'pos':>4} {'mg_cos[0]':>12} {'hf_cos[0]':>12} {'mg_cos[half]':>14} {'hf_cos[half]':>14} {'mg_cos[1]':>12} {'hf_cos[1]':>12}"
    )
    for pos in range(min(4, n)):
        print(
            f"  {pos:>4} "
            f"{mg_cos[pos, 0].item():>12.6f} {hf_cos[pos, 0].item():>12.6f} "
            f"{mg_cos[pos, half].item():>14.6f} {hf_cos[pos, half].item():>14.6f} "
            f"{mg_cos[pos, 1].item():>12.6f} {hf_cos[pos, 1].item():>12.6f}"
        )
    # If mg_cos[pos, 0] != mg_cos[pos, half], megatron's freqs are NOT
    # duplicate-cat layout — that mismatch with HF's duplicate-cat would
    # cause exactly this kind of attention divergence.
    same_layout = torch.allclose(mg_cos[:, 0], mg_cos[:, half], atol=1e-5)
    hf_same_layout = torch.allclose(hf_cos[:, 0], hf_cos[:, half], atol=1e-5)
    print(f"\n  megatron uses duplicate-cat layout (cos[0]==cos[half]): {same_layout}")
    print(f"  hf       uses duplicate-cat layout (cos[0]==cos[half]): {hf_same_layout}")
    if same_layout != hf_same_layout:
        print("  ✗ LAYOUT MISMATCH — this is the RoPE bug")


def _compare_megatron_qkv_split_vs_hf(mg_dict: dict, hf_dict: dict, aligned_n: int) -> None:
    """Megatron's fused ``linear_qkv`` output for GQA is laid out per-group:

    [g0_q0..q5, g0_k, g0_v, g1_q0..q5, g1_k, g1_v] each block being head_dim
    wide. Split it back into Q/K/V and compare against HF's separate
    q_proj/k_proj/v_proj outputs to isolate whether (RMSNorm+QKV-projection) is
    correct, independent of RoPE/attention math.
    """
    mg_key = "layer.00.attn.linear_qkv"
    if mg_key not in mg_dict:
        return
    # Find HF Q/K/V; they may be missing if not hooked.
    hf_q = hf_dict.get("layer.00.attn.q_proj")
    hf_k = hf_dict.get("layer.00.attn.k_proj")
    hf_v = hf_dict.get("layer.00.attn.v_proj")
    if hf_q is None or hf_k is None or hf_v is None:
        return

    mg_qkv = mg_dict[mg_key]  # already BSH-converted: (1, T, 2048) typical
    # Hardcode shapes for dots.mocr (Qwen2 GQA 12/2 with head_dim=128):
    num_q_heads = 12
    num_kv_heads = 2
    head_dim = hf_q.shape[-1] // num_q_heads
    # q_dim = num_q_heads * head_dim
    # kv_dim = num_kv_heads * head_dim
    group_size = (num_q_heads // num_kv_heads) * head_dim + 2 * head_dim  # = 6*128 + 128 + 128 = 1024
    expected_dim = num_kv_heads * group_size  # = 2048

    if mg_qkv.shape[-1] != expected_dim:
        print(
            f"\n[qkv-split] skip: linear_qkv last-dim={mg_qkv.shape[-1]} "
            f"!= expected per-group GQA layout ({expected_dim}) — model may not be standard GQA"
        )
        return

    # Split per-group: each group of size 1024 holds (6*128 Q, 128 K, 128 V)
    mg_q_chunks, mg_k_chunks, mg_v_chunks = [], [], []
    heads_per_group = num_q_heads // num_kv_heads  # 6
    for g in range(num_kv_heads):
        base = g * group_size
        mg_q_chunks.append(mg_qkv[..., base : base + heads_per_group * head_dim])
        mg_k_chunks.append(
            mg_qkv[..., base + heads_per_group * head_dim : base + heads_per_group * head_dim + head_dim]
        )
        mg_v_chunks.append(mg_qkv[..., base + heads_per_group * head_dim + head_dim : base + group_size])
    mg_q = torch.cat(mg_q_chunks, dim=-1)
    mg_k = torch.cat(mg_k_chunks, dim=-1)
    mg_v = torch.cat(mg_v_chunks, dim=-1)

    print("\n========== layer 0 QKV split (megatron unfused vs hf q/k/v_proj) ==========")
    print(
        f"  layout assumed: per-group [Q×{heads_per_group}, K, V], "
        f"head_dim={head_dim}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"
    )
    print(f"  {'tensor':<10} {'shape_mg':<20} {'shape_hf':<20} {'cos':>10} {'|Δ|mean':>10} {'|Δ|max':>10}")
    for label, mg_t, hf_t in (("Q", mg_q, hf_q), ("K", mg_k, hf_k), ("V", mg_v, hf_v)):
        a, b = _trim_to_aligned_prefix(mg_t, hf_t, aligned_n)
        s = _stats_one_tensor(a, b)
        print(
            f"  {label:<10} {str(tuple(mg_t.shape)):<20} {str(tuple(hf_t.shape)):<20} "
            f"{s['cos']:>10.6f} {s['abs_mean']:>10.4f} {s['abs_max']:>10.4f}"
        )
    print()


def compare_intermediates(dump_dir: Path, aligned_n: int) -> None:
    """Walk per-layer dumps from megatron/{side}.intermediates.pt + hf/...

    and report cosine + |Δ| per layer so we can localize the first layer where
    the two diverge meaningfully (cos drops below ~0.99).
    """
    mg_path = _intermediates_path(dump_dir, "megatron")
    hf_path = _intermediates_path(dump_dir, "hf")
    if not mg_path.exists() or not hf_path.exists():
        print(f"\n[compare-intermediates] skipped: need both {mg_path.name} and {hf_path.name}")
        return

    mg = torch.load(mg_path, map_location="cpu", weights_only=True)
    hf = torch.load(hf_path, map_location="cpu", weights_only=True)
    common = [k for k in mg.keys() if k in hf]
    print("\n========== per-layer intermediates (megatron vs hf) ==========")
    print(f"dump_dir={dump_dir}  trim_to_aligned_prefix={aligned_n}")
    print(f"{'layer':<14} {'shape_mg':<22} {'shape_hf':<22} {'cos':>10} {'|Δ|mean':>10} {'|Δ|max':>10} {'rel':>10}")

    def sort_key(name: str) -> tuple:
        # embed < layer.NN (whole) < layer.NN.attn_out < layer.NN.mlp_out < final_norm < lm_head
        order = {"embed": (0, 0, 0), "final_norm": (2, 0, 0), "lm_head": (3, 0, 0)}
        if name.startswith("layer."):
            parts = name.split(".")
            idx = int(parts[1])
            sub = 0
            if len(parts) > 2:
                sub = {"attn_out": 1, "mlp_out": 2}.get(parts[2], 3)
            return (1, idx, sub)
        return order.get(name, (4, 0, 0))

    first_bad: Optional[str] = None
    for name in sorted(common, key=sort_key):
        a, b = mg[name], hf[name]
        a, b = _trim_to_aligned_prefix(a, b, aligned_n)
        s = _stats_one_tensor(a, b)
        marker = "" if s["cos"] > 0.999 else " <- DIVERGES" if first_bad is None else ""
        if s["cos"] <= 0.999 and first_bad is None:
            first_bad = name
        print(
            f"{name:<14} {str(s['shape_a']):<22} {str(s['shape_b']):<22} "
            f"{s['cos']:>10.6f} {s['abs_mean']:>10.4f} {s['abs_max']:>10.4f} {s['rel_mean']:>10.4f}{marker}"
        )
    if first_bad:
        print(f"\nfirst layer with cos<=0.999: {first_bad}")
    else:
        print("\nall layers cos > 0.999 — divergence must be downstream of lm_head (sampling/dtype)")

    # Dedicated layer-0 QKV split comparison (Megatron fused linear_qkv vs HF q/k/v_proj).
    _compare_megatron_qkv_split_vs_hf(mg, hf, aligned_n)
    # RoPE freq comparison (cos/sin tensors from rotary embedding module).
    _compare_rope_freqs(mg, hf)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    dump_dir = Path(args.dump_dir)

    # Run order matters: HF + megatron load the full model into our process
    # before sglang launches its subprocess, so we go megatron→hf→sglang and
    # rely on each side's del+empty_cache to free memory before the next.
    if args.side in ("megatron", "all"):
        save_record(run_megatron(args), dump_dir)

    if args.side in ("hf", "all"):
        save_record(run_hf(args), dump_dir)

    if args.side in ("sglang", "all"):
        save_record(run_sglang(args), dump_dir)

    if args.side in ("compare", "all"):
        records: dict[str, DumpRecord] = {}
        for side in ("megatron", "hf", "sglang"):
            path = _dump_path(dump_dir, side)
            if path.exists():
                records[side] = load_record(dump_dir, side)
            else:
                print(f"[compare] skip {side}: no dump at {path}")
        compare(records, top_n_worst=args.top_n_worst)

        # Per-layer hidden-state comparison (megatron vs hf). Restrict to the
        # aligned id-prefix so image-pad sentinel positions don't pollute the
        # stats — image-token expansion differs across engines but pre-image
        # positions are 1:1 comparable.
        if "megatron" in records and "hf" in records:
            mg, hf = records["megatron"], records["hf"]
            n = min(len(mg.input_ids), len(hf.input_ids))
            aligned_n = 0
            for i in range(n):
                if mg.input_ids[i] != hf.input_ids[i]:
                    break
                aligned_n = i + 1
            compare_intermediates(dump_dir, aligned_n=aligned_n or 10**9)

    return 0


if __name__ == "__main__":
    sys.exit(main())
