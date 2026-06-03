# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert compressed-tensors W4A16 quantized HF checkpoints to BF16.

Adapted from slime/tools/convert_k2_thinking_int4_to_bf16.py.

Default output is a sibling directory ``<model-dir>_bf16``. The output ``config.json`` has
``quantization_config`` removed; the original block is written to a sidecar
``quantization_config.json`` so QAT paths can read it back when needed.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from compressed_tensors.compressors import unpack_from_int32
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _quant_config(cfg: dict) -> dict:
    """Return the quantization_config block, falling back to text_config (VLM
    layout)."""
    qc = cfg.get("quantization_config")
    if qc:
        return qc
    return cfg.get("text_config", {}).get("quantization_config") or {}


def read_group_size(model_dir: str, config_path: str | None = None) -> int:
    cfg_path = config_path or os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    return int(
        _quant_config(cfg).get("config_groups", {}).get("group_0", {}).get("weights", {}).get("group_size", 128)
    )


def _dequantize_tensor(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if isinstance(weight_shape, torch.Tensor):
        shape = tuple(int(v) for v in weight_shape.view(-1).tolist())
    else:
        shape = tuple(weight_shape)

    weight = unpack_from_int32(weight_packed, 4, shape)

    if group_size > 0:
        scale = weight_scale.to(torch.float32)
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)
        scales = torch.repeat_interleave(scale, repeats=group_size, dim=1)
    else:
        scales = weight_scale.to(torch.float32)

    if scales.shape != weight.shape:
        if scales.numel() == weight.numel():
            scales = scales.reshape_as(weight)
        else:
            raise ValueError(f"scale shape {scales.shape} incompatible with weight shape {weight.shape}")

    return (weight.to(torch.float32) * scales).to(torch.bfloat16).contiguous()


def _is_quantized_weight_key(key: str) -> bool:
    if ".mlp.experts." not in key or ".shared_experts." in key:
        return False
    suffixes = ("weight_packed", "weight_scale", "weight_shape")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in suffixes:
            if key.endswith(f".{proj}.{suffix}"):
                return True
    return False


def _convert_file(input_path: str, output_path: str, group_size: int, skip_existing: bool) -> None:
    if skip_existing and os.path.exists(output_path):
        return

    # Memory ceiling: this loads ALL tensors of a single safetensors shard into
    # GPU memory at once (one shard at a time). K2.6 shards are ~5GB packed →
    # ~10GB BF16 dequantized, so a 40GB+ GPU is comfortable. If shards grow
    # past that, stream key-by-key and keep only expert triplets resident.
    tensors: dict[str, torch.Tensor] = {}
    expert_buffers: dict[str, dict[str, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with safe_open(input_path, framework="pt", device=device) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            if not _is_quantized_weight_key(key):
                tensors[key] = tensor
                continue
            parts = key.split(".")
            try:
                expert_idx = parts.index("experts")
            except ValueError:
                tensors[key] = tensor
                continue
            prefix = ".".join(parts[: expert_idx + 2])
            project = parts[-2]
            suffix = parts[-1]
            expert_buffers[prefix][project][suffix] = tensor

    for prefix, components in expert_buffers.items():
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj_data = components.get(proj_name, {})
            required = {"weight_packed", "weight_scale", "weight_shape"}
            if not required.issubset(proj_data.keys()):
                for suffix, value in proj_data.items():
                    tensors[f"{prefix}.{proj_name}.{suffix}"] = value
                continue
            bf16_weight = _dequantize_tensor(
                proj_data["weight_packed"].to(torch.int32),
                proj_data["weight_scale"].to(torch.float32),
                proj_data["weight_shape"],
                group_size,
            )
            tensors[f"{prefix}.{proj_name}.weight"] = bf16_weight

    cpu_tensors = {k: v.cpu() for k, v in tensors.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(cpu_tensors, output_path)


def _derive_extra_ignore_namespaces(src: str) -> list[str]:
    """Return top-level module names whose subtree has plain ``.weight`` keys
    but zero ``.weight_packed`` triplets in the source checkpoint.

    These namespaces are definitively not quantized; surfacing them in the
    sidecar ignore list lets downstream quantizers reject them without model-
    specific name knowledge.
    """
    namespaces: dict[str, dict[str, bool]] = defaultdict(lambda: {"plain": False, "packed": False})
    for fname in os.listdir(src):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(src, fname), framework="pt") as reader:
            for key in reader.keys():
                top = key.split(".", 1)[0]
                if key.endswith(".weight_packed"):
                    namespaces[top]["packed"] = True
                elif key.endswith(".weight"):
                    namespaces[top]["plain"] = True
    return sorted(top for top, info in namespaces.items() if info["plain"] and not info["packed"])


def _copy_aux_files(src: str, dst: str, *, strip_quantization_config: bool) -> dict | None:
    src_path = Path(src)
    dst_path = Path(dst)
    stripped: dict | None = None

    for fname in os.listdir(src_path):
        # Safetensors shards are produced by _convert_file; the index is rewritten
        # after the cast loop. Everything else (config.json, *.py, tokenizer*,
        # chat_template.jinja, tiktoken.model, README/LICENSE, …) is copied
        # verbatim so trust_remote_code modules find their sidecars.
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        full = src_path / fname
        if not full.is_file():
            continue
        target = dst_path / fname
        if fname == "config.json" and strip_quantization_config:
            with open(full) as f:
                cfg = json.load(f)
            stripped = cfg.pop("quantization_config", None)
            text_cfg = cfg.get("text_config")
            if stripped is None and isinstance(text_cfg, dict):
                stripped = text_cfg.pop("quantization_config", None)
            with open(target, "w") as f:
                json.dump(cfg, f, indent=2)
        else:
            shutil.copy2(full, target)
    return stripped


def cast(
    src: str,
    dst: str,
    *,
    group_size: int | None = None,
    files: list[str] | None = None,
    config_path: str | None = None,
    overwrite: bool = False,
    strip_quantization_config: bool = True,
) -> None:
    """Cast a compressed-tensors W4A16 HF checkpoint at ``src`` into BF16 at
    ``dst``."""
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"model directory not found: {src}")

    os.makedirs(dst, exist_ok=True)
    if group_size is None:
        group_size = read_group_size(src, config_path)
    logger.info(f"int4 → bf16 cast: src={src} dst={dst} group_size={group_size}")

    if files:
        targets = [os.path.join(src, name) for name in files]
    else:
        targets = sorted(os.path.join(src, name) for name in os.listdir(src) if name.endswith(".safetensors"))

    if not targets:
        logger.warning("no safetensors found in source directory")
        return

    for path in tqdm(targets, desc="int4 → bf16", unit="file"):
        if not os.path.isfile(path):
            continue
        rel = os.path.relpath(path, src)
        _convert_file(path, os.path.join(dst, rel), group_size, skip_existing=not overwrite)

    stripped = _copy_aux_files(src, dst, strip_quantization_config=strip_quantization_config)
    if stripped is not None:
        # Augment the sidecar's ignore list with top-level namespaces that have
        # no weight_packed triplets in source — these are guaranteed-not-quantized
        # (for K2.x VLMs that is vision_tower / mm_projector, which the published
        # config.ignore does not list). Lets the generic quantizer stay model-
        # agnostic and decide skip purely from the config.
        extra_ignore = _derive_extra_ignore_namespaces(src)
        if extra_ignore:
            existing = list(stripped.get("ignore", []))
            stripped["ignore"] = existing + [ns for ns in extra_ignore if ns not in existing]
            logger.info(f"sidecar quantization_config.ignore extended with {extra_ignore}")
        with open(os.path.join(dst, "quantization_config.json"), "w") as f:
            json.dump(stripped, f, indent=2)

    weight_map: dict[str, str] = {}
    for fname in sorted(os.listdir(dst)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(dst, fname), framework="pt") as reader:
            for key in reader.keys():
                weight_map[key] = fname
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

    logger.info(f"int4 → bf16 cast complete: {dst}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert compressed-tensors W4A16 MoE experts to BF16.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default=None, help="Default: <model-dir>_bf16")
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-quantization-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or f"{os.path.abspath(args.model_dir)}_bf16"
    cast(
        args.model_dir,
        output_dir,
        files=args.files,
        config_path=args.config_path,
        overwrite=args.overwrite,
        strip_quantization_config=not args.keep_quantization_config,
    )


if __name__ == "__main__":
    main()
