# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert FP8 (e4m3, block-quantized) HF safetensors checkpoints back to BF16.

Each FP8 ``weight`` is paired with its ``weight_scale_inv`` and dequantized via
a Triton kernel. Shards are processed in parallel; scale tensors that live in a
different shard are pulled on demand via ``safetensors.safe_open``. The output
``config.json`` has its ``quantization_config`` block stripped, and
``model.safetensors.index.json`` is rewritten without the obsolete
``_scale_inv`` entries.
"""

import argparse
import gc
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import safetensors
import safetensors.torch
import torch
import triton
import triton.language as tl
from tqdm import tqdm

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.bfloat16)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


class ConversionResult:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.weight_map: dict[str, str] = {}
        self.param_count: int = 0

    def add_result(self, filename: str, weights: dict[str, torch.Tensor]) -> None:
        with self.lock:
            for k, v in weights.items():
                self.weight_map[k] = filename
                self.param_count += len(v)


def _process_file(
    input_path: str,
    output_path: str,
    filename: str,
    weight_map: dict[str, str],
    result_collector: ConversionResult,
) -> None:
    logger.info(f"Processing {filename}, memory usage: {torch.cuda.memory_allocated()}")
    local_weights: dict[str, torch.Tensor] = {}
    new_weights: dict[str, torch.Tensor] = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device="cuda") as f:
        for k in f.keys():
            local_weights[k] = f.get_tensor(k)

    def _get_scale_inv(scale_inv_name: str) -> torch.Tensor | None:
        if scale_inv_name in local_weights:
            return local_weights[scale_inv_name]
        scale_inv_file = weight_map.get(scale_inv_name)
        if scale_inv_file is None:
            return None
        with safetensors.safe_open(os.path.join(input_path, scale_inv_file), framework="pt", device="cuda") as sf:
            return sf.get_tensor(scale_inv_name)

    for name, weight in local_weights.items():
        if name.endswith("_scale_inv"):
            continue
        if weight.element_size() == 1:  # FP8 weight
            scale_inv = _get_scale_inv(f"{name}_scale_inv")
            if scale_inv is None:
                logger.warning(f"Missing scale_inv tensor for {name}, skipping conversion")
                new_weights[name] = weight
                continue
            new_weights[name] = weight_dequant(weight, scale_inv)
        else:
            new_weights[name] = weight

    safetensors.torch.save_file(new_weights, os.path.join(output_path, filename), metadata={"format": "pt"})

    result_collector.add_result(filename, new_weights)


def convert_bf16(
    input_path: str,
    output_path: str,
    max_workers: int,
) -> None:
    input_path = os.path.abspath(input_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(model_index_file) as f:
        weight_map = json.load(f)["weight_map"]

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    result_collector = ConversionResult()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in safetensors_files:
            future = executor.submit(_process_file, input_path, output_path, filename, weight_map, result_collector)
            futures.append(future)

        for future in tqdm(futures, desc="Processing files"):
            future.result()

    # Output is plain BF16; drop the FP8 quantization_config so downstream
    # loaders don't try to dequantize the already-dequantized weights.
    config_path = Path(input_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        cfg.pop("quantization_config", None)
        with open(Path(output_path) / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    index_dict = {"weight_map": result_collector.weight_map, "metadata": {"total_size": result_collector.param_count}}
    with open(Path(output_path) / "model.safetensors.index.json", "w") as f:
        json.dump(index_dict, f, indent=2)

    gc.collect()
    torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=str, required=True, help="Path to the directory of the FP8 HF safetensors model."
    )
    parser.add_argument(
        "--save-dir", type=str, required=True, help="Path to the directory to save the converted BF16 model."
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker threads for parallel processing")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.path.exists(args.save_dir):
        logger.info(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_bf16(args.model_dir, args.save_dir, args.max_workers)
    logger.info(f"Conversion complete, output saved to {args.save_dir}")


if __name__ == "__main__":
    main()
