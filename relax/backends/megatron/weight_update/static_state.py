# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Sequence

import torch


def relocate_colocate_static_tensors(model_chunks: Sequence[torch.nn.Module]) -> int:
    """Move Qwen3-VL static CUDA state out of the no-backup model region."""
    relocated_bytes = 0
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            if type(module).__name__ != "Qwen3VLMultimodalRotaryEmbedding":
                continue
            inv_freq = getattr(module, "inv_freq", None)
            if not isinstance(inv_freq, torch.Tensor) or isinstance(inv_freq, torch.nn.Parameter):
                raise RuntimeError(
                    "--colocate-weight-handoff requires Qwen3-VL rotary inv_freq to be a non-parameter tensor"
                )
            module.inv_freq = inv_freq.detach().clone()
            relocated_bytes += inv_freq.numel() * inv_freq.element_size()
    if relocated_bytes == 0:
        raise RuntimeError("--colocate-weight-handoff could not find Qwen3-VL rotary inv_freq")
    return relocated_bytes
