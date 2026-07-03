# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel

from relax.models.dots_ocr.configuration import DotsVisionConfig
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _flash_attn_varlen_func(*args, **kwargs):
    """Lazy import of flash_attn.

    Importing flash_attn at module top-level eagerly initializes CUDA, which
    causes the SGLang HTTP server subprocess (which only runs the tokenizer
    manager + uvicorn, with no model on GPU) to fork CUDA-tainted children
    inside SGLangBaseProcessor's ProcessPoolExecutor, deadlocking startup.
    Importing inside the call site keeps the import out of CPU-only paths.
    """
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(*args, **kwargs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    return ((tensor * cos) + (rotate_half(tensor) * sin)).to(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class PatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        pre_norm: str = "layernorm",
        init_merger_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.pre_norm = pre_norm
        if self.pre_norm == "layernorm":
            self.ln_q = LayerNorm(context_dim, eps=1e-6)
        elif self.pre_norm == "rmsnorm":
            self.ln_q = RMSNorm(context_dim, eps=1e-6)
        else:
            logger.warning("DotsOCR PatchMerger is running without pre-norm.")
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )
        if init_merger_std is not None:
            nn.init.normal_(self.mlp[0].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.normal_(self.mlp[2].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            return self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return self.mlp(x.view(-1, self.hidden_size))


class VisionFlashAttention2(nn.Module):
    def __init__(self, config: DotsVisionConfig, dim: int, num_heads: int = 16, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.is_causal = config.is_causal

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = _flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=self.is_causal
        ).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionSdpaAttention(nn.Module):
    def __init__(self, config: DotsVisionConfig, dim: int, num_heads: int = 16, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        attn_output = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attention_mask, dropout_p=0.0
        )
        return self.proj(attn_output.transpose(0, 1).reshape(seq_length, -1))


DOTS_VISION_ATTENTION_CLASSES = {
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
}


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, config: DotsVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.intermediate_size, bias=config.use_bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.embed_dim, bias=config.use_bias)
        self.fc3 = nn.Linear(config.embed_dim, config.intermediate_size, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)) * self.fc3(x))


class DotsPatchEmbed(nn.Module):
    def __init__(self, config: DotsVisionConfig) -> None:
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        x = x.view(-1, self.num_channels, self.temporal_patch_size, self.patch_size, self.patch_size)[:, :, 0]
        return self.norm(self.proj(x).view(-1, self.embed_dim))


class DotsViTPreprocessor(nn.Module):
    def __init__(self, config: DotsVisionConfig) -> None:
        super().__init__()
        self.patchifier = DotsPatchEmbed(config)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        return self.patchifier(x, grid_thw)


class DotsVisionBlock(nn.Module):
    def __init__(self, config: DotsVisionConfig, attn_implementation: str = "flash_attention_2") -> None:
        super().__init__()
        attention_cls = DOTS_VISION_ATTENTION_CLASSES.get(attn_implementation, VisionFlashAttention2)
        self.attn = attention_cls(config, config.embed_dim, num_heads=config.num_attention_heads, bias=config.use_bias)
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class DotsVisionTransformer(PreTrainedModel):
    config_class = DotsVisionConfig

    def __init__(self, config: DotsVisionConfig) -> None:
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = DotsViTPreprocessor(config)
        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [DotsVisionBlock(config, config.attn_implementation) for _ in range(config.num_hidden_layers)]
        )
        if self.config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
            pre_norm="layernorm",
            init_merger_std=self.config.init_merger_std,
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def get_pos_ids_by_grid(self, grid_thw: torch.Tensor) -> List[torch.Tensor]:
        pos_ids = []
        for _, h, w in grid_thw:
            hpos_ids = torch.arange(h, device=grid_thw.device).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
            wpos_ids = torch.arange(w, device=grid_thw.device).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(_, 1))
        return pos_ids

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = torch.cat(self.get_pos_ids_by_grid(grid_thw), dim=0)
        rotary_pos_emb_full = self.rotary_pos_emb(grid_thw[:, 1:].max())
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, bf16: bool = True) -> torch.Tensor:
        if bf16:
            hidden_states = hidden_states.bfloat16()
        hidden_states = self.patch_embed(hidden_states, grid_thw)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, rotary_pos_emb, use_reentrant=False
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)
        return self.merger(hidden_states)
