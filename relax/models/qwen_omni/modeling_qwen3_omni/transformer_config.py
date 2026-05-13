# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from dataclasses import dataclass, field
from typing import List, Optional

from megatron.core.transformer.transformer_config import TransformerConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig


@dataclass
class Qwen3OmniTransformerConfig(TransformerConfig):
    """Configuration for Qwen3-VL transformer with vision and language
    components."""

    vocab_size: int = 64000
    language_max_sequence_length: int = 4096

    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2304

    apply_rotary_pos_emb_in_fp32: bool = False
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    fp16_lm_cross_entropy: bool = False
    share_embeddings_and_output_weights: bool = False
    rotary_percent: float = 1.0
    rotary_base: float = 10000

    # Multimodal rope section for [temporal, height, width] dimensions
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    apply_rope_fusion: bool = False

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    hf_text_config: Optional[Qwen3OmniMoeTextConfig] = None
