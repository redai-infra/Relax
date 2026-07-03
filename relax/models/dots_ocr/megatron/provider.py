# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch.nn.functional as F
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

from relax.models.dots_ocr.configuration import DotsVisionConfig
from relax.models.dots_ocr.megatron.model import DotsOCRModel


@dataclass
class DotsOCRModelProvider(GPTModelProvider):
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    hidden_dropout: float = 0.0
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    qk_layernorm: bool = False
    masked_softmax_fusion: bool = False
    gradient_accumulation_fusion: bool = False
    apply_rotary_pos_emb_in_fp32: bool = False
    apply_rope_fusion: bool = False
    attention_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    scatter_embedding_sequence_parallel: bool = False
    share_embeddings_and_output_weights: bool = False
    # GPTModelProvider defaults to "learned_absolute", which makes GPTModel skip
    # building self.rotary_pos_emb — LM forward then runs with no positional
    # encoding at all and silently produces garbage attention. dots.mocr is a
    # Qwen2-arch model with RoPE, so this MUST be "rope".
    position_embedding_type: str = "rope"

    language_max_sequence_length: int = 131072
    image_token_id: int = 151665
    video_token_id: int = 151656
    vision_config: Optional[DotsVisionConfig] = None

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    vision_dp_when_cp: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=self.qk_layernorm,
        )
        model = DotsOCRModel(
            language_transformer_config=self,
            language_transformer_layer_spec=language_transformer_layer_spec,
            vision_transformer_config=self.vision_config,
            pre_process=pre_process,
            post_process=post_process,
        )
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model
