# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang external model for dotsocr2."""

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens, general_mm_embed_routine
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.utils import add_prefix, logging

from relax.models.dots_ocr.configuration import DotsVisionConfig
from relax.models.dots_ocr.vision import DotsVisionTransformer


class DotsOCRForCausalLM(nn.Module):
    default_bitsandbytes_target_modules = [
        ".fc2.",
        ".fc1.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        vision_config = config.vision_config
        if isinstance(vision_config, dict):
            vision_config = DotsVisionConfig(**vision_config)
        self.vision_tower = DotsVisionTransformer(vision_config)
        self.model = Qwen2Model(config, quant_config, prefix=add_prefix("model", prefix))
        if config.tie_word_embeddings:
            logging.warning("tied word embeddings are not supported in SGLang DotsOCRForCausalLM.")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        target_device = self.vision_tower.device
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=target_device, dtype=self.vision_tower.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0).to(target_device)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        return self.vision_tower(pixel_values, grid_thw=image_grid_thw)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        target_device = self.vision_tower.device
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=target_device, dtype=self.vision_tower.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0).to(target_device)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        return self.vision_tower(pixel_values, grid_thw=video_grid_thw)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not get_embedding:
            out = self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        else:
            out = self.pooler(hidden_states, forward_batch)
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "vision_tower" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [DotsOCRForCausalLM]
