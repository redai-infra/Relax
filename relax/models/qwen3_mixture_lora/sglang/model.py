# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang Qwen3 external model with token-routed LoRA experts."""

from sglang.srt.models.qwen3 import Qwen3ForCausalLM as SGLangQwen3ForCausalLM
from torch import nn

from relax.models.mixture_lora_sglang import MixtureLoraSGLangModelMixin


# SGLang registers external models by the checkpoint architecture name. Keeping
# this name replaces its built-in Qwen3 entry while preserving checkpoint metadata.
class Qwen3ForCausalLM(MixtureLoraSGLangModelMixin, SGLangQwen3ForCausalLM):
    """Qwen3 external model that routes LoRA experts at attention
    projections."""

    def mixture_lora_site_modules(self, layer_id: int) -> dict[str, nn.Module]:
        attention = self.model.layers[layer_id].self_attn
        return {"linear_qkv": attention.qkv_proj, "linear_proj": attention.o_proj}


EntryClass = Qwen3ForCausalLM
