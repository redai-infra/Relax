# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen3 Omni model providers and configurations."""

# Core model components
# Bridges for HuggingFace to Megatron conversion
from relax.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniMoeModel  # noqa: F401
from relax.models.qwen_omni.qwen3_omni_bridge import Qwen3OmniMoEBridge

# Dense and MoE model providers
from relax.models.qwen_omni.qwen3_omni_provider import Qwen3OmniModelProvider


__all__ = [
    "Qwen3OmniMoeModel",
    "Qwen3OmniMoEBridge",
    "Qwen3OmniModelProvider",
]
