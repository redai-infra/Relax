# Copyright (c) 2026 Relax Authors. All Rights Reserved.

try:
    from megatron.bridge.models.qwen_omni import (  # type: ignore[attr-defined]  # noqa: F401
        Qwen3OmniModelProvider,
        Qwen3OmniMoEBridge,
        Qwen3OmniMoeModel,
    )
except (ImportError, AttributeError):
    from relax.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniMoeModel  # noqa: F811
    from relax.models.qwen_omni.qwen3_omni_bridge import Qwen3OmniMoEBridge  # noqa: F811
    from relax.models.qwen_omni.qwen3_omni_provider import Qwen3OmniModelProvider  # noqa: F811


__all__ = [
    "Qwen3OmniMoEBridge",
    "Qwen3OmniMoeModel",
    "Qwen3OmniModelProvider",
]
