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

# Import glm_moe_dsa in its own try/except so a failure above does not block
# the GLM5Bridge @register_bridge decorator from running. Without this, an
# unrelated qwen_omni circular-import error prevents GLM5Bridge from being
# registered, and AutoBridge silently falls back to the generic MLA bridge,
# bypassing the fused DSAMLASelfAttention spec.
try:
    from relax.models import glm_moe_dsa  # noqa: F401
except Exception as _e:
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).warning("Failed to import relax.models.glm_moe_dsa: %s", _e)


__all__ = [
    "Qwen3OmniMoEBridge",
    "Qwen3OmniMoeModel",
    "Qwen3OmniModelProvider",
]
