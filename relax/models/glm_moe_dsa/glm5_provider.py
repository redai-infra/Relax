"""GLM5 uses MLAModelProvider directly.

This module is kept for import compatibility.
"""

from megatron.bridge.models.mla_provider import MLAModelProvider as GLM5ModelProvider  # noqa: F401


__all__ = ["GLM5ModelProvider"]
