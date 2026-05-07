# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock


try:
    import transformer_engine.pytorch as te  # noqa: F401 # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

te_checkpoint = None
if HAVE_TE:
    pass


class Qwen3OmniTransformerBlock(Qwen3VLTransformerBlock):
    """Qwen3 Omni Transformer Block extending Qwen3VL functionality.

    This block extends the Qwen3VL transformer block with Omni-specific
    features for handling multimodal inputs including audio, images, and
    videos.
    """

    pass
