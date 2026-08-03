# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Monkey-patch for SGLang ``QwenVLImageProcessor.process_mm_data_async``:
support **pre-tokenized multi-turn rollout input_ids**.

Why
---
DeepEyes multi-turn VLM rollout sends **already-tokenized** ``input_ids``
(``list[int]``) to SGLang instead of raw text — see
``examples/deepeyes/rollout.py::_run_inference_step``. The accumulated
``sample.tokens`` carry, per image, N ``<|image_pad|>`` tokens (one per visual
patch) produced by a *previous* round's processor.

SGLang's stock ``process_mm_data_async`` mishandles such a prompt:

* **Problem A — decode→retokenize drift.** ``load_mm_data`` decodes the int
  list back to text and re-tokenizes it, which is not lossless (special
  tokens / whitespace). The re-tokenized length can differ from the caller's
  ``input_ids``, misaligning mrope positions and breaking loss_mask / log-prob
  bookkeeping.
* **Problem B — N×M image-pad explosion.** The accumulated sequence already
  contains N ``<|image_pad|>`` per image; ``load_mm_data`` treats each as a
  fresh image placeholder and re-expands it to M patch tokens, yielding
  ``N×M`` image-pad tokens and wrecking mrope.

What this patch does
--------------------
Only two surgical changes over the upstream method, both no-ops for raw-text
(``str``) prompts so normal single-turn Qwen-VL requests are unaffected:

1. ``_strip_image_token`` — collapse each run of consecutive
   ``<|image_pad|>`` into a single placeholder *before* ``load_mm_data``, so
   downstream sees exactly one placeholder per image and re-expands to the
   correct M patch tokens.
2. ``original_input_ids`` save/restore — when the caller passed a ``list[int]``,
   remember it, let the mm pipeline build ``pixel_values`` / grids off the
   stripped prompt, then **restore the original ``input_ids``** and
   **recompute mrope** from it (the mm pipeline's grids are image-data-derived
   and stay valid; only the token sequence must match the caller).

   Re-computing mrope from the *restored* ids is the crux: a plain wrapper
   that only rewrites ``input_ids`` in the return value would leave mrope
   computed from the re-tokenized sequence, misaligning it with the restored
   ids — exactly the bug this patch fixes. Hence the method body is
   re-implemented here, delegating to the upstream primitives
   ``self.load_mm_data`` / ``self.process_and_combine_mm_data`` /
   ``MRotaryEmbedding.get_rope_index`` / upstream module-level
   ``preprocess_video``. The unrelated helpers (``smart_resize``,
   ``preprocess_video``, ``get_mm_data``, ``__init__`` …) are **not** copied —
   they stay on the upstream class / module.

This is the runtime equivalent of the old
``examples/deepeyes/qwen_vl.py`` ``cp`` overlay, minus the whole-file copy.

Usage
-----
Applied by ``_launch_server_with_patches()`` in ``sglang_engine.py`` when the
env flag ``RELAX_DEEPEYES_QWEN_VL_PATCH=1`` is set. Forward the flag to every
Ray worker via ``RELAX_PROPAGATE_ENV_VARS`` (see ``run_deepeyes_r3.sh``);
``post_process_env`` then injects it into the Ray ``runtime_env`` so the
SGLang subprocess (spawned via ``multiprocessing``) inherits it. Idempotent:
applying twice is a no-op.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch


logger = logging.getLogger(__name__)

ENV_FLAG = "RELAX_DEEPEYES_QWEN_VL_PATCH"
_PATCH_FLAG = "_relax_deepeyes_patched"
_TESTED_SGLANG = "sglang 0.5.12.post1"


def _strip_image_token(input_ids, image_token_id: int = 151655):
    """Collapse consecutive ``<|image_pad|>`` tokens into a single placeholder.

    Transform::

        <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
        -> <|vision_start|><|image_pad|><|vision_end|>

    Why: the caller may pass pre-tokenized ``input_ids`` in which each image
    has already been expanded into N ``<|image_pad|>`` tokens (one per visual
    patch). However ``load_mm_data`` downstream expects exactly *one*
    ``<|image_pad|>`` placeholder per image and re-expands it itself based on
    the actual patch count. Without this collapse the pipeline would see
    ``N x M`` image-pad tokens and miscount positions, breaking mrope
    bookkeeping. Raw text prompts are returned unchanged.

    Args:
        input_ids: List of token ids, or any non-list value (passed through).
        image_token_id: Id of ``<|image_pad|>`` (151655 for Qwen-VL family).

    Returns:
        ``input_ids`` with each run of consecutive ``image_token_id`` reduced
        to a single occurrence, or the input untouched if it is not a list.
    """
    # Raw text prompts (str) and other non-list inputs require no rewrite.
    if not isinstance(input_ids, list):
        return input_ids

    input_id_arr = np.array(input_ids)

    # mask[i] == True means "keep token at index i"; start by keeping all.
    mask = np.ones(len(input_id_arr), dtype=bool)

    # Boolean array marking every <|image_pad|> position.
    is_value = input_id_arr == image_token_id

    # A token at index i is a redundant duplicate iff both it and its left
    # neighbour are <|image_pad|>. Dropping those keeps the first occurrence
    # in each run and removes the rest. Index 0 has no left neighbour so it
    # is always kept (mask[0] stays True).
    mask[1:] &= ~(is_value[1:] & is_value[:-1])
    return input_id_arr[mask].tolist()


async def _patched_process_mm_data_async(
    self,
    image_data,
    input_text,
    request_obj,
    *args,
    **kwargs,
):
    """Drop-in replacement for ``QwenVLImageProcessor.process_mm_data_async``.

    Identical to the upstream method body except for the two Relax changes
    documented in this module's docstring (strip before ``load_mm_data``;
    save/restore ``input_ids`` and recompute mrope). ``preprocess_video`` and
    ``MRotaryEmbedding`` are imported from upstream — not redefined here.
    """
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
    from sglang.srt.multimodal.processors.qwen_vl import preprocess_video

    entry_time = time.perf_counter()

    # When the caller provides pre-tokenized input_ids (list of ints) instead
    # of raw text, save them. The mm pipeline will decode→retokenize internally
    # which can alter the token count. We restore the original input_ids
    # afterwards so the engine's token bookkeeping stays consistent with the
    # caller.
    original_input_ids = None
    if isinstance(input_text, list) and len(input_text) > 0 and isinstance(input_text[0], int):
        original_input_ids = input_text

    base_output = self.load_mm_data(
        prompt=_strip_image_token(input_text),
        image_data=image_data,
        video_data=request_obj.video_data,
        audio_data=request_obj.audio_data,
        multimodal_tokens=self.mm_tokens,
    )
    load_time = time.perf_counter()
    rid = getattr(request_obj, "rid", "anonymous_rid")

    video_metadata = None
    if base_output.videos:
        videos_processed = [
            await preprocess_video(video, video_config=self.video_config) for video in base_output.videos
        ]
        base_output.videos, video_metadata = map(list, zip(*videos_processed))

    preprocess_time = time.perf_counter()

    # NOTE: for qwen3-vl, video_meta need to be passed in, since do_sample_frames is already done in preprocess_video
    if self.hf_config.model_type in ("qwen3_vl", "qwen3_vl_moe"):
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output,
            self.mm_tokens,
            video_metadata=video_metadata,
            do_sample_frames=False,
        )
    else:
        mm_items, input_ids, ret = self.process_and_combine_mm_data(base_output, self.mm_tokens)

    audio_feature_lengths = None

    if self.model_type == "qwen3_omni_moe":
        audio_item = next((mm for mm in mm_items if mm.is_audio()), None)
        if audio_item:
            audio_feature_lengths = torch.sum(audio_item.feature_attention_mask, dim=1)

    second_per_grid_ts = getattr(ret, "second_per_grid_ts", None)
    if second_per_grid_ts is None:
        second_per_grid_ts = getattr(ret, "video_second_per_grid", None)

    process_time = time.perf_counter()

    input_ids = input_ids.flatten()

    # Restore original input_ids to avoid decode→retokenize mismatches.
    # The mm_items (pixel_values, etc.) are image-data-dependent and remain
    # valid; only the token sequence itself needs to match the caller's version.
    if original_input_ids is not None:
        input_ids = torch.tensor(original_input_ids, dtype=input_ids.dtype)

    image_grid_thw = None
    if hasattr(ret, "image_grid_thw"):
        image_grid_thw = ret.image_grid_thw

    if image_grid_thw is None and image_data and isinstance(image_data[0], dict):
        image_grid_thw = image_data[0].get("image_grid_thw")

    video_grid_thw = None
    if hasattr(ret, "video_grid_thw"):
        video_grid_thw = ret.video_grid_thw

    if video_grid_thw is None and request_obj.video_data:
        first_video = request_obj.video_data[0]
        if isinstance(first_video, dict):
            video_grid_thw = first_video.get("video_grid_thw")

    mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
        image_token_id=self.mm_tokens.image_token_id,
        video_token_id=self.mm_tokens.video_token_id,
        vision_start_token_id=self.vision_start_token_id,
        model_type=self.model_type,
        tokens_per_second=getattr(self.hf_config.vision_config, "tokens_per_second", None),
        # use the expanded token ids
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=getattr(ret, "image_grid_thw", None),
        video_grid_thw=getattr(ret, "video_grid_thw", None),
        second_per_grid_ts=second_per_grid_ts,
        use_audio_in_video=False,
        audio_seqlens=audio_feature_lengths,
        audio_token_id=getattr(self.hf_config, "audio_token_id", None),
        audio_start_token_id=self.audio_start_token_id,
        position_id_per_seconds=getattr(self.hf_config, "position_id_per_seconds", None),
    )
    mrope_positions = mrope_positions.squeeze(1)
    get_rope_index_time = time.perf_counter()
    logger.debug(
        f"[QwenVLProcessor Perf] {rid=}, "
        f"load_time: {(load_time - entry_time) * 1000:.2f} ms, "
        f"preprocess_time: {(preprocess_time - load_time) * 1000:.2f} ms, "
        f"process_time: {(process_time - preprocess_time) * 1000:.2f} ms, "
        f"get_rope_index_time: {(get_rope_index_time - process_time) * 1000:.2f} ms, "
        f"total_time: {(get_rope_index_time - entry_time) * 1000:.2f} ms"
    )

    ret_dict = {
        "input_ids": input_ids.tolist(),
        "mm_items": mm_items,
        "im_start_id": self.vision_start_token_id,
        "im_end_id": self.vision_end_token_id,
        "im_token_id": self.mm_tokens.image_token_id,
        "video_token_id": self.mm_tokens.video_token_id,
        "audio_token_id": self.mm_tokens.audio_token_id,
        "mrope_positions": mrope_positions,
        "mrope_position_delta": mrope_position_delta,
    }
    # sglang >= 0.5.12 replaced the plain-dict return value of
    # ``process_mm_data_async`` with a typed ``MultimodalProcessorOutput``
    # dataclass; the TokenizerManager consumes it via attribute access. Wrap
    # our dict into that object when the type exists, and fall back to the raw
    # dict on older sglang so this patch stays version-agnostic (mirrors
    # ``relax.utils.opd.opd_sglang_patch``).
    try:
        from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput

        return MultimodalProcessorOutput.from_dict(ret_dict)
    except Exception:
        return ret_dict


def _verify_upstream_api() -> None:
    """Fail fast with a clear message if the upstream API this patch binds to
    has moved or been renamed.

    Raises ``RuntimeError`` listing the missing symbol(s) and the sglang
    version this patch was validated against.
    """
    missing = []
    try:
        from sglang.srt.multimodal.processors.qwen_vl import (
            QwenVLImageProcessor,
            preprocess_video,
        )
    except Exception as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            f"DeepEyes qwen_vl patch: cannot import QwenVLImageProcessor / "
            f"preprocess_video from sglang ({exc!r}). Relax requires "
            f"{_TESTED_SGLANG} with sglang.srt.multimodal.processors.qwen_vl."
        ) from exc

    for attr in ("process_mm_data_async", "load_mm_data", "process_and_combine_mm_data"):
        if not hasattr(QwenVLImageProcessor, attr):
            missing.append(f"QwenVLImageProcessor.{attr}")
    try:
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

        if not hasattr(MRotaryEmbedding, "get_rope_index"):
            missing.append("MRotaryEmbedding.get_rope_index")
    except Exception as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            f"DeepEyes qwen_vl patch: cannot import MRotaryEmbedding ({exc!r}). Relax requires {_TESTED_SGLANG}."
        ) from exc

    # ``preprocess_video`` must be the upstream module-level function.
    if not callable(preprocess_video):
        missing.append("preprocess_video (module-level)")

    if missing:
        raise RuntimeError(
            "DeepEyes qwen_vl patch: upstream SGLang API changed — missing: "
            f"{', '.join(missing)}. Relax requires {_TESTED_SGLANG}; "
            "revisit relax/backends/sglang/patches/qwen_vl_patch.py."
        )


def apply_qwen_vl_patches() -> bool:
    """Bind ``_patched_process_mm_data_async`` onto ``QwenVLImageProcessor``.

    Idempotent: a second call is a no-op. Returns ``True`` if the patch was
    applied this call, ``False`` if it was already in place.
    """
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    if getattr(QwenVLImageProcessor, _PATCH_FLAG, False):
        logger.info("[deepeyes-qwen-vl-patch] already applied, skip")
        return False

    _verify_upstream_api()

    QwenVLImageProcessor.process_mm_data_async = _patched_process_mm_data_async
    setattr(QwenVLImageProcessor, _PATCH_FLAG, True)
    logger.info(
        "[deepeyes-qwen-vl-patch] applied to QwenVLImageProcessor.process_mm_data_async "
        "(pre-tokenized multi-turn input_ids support)"
    )
    return True


def apply_deepeyes_qwen_vl_patch() -> None:
    """Apply the DeepEyes Qwen-VL processor patch in the current (server)
    process.

    Never blocks server startup: failures are logged as a warning.
    """
    try:
        apply_qwen_vl_patches()
    except Exception as e:
        logger.warning("Failed to apply DeepEyes qwen_vl patch: %r", e)


# Apply on import ONLY when explicitly enabled via the env flag. Merely
# importing this module must stay a complete no-op unless
# ``RELAX_DEEPEYES_QWEN_VL_PATCH=1`` is set, so sglang and its other modules
# are left untouched when the feature is disabled.
if os.environ.get(ENV_FLAG, "0") == "1":
    apply_deepeyes_qwen_vl_patch()
