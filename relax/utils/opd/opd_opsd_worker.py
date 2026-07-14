# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from relax.utils.types import Sample

logger = get_logger(__name__)

PREEXPANDED_RAW_FORMAT = "opd_preexpanded_raw"


class OpsdWorker:
    def __init__(self, *, is_opsd: bool = False):
        self.is_opsd = is_opsd

    @classmethod
    def from_args(cls, args) -> "OpsdWorker":
        teacher_prompt_key = getattr(args, "opd_teacher_prompt_key", None)
        teacher_image_key = getattr(args, "opd_teacher_image_key", None)
        return cls(is_opsd=teacher_prompt_key is not None or teacher_image_key is not None)

    async def build_teacher_inputs(self, args, sample: "Sample") -> None:
        """Pre-expand teacher inputs on the client (rollout) side.

        Populates, on the ``sample``:

        - ``teacher_tokens``: expanded teacher prompt ids + student response ids.
          For multimodal OPSD the HF processor expands ``<image_pad>`` placeholders
          into N image-pad tokens, so this is the real prompt length the teacher
          prefill will see.
        - ``teacher_image_b64_list`` / ``teacher_image_grid_thw`` (multimodal only):
          raw base64-PNG list + client-precomputed grid_thw. The patched teacher
          SGLang runs its own HF image processor on the raw bytes.
        - ``teacher_prompt_length``: expanded prompt token count.

        No-op when OPSD is not active, or when the sample has no response yet.
        """
        if not self.is_opsd:
            return

        teacher_prompt = getattr(sample, "teacher_prompt", None)
        teacher_mm = getattr(sample, "teacher_multimodal_inputs", None)
        teacher_has_media = teacher_mm is not None and bool(teacher_mm.get("images"))

        if teacher_prompt is None and not teacher_has_media:
            # Plain OPD: reuse student-side processed vision inputs so teacher
            # logprob requests use the pre-expanded path and logprob_start_len
            # stays aligned with SGLang.
            if sample.tokens and int(sample.response_length or 0) > 0:
                student_mm_train_inputs = getattr(sample, "multimodal_train_inputs", None)
                student_grid_thw = (student_mm_train_inputs or {}).get("image_grid_thw")
                student_mm_in = sample.multimodal_inputs or {}
                student_raw_images = student_mm_in.get("images") or []
                if student_grid_thw is not None and student_raw_images:
                    cached = student_mm_in.get("_teacher_image_b64_cache")
                    if cached is None:
                        from relax.utils.data.processing_utils import async_encode_image_for_rollout_engine

                        cached = list(await _gather_encode(student_raw_images, async_encode_image_for_rollout_engine))
                        student_mm_in["_teacher_image_b64_cache"] = cached
                    sample.teacher_image_b64_list = cached
                    sample.teacher_image_grid_thw = student_grid_thw
            return

        if not sample.tokens or int(sample.response_length or 0) <= 0:
            return

        # Lazy import to avoid a module-level import cycle: sglang_rollout imports
        # on_policy_distillation, whose OpdManager.prefill calls into
        # this module. GenerateState is a singleton.
        from relax.engine.rollout.sglang_rollout import GenerateState, _run_image_processor

        state = GenerateState(args)
        teacher_prompt_for_tokenize = teacher_prompt if teacher_prompt is not None else sample.prompt

        if state.processor is not None and teacher_has_media:
            teacher_prompt_ids, teacher_mm_train_inputs, _ = await _run_image_processor(
                state, args, teacher_prompt_for_tokenize, teacher_mm
            )
            if teacher_mm_train_inputs:
                sample.teacher_image_grid_thw = teacher_mm_train_inputs.get("image_grid_thw")
            # Encode raw teacher images to base64 (cache on shared dict for group dedup).
            teacher_raw_images = (teacher_mm or {}).get("images") or []
            if teacher_raw_images:
                cached = teacher_mm.get("_teacher_image_b64_cache") if teacher_mm else None
                if cached is None:
                    from relax.utils.data.processing_utils import async_encode_image_for_rollout_engine

                    cached = list(await _gather_encode(teacher_raw_images, async_encode_image_for_rollout_engine))
                    if teacher_mm is not None:
                        teacher_mm["_teacher_image_b64_cache"] = cached
                sample.teacher_image_b64_list = cached
        elif isinstance(teacher_prompt_for_tokenize, str):
            teacher_prompt_ids = state.tokenizer.encode(teacher_prompt_for_tokenize, add_special_tokens=False)
        else:
            teacher_prompt_ids = state.tokenizer.apply_chat_template(
                teacher_prompt_for_tokenize,
                tokenize=True,
                add_generation_prompt=True,
            )

        response_ids = list(sample.tokens[-int(sample.response_length) :])
        sample.teacher_tokens = list(teacher_prompt_ids) + response_ids
        sample.teacher_prompt_length = len(teacher_prompt_ids)

    @staticmethod
    def _to_jsonable(x):
        if x is None:
            return None
        tolist = getattr(x, "tolist", None)
        return tolist() if callable(tolist) else x

    def build_preexpanded_image_data(self, sample: "Sample") -> list | None:
        """Build the SGLang ``image_data=[{"format": "opd_preexpanded_raw",

        ...}]`` payload from the sample's teacher_image_b64_list +
        image_grid_thw.

        Ships raw base64 image bytes (compressed) plus the client-precomputed
        image_grid_thw. The patched teacher SGLang runs its own HF image processor
        on the raw bytes to get pixel_values, then runs its own vision tower on
        the expanded input_ids — bypassing decode->re-tokenize.

        Returns None when there are no images (text-only path).
        """
        image_b64_list = getattr(sample, "teacher_image_b64_list", None)
        image_grid_thw = getattr(sample, "teacher_image_grid_thw", None)
        if not image_b64_list or image_grid_thw is None:
            return None
        return [
            {
                "format": PREEXPANDED_RAW_FORMAT,
                "images_b64": list(image_b64_list),
                "image_grid_thw": self._to_jsonable(image_grid_thw),
            }
        ]

    def teacher_input_ids(self, sample: "Sample", response_length: int) -> list[int]:
        """Return the teacher input_ids for the prefill request.

        When OPSD has pre-expanded teacher_tokens (with image_pad
        placeholders), send them AS-IS so the server prompt length matches
        exactly. Otherwise fall back to the student's collapsed rollout_tokens
        (byte-identical to expanded for text-only).
        """
        teacher_tokens = getattr(sample, "teacher_tokens", None)
        if teacher_tokens is not None:
            return teacher_tokens
        return sample.rollout_tokens or sample.tokens

    def teacher_prompt_len(self, sample: "Sample", response_length: int) -> int:
        """Return the teacher prompt length for logprob_start_len
        derivation."""
        teacher_tokens = getattr(sample, "teacher_tokens", None)
        if teacher_tokens is not None:
            return len(teacher_tokens) - response_length
        return len(sample.tokens) - response_length


async def _gather_encode(images: list, encode_fn) -> list:
    """Parallel-encode a list of raw images to base64 via the executor pool."""
    import asyncio

    return await asyncio.gather(*(encode_fn(img) for img in images))
