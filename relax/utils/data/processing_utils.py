# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import base64
import io
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import imageio.v2 as imageio
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Shared thread pool for offloading CPU-intensive media encoding from the asyncio event loop.
# PNG compression (libpng), H.264 encoding (libx264), and base64 encoding are all C-level
# operations that release the GIL, so a thread pool achieves true parallelism without the
# serialization overhead of a process pool.
# FIXME: hardcode
_ENCODE_EXECUTOR = ThreadPoolExecutor(max_workers=32)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    # Multimodal models like Qwen3-Omni ship the chat template in a standalone
    # chat_template.json (loaded by AutoProcessor) rather than tokenizer_config.json,
    # so AutoTokenizer leaves chat_template unset. Backfill from the sidecar file.
    if getattr(tokenizer, "chat_template", None) is None and os.path.isdir(name_or_path):
        chat_template_path = os.path.join(name_or_path, "chat_template.json")
        if os.path.isfile(chat_template_path):
            with open(chat_template_path) as f:
                chat_template = json.load(f).get("chat_template")
            if chat_template:
                tokenizer.chat_template = chat_template
                logger.info(f"Loaded chat_template from {chat_template_path}")
    return tokenizer


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:
    forced = {
        # force return_tensors to None for input_ids
        "return_tensors": None,
    }
    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    result.update(forced)

    # set return_tensors="pt" for modality-specific outputs
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def process_vision_info(prompt, processor, use_audio_in_video, config=None):
    # temporary solution, will write image utils for slime later
    from relax.utils.multimodal.process import process_multimodal_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos, audios = process_multimodal_info(
        prompt, image_patch_size=image_patch_size, use_audio_in_video=use_audio_in_video, config=config
    )
    multimodal_inputs = {"images": images, "videos": videos, "audio": audios}
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_video_tensor_for_rollout_engine(video: torch.Tensor) -> str:
    """
    video: Tensor[T, C, H, W], RGB, uint8
    return: base64 encoded mp4
    """
    if video.dtype != torch.uint8:
        video = video.clamp(0, 255).to(torch.uint8)

    video_np = video.permute(0, 2, 3, 1).cpu().numpy()  # T,H,W,C

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        writer = imageio.get_writer(
            f.name,
            fps=4,
            codec="libx264",
        )
        for frame in video_np:
            writer.append_data(frame)
        writer.close()

        with open(f.name, "rb") as rf:
            return base64.b64encode(rf.read()).decode("utf-8")


def encode_audio_for_rollout_engine(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> str:
    """Encode audio waveform into WAV base64 string for sglang rollout.

    Args:
        audio: np.ndarray, shape (N,) or (N, C), float32
        sample_rate: audio sampling rate

    Returns:
        base64 encoded wav string
    """
    if audio is None:
        return None

    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)

    audio = audio.astype(np.float32)

    buffer = io.BytesIO()
    sf.write(
        buffer,
        audio,
        samplerate=sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    # For sglang, it needs base64 format audio file starting with "data:,"
    return "data:," + base64.b64encode(buffer.getvalue()).decode("utf-8")


async def async_encode_image_for_rollout_engine(image) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ENCODE_EXECUTOR, encode_image_for_rollout_engine, image)


async def async_encode_video_tensor_for_rollout_engine(video: torch.Tensor) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ENCODE_EXECUTOR, encode_video_tensor_for_rollout_engine, video)


async def async_encode_audio_for_rollout_engine(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ENCODE_EXECUTOR, encode_audio_for_rollout_engine, audio, sample_rate)
