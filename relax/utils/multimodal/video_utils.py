# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import os
from io import BytesIO
from typing import Any, ByteString, Dict, List, Optional, Tuple, Union

import av
import librosa
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.io.video import _read_from_stream
from torchvision.transforms import InterpolationMode

from relax.utils.logging_utils import get_logger

from .config import (
    MultimodalConfig,
    get_audio_sample_rate,
    get_frame_factor,
    get_video_fps,
    get_video_fps_max_frames,
    get_video_fps_min_frames,
    get_video_max_token_num,
    get_video_min_token_num,
)
from .image_utils import SPATIAL_MERGE_SIZE, decode_data_uri, get_resize_height_width


logger = get_logger(__name__)

if not hasattr(av, "AVError"):
    try:
        from av.error import AVError  # noqa: F401
    except (ImportError, AttributeError):
        av.AVError = OSError

VideoInput = Union[
    List["Image.Image"],
    Dict[str, "np.ndarray"],
    List[bytes],
    ByteString,
    str,
]


def video_smart_resize(
    video: torch.Tensor,
    height: int,
    width: int,
    scale_factor: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    max_ratio: Optional[float] = None,
    config: MultimodalConfig = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Resize a video tensor (T,C,H,W) while respecting pixel and aspect
    constraints.

    Parameters
    - video: Input video as a `torch.Tensor` with shape (T, C, H, W) or a single
      image tensor. The function uses `torchvision.transforms.functional.resize`.
    - height, width: Baseline height/width used to compute target dimensions.
    - scale_factor: Optional alignment factor for resulting dimensions.
    - video_min_pixels, video_max_pixels: Optional pixel bounds.
    - max_ratio: Optional maximum allowed aspect ratio.

    Returns
    - Resized video as a `torch.Tensor` (float dtype).
    """
    video_max_pixels = (
        video_max_pixels if video_max_pixels is not None else (get_video_max_token_num(config) * scale_factor**2)
    )
    video_min_pixels = (
        video_min_pixels if video_min_pixels is not None else (get_video_min_token_num(config) * scale_factor**2)
    )
    assert video_max_pixels >= video_min_pixels, "The max_pixels of video must be greater than or equal to min_pixels."
    h_bar, w_bar = get_resize_height_width(max_ratio, height, width, scale_factor, video_max_pixels, video_min_pixels)
    video = transforms.functional.resize(
        video,
        [h_bar, w_bar],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


def smart_video_nframes(
    info: Dict[str, Any],
    video: torch.Tensor,
    video_meta: Dict[str, int],
    config: MultimodalConfig = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Union[float, int]]]:
    fps = info.get("fps", get_video_fps(config))
    frame_factor = info.get("frame_factor", get_frame_factor(config))
    min_frames = info.get("min_frames", get_video_fps_min_frames(config))
    min_frames = math.ceil(min_frames / frame_factor) * frame_factor
    max_frames = info.get("max_frames", get_video_fps_max_frames(config))
    max_frames = math.floor(max_frames / frame_factor) * frame_factor

    video_fps = video_meta["fps"]
    total_frames = video_meta["total_num_frames"]

    nframes = info.get("nframes", total_frames / video_fps * fps)
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = round(nframes / frame_factor) * frame_factor

    if not (frame_factor <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{frame_factor}, {total_frames}], but got {nframes}.")

    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]

    return video, {"fps": video_fps, "sample_fps": sample_fps, "frames_indices": idx, "total_num_frames": total_frames}


def smart_audio_nframes(
    info: Dict[str, Any],
    audio: Optional[np.ndarray],
    audio_meta: Optional[Dict[str, int]],
    config: MultimodalConfig = None,
    **kwargs: Any,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
    """Resample and return audio plus metadata adapted to requested sample
    rate.

    Parameters
    - info: Mapping with optional `sample_rate` key.
    - audio: 1-D numpy array or None.
    - audio_meta: Metadata mapping with at least `fps` and `total_num_frames` keys.

    Returns
    - Tuple `(audio, meta)` where `audio` is the resampled numpy array (or None)
      and `meta` contains fps/sample counts or None if input audio is None.
    """
    if audio is None:
        return None, None
    sample_rate = info.get("sample_rate", get_audio_sample_rate(config))
    audio_fps = audio_meta["fps"]
    num_frames = audio_meta["total_num_frames"]
    if audio_fps != sample_rate:
        audio = librosa.resample(y=audio, orig_sr=audio_fps, target_sr=sample_rate)
    sample_num_frames = len(audio)
    return audio, {
        "fps": audio_fps,
        "sample_fps": sample_rate,
        "sample_num_frames": sample_num_frames,
        "total_num_frames": num_frames,
    }


def load_video_from_path(
    video: str,
    use_audio_in_video: bool = True,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, int], Optional[np.ndarray], Optional[Dict[str, int]]]:
    """Load video and optionally extract audio from a path or URL.

    Returns `(video_tensor, video_metadata, audio, audio_metadata)`.
    """
    if "http://" in video or "https://" in video:
        from packaging import version

        if version.parse(torchvision.__version__) < version.parse("0.19.0"):
            logger.warning_once(
                "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0."
            )
    else:
        if "file://" in video:
            video = video[7:]
        assert os.path.exists(video), f"Video path {video} does not exist."

    video, _audio, read_info = torchvision.io.read_video(
        video,
        start_pts=0.0,
        end_pts=None,
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.shape[0], read_info["video_fps"]
    video_metadata = {"fps": video_fps, "total_num_frames": total_frames}

    audio, audio_metadata = None, None
    if use_audio_in_video and _audio.numel() > 0:
        # Average across channels if multi-channel
        audio = torch.mean(_audio, dim=0).numpy()
        audio_fps = read_info["audio_fps"]
        audio_metadata = {"fps": audio_fps, "total_num_frames": _audio.shape[0]}

    return video, video_metadata, audio, audio_metadata


def load_video_from_bytes_list(
    video: Union[List[bytes], np.ndarray],
    use_audio_in_video: bool = False,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, int], Optional[np.ndarray], Optional[Dict[str, int]]]:
    """Loads video frames from a list of bytes with memory optimization.

    Expects 'fps' in kwargs for metadata.
    """
    if use_audio_in_video:
        raise ValueError("load_video_from_bytes_list not support to load audio")
    if isinstance(video, np.ndarray):
        video = video.tolist()
    if not video:
        raise ValueError("Input video frame list is empty")

    fps_val = kwargs.get("fps", 2.0)
    nframes = len(video)

    # Decode first frame to get dimensions
    with Image.open(BytesIO(video[0])) as img:
        img = img.convert("RGB")
        w, h = img.size

    T, C = nframes, 3
    # Memory optimization: Allocate uint8 tensor
    video_tensor = torch.empty((T, C, h, w), dtype=torch.uint8)

    for i, frame_bytes in enumerate(video):
        with Image.open(BytesIO(frame_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            frame_arr = np.array(img)
            # Convert to Tensor (C, H, W)
            frame_t = torch.from_numpy(frame_arr).permute(2, 0, 1)
            video_tensor[i] = frame_t

    video_metadata = {"fps": fps_val, "total_num_frames": nframes}
    return video_tensor, video_metadata, None, None


def load_video_from_bytes(
    video: bytes,
    use_audio_in_video: bool = True,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, int], Optional[np.ndarray], Optional[Dict[str, int]]]:
    container = av.open(BytesIO(video))
    video_frames = _read_from_stream(
        container,
        0.0,
        float("inf"),
        "sec",
        container.streams.video[0],
        {"video": 0},
    )
    video_fps = container.streams.video[0].average_rate
    video_metadata = {"fps": video_fps, "total_num_frames": len(video_frames)}
    vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
    video = torch.as_tensor(np.stack(vframes_list)).permute(0, 3, 1, 2)  # t,c,h,w

    audio, audio_metadata = None, None
    if use_audio_in_video and len(container.streams.audio) > 0:
        audio_frames = _read_from_stream(
            container,
            0.0,
            float("inf"),
            "sec",
            container.streams.audio[0],
            {"audio": 0},
        )

        aframes_list = [frame.to_ndarray() for frame in audio_frames]
        if len(aframes_list) > 0:
            aframes = np.concatenate(aframes_list, 1)
            aframes = np.mean(aframes, axis=0)
            audio_fps = container.streams.audio[0].rate
            audio_metadata = {"fps": audio_fps, "total_num_frames": len(aframes_list)}
            audio = aframes

    return video, video_metadata, audio, audio_metadata


def load_video(
    video: VideoInput,
    use_audio_in_video: bool,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, int], Optional[np.ndarray], Optional[Dict[str, int]]]:
    """Dispatch loader based on `video` input type.

    Supports local path/URL (`str`), `data:` URI string, raw bytes (`bytes`),
    list/array of frame bytes, or a `{path|bytes|base64}` dict. Returns
    `(video, video_meta, audio, audio_meta)`.
    """
    if isinstance(video, str):
        if video.startswith("data:"):
            return load_video_from_bytes(decode_data_uri(video), use_audio_in_video, **kwargs)
        return load_video_from_path(video, use_audio_in_video, **kwargs)
    if isinstance(video, (bytes, bytearray)):
        return load_video_from_bytes(bytes(video), use_audio_in_video, **kwargs)
    if isinstance(video, (list, np.ndarray)):
        return load_video_from_bytes_list(video, use_audio_in_video, **kwargs)
    if isinstance(video, dict):
        raw = video.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return load_video_from_bytes(bytes(raw), use_audio_in_video, **kwargs)
        b64 = video.get("base64")
        if isinstance(b64, str):
            import base64 as _b64

            return load_video_from_bytes(_b64.b64decode(b64), use_audio_in_video, **kwargs)
        path = video.get("path")
        if isinstance(path, str):
            if path.startswith("data:"):
                return load_video_from_bytes(decode_data_uri(path), use_audio_in_video, **kwargs)
            return load_video_from_path(path, use_audio_in_video, **kwargs)
    raise NotImplementedError(f"Unsupported video input type: {type(video)}")


def fetch_video(
    info: Dict[str, Any],
    image_patch_size: int = 14,
    use_audio_in_video: bool = True,
    config: MultimodalConfig = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, Any], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Load and process a video according to `info` metadata.

    Parameters
    - info: Mapping with `"video"` key and optional resizing metadata
      (`resized_height`, `resized_width`, `min_pixels`, `max_pixels`, etc.).
    - image_patch_size: Base patch size used to compute alignment and pixel limits.
    - use_audio_in_video: Whether to extract audio from the video track.

    Returns
    - Tuple `(processed_video, processed_video_meta, processed_audio, processed_audio_meta)`.
    """
    video_info = info["video"]
    video, video_meta, audio, audio_meta = load_video(video_info, use_audio_in_video, **kwargs)
    processed_video, processed_video_meta = smart_video_nframes(info, video, video_meta, config=config)
    processed_audio, processed_audio_meta = smart_audio_nframes(info, audio, audio_meta, config=config)

    image_factor = image_patch_size * SPATIAL_MERGE_SIZE

    # resize
    if "resized_height" in info and "resized_width" in info:
        processed_video = video_smart_resize(
            processed_video,
            info["resized_height"],
            info["resized_width"],
            scale_factor=image_factor,
            config=config,
        )
    else:
        VIDEO_FRAME_MIN_PIXELS = get_video_min_token_num(config) * image_factor**2
        VIDEO_FRAME_MAX_PIXELS = get_video_max_token_num(config) * image_factor**2
        _, _, height, width = processed_video.shape
        min_pixels = info.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
        max_pixels = max(VIDEO_FRAME_MAX_PIXELS, int(min_pixels * 1.05))

        max_pixels_supposed = info.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)

        processed_video = video_smart_resize(
            processed_video,
            height,
            width,
            scale_factor=image_factor,
            video_min_pixels=min_pixels,
            video_max_pixels=max_pixels,
            config=config,
        )

    return processed_video, processed_video_meta, processed_audio, processed_audio_meta
