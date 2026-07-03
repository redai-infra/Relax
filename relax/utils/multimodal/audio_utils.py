# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from io import BytesIO
from typing import Any, ByteString, Union

import audioread
import librosa
import numpy as np

from .config import MultimodalConfig, get_audio_sample_rate


AudioInput = Union[
    np.ndarray,
    ByteString,
    str,
]


def load_audio_from_bytes(audio_bytes: bytes, config: MultimodalConfig = None, **kwargs: Any) -> np.ndarray:
    """Load audio waveform from raw bytes.

    Parameters
    - audio_bytes: Raw audio file bytes (for example WAV data).
    - config: MultimodalConfig object.

    Returns
    - 1-D numpy array with audio samples.
    """
    sample_rate = get_audio_sample_rate(config)
    with BytesIO(audio_bytes) as wav_io:
        audio, _ = librosa.load(wav_io, sr=sample_rate)
    return audio


def load_audio_from_path(audio_path: str, config: MultimodalConfig = None, **kwargs: Any) -> np.ndarray:
    """Load audio from a filesystem path, HTTP(S) URL, or tar-offset spec.

    Tar-offset spec: ``<tar_path>:<wav_name>:<byte_offset>:<byte_size>`` —
    bytes are read in-place from the tar with no extraction, then handed to
    librosa.  Recognised by the literal substring ``.tar:`` in the path.
    """
    sample_rate = get_audio_sample_rate(config)
    if ".tar:" in audio_path:
        return _load_audio_from_tar_offset(audio_path, sample_rate)
    if audio_path.startswith(("http://", "https://")):
        return librosa.load(audioread.ffdec.FFmpegAudioFile(audio_path), sr=sample_rate)[0]
    return librosa.load(audio_path, sr=sample_rate)[0]


def _load_audio_from_tar_offset(spec: str, sample_rate: int) -> np.ndarray:
    """Read a wav payload from a tar at a known offset and decode it.

    spec format: ``<tar_path>:<wav_name>:<byte_offset>:<byte_size>``
    """
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"unexpected audio spec (want tar:name:off:size): {spec!r}")
    tar_path, _wav_name, off_str, sz_str = parts
    off, sz = int(off_str), int(sz_str)
    if off < 0 or sz <= 0:
        raise ValueError(f"invalid offset/size in spec {spec!r}: off={off}, sz={sz}")
    with open(tar_path, "rb") as f:
        f.seek(off)
        blob = f.read(sz)
    if len(blob) != sz:
        raise OSError(f"short read at {tar_path}@{off}: got {len(blob)} bytes, want {sz}")
    with BytesIO(blob) as wav_io:
        audio, _ = librosa.load(wav_io, sr=sample_rate)
    return audio


def load_audio(audio: AudioInput, config: MultimodalConfig = None, **kwargs: Any) -> np.ndarray:
    """Unified loader for different audio input types.

    Parameters
    - audio: One of:
        - `np.ndarray`: a waveform already in memory (returned unchanged).
        - `bytes`: raw audio bytes (WAV/etc.) — handled by `load_audio_from_bytes`.
        - `str`: a path, URL, or `data:` URI — handled by `load_audio_from_path`
          or decoded inline for `data:` URIs.
        - `dict`: with one of `bytes`, `base64`, or `path` fields.
    - config: MultimodalConfig object

    Returns
    - 1-D numpy array with audio samples.
    """
    from relax.utils.multimodal.image_utils import decode_data_uri

    if isinstance(audio, np.ndarray):
        return audio
    if isinstance(audio, str):
        if audio.startswith("data:"):
            return load_audio_from_bytes(decode_data_uri(audio), config=config, **kwargs)
        return load_audio_from_path(audio, config=config, **kwargs)
    if isinstance(audio, (bytes, bytearray)):
        return load_audio_from_bytes(bytes(audio), config=config, **kwargs)
    if isinstance(audio, dict):
        raw = audio.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return load_audio_from_bytes(bytes(raw), config=config, **kwargs)
        b64 = audio.get("base64")
        if isinstance(b64, str):
            import base64 as _b64

            return load_audio_from_bytes(_b64.b64decode(b64), config=config, **kwargs)
        path = audio.get("path")
        if isinstance(path, str):
            if path.startswith("data:"):
                return load_audio_from_bytes(decode_data_uri(path), config=config, **kwargs)
            return load_audio_from_path(path, config=config, **kwargs)
    raise NotImplementedError(f"Unsupported audio input type: {type(audio)}")


def fetch_audio(info: dict, config: MultimodalConfig = None, **kwargs: Any) -> np.ndarray:
    """Convenience helper to extract and load audio from an `info` mapping.

    Parameters
    - info: Mapping that must contain an `"audio"` key whose value is an
        `AudioInput` (see `load_audio`).
    - config: Optional MultimodalConfig object for audio processing parameters.

    Returns
    - 1-D numpy array with audio samples.
    """
    audio_info = info["audio"]
    audio = load_audio(audio_info, config=config, **kwargs)
    return audio
