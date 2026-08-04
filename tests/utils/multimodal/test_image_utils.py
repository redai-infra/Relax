# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for relax.utils.multimodal.image_utils.

All image inputs are constructed in memory (PIL / PNG bytes / data URI); no
real image files or network access are required. The patch-alignment test
derives patch_size / merge_size from the preprocessor config of the
Qwen3-VL-4B-Instruct model pointed to by the RELAX_TEST_QWEN3_VL_4B
environment variable, and is skipped when the model directory is unavailable.
"""

import base64
import io
import json
import os

import pytest
from PIL import Image

from relax.utils.multimodal.image_utils import (
    decode_data_uri,
    get_resize_height_width,
    image_smart_resize,
    load_image,
    to_rgb,
)


# Pixel bounds used across the resize tests (aligned to the 28x28 grid).
MAX_PIXELS = 16384 * 28 * 28
MIN_PIXELS = 4 * 28 * 28


def _png_bytes(width: int = 200, height: int = 100, color: tuple = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _png_data_uri(width: int = 200, height: int = 100) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(width, height)).decode()


# ── get_resize_height_width ──────────────────────────────────────────────────


def test_get_resize_height_width_aligns_up_to_scale_factor():
    # round(100/28)=4 -> 112, round(133/28)=5 -> 140; well within pixel bounds.
    h_bar, w_bar = get_resize_height_width(None, 100, 133, 28, MAX_PIXELS, MIN_PIXELS)
    assert (h_bar, w_bar) == (112, 140)
    assert h_bar % 28 == 0 and w_bar % 28 == 0


def test_get_resize_height_width_clamps_to_max_pixels():
    # 2000x1000 = 2,000,000 px > MAX_PIXELS; must shrink onto the 28-grid and
    # never exceed the pixel budget.
    max_pixels = 100 * 28 * 28  # 78,400
    h_bar, w_bar = get_resize_height_width(None, 2000, 1000, 28, max_pixels, MIN_PIXELS)
    assert h_bar % 28 == 0 and w_bar % 28 == 0
    assert h_bar * w_bar <= max_pixels
    # beta = sqrt(2e6/78400) ≈ 5.05 -> floor(14.14)*28=392, floor(7.07)*28=196
    assert (h_bar, w_bar) == (392, 196)


def test_get_resize_height_width_boosts_to_min_pixels():
    # 28x28 = 784 px < min_pixels; must grow onto the 28-grid.
    min_pixels = 8 * 28 * 28  # 6,272
    h_bar, w_bar = get_resize_height_width(None, 28, 28, 28, MAX_PIXELS, min_pixels)
    assert h_bar % 28 == 0 and w_bar % 28 == 0
    assert h_bar * w_bar >= min_pixels
    # beta = sqrt(6272/784) ≈ 2.83 -> ceil(2.83)*28 = 84
    assert (h_bar, w_bar) == (84, 84)


def test_get_resize_height_width_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError, match="aspect ratio"):
        get_resize_height_width(2.0, 100, 400, 28, MAX_PIXELS, MIN_PIXELS)


def test_get_resize_height_width_without_scale_factor_keeps_size():
    h_bar, w_bar = get_resize_height_width(None, 101, 133, None, MAX_PIXELS, MIN_PIXELS)
    assert (h_bar, w_bar) == (101, 133)


def test_get_resize_height_width_without_scale_factor_clamps_to_max_pixels():
    # 2000x1000 = 2,000,000 px > max_pixels; must shrink without grid alignment.
    max_pixels = 100 * 28 * 28  # 78,400
    h_bar, w_bar = get_resize_height_width(None, 2000, 1000, None, max_pixels, MIN_PIXELS)
    assert h_bar * w_bar <= max_pixels
    # beta = sqrt(2e6/78400) ≈ 5.05 -> floor(395.98)=395, floor(197.99)=197
    assert (h_bar, w_bar) == (395, 197)


def test_get_resize_height_width_without_scale_factor_boosts_to_min_pixels():
    # 28x28 = 784 px < min_pixels; must grow without grid alignment.
    min_pixels = 8 * 28 * 28  # 6,272
    h_bar, w_bar = get_resize_height_width(None, 28, 28, None, MAX_PIXELS, min_pixels)
    assert h_bar * w_bar >= min_pixels
    # beta = sqrt(6272/784) ≈ 2.83 -> ceil(79.20) = 80
    assert (h_bar, w_bar) == (80, 80)


# ── image_smart_resize ───────────────────────────────────────────────────────


def test_image_smart_resize_produces_aligned_size():
    image = Image.new("RGB", (200, 100))  # PIL size is (width, height)
    resized = image_smart_resize(
        image,
        100,
        200,
        scale_factor=28,
        image_min_pixels=MIN_PIXELS,
        image_max_pixels=MAX_PIXELS,
    )
    # round(100/28)*28=112, round(200/28)*28=196; PIL size is (width, height).
    assert resized.size == (196, 112)
    assert resized.size[0] % 28 == 0 and resized.size[1] % 28 == 0


def test_image_smart_resize_rejects_inverted_pixel_bounds():
    image = Image.new("RGB", (200, 100))
    with pytest.raises(AssertionError, match="max_pixels"):
        image_smart_resize(image, 100, 200, scale_factor=28, image_min_pixels=MAX_PIXELS, image_max_pixels=MIN_PIXELS)


# ── to_rgb ───────────────────────────────────────────────────────────────────


def test_to_rgb_composites_rgba_over_white():
    rgba = Image.new("RGBA", (1, 2))
    rgba.putpixel((0, 0), (255, 0, 0, 128))  # half-transparent red
    rgba.putpixel((0, 1), (0, 0, 0, 0))  # fully transparent
    rgb = to_rgb(rgba)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((0, 0)) == (255, 127, 127)  # red over white at alpha=128
    assert rgb.getpixel((0, 1)) == (255, 255, 255)  # transparent -> white


def test_to_rgb_converts_grayscale_to_rgb():
    gray = Image.new("L", (1, 1), 128)
    rgb = to_rgb(gray)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((0, 0)) == (128, 128, 128)


# ── decode_data_uri ──────────────────────────────────────────────────────────


def test_decode_data_uri_base64_payload():
    payload = b"\x89PNG\r\n\x1a\nfake-image-bytes"
    uri = "data:image/png;base64," + base64.b64encode(payload).decode()
    assert decode_data_uri(uri) == payload


def test_decode_data_uri_url_encoded_payload():
    assert decode_data_uri("data:text/plain,hello%20world%21") == b"hello world!"


def test_decode_data_uri_rejects_malformed_uri():
    with pytest.raises(ValueError, match="Malformed data URI"):
        decode_data_uri("data:missing-comma")


# ── load_image ───────────────────────────────────────────────────────────────


def test_load_image_passthrough_pil():
    image = Image.new("RGB", (200, 100))
    assert load_image(image) is image


def test_load_image_from_bytes_and_bytearray():
    raw = _png_bytes()
    for variant in (raw, bytearray(raw)):
        image = load_image(variant)
        assert isinstance(image, Image.Image)
        assert image.size == (200, 100)
        assert image.getpixel((0, 0)) == (10, 20, 30)


def test_load_image_from_data_uri_string():
    image = load_image(_png_data_uri())
    assert isinstance(image, Image.Image)
    assert image.size == (200, 100)


def test_load_image_from_local_path_and_file_uri(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(_png_bytes())
    for variant in (str(path), f"file://{path}"):
        image = load_image(variant)
        assert isinstance(image, Image.Image)
        assert image.size == (200, 100)


def test_load_image_from_dict_variants(tmp_path):
    path = tmp_path / "sample.png"
    raw = _png_bytes()
    path.write_bytes(raw)
    variants = (
        {"bytes": raw},
        {"base64": base64.b64encode(raw).decode()},
        {"path": str(path)},
        {"path": _png_data_uri()},
    )
    for variant in variants:
        image = load_image(variant)
        assert isinstance(image, Image.Image)
        assert image.size == (200, 100)


def test_load_image_rejects_unsupported_type():
    with pytest.raises(NotImplementedError, match="Unsupported image input type"):
        load_image(123)


# ── patch alignment with the real Qwen3-VL processor ─────────────────────────

_MODEL_DIR = os.environ.get("RELAX_TEST_QWEN3_VL_4B")


def _qwen3_vl_patch_grid(model_dir: str) -> tuple:
    """Return (patch_size, merge_size) from the model's preprocessor config."""
    with open(os.path.join(model_dir, "preprocessor_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    return config["patch_size"], config["merge_size"]


@pytest.mark.skipif(
    not _MODEL_DIR or not os.path.isdir(_MODEL_DIR),
    reason="RELAX_TEST_QWEN3_VL_4B is not set or the model directory is missing",
)
def test_smart_resize_aligns_with_qwen3_vl_patch_grid():
    patch_size, merge_size = _qwen3_vl_patch_grid(_MODEL_DIR)
    factor = patch_size * merge_size

    image = Image.new("RGB", (401, 237))  # deliberately unaligned size
    resized = image_smart_resize(
        image,
        237,
        401,
        scale_factor=factor,
        image_min_pixels=4 * factor * factor,
        image_max_pixels=16384 * factor * factor,
    )

    out_w, out_h = resized.size
    # Every resized side must land on the patch*merge grid, so the ViT token
    # count (h/patch) x (w/patch) is an integer multiple of the merge group.
    assert out_w % factor == 0
    assert out_h % factor == 0
    assert (out_w // patch_size) % merge_size == 0
    assert (out_h // patch_size) % merge_size == 0
    # Alignment of a 237x401 image on a 32-grid (Qwen3-VL: patch 16, merge 2):
    # round(237/32)*32 = 224, round(401/32)*32 = 416.
    assert (out_h, out_w) == (
        max(factor, round(237 / factor) * factor),
        max(factor, round(401 / factor) * factor),
    )
