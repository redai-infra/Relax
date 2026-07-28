# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tests for text and multimodal prompt length filtering."""

import math
import os
from pathlib import Path

import pytest
from PIL import Image

from relax.utils.data.data_utils import check_sample_length
from relax.utils.types import Sample


def _model_path_or_skip(env_name: str) -> str:
    model_path = os.environ.get(env_name)
    if not model_path:
        pytest.skip(f"{env_name} is not set")
    if not Path(model_path).is_dir():
        pytest.skip(f"{env_name} does not point to an existing directory: {model_path}")
    return model_path


@pytest.fixture(scope="module")
def text_tokenizer():
    transformers = pytest.importorskip("transformers")
    model_path = _model_path_or_skip("RELAX_TEST_QWEN3_4B")
    return transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)


@pytest.fixture(scope="module")
def vl_processor():
    transformers = pytest.importorskip("transformers")
    model_path = _model_path_or_skip("RELAX_TEST_QWEN3_VL_4B")
    return transformers.AutoProcessor.from_pretrained(model_path, local_files_only=True)


def _image_prompt(processor) -> str:
    return (
        f"{processor.vision_start_token}{processor.image_token}{processor.vision_end_token}\n"
        "Describe this image briefly."
    )


def _multimodal_sample(processor, image: Image.Image) -> Sample:
    return Sample(
        prompt=_image_prompt(processor),
        multimodal_inputs={"images": [image]},
    )


def _multimodal_length(sample: Sample, processor) -> int:
    output = processor(text=sample.prompt, images=sample.multimodal_inputs["images"])
    return len(output["input_ids"][0])


def test_multimodal_image_expands_prompt_tokens(vl_processor):
    sample = _multimodal_sample(vl_processor, Image.new("RGB", (256, 256), (255, 0, 0)))
    raw_ids = vl_processor.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    output = vl_processor(text=sample.prompt, images=sample.multimodal_inputs["images"])
    expanded_ids = output["input_ids"][0]
    raw_image_tokens = sum(token_id == vl_processor.image_token_id for token_id in raw_ids)
    expanded_image_tokens = sum(token_id == vl_processor.image_token_id for token_id in expanded_ids)
    image_grid = output["image_grid_thw"][0]
    if hasattr(image_grid, "tolist"):
        image_grid = image_grid.tolist()
    expected_image_tokens = math.prod(int(value) for value in image_grid) // vl_processor.image_processor.merge_size**2

    assert raw_image_tokens == 1, "the raw prompt must contain one image placeholder"
    assert expanded_image_tokens == expected_image_tokens, "image tokens must match the processor grid"
    assert expanded_image_tokens > raw_image_tokens, "the processor must expand the image placeholder"


def test_multimodal_length_boundary_is_inclusive(vl_processor):
    sample = _multimodal_sample(vl_processor, Image.new("RGB", (256, 256), (0, 255, 0)))
    reference_length = _multimodal_length(sample, vl_processor)

    assert check_sample_length(
        sample,
        vl_processor.tokenizer,
        vl_processor,
        max_length=reference_length,
    ), "a prompt exactly at max_length must be retained"
    assert not check_sample_length(
        sample,
        vl_processor.tokenizer,
        vl_processor,
        max_length=reference_length - 1,
    ), "a prompt one token over max_length must be rejected"


def test_larger_image_produces_more_tokens(vl_processor):
    small_sample = _multimodal_sample(vl_processor, Image.new("RGB", (256, 256), (0, 0, 255)))
    large_sample = _multimodal_sample(vl_processor, Image.new("RGB", (512, 512), (0, 0, 255)))
    small_length = _multimodal_length(small_sample, vl_processor)
    large_length = _multimodal_length(large_sample, vl_processor)

    assert large_length > small_length, "a higher-resolution image must produce more prompt tokens"
    assert check_sample_length(
        small_sample,
        vl_processor.tokenizer,
        vl_processor,
        max_length=small_length,
    ), "the small image must fit its dynamically calculated limit"
    assert not check_sample_length(
        large_sample,
        vl_processor.tokenizer,
        vl_processor,
        max_length=small_length,
    ), "the large image must exceed the small image's token limit"


def test_text_prompt_length_filter_uses_tokenizer_length(text_tokenizer):
    prompt = "Relax length filtering should use tokenizer output. " * 32
    sample = Sample(prompt=prompt)
    reference_length = len(text_tokenizer(prompt, add_special_tokens=False)["input_ids"])

    assert check_sample_length(
        sample,
        text_tokenizer,
        processor=None,
        max_length=reference_length,
    ), "a text prompt exactly at max_length must be retained"
    assert not check_sample_length(
        sample,
        text_tokenizer,
        processor=None,
        max_length=reference_length - 1,
    ), "a text prompt one token over max_length must be rejected"
