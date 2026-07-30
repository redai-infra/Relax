# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.backends.megatron.weight_update.static_state import relocate_colocate_static_tensors
from relax.utils.weight_handoff import get_memory_preflight_error, validate_export_response


NAMES = ["model.language_model.embed_tokens.weight", "lm_head.weight"]


class Qwen3VLMultimodalRotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_freq = torch.arange(8, dtype=torch.float32)


class OtherRotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_freq = torch.arange(4, dtype=torch.float32)


def _response(**overrides):
    response = {
        "success": True,
        "message": "Success",
        "weight_version": "7",
        "serialized_named_tensors": "ipc-payload",
        "metadata": [{"name": name, "shape": [2, 2], "dtype": "bfloat16"} for name in NAMES],
    }
    response.update(overrides)
    return response


def test_export_response_validation_accepts_matching_contract():
    metadata, serialized = validate_export_response(_response(), NAMES, expected_version=7)

    assert [item["name"] for item in metadata] == NAMES
    assert serialized == "ipc-payload"


def test_export_response_validation_rejects_stale_version():
    with pytest.raises(RuntimeError, match="version mismatch"):
        validate_export_response(_response(weight_version="6"), NAMES, expected_version=7)


def test_export_response_validation_rejects_missing_field():
    with pytest.raises(RuntimeError, match="metadata count mismatch"):
        validate_export_response(_response(metadata=_response()["metadata"][:1]), NAMES, expected_version=7)


def test_export_response_validation_rejects_reordered_names():
    metadata = list(reversed(_response()["metadata"]))
    with pytest.raises(RuntimeError, match="metadata names"):
        validate_export_response(_response(metadata=metadata), NAMES, expected_version=7)


def test_memory_preflight_reports_all_allocation_components():
    gib = 1024**3
    error = get_memory_preflight_error(
        rank=1,
        target="Megatron",
        free_bytes=3 * gib,
        total_bytes=8 * gib,
        target_bytes=2 * gib,
        bucket_bytes=1 * gib,
        margin_bytes=1 * gib,
    )

    assert error is not None
    assert "rank=1 target=Megatron" in error
    assert "target_weights=2.00 GiB" in error
    assert "conversion_bucket=1.00 GiB" in error
    assert "margin=1.00 GiB" in error


def test_memory_preflight_accepts_exact_capacity():
    assert (
        get_memory_preflight_error(
            rank=0,
            target="SGLang",
            free_bytes=4,
            total_bytes=8,
            target_bytes=2,
            bucket_bytes=1,
            margin_bytes=1,
        )
        is None
    )


def test_relocate_colocate_static_tensors_preserves_value_in_new_storage():
    rotary = Qwen3VLMultimodalRotaryEmbedding()
    original = rotary.inv_freq
    model = torch.nn.ModuleList([rotary])

    relocated_bytes = relocate_colocate_static_tensors([model])

    assert torch.equal(rotary.inv_freq, original)
    assert rotary.inv_freq.data_ptr() != original.data_ptr()
    assert relocated_bytes == original.numel() * original.element_size()


def test_relocate_colocate_static_tensors_ignores_other_rotary_modules():
    rotary = OtherRotaryEmbedding()
    original = rotary.inv_freq

    with pytest.raises(RuntimeError, match="could not find Qwen3-VL rotary inv_freq"):
        relocate_colocate_static_tensors([rotary])

    assert rotary.inv_freq is original
