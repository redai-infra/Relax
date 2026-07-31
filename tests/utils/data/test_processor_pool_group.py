# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.data.group_processor import (
    MM_GROUP_REF_KEY,
    MM_GROUP_SOURCE_KEY,
    copy_group_processor_output,
    get_reusable_group_processor_input,
    pack_group_multimodal_train_inputs,
    unpack_group_multimodal_train_inputs,
)


def _sample(prompt, multimodal_inputs):
    return SimpleNamespace(prompt=prompt, multimodal_inputs=multimodal_inputs)


def test_group_processor_input_reuses_same_prompt_and_media_object():
    media = {"images": [object()], "videos": [], "audio": []}
    samples = [_sample("question", media) for _ in range(8)]

    reusable = get_reusable_group_processor_input(samples)

    assert reusable == ("question", media)


def test_group_processor_input_rejects_prompt_or_media_identity_mismatch():
    media = {"images": [object()], "videos": [], "audio": []}
    same_value_different_object = dict(media)

    assert get_reusable_group_processor_input([_sample("a", media), _sample("b", media)]) is None
    assert get_reusable_group_processor_input([_sample("a", media), _sample("a", same_value_different_object)]) is None


def test_group_processor_input_rejects_empty_media():
    media = {"images": [], "videos": [], "audio": []}

    assert get_reusable_group_processor_input([_sample("question", media)]) is None


def test_group_processor_output_copies_containers_and_shares_tensor_storage():
    tensor = object()
    prompt_ids = [1, 2]
    train_inputs = {"pixel_values": tensor, "image_grid_thw": [tensor]}

    copied_prompt_ids, copied_train_inputs = copy_group_processor_output(prompt_ids, train_inputs)

    assert copied_prompt_ids == prompt_ids
    assert copied_prompt_ids is not prompt_ids
    assert copied_train_inputs is not train_inputs
    assert copied_train_inputs["image_grid_thw"] is not train_inputs["image_grid_thw"]
    assert copied_train_inputs["pixel_values"] is tensor
    assert copied_train_inputs["image_grid_thw"][0] is tensor


def test_group_multimodal_transfer_keeps_one_payload_and_reconstructs_aliases():
    tensor = object()
    samples = [
        SimpleNamespace(group_index=9, multimodal_train_inputs={"pixel_values": tensor, "image_grid_thw": [tensor]})
        for _ in range(8)
    ]

    packed, source_count, ref_count = pack_group_multimodal_train_inputs(samples, group_size=8)

    assert source_count == 1
    assert ref_count == 7
    assert packed[0][MM_GROUP_SOURCE_KEY] == 9
    assert packed[0]["pixel_values"] is tensor
    assert packed[1:] == [{MM_GROUP_REF_KEY: 9}] * 7
    assert sum("pixel_values" in item for item in packed) == 1

    unpacked = unpack_group_multimodal_train_inputs(list(reversed(packed)), group_size=8)

    assert len(unpacked) == 8
    assert all(item["pixel_values"] is tensor for item in unpacked)
    assert all(MM_GROUP_SOURCE_KEY not in item and MM_GROUP_REF_KEY not in item for item in unpacked)
    assert unpacked[0] is not unpacked[1]
    assert unpacked[0]["image_grid_thw"] is not unpacked[1]["image_grid_thw"]


def test_group_multimodal_transfer_falls_back_when_storage_is_not_shared():
    samples = [SimpleNamespace(group_index=3, multimodal_train_inputs={"pixel_values": object()}) for _ in range(8)]

    packed, source_count, ref_count = pack_group_multimodal_train_inputs(samples, group_size=8)

    assert source_count == 0
    assert ref_count == 0
    assert packed == [sample.multimodal_train_inputs for sample in samples]


def test_group_multimodal_transfer_rejects_missing_source():
    with pytest.raises(ValueError, match="missing multimodal group source"):
        unpack_group_multimodal_train_inputs([{MM_GROUP_REF_KEY: 12}] * 8, group_size=8)


def test_group_multimodal_transfer_rejects_incomplete_group():
    tensor = object()
    packed = [{"pixel_values": tensor, MM_GROUP_SOURCE_KEY: 6}] + [{MM_GROUP_REF_KEY: 6}] * 6 + [None]

    with pytest.raises(ValueError, match="has 7 rows, expected group_size 8"):
        unpack_group_multimodal_train_inputs(packed, group_size=8)


def test_group_multimodal_transfer_rejects_duplicate_source():
    tensor = object()
    packed = [
        {"pixel_values": tensor, MM_GROUP_SOURCE_KEY: 6},
        {"pixel_values": tensor, MM_GROUP_SOURCE_KEY: 6},
        {MM_GROUP_REF_KEY: 6},
        {MM_GROUP_REF_KEY: 6},
    ]

    with pytest.raises(ValueError, match="duplicate multimodal group source"):
        unpack_group_multimodal_train_inputs(packed, group_size=4)


def test_group_multimodal_transfer_supports_interleaved_complete_groups():
    first_tensor = object()
    second_tensor = object()
    samples = [
        SimpleNamespace(group_index=group_index, multimodal_train_inputs={"pixel_values": tensor})
        for group_index, tensor in [
            (10, first_tensor),
            (20, second_tensor),
            (10, first_tensor),
            (20, second_tensor),
            (10, first_tensor),
            (20, second_tensor),
            (10, first_tensor),
            (20, second_tensor),
        ]
    ]

    packed, source_count, ref_count = pack_group_multimodal_train_inputs(samples, group_size=4)
    unpacked = unpack_group_multimodal_train_inputs(packed, group_size=4)

    assert source_count == 2
    assert ref_count == 6
    assert [item["pixel_values"] for item in unpacked] == [
        first_tensor,
        second_tensor,
        first_tensor,
        second_tensor,
        first_tensor,
        second_tensor,
        first_tensor,
        second_tensor,
    ]


def test_group_multimodal_transfer_leaves_unmarked_inputs_unchanged():
    inline = {"text": "caption-only"}
    unpacked = unpack_group_multimodal_train_inputs([None, inline, None, None], group_size=4)

    assert unpacked == [None, inline, None, None]
    assert unpacked[1] is inline


def test_group_multimodal_transfer_rejects_reserved_marker_collision():
    tensor = object()
    samples = [
        SimpleNamespace(
            group_index=4,
            multimodal_train_inputs={"pixel_values": tensor, MM_GROUP_REF_KEY: 4},
        )
        for _ in range(8)
    ]

    with pytest.raises(ValueError, match="collides with Relax group transfer marker keys"):
        pack_group_multimodal_train_inputs(samples, group_size=8)
