# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.data.group_processor import (
    MM_GROUP_ID_KEY,
    MM_GROUP_OWNER_KEY,
    copy_group_processor_output,
    get_reusable_group_processor_input,
    pack_group_multimodal_train_inputs,
    unpack_group_multimodal_train_inputs,
)


def _sample(prompt, multimodal_inputs):
    return SimpleNamespace(prompt=prompt, multimodal_inputs=multimodal_inputs)


def _packed_rows(packed):
    return [dict(row.items()) for row in packed.unbind(0)]


def _torch():
    return pytest.importorskip("torch")


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
    torch = _torch()
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    grid = torch.tensor([[1, 2, 2]])
    samples = [
        SimpleNamespace(group_index=9, multimodal_train_inputs={"pixel_values": tensor, "image_grid_thw": grid})
        for _ in range(8)
    ]

    packed, source_count, ref_count = pack_group_multimodal_train_inputs(samples, group_size=8)

    assert type(packed).__name__ == "TensorDict"
    assert source_count == 1
    assert ref_count == 7
    rows = _packed_rows(packed)
    assert rows[0][MM_GROUP_ID_KEY] == 9
    assert rows[0][MM_GROUP_OWNER_KEY]
    assert rows[0]["pixel_values"].numel() == tensor.numel()
    assert all(row["pixel_values"].numel() == 0 for row in rows[1:])

    unpacked = unpack_group_multimodal_train_inputs(list(reversed(rows)), group_size=8)

    assert len(unpacked) == 8
    assert len({item["pixel_values"].untyped_storage().data_ptr() for item in unpacked}) == 1
    assert all(MM_GROUP_ID_KEY not in item and MM_GROUP_OWNER_KEY not in item for item in unpacked)
    assert unpacked[0] is not unpacked[1]


def test_group_multimodal_transfer_falls_back_when_storage_is_not_shared():
    torch = _torch()
    samples = [
        SimpleNamespace(group_index=3, multimodal_train_inputs={"pixel_values": torch.ones(2, 4)}) for _ in range(8)
    ]

    packed, source_count, ref_count = pack_group_multimodal_train_inputs(samples, group_size=8)

    assert source_count == 0
    assert ref_count == 0
    assert packed == [sample.multimodal_train_inputs for sample in samples]


def test_group_multimodal_transfer_rejects_missing_source():
    with pytest.raises(ValueError, match="missing multimodal group source"):
        unpack_group_multimodal_train_inputs(
            [{MM_GROUP_ID_KEY: 12, MM_GROUP_OWNER_KEY: False}] * 8,
            group_size=8,
        )


def test_group_multimodal_transfer_rejects_incomplete_group():
    torch = _torch()
    tensor = torch.ones(2, 4)
    packed = (
        [{"pixel_values": tensor, MM_GROUP_ID_KEY: 6, MM_GROUP_OWNER_KEY: True}]
        + [{MM_GROUP_ID_KEY: 6, MM_GROUP_OWNER_KEY: False}] * 6
        + [None]
    )

    with pytest.raises(ValueError, match="has 7 rows, expected group_size 8"):
        unpack_group_multimodal_train_inputs(packed, group_size=8)


def test_group_multimodal_transfer_rejects_duplicate_source():
    torch = _torch()
    tensor = torch.ones(2, 4)
    packed = [
        {"pixel_values": tensor, MM_GROUP_ID_KEY: 6, MM_GROUP_OWNER_KEY: True},
        {"pixel_values": tensor, MM_GROUP_ID_KEY: 6, MM_GROUP_OWNER_KEY: True},
        {MM_GROUP_ID_KEY: 6, MM_GROUP_OWNER_KEY: False},
        {MM_GROUP_ID_KEY: 6, MM_GROUP_OWNER_KEY: False},
    ]

    with pytest.raises(ValueError, match="duplicate multimodal group source"):
        unpack_group_multimodal_train_inputs(packed, group_size=4)


def test_group_multimodal_transfer_supports_interleaved_complete_groups():
    torch = _torch()
    first_tensor = torch.ones(2, 4)
    second_tensor = torch.full((3, 4), 2.0)
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
    unpacked = unpack_group_multimodal_train_inputs(_packed_rows(packed), group_size=4)

    assert source_count == 2
    assert ref_count == 6
    assert [item["pixel_values"].shape for item in unpacked] == [
        first_tensor.shape,
        second_tensor.shape,
    ] * 4
    assert len({unpacked[index]["pixel_values"].untyped_storage().data_ptr() for index in (0, 2, 4, 6)}) == 1
    assert len({unpacked[index]["pixel_values"].untyped_storage().data_ptr() for index in (1, 3, 5, 7)}) == 1


def test_group_multimodal_transfer_leaves_unmarked_inputs_unchanged():
    inline = {"text": "caption-only"}
    unpacked = unpack_group_multimodal_train_inputs([None, inline, None, None], group_size=4)

    assert unpacked == [None, inline, None, None]
    assert unpacked[1] is inline


def test_group_multimodal_transfer_rejects_reserved_marker_collision():
    torch = _torch()
    tensor = torch.ones(2, 4)
    samples = [
        SimpleNamespace(
            group_index=4,
            multimodal_train_inputs={"pixel_values": tensor, MM_GROUP_ID_KEY: torch.tensor(4)},
        )
        for _ in range(8)
    ]

    with pytest.raises(ValueError, match="collides with Relax group transfer marker keys"):
        pack_group_multimodal_train_inputs(samples, group_size=8)
