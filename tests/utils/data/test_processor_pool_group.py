# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import gc
from types import SimpleNamespace

import pytest

from relax.utils.data.group_processor import (
    MM_GROUP_ID_KEY,
    MM_GROUP_OWNER_KEY,
    copy_group_processor_output,
    get_reusable_group_processor_input,
    pack_group_multimodal_train_inputs,
    unpack_group_multimodal_train_inputs,
    validate_group_multimodal_transport_dp_size,
)


def _sample(prompt, multimodal_inputs):
    return SimpleNamespace(prompt=prompt, multimodal_inputs=multimodal_inputs)


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


def test_group_multimodal_transport_requires_group_atomic_dp_layout():
    validate_group_multimodal_transport_dp_size(1)

    with pytest.raises(ValueError, match="requires actor data-parallel size 1"):
        validate_group_multimodal_transport_dp_size(2)


def test_shared_train_data_pack_entry_rejects_non_atomic_dp_layout():
    from relax.utils.utils import convert_samples_to_train_data

    args = SimpleNamespace(
        context_parallel_size=2,
        debug_train_only=False,
        hybrid=True,
        mm_processor_group_dedup=True,
        multimodal_keys=["pixel_values"],
        pipeline_model_parallel_size=1,
        resource={"actor": [1, 8]},
        tensor_model_parallel_size=2,
    )

    with pytest.raises(ValueError, match="requires actor data-parallel size 1"):
        convert_samples_to_train_data(args, [])


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


def test_group_processor_output_survives_packed_transport_with_shared_tensor_storage():
    torch = _torch()
    pixel_values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    image_grid_thw = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    processor_output = {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    samples = []
    for _ in range(8):
        _prompt_ids, copied_output = copy_group_processor_output([1, 2, 3], processor_output)
        samples.append(SimpleNamespace(group_index=17, multimodal_train_inputs=copied_output))

    packed, source_count, ref_count = pack_group_multimodal_train_inputs(samples, group_size=8)
    unpacked = unpack_group_multimodal_train_inputs(packed, group_size=8)
    source_ptr = unpacked[0]["pixel_values"].untyped_storage().data_ptr()

    del packed, samples, processor_output, pixel_values, image_grid_thw
    gc.collect()

    assert source_count == 1
    assert ref_count == 7
    assert all(item["pixel_values"].shape == (6, 4) for item in unpacked)
    assert all(item["pixel_values"].dtype == torch.float32 for item in unpacked)
    assert all(item["image_grid_thw"].dtype == torch.int64 for item in unpacked)
    assert all(item["pixel_values"].untyped_storage().data_ptr() == source_ptr for item in unpacked)
    assert torch.equal(unpacked[-1]["pixel_values"], torch.arange(24, dtype=torch.float32).reshape(6, 4))
    assert torch.equal(unpacked[-1]["image_grid_thw"], torch.tensor([[1, 2, 3]], dtype=torch.int64))


def test_group_multimodal_transfer_keeps_one_payload_and_reconstructs_aliases():
    torch = _torch()
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    grid = torch.tensor([[1, 2, 2]])
    samples = [
        SimpleNamespace(group_index=9, multimodal_train_inputs={"pixel_values": tensor, "image_grid_thw": grid})
        for _ in range(8)
    ]

    packed, source_count, ref_count = pack_group_multimodal_train_inputs(samples, group_size=8)

    assert isinstance(packed, list)
    assert source_count == 1
    assert ref_count == 7
    assert packed[0][MM_GROUP_ID_KEY] == 9
    assert packed[0][MM_GROUP_OWNER_KEY]
    assert packed[0]["pixel_values"] is tensor
    assert all(row == {MM_GROUP_ID_KEY: 9, MM_GROUP_OWNER_KEY: False} for row in packed[1:])

    unpacked = unpack_group_multimodal_train_inputs(list(reversed(packed)), group_size=8)

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
    unpacked = unpack_group_multimodal_train_inputs(packed, group_size=4)

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


def test_group_multimodal_transfer_supports_transfer_queue_position_selection():
    torch = _torch()
    tensordict = pytest.importorskip("tensordict")
    manager_module = pytest.importorskip("transfer_queue.storage.managers.simple_storage_manager")
    serial_utils = pytest.importorskip("transfer_queue.utils.serial_utils")
    tensor = torch.arange(100_000, dtype=torch.uint8)
    samples = [SimpleNamespace(group_index=9, multimodal_train_inputs={"pixel_values": tensor}) for _ in range(8)]
    packed, source_count, ref_count = pack_group_multimodal_train_inputs(samples, group_size=8)
    rollout_data = tensordict.TensorDict({"multimodal_train_inputs": packed}, batch_size=[8])
    field = rollout_data["multimodal_train_inputs"]

    selected_by_position = {}
    for positions in ([0, 2, 5], [1, 3, 7], [4, 6]):
        selected = manager_module.AsyncSimpleStorageManager._select_by_positions(field, positions)
        selected_by_position.update(zip(positions, selected, strict=True))
    merged = [selected_by_position[position] for position in range(8)]
    unpacked = unpack_group_multimodal_train_inputs(merged, group_size=8)

    assert source_count == 1
    assert ref_count == 7
    assert len({item["pixel_values"].untyped_storage().data_ptr() for item in unpacked}) == 1

    packed_frames = serial_utils.encode(
        {
            "multimodal_train_inputs": manager_module.AsyncSimpleStorageManager._select_by_positions(
                field,
                list(range(8)),
            )
        }
    )
    dense_field = tensordict.TensorDict(
        {"multimodal_train_inputs": [{"pixel_values": tensor} for _ in range(8)]},
        batch_size=[8],
    )["multimodal_train_inputs"]
    dense_frames = serial_utils.encode(
        {
            "multimodal_train_inputs": manager_module.AsyncSimpleStorageManager._select_by_positions(
                dense_field,
                list(range(8)),
            )
        }
    )

    assert sum(len(frame) for frame in packed_frames) * 4 < sum(len(frame) for frame in dense_frames)
