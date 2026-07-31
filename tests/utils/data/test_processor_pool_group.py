# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from relax.utils.data.group_processor import copy_group_processor_output, get_reusable_group_processor_input


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
