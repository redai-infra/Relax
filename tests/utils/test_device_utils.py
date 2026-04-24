# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.device_utils import get_visible_devices, get_visible_devices_env_name, to_local_visible_device_index


def test_prefers_cuda_visible_devices():
    env = {
        "CUDA_VISIBLE_DEVICES": "4,5",
        "ROCR_VISIBLE_DEVICES": "1,2",
    }

    assert get_visible_devices_env_name(env) == "CUDA_VISIBLE_DEVICES"
    assert get_visible_devices(env) == ["4", "5"]


def test_falls_back_to_rocr_visible_devices():
    env = {
        "ROCR_VISIBLE_DEVICES": "6,7",
    }

    assert get_visible_devices_env_name(env) == "ROCR_VISIBLE_DEVICES"
    assert get_visible_devices(env) == ["6", "7"]


def test_maps_physical_id_to_local_index():
    env = {
        "ROCR_VISIBLE_DEVICES": "4,6,7",
    }

    assert to_local_visible_device_index(6, env) == 1


def test_accepts_local_index_when_already_remapped():
    env = {
        "HIP_VISIBLE_DEVICES": "4,6,7",
    }

    assert to_local_visible_device_index(2, env) == 2
