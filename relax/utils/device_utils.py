# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import shutil
import subprocess


VISIBLE_DEVICE_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
)


def get_visible_devices_env_name(env: dict[str, str] | None = None) -> str | None:
    env = env or os.environ
    for name in VISIBLE_DEVICE_ENV_VARS:
        if env.get(name):
            return name
    return None


def get_visible_devices(env: dict[str, str] | None = None) -> list[str]:
    env = env or os.environ
    env_name = get_visible_devices_env_name(env)
    if env_name is None:
        return []
    return [item.strip() for item in env[env_name].split(",") if item.strip() != ""]


def to_local_visible_device_index(device_id: int, env: dict[str, str] | None = None) -> int:
    visible = get_visible_devices(env)
    if not visible:
        return device_id

    visible_ints = [int(item) for item in visible]
    if device_id in visible_ints:
        return visible_ints.index(device_id)
    if 0 <= device_id < len(visible_ints):
        return device_id

    env_name = get_visible_devices_env_name(env)
    raise RuntimeError(
        f"GPU id {device_id} is not valid under {env_name}={','.join(visible)}. "
        f"Expected one of {visible_ints} (physical) or 0..{len(visible_ints) - 1} (local)."
    )


def detect_fast_interconnect() -> bool:
    if shutil.which("nvidia-smi"):
        output = subprocess.run(
            ["bash", "-lc", "nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l"],
            capture_output=True,
            text=True,
            check=False,
        )
        return int((output.stdout or "0").strip() or "0") > 0

    if shutil.which("rocm-smi"):
        output = subprocess.run(
            ["bash", "-lc", "rocm-smi --showtopotype --csv 2>/dev/null | grep -Eci 'XGMI|MGMI'"],
            capture_output=True,
            text=True,
            check=False,
        )
        return int((output.stdout or "0").strip() or "0") > 0

    return False
