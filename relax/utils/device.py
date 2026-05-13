# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Multi-hardware backend abstraction layer.
#
# Inspired by verl (https://github.com/verl-project/verl) device.py
# and slime (https://github.com/THUDM/slime) plugin architecture.
#
# This module provides a unified device abstraction that allows Relax to run
# on multiple hardware backends (NVIDIA CUDA, Ascend NPU, AMD ROCm, Kunlunxin XPU,
# PPU, etc.) with minimal code changes throughout the framework.
#
# Usage:
#   from relax.utils.device import get_device_name, get_torch_device, ...
#
# The module auto-detects the available accelerator at import time and exposes
# a consistent API regardless of the underlying hardware.

import os
from enum import Enum
from functools import lru_cache
from typing import Optional

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Accelerator type enum
# ---------------------------------------------------------------------------
class AcceleratorType(str, Enum):
    """Supported hardware accelerator types."""

    CUDA = "cuda"  # NVIDIA GPU
    NPU = "npu"  # Ascend NPU (Huawei)
    XPU = "xpu"  # Intel / Kunlunxin XPU
    PPU = "ppu"  # PPU (Enflame / custom)
    ROCM = "rocm"  # AMD ROCm (uses 'cuda' device in PyTorch but HIP backend)
    CPU = "cpu"  # CPU fallback


# ---------------------------------------------------------------------------
# Detection helpers (cached — hardware won't change at runtime)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _detect_accelerator() -> AcceleratorType:
    """Detect the available hardware accelerator.

    Detection order follows specificity: NPU > XPU > PPU > CUDA/ROCm > CPU.
    Environment variable ``RELAX_DEVICE_TYPE`` can override auto-detection.
    """
    # Allow explicit override via environment variable
    override = os.environ.get("RELAX_DEVICE_TYPE", "").lower().strip()
    if override:
        for accel in AcceleratorType:
            if override == accel.value:
                logger.info(f"Device type overridden by RELAX_DEVICE_TYPE={override}")
                return accel
        logger.warning(f"Unknown RELAX_DEVICE_TYPE='{override}', falling back to auto-detection")

    # Ascend NPU
    if _is_npu_available():
        return AcceleratorType.NPU

    # Kunlunxin / Intel XPU
    if _is_xpu_available():
        return AcceleratorType.XPU

    # PPU (Enflame)
    if _is_ppu_available():
        return AcceleratorType.PPU

    # NVIDIA CUDA or AMD ROCm (both expose torch.cuda)
    if torch.cuda.is_available():
        if _is_rocm():
            return AcceleratorType.ROCM
        return AcceleratorType.CUDA

    return AcceleratorType.CPU


def _is_npu_available() -> bool:
    """Check if Ascend NPU is available."""
    try:
        if not hasattr(torch, "npu"):
            return False
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def _is_xpu_available() -> bool:
    """Check if XPU (Intel / Kunlunxin) is available."""
    try:
        if not hasattr(torch, "xpu"):
            return False
        return torch.xpu.is_available()
    except (ImportError, AttributeError):
        return False


def _is_ppu_available() -> bool:
    """Check if PPU is available."""
    try:
        if not hasattr(torch, "ppu"):
            return False
        return torch.ppu.is_available()
    except (ImportError, AttributeError):
        return False


def _is_rocm() -> bool:
    """Check if the current CUDA build is actually AMD ROCm/HIP."""
    return getattr(torch.version, "hip", None) is not None


# ---------------------------------------------------------------------------
# Public API — device info
# ---------------------------------------------------------------------------
def get_accelerator_type() -> AcceleratorType:
    """Return the detected :class:`AcceleratorType`."""
    return _detect_accelerator()


def get_device_name() -> str:
    """Return the PyTorch device type string (``'cuda'``, ``'npu'``, ``'xpu'``,
    etc.).

    For ROCm, returns ``'cuda'`` because PyTorch ROCm uses the CUDA device
    namespace.
    """
    accel = _detect_accelerator()
    if accel == AcceleratorType.ROCM:
        return "cuda"  # ROCm uses torch.cuda namespace
    if accel == AcceleratorType.CPU:
        return "cpu"
    return accel.value


def get_torch_device_module():
    """Return the ``torch.<device>`` module (e.g. ``torch.cuda``,
    ``torch.npu``).

    This is the namespace that provides ``current_device()``,
    ``synchronize()``, ``empty_cache()``, etc.
    """
    name = get_device_name()
    try:
        return getattr(torch, name)
    except AttributeError:
        logger.warning(f"torch.{name} not found, falling back to torch.cuda")
        return torch.cuda


# ---------------------------------------------------------------------------
# Public API — distributed backend
# ---------------------------------------------------------------------------

# Mapping from accelerator type to the default collective communication backend
_DIST_BACKEND_MAP = {
    AcceleratorType.CUDA: "nccl",
    AcceleratorType.ROCM: "nccl",  # ROCm uses RCCL which is NCCL-compatible
    AcceleratorType.NPU: "hccl",
    AcceleratorType.XPU: "xccl",
    AcceleratorType.PPU: "eccl",
    AcceleratorType.CPU: "gloo",
}


def get_dist_backend() -> str:
    """Return the default distributed communication backend name.

    Returns ``'nccl'`` for NVIDIA/AMD, ``'hccl'`` for Ascend NPU, etc.

    Uses :func:`_current_accelerator` so callers on a CPU-only Ray driver/head
    (e.g. argparse defaults) get the cluster's backend rather than ``'gloo'``.
    """
    return _DIST_BACKEND_MAP.get(_current_accelerator(), "nccl")


# ---------------------------------------------------------------------------
# Public API — environment variables
# ---------------------------------------------------------------------------

# Mapping from accelerator type to the visible-devices environment variable
_VISIBLE_DEVICES_ENV_MAP = {
    AcceleratorType.CUDA: "CUDA_VISIBLE_DEVICES",
    AcceleratorType.ROCM: "CUDA_VISIBLE_DEVICES",  # ROCm also uses this (or HIP_VISIBLE_DEVICES)
    AcceleratorType.NPU: "ASCEND_RT_VISIBLE_DEVICES",
    AcceleratorType.XPU: "XPU_VISIBLE_DEVICES",
    AcceleratorType.PPU: "PPU_VISIBLE_DEVICES",
    AcceleratorType.CPU: "",
}


def get_visible_devices_env_var() -> str:
    """Return the environment variable name for controlling visible devices.

    E.g. ``'CUDA_VISIBLE_DEVICES'`` for NVIDIA, ``'ASCEND_RT_VISIBLE_DEVICES'``
    for Ascend NPU.

    Uses :func:`_current_accelerator` so a CPU-only Ray driver/head still gets
    the right env var name to read (e.g. when forwarding it to actors).
    """
    return _VISIBLE_DEVICES_ENV_MAP.get(_current_accelerator(), "CUDA_VISIBLE_DEVICES")


def get_visible_devices() -> Optional[str]:
    """Return the value of the visible-devices environment variable, or
    None."""
    env_var = get_visible_devices_env_var()
    if not env_var:
        return None
    return os.environ.get(env_var)


# ---------------------------------------------------------------------------
# Public API — Ray resource name
# ---------------------------------------------------------------------------

_RAY_RESOURCE_MAP = {
    AcceleratorType.CUDA: "GPU",
    AcceleratorType.ROCM: "GPU",
    AcceleratorType.NPU: "NPU",
    AcceleratorType.XPU: "XPU",
    AcceleratorType.PPU: "PPU",
    AcceleratorType.CPU: "CPU",
}

# Ray resource name → AcceleratorType, used for cluster-based detection.
_RAY_RESOURCE_TO_ACCEL = {
    "NPU": AcceleratorType.NPU,
    "XPU": AcceleratorType.XPU,
    "PPU": AcceleratorType.PPU,
    "GPU": AcceleratorType.CUDA,
}


def _detect_accelerator_from_ray_cluster() -> Optional[AcceleratorType]:
    """Infer the accelerator type from Ray cluster resources.

    Used as a fallback when the local process has no accelerator (e.g. the Ray
    head node).  Queries ``ray.cluster_resources()`` and maps the first
    non-zero accelerator resource back to an :class:`AcceleratorType`.

    Returns ``None`` if Ray is not initialised or the cluster has no
    accelerator resources.
    """
    try:
        import ray

        if not ray.is_initialized():
            return None
        resources = ray.cluster_resources()
        # Priority order matches _detect_accelerator(): NPU > XPU > PPU > GPU.
        for ray_key in ("NPU", "XPU", "PPU", "GPU"):
            if resources.get(ray_key, 0) > 0:
                accel = _RAY_RESOURCE_TO_ACCEL[ray_key]
                logger.info(
                    f"Local process has no accelerator; detected '{ray_key}' "
                    f"from Ray cluster resources — using {accel.value}"
                )
                return accel
        return None
    except Exception as e:
        logger.debug(f"Ray cluster accelerator detection failed: {e}")
        return None


def _current_accelerator() -> AcceleratorType:
    """Detect accelerator with Ray-cluster fallback, for actor-configuration
    values.

    Use this for values that configure REMOTE actors (dist backend, Ray
    resource name, visible-devices env var). On a CPU-only Ray driver/head,
    the local probe returns CPU but the cluster usually has GPUs/NPUs/etc.;
    we query ``ray.cluster_resources()`` to recover the right answer.

    LOCAL operations (set_device, synchronize, current_device, ...) must keep
    using :func:`_detect_accelerator` so they don't pretend the local process
    owns a GPU.

    Resolution order:
    1. Local accelerator if present.
    2. Ray cluster resources if Ray is initialised.
    3. CUDA default — Relax trains on GPUs, and the typical "neither fires"
       case is the driver at parse-args time, before ``ray.init()`` runs.

    Not cached: a call before ``ray.init`` must not poison later calls that
    happen after the cluster is up.
    """
    accel = _detect_accelerator()
    if accel != AcceleratorType.CPU:
        return accel

    try:
        import ray

        ray_initialized = ray.is_initialized()
    except Exception:
        ray_initialized = False

    if ray_initialized:
        cluster_accel = _detect_accelerator_from_ray_cluster()
        if cluster_accel is not None:
            return cluster_accel
        # Ray is up and the cluster is genuinely accelerator-free.
        return AcceleratorType.CPU

    # Ray not initialised yet — common at parse-args time on the driver.
    # Default to CUDA so actor-configuration values stay usable.
    return AcceleratorType.CUDA


def get_ray_accelerator_name() -> str:
    """Return the Ray resource name for the current accelerator.

    E.g. ``'GPU'`` for NVIDIA/AMD, ``'NPU'`` for Ascend.

    When the local process has no accelerator (e.g. a CPU-only Ray head node),
    falls back to querying Ray cluster resources via
    :func:`_current_accelerator` so placement groups are created with the
    correct resource type.
    """
    return _RAY_RESOURCE_MAP.get(_current_accelerator(), "GPU")


# ---------------------------------------------------------------------------
# Public API — device operations (thin wrappers)
# ---------------------------------------------------------------------------
def current_device() -> int:
    """Return the index of the current device."""
    mod = get_torch_device_module()
    return mod.current_device()


def set_device(device) -> None:
    """Set the current device.

    Args:
        device: Device index (int) or device string (e.g. ``'cuda:0'``).
    """
    mod = get_torch_device_module()
    mod.set_device(device)


def device_count() -> int:
    """Return the number of available accelerator devices."""
    mod = get_torch_device_module()
    return mod.device_count()


def synchronize(device=None) -> None:
    """Synchronize the current (or specified) device."""
    accel = _detect_accelerator()
    if accel == AcceleratorType.CPU:
        return  # no-op for CPU
    mod = get_torch_device_module()
    if device is not None:
        mod.synchronize(device)
    else:
        mod.synchronize()


def empty_cache() -> None:
    """Release all unoccupied cached memory."""
    accel = _detect_accelerator()
    if accel == AcceleratorType.CPU:
        return
    mod = get_torch_device_module()
    mod.empty_cache()


def memory_allocated(device=None) -> int:
    """Return the current GPU memory occupied by tensors in bytes."""
    mod = get_torch_device_module()
    if device is not None:
        return mod.memory_allocated(device)
    return mod.memory_allocated()


def memory_reserved(device=None) -> int:
    """Return the current GPU memory managed by the caching allocator in
    bytes."""
    mod = get_torch_device_module()
    if device is not None:
        return mod.memory_reserved(device)
    return mod.memory_reserved()


def mem_get_info(device=None):
    """Return ``(free, total)`` memory in bytes for the given device."""
    mod = get_torch_device_module()
    if device is not None:
        return mod.mem_get_info(device)
    return mod.mem_get_info()


def get_device_properties(device=None):
    """Return device properties for the given device."""
    mod = get_torch_device_module()
    if device is not None:
        return mod.get_device_properties(device)
    return mod.get_device_properties(mod.current_device())


def current_stream(device=None):
    """Return the currently selected stream for the given device."""
    mod = get_torch_device_module()
    if device is not None:
        return mod.current_stream(device)
    return mod.current_stream()


def Stream(device=None, **kwargs):
    """Create a new stream on the given device."""
    mod = get_torch_device_module()
    if device is not None:
        return mod.Stream(device=device, **kwargs)
    return mod.Stream(**kwargs)


def Event(**kwargs):
    """Create a new event."""
    mod = get_torch_device_module()
    return mod.Event(**kwargs)


def stream_context(stream):
    """Return a context manager that sets the given stream as the current
    stream.

    Equivalent to ``torch.cuda.stream(s)`` but dispatches to the correct device
    backend (e.g. ``torch.npu.stream(s)`` on Ascend NPU).
    """
    mod = get_torch_device_module()
    return mod.stream(stream)


def is_initialized() -> bool:
    """Return True if the device backend has been initialized.

    Equivalent to ``torch.cuda.is_initialized()`` but dispatches to the correct
    device backend.
    """
    mod = get_torch_device_module()
    if hasattr(mod, "is_initialized"):
        return mod.is_initialized()
    # Fallback: if the backend doesn't expose is_initialized, check if
    # any device is available (conservative — assumes initialized if available).
    return is_available()


# ---------------------------------------------------------------------------
# Public API — device string helpers
# ---------------------------------------------------------------------------
def make_device_string(index: Optional[int] = None) -> str:
    """Build a device string like ``'cuda:0'`` or ``'npu:2'``.

    Args:
        index: Device index. If None, uses :func:`current_device`.
    """
    name = get_device_name()
    if name == "cpu":
        return "cpu"
    if index is None:
        index = current_device()
    return f"{name}:{index}"


def make_current_torch_device() -> torch.device:
    """Return a ``torch.device`` for the current accelerator and device
    index."""
    return torch.device(make_device_string())


# ---------------------------------------------------------------------------
# Public API — NUMA affinity
# ---------------------------------------------------------------------------
def set_numa_affinity(local_rank: int) -> None:
    """Set NUMA affinity for the given local rank.

    On NVIDIA GPUs, uses pynvml. On other backends, this is a no-op with a
    warning.
    """
    accel = _detect_accelerator()
    if accel in (AcceleratorType.CUDA,):
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            pynvml.nvmlDeviceSetCpuAffinity(handle)
            logger.info(f"Set NUMA affinity for GPU {local_rank}")
            pynvml.nvmlShutdown()
        except ImportError:
            logger.info("pynvml not available, skipping NUMA affinity setup")
        except Exception as e:
            logger.info(f"Failed to set NUMA affinity: {e}")
    elif accel == AcceleratorType.ROCM:
        logger.info("ROCm/HIP environment detected, skipping NUMA affinity setup")
    elif accel == AcceleratorType.NPU:
        logger.info("Ascend NPU environment, skipping NUMA affinity setup (not yet supported)")
    else:
        logger.info(f"NUMA affinity not supported for {accel.value}, skipping")


# ---------------------------------------------------------------------------
# Public API — expandable segments (CUDA-specific, no-op on others)
# ---------------------------------------------------------------------------
def set_expandable_segments(enable: bool) -> None:
    """Configure CUDA memory allocator expandable segments.

    Only effective on NVIDIA CUDA. No-op on other backends.
    """
    if _detect_accelerator() == AcceleratorType.CUDA:
        try:
            torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
        except Exception as e:
            logger.warning(f"Failed to set expandable_segments: {e}")


# ---------------------------------------------------------------------------
# Public API — availability check
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """Return True if any accelerator device is available (not CPU-only)."""
    return _detect_accelerator() != AcceleratorType.CPU


# ---------------------------------------------------------------------------
# Convenience: boolean flags (for backward compatibility / quick checks)
# ---------------------------------------------------------------------------
is_cuda_available: bool = torch.cuda.is_available()
is_npu_available: bool = _is_npu_available()
is_xpu_available: bool = _is_xpu_available()
is_ppu_available: bool = _is_ppu_available()
is_rocm: bool = _is_rocm()
