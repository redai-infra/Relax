# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import gc

import torch
import torch.distributed as dist

from relax.utils import device as device_utils
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def clear_memory(clear_host_memory: bool = False):
    device_utils.synchronize()
    gc.collect()
    device_utils.empty_cache()
    if clear_host_memory:
        if device_utils.is_npu_available:
            torch.npu.host_empty_cache()
        else:
            torch._C._host_emptyCache()


def available_memory():
    dev = device_utils.current_device()
    free, total = device_utils.mem_get_info(dev)
    return {
        "device": str(dev),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(device_utils.memory_allocated(dev)),
        "reserved_GB": _byte_to_gb(device_utils.memory_reserved(dev)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
