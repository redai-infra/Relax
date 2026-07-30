# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any

import ray
import torch

from relax.utils import device as device_utils
from relax.utils.weight_handoff import validate_export_response

from ..sglang import MultiprocessingSerializer, monkey_patch_torch_reductions
from .bridge_converter import BridgeConverter


class UpdateWeightFromSGLang:
    """Restore Megatron shards from partial same-GPU SGLang CUDA IPC
    exports."""

    def __init__(self, converter: BridgeConverter) -> None:
        self.converter = converter
        self.names = converter.get_required_hf_names()
        self.task_groups = converter.get_hf_to_megatron_task_groups(int(converter._args.update_weight_buffer_size))
        self.last_source_bytes = 0

    @torch.no_grad()
    def update_weights(self, ipc_engine: Any, expected_version: int | str) -> int:
        if ipc_engine is None:
            raise RuntimeError("No colocated SGLang engine is paired with this Megatron rank")

        monkey_patch_torch_reductions()
        converted_count = 0
        source_bytes = 0
        for tasks, names in self.task_groups:
            response = ray.get(ipc_engine.export_weights_to_tensor.remote(names=names, load_format="hf"))
            metadata, serialized = validate_export_response(response, names, expected_version)
            named_tensors = MultiprocessingSerializer.deserialize(serialized)
            if [name for name, _ in named_tensors] != names:
                raise RuntimeError("SGLang exported tensor names do not match the requested Bridge weights")

            state = {}
            for (name, tensor), item in zip(named_tensors, metadata, strict=True):
                expected_shape = tuple(item.get("shape", ()))
                expected_dtype = item.get("dtype")
                if tuple(tensor.shape) != expected_shape:
                    raise RuntimeError(
                        f"SGLang export shape mismatch for {name}: metadata={expected_shape}, "
                        f"tensor={tuple(tensor.shape)}"
                    )
                if str(tensor.dtype).removeprefix("torch.") != expected_dtype:
                    raise RuntimeError(
                        f"SGLang export dtype mismatch for {name}: metadata={expected_dtype}, tensor={tensor.dtype}"
                    )
                if not tensor.is_cuda:
                    raise RuntimeError(f"SGLang export for {name} is not a CUDA IPC tensor")
                state[name] = tensor
                source_bytes += tensor.numel() * tensor.element_size()

            converted_count += self.converter.load_hf_views(state, tasks=tasks)
            # The sender may recycle this task group's temporary IPC storage
            # once Bridge has consumed all mappings.
            device_utils.synchronize()
            del state, named_tensors, serialized, response

        # All consumers must finish before the source engine can release its weight region.
        device_utils.synchronize()
        self.last_source_bytes = source_bytes
        return converted_count
