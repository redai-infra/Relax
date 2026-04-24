# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import torch.nn as nn

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _install_peft_stub_modules() -> None:
    canonical_name = "megatron.bridge.peft.canonical_lora"
    if canonical_name not in sys.modules:
        canonical_mod = types.ModuleType(canonical_name)

        class ModuleDict(nn.ModuleDict):
            def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
                state = {}
                for key, layer in self.items():
                    if hasattr(layer, "sharded_state_dict"):
                        state.update(layer.sharded_state_dict(f"{prefix}{key}.", sharded_offsets, metadata))
                return state

        canonical_mod.ModuleDict = ModuleDict
        sys.modules[canonical_name] = canonical_mod

    lora_name = "megatron.bridge.peft.lora"
    if lora_name not in sys.modules:
        lora_mod = types.ModuleType(lora_name)

        class LoRAMerge:
            def merge(self, *args, **kwargs):
                raise RuntimeError(
                    "Megatron-Bridge PEFT merge requires transformer_engine, which is unavailable in this runtime."
                )

        lora_mod.LoRAMerge = LoRAMerge
        sys.modules[lora_name] = lora_mod


def _raise_modelopt_unavailable(feature: str) -> None:
    raise RuntimeError(f"{feature} requires modelopt, which is unavailable in this runtime.")


def _install_modelopt_stub_modules() -> None:
    modelopt_name = "modelopt"
    if modelopt_name not in sys.modules:
        sys.modules[modelopt_name] = types.ModuleType(modelopt_name)

    torch_name = "modelopt.torch"
    if torch_name not in sys.modules:
        torch_mod = types.ModuleType(torch_name)
        sys.modules[torch_name] = torch_mod
        sys.modules[modelopt_name].torch = torch_mod

    distill_name = "modelopt.torch.distill"
    if distill_name not in sys.modules:
        distill_mod = types.ModuleType(distill_name)

        class DistillationModel(nn.Module):
            pass

        def convert(*args, **kwargs):
            _raise_modelopt_unavailable("Megatron-Bridge distillation")

        distill_mod.DistillationModel = DistillationModel
        distill_mod.convert = convert
        sys.modules[distill_name] = distill_mod
        sys.modules[torch_name].distill = distill_mod

    plugins_name = "modelopt.torch.distill.plugins"
    if plugins_name not in sys.modules:
        plugins_mod = types.ModuleType(plugins_name)
        sys.modules[plugins_name] = plugins_mod
        sys.modules[distill_name].plugins = plugins_mod

    megatron_name = "modelopt.torch.distill.plugins.megatron"
    if megatron_name not in sys.modules:
        megatron_mod = types.ModuleType(megatron_name)

        @dataclass
        class DistillationConfig:
            pass

        def setup_distillation_config(*args, **kwargs):
            _raise_modelopt_unavailable("Megatron-Bridge distillation")

        def adjust_distillation_model_for_mcore(*args, **kwargs):
            _raise_modelopt_unavailable("Megatron-Bridge distillation")

        megatron_mod.DistillationConfig = DistillationConfig
        megatron_mod.setup_distillation_config = setup_distillation_config
        megatron_mod.adjust_distillation_model_for_mcore = adjust_distillation_model_for_mcore
        sys.modules[megatron_name] = megatron_mod
        sys.modules[plugins_name].megatron = megatron_mod


def ensure_megatron_bridge_importable() -> None:
    try:
        import transformer_engine.pytorch  # noqa: F401

    except Exception:
        logger.warning(
            "transformer_engine is unavailable; install Megatron-Bridge PEFT compatibility shims for bridge import."
        )
        _install_peft_stub_modules()

    try:
        import modelopt.torch.distill  # noqa: F401

        return
    except Exception:
        logger.warning(
            "modelopt is unavailable; install Megatron-Bridge distillation compatibility shims for bridge import."
        )
        _install_modelopt_stub_modules()
