# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def megatron_arguments_module(monkeypatch):
    """Load the real Relax Megatron argument resolver with lightweight
    dependencies."""

    backend_package = importlib.import_module("relax.backends.megatron")
    module_name = "relax.backends.megatron.arguments"
    original_module = sys.modules.get(module_name)
    had_package_attribute = hasattr(backend_package, "arguments")
    original_package_attribute = getattr(backend_package, "arguments", None)

    class OptimizerConfig:
        def __init__(self):
            self.initial_loss_scale = 2**32
            self.min_loss_scale = 1.0
            self.use_precision_aware_optimizer = False
            self.store_param_remainders = True

    megatron = ModuleType("megatron")
    megatron.__path__ = []
    megatron_core = ModuleType("megatron.core")
    megatron_core.__path__ = []
    optimizer = ModuleType("megatron.core.optimizer")
    optimizer.OptimizerConfig = OptimizerConfig
    megatron_training = ModuleType("megatron.training")
    megatron_training.__path__ = []
    megatron_training_arguments = ModuleType("megatron.training.arguments")
    megatron_training_arguments.parse_args = lambda **_kwargs: None
    megatron_training_arguments.validate_args = lambda args: args
    megatron_tokenizer = ModuleType("megatron.training.tokenizer")
    megatron_tokenizer.__path__ = []
    megatron_tokenizer_impl = ModuleType("megatron.training.tokenizer.tokenizer")
    megatron_tokenizer_impl._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    transformers = ModuleType("transformers")
    transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: None)

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer", optimizer)
    monkeypatch.setitem(sys.modules, "megatron.training", megatron_training)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", megatron_training_arguments)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", megatron_tokenizer)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", megatron_tokenizer_impl)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        if had_package_attribute:
            backend_package.arguments = original_package_attribute
        else:
            delattr(backend_package, "arguments")
