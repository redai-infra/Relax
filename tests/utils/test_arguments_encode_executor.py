# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import importlib
import sys
import types

import pytest


@pytest.fixture(scope="module")
def parser():
    module_names = ("relax.backends.sglang.arguments", "relax.utils.arguments")
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    sglang_arguments = types.ModuleType("relax.backends.sglang.arguments")

    def no_op(*args, **kwargs):
        return None

    sglang_arguments.sglang_parse_args = no_op
    sglang_arguments.validate_args = no_op
    sys.modules["relax.backends.sglang.arguments"] = sglang_arguments
    sys.modules.pop("relax.utils.arguments", None)

    arguments = importlib.import_module("relax.utils.arguments")
    parser = argparse.ArgumentParser()
    arguments.get_slime_extra_args_provider()(parser)
    yield parser

    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_encode_max_workers_cli_defaults_to_none(parser):
    args = parser.parse_args([])

    assert args.encode_max_workers is None
    assert args.mm_processor_pool_size == 0


@pytest.mark.parametrize("value", ["1", "8", "32", "64"])
def test_encode_max_workers_cli_accepts_positive_integer(parser, value):
    args = parser.parse_args(["--encode-max-workers", value])

    assert args.encode_max_workers == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_encode_max_workers_cli_rejects_invalid_value(parser, value):
    with pytest.raises(SystemExit):
        parser.parse_args(["--encode-max-workers", value])
