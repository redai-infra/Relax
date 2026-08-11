# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest


pytest.importorskip("ray", reason="Ray is an optional test dependency")
pytest.importorskip("megatron", reason="Megatron is an optional test dependency")

from relax.utils.utils import process_args  # noqa: E402


def test_actor_fwd_preserves_actor_checkpoint_when_zero_kl_removes_reference() -> None:
    args = Namespace(
        ref_actor_config=None,
        log_probs_max_tokens_per_gpu=4096,
        max_tokens_per_gpu=None,
        only_load_weight=False,
        ref_load=None,
        load="/checkpoints/actor",
        hf_checkpoint="/checkpoints/hf",
    )

    process_args(args, "actor_fwd")

    assert args.load == "/checkpoints/actor"
    assert args.only_load_weight is True


def test_actor_fwd_keeps_explicit_reference_checkpoint_precedence() -> None:
    args = Namespace(
        ref_actor_config=None,
        log_probs_max_tokens_per_gpu=4096,
        max_tokens_per_gpu=None,
        only_load_weight=False,
        ref_load="/checkpoints/reference",
        load="/checkpoints/actor",
        hf_checkpoint="/checkpoints/hf",
    )

    process_args(args, "actor_fwd")

    assert args.load == "/checkpoints/reference"
