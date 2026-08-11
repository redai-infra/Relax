# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

from relax.utils.utils import process_args


def _args(*, load: str, ref_load: str | None) -> Namespace:
    return Namespace(
        load=load,
        ref_load=ref_load,
        ref_actor_config=None,
        max_tokens_per_gpu=1024,
        log_probs_max_tokens_per_gpu=2048,
        only_load_weight=False,
    )


def test_process_args_actor_fwd_preserves_resolved_load_without_ref_load() -> None:
    args = _args(load="/hf-checkpoint", ref_load=None)

    process_args(args, "actor_fwd")

    assert args.load == "/hf-checkpoint"


def test_process_args_actor_fwd_prefers_explicit_ref_load() -> None:
    args = _args(load="/actor-checkpoint", ref_load="/reference-checkpoint")

    process_args(args, "actor_fwd")

    assert args.load == "/reference-checkpoint"


def test_process_args_reference_preserves_resolved_load_without_ref_load() -> None:
    args = _args(load="/hf-checkpoint", ref_load=None)

    process_args(args, "reference")

    assert args.load == "/hf-checkpoint"


def test_process_args_reference_prefers_explicit_ref_load() -> None:
    args = _args(load="/actor-checkpoint", ref_load="/reference-checkpoint")

    process_args(args, "reference")

    assert args.load == "/reference-checkpoint"
