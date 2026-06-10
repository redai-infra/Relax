# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MTP training forwards must keep labels and loss masks aligned."""

from argparse import Namespace

import pytest


try:
    from relax.backends.megatron.model import _attach_mtp_forward_kwargs
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.megatron.model unavailable: {_exc}", allow_module_level=True)


def _mk_args(enable_mtp_training: bool) -> Namespace:
    return Namespace(enable_mtp_training=enable_mtp_training)


def test_attach_mtp_forward_kwargs_noop_when_disabled():
    forward_kwargs = {"loss_mask": None}
    original = forward_kwargs.copy()

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=False), {}, forward_kwargs)

    assert forward_kwargs == original


def test_attach_mtp_forward_kwargs_preserves_existing_loss_mask():
    tokens = object()
    existing_loss_mask = object()
    full_loss_masks = object()
    batch = {"tokens": tokens, "full_loss_masks": full_loss_masks}
    forward_kwargs = {"loss_mask": existing_loss_mask}

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    assert forward_kwargs["mtp_kwargs"]["mtp_labels"] is tokens
    assert forward_kwargs["loss_mask"] is existing_loss_mask


def test_attach_mtp_forward_kwargs_restores_bridge_unsplit_loss_mask():
    tokens = object()
    full_loss_masks = object()
    batch = {"tokens": tokens, "full_loss_masks": full_loss_masks}
    forward_kwargs = {"loss_mask": None}

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    assert forward_kwargs["mtp_kwargs"]["mtp_labels"] is tokens
    assert forward_kwargs["loss_mask"] is full_loss_masks
