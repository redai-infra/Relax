# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch


pytest.importorskip("megatron.core")

from megatron.core.transformer.enums import AttnBackend  # noqa: E402

from relax.backends.megatron import model  # noqa: E402


@pytest.mark.parametrize(
    ("cp_rank", "expected_rows"),
    [
        (0, [0, 1, 6, 7]),
        (1, [2, 3, 4, 5]),
    ],
)
def test_rloo_native_cp_attention_mask_selects_zigzag_query_rows(monkeypatch, cp_rank, expected_rows):
    monkeypatch.setattr(model.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(model.mpu, "get_context_parallel_rank", lambda: cp_rank)
    args = Namespace(
        advantage_estimator="rloo",
        qkv_format="bshd",
        attention_backend=AttnBackend.local,
    )
    batch = {"tokens": torch.zeros((1, 4), dtype=torch.long)}

    monkeypatch.setattr(model.torch, "triu", lambda *_args, **_kwargs: pytest.fail("global mask allocated"))
    actual = model._get_rloo_native_cp_attention_mask(args, batch)

    query_positions = torch.tensor(expected_rows)
    key_positions = torch.arange(8)
    expected = (key_positions.unsqueeze(0) > query_positions.unsqueeze(1)).reshape(1, 1, 4, 8)
    assert actual.shape == (1, 1, 4, 8)
    assert actual.numel() == 4 * 8
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("advantage_estimator", "cp_size", "qkv_format", "attention_backend"),
    [
        ("grpo", 2, "bshd", AttnBackend.local),
        ("rloo", 1, "bshd", AttnBackend.local),
        ("rloo", 2, "thd", AttnBackend.local),
        ("rloo", 2, "bshd", AttnBackend.flash),
    ],
)
def test_rloo_native_cp_attention_mask_leaves_other_paths_unchanged(
    monkeypatch,
    advantage_estimator,
    cp_size,
    qkv_format,
    attention_backend,
):
    monkeypatch.setattr(model.mpu, "get_context_parallel_world_size", lambda: cp_size)
    args = Namespace(
        advantage_estimator=advantage_estimator,
        qkv_format=qkv_format,
        attention_backend=attention_backend,
    )

    assert model._get_rloo_native_cp_attention_mask(args, {"tokens": torch.zeros((1, 4))}) is None


def test_rloo_native_cp_attention_mask_rejects_odd_local_sequence(monkeypatch):
    monkeypatch.setattr(model.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(model.mpu, "get_context_parallel_rank", lambda: 0)
    args = Namespace(
        advantage_estimator="rloo",
        qkv_format="bshd",
        attention_backend=AttnBackend.local,
    )

    with pytest.raises(ValueError, match="even local sequence length"):
        model._get_rloo_native_cp_attention_mask(args, {"tokens": torch.zeros((1, 3))})
