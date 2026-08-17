# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU CP2 parity tests for P3O full-sequence attention."""

from __future__ import annotations

import os
import socket
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.megatron.cp_utils import gdn_cp_slice
from relax.backends.megatron.model import (
    P3O_CP_ATTENTION_OOM_ERROR,
    _install_p3o_full_sequence_attention,
)


class _AttentionRoot(torch.nn.Module):
    def __init__(self, attention: torch.nn.Module):
        super().__init__()
        self.attention = attention


class SelfAttention(torch.nn.Module):
    """Minimal Megatron-shaped attention used to exercise the adapter."""

    def __init__(self, cp_group: object, weight: torch.Tensor):
        super().__init__()
        self.pg_collection = SimpleNamespace(cp=cp_group)
        self.weight = torch.nn.Parameter(weight.clone())
        self.seen_local_cp_sizes: list[int | None] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        key_value_states: torch.Tensor | None = None,
        inference_context: object | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: object | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del (
            attention_mask,
            key_value_states,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            sequence_len_offset,
            inference_params,
        )
        self.seen_local_cp_sizes.append(getattr(packed_seq_params, "local_cp_size", None))
        projected = hidden_states @ self.weight
        cu_seqlens = packed_seq_params.cu_seqlens_q.tolist()
        output = torch.cat(
            [
                torch.cumsum(projected[start:end], dim=0)
                for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)
            ],
            dim=0,
        )
        return output, torch.zeros(projected.shape[-1], dtype=projected.dtype)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _p3o_args(scope: str, **overrides: object) -> Namespace:
    values = {
        "advantage_estimator": "p3o",
        "context_parallel_size": 2,
        "p3o_ess_scope": scope,
    }
    values.update(overrides)
    return Namespace(**values)


def _cp2_parity_worker(rank: int, world_size: int, port: int, scope: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(17)
        total_lengths = [5, 5]
        cp_padded_lengths = [8, 8]
        cu_seqlens = [0, 8, 16]
        canonical_cu_seqlens = [0, 5, 10, 16]
        hidden_size = 3
        real_samples = [torch.randn(length, 1, hidden_size, dtype=torch.float64) for length in total_lengths]
        cp_padded_samples = [
            torch.cat([sample, torch.zeros(padded - len(sample), 1, hidden_size, dtype=torch.float64)])
            for sample, padded in zip(real_samples, cp_padded_lengths, strict=True)
        ]
        full_hidden = torch.cat(cp_padded_samples)
        canonical_hidden = torch.cat([*real_samples, torch.zeros(6, 1, hidden_size, dtype=torch.float64)])
        weight = torch.randn(hidden_size, hidden_size, dtype=torch.float64)

        reference_hidden = canonical_hidden.clone().requires_grad_(True)
        reference_weight = weight.clone().requires_grad_(True)
        reference_projected = reference_hidden @ reference_weight
        reference_output = torch.cat(
            [
                torch.cumsum(reference_projected[start:end], dim=0)
                for start, end in zip(canonical_cu_seqlens[:-1], canonical_cu_seqlens[1:], strict=True)
            ]
        )
        reference_mask = torch.zeros(len(reference_output), 1, 1, dtype=torch.float64)
        reference_mask[: sum(total_lengths)] = 1.0
        (reference_output * reference_mask).square().sum().backward()

        local_hidden = gdn_cp_slice(full_hidden, cu_seqlens, world_size, rank).clone().requires_grad_(True)
        attention = SelfAttention(dist.group.WORLD, weight)
        root = _AttentionRoot(attention)
        assert _install_p3o_full_sequence_attention(_p3o_args(scope), root) == 1

        packed_seq_params = SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32),
            cu_seqlens_kv=torch.tensor(cu_seqlens, dtype=torch.int32),
            local_cp_size=None,
            cp_group=None,
            _relax_total_lengths=total_lengths,
            _relax_attention_pad_multiple=8,
            _relax_cu_seqlens_cpu=cu_seqlens,
        )
        output, bias = attention(local_hidden, None, packed_seq_params=packed_seq_params)
        expected_full_output = torch.zeros_like(full_hidden)
        source_offset = 0
        canonical_offset = 0
        for total_length, padded_length in zip(total_lengths, cp_padded_lengths, strict=True):
            expected_full_output[source_offset : source_offset + total_length] = reference_output.detach()[
                canonical_offset : canonical_offset + total_length
            ]
            source_offset += padded_length
            canonical_offset += total_length
        expected_output = gdn_cp_slice(expected_full_output, cu_seqlens, world_size, rank)
        assert torch.equal(output.detach(), expected_output)
        assert torch.equal(bias, torch.zeros(hidden_size, dtype=torch.float64))
        assert attention.seen_local_cp_sizes == [1]

        full_loss_mask = torch.zeros(len(full_hidden), 1, 1, dtype=torch.float64)
        source_offset = 0
        for total_length, padded_length in zip(total_lengths, cp_padded_lengths, strict=True):
            full_loss_mask[source_offset : source_offset + total_length] = 1.0
            source_offset += padded_length
        local_loss_mask = gdn_cp_slice(full_loss_mask, cu_seqlens, world_size, rank)
        (output * local_loss_mask).square().sum().backward()
        expected_full_input_grad = torch.zeros_like(full_hidden)
        source_offset = 0
        canonical_offset = 0
        for total_length, padded_length in zip(total_lengths, cp_padded_lengths, strict=True):
            expected_full_input_grad[source_offset : source_offset + total_length] = reference_hidden.grad[
                canonical_offset : canonical_offset + total_length
            ]
            source_offset += padded_length
            canonical_offset += total_length
        expected_input_grad = gdn_cp_slice(expected_full_input_grad, cu_seqlens, world_size, rank)
        torch.testing.assert_close(local_hidden.grad, expected_input_grad, atol=1e-10, rtol=1e-10)
        dist.all_reduce(attention.weight.grad, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(attention.weight.grad, reference_weight.grad, atol=1e-10, rtol=1e-10)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scope", ["micro-batch", "step"])
def test_p3o_full_sequence_attention_cpu_cp2_matches_cp1(scope: str) -> None:
    mp.spawn(_cp2_parity_worker, args=(2, _free_port(), scope), nprocs=2, join=True)


def test_p3o_full_sequence_attention_gate_is_p3o_cp_only() -> None:
    attention = SelfAttention(SimpleNamespace(size=lambda: 1, rank=lambda: 0), torch.eye(2))
    root = _AttentionRoot(attention)
    original_forward = attention.forward

    assert _install_p3o_full_sequence_attention(_p3o_args("micro-batch", context_parallel_size=1), root) == 0
    assert attention.forward == original_forward
    assert (
        _install_p3o_full_sequence_attention(
            _p3o_args("micro-batch", advantage_estimator="grpo", context_parallel_size=2), root
        )
        == 0
    )
    assert attention.forward == original_forward
    assert _install_p3o_full_sequence_attention(_p3o_args("micro-batch"), root) == 1
    installed_forward = attention.forward
    assert _install_p3o_full_sequence_attention(_p3o_args("micro-batch"), root) == 0
    assert attention.forward == installed_forward


def test_p3o_full_sequence_attention_oom_refuses_native_cp_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from relax.backends.megatron import model as model_module

    cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    attention = SelfAttention(cp_group, torch.eye(2))
    root = _AttentionRoot(attention)
    assert _install_p3o_full_sequence_attention(_p3o_args("micro-batch"), root) == 1

    def raise_oom(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(model_module, "_p3o_cp_gather_full", raise_oom)
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 8], dtype=torch.int32),
        local_cp_size=None,
        cp_group=None,
    )
    with pytest.raises(RuntimeError, match=P3O_CP_ATTENTION_OOM_ERROR):
        attention(torch.randn(4, 1, 2), None, packed_seq_params=packed_seq_params)
    assert attention.seen_local_cp_sizes == []
    assert attention.pg_collection.cp is cp_group
