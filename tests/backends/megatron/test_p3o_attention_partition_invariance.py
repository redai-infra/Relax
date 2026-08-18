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
    _p3o_full_sequence_embedding,
    _p3o_full_sequence_post_process,
)


class _LayerRoot(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.layer = layer


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


class TransformerLayer(torch.nn.Module):
    """Megatron-shaped layer with a token-shape-sensitive post-attention
    MLP."""

    def __init__(self, cp_group: object, attention_weight: torch.Tensor, mlp_weight: torch.Tensor):
        super().__init__()
        self.self_attention = SelfAttention(cp_group, attention_weight)
        self.mlp_weight = torch.nn.Parameter(mlp_weight.clone())
        self.seen_token_counts: list[int] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        inference_context: object | None = None,
        packed_seq_params: object | None = None,
        sequence_len_offset: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        *,
        inference_params: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del context_mask, padding_mask
        attention_output, _ = self.self_attention(
            hidden_states,
            attention_mask,
            None,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            packed_seq_params,
            sequence_len_offset,
            inference_params=inference_params,
        )
        token_count = attention_output.shape[0]
        self.seen_token_counts.append(token_count)
        # Real fused MLP/residual kernels are numerically shape-sensitive. Make
        # that boundary explicit so an attention-only adapter cannot satisfy
        # this whole-layer partition-invariance test accidentally.
        output = (attention_output @ self.mlp_weight) * (1.0 + token_count / 100.0)
        return output, context


class _TokenShapePostProcess(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, *, returns_tuple: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.returns_tuple = returns_tuple
        self.seen_token_counts: list[int] = []

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        del args, kwargs
        self.seen_token_counts.append(hidden_states.shape[0])
        output = (hidden_states @ self.weight) * (1.0 + hidden_states.shape[0] / 100.0)
        return (output, None) if self.returns_tuple else output


class _TokenEmbedding(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding.from_pretrained(weight.clone(), freeze=False)
        self.seen_token_counts: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        tokentype_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del position_ids
        assert tokentype_ids is None
        self.seen_token_counts.append(input_ids.shape[1])
        return self.word_embeddings(input_ids).transpose(0, 1).contiguous()


class _PostProcessRoot(torch.nn.Module):
    def __init__(
        self,
        layer: TransformerLayer,
        final_layernorm: torch.nn.Module,
        output_layer: torch.nn.Module,
        embedding: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.embedding = embedding
        self.decoder = torch.nn.Module()
        self.decoder.final_layernorm = final_layernorm
        self.output_layer = output_layer


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
        attention_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float64)
        mlp_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float64)

        reference_hidden = canonical_hidden.clone().requires_grad_(True)
        reference_layer = TransformerLayer(
            SimpleNamespace(size=lambda: 1, rank=lambda: 0), attention_weight, mlp_weight
        )
        reference_packed_seq_params = SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor(canonical_cu_seqlens, dtype=torch.int32),
            cu_seqlens_kv=torch.tensor(canonical_cu_seqlens, dtype=torch.int32),
            local_cp_size=1,
            cp_group=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
        )
        reference_output, _ = reference_layer(
            reference_hidden,
            packed_seq_params=reference_packed_seq_params,
        )
        reference_mask = torch.zeros(len(reference_output), 1, 1, dtype=torch.float64)
        reference_mask[: sum(total_lengths)] = 1.0
        (reference_output * reference_mask).square().sum().backward()

        local_hidden = gdn_cp_slice(full_hidden, cu_seqlens, world_size, rank).clone().requires_grad_(True)
        layer = TransformerLayer(dist.group.WORLD, attention_weight, mlp_weight)
        root = _LayerRoot(layer)
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
        output, context = layer(local_hidden, packed_seq_params=packed_seq_params)
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
        assert context is None
        assert layer.self_attention.seen_local_cp_sizes == [1]
        assert layer.seen_token_counts == [len(canonical_hidden)]

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
        for parameter, reference_parameter in zip(layer.parameters(), reference_layer.parameters(), strict=True):
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=1e-10, rtol=1e-10)

        embedding_weight = torch.randn(17, hidden_size, dtype=torch.float64)
        real_token_ids = [
            torch.tensor([1, 2, 3, 2, 1], dtype=torch.long),
            torch.tensor([4, 5, 4, 6, 4], dtype=torch.long),
        ]
        source_token_ids = torch.cat(
            [
                torch.cat([tokens, torch.zeros(padded - len(tokens), dtype=torch.long)])
                for tokens, padded in zip(real_token_ids, cp_padded_lengths, strict=True)
            ]
        )
        canonical_token_ids = torch.cat([*real_token_ids, torch.zeros(6, dtype=torch.long)]).unsqueeze(0)
        reference_embedding = _TokenEmbedding(embedding_weight)
        reference_embedding_output = reference_embedding(canonical_token_ids, None)
        (reference_embedding_output * reference_mask).square().sum().backward()

        embedding = _TokenEmbedding(embedding_weight)
        embedding_root = _PostProcessRoot(
            layer,
            torch.nn.Identity(),
            torch.nn.Identity(),
            embedding,
        )
        local_token_ids = gdn_cp_slice(source_token_ids, cu_seqlens, world_size, rank).unsqueeze(0)
        with _p3o_full_sequence_embedding(_p3o_args(scope), embedding_root, packed_seq_params):
            local_embedding_output = embedding(local_token_ids, None)
        assert embedding.seen_token_counts == [len(canonical_hidden)]
        (local_embedding_output * local_loss_mask).square().sum().backward()
        dist.all_reduce(embedding.word_embeddings.weight.grad, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(
            embedding.word_embeddings.weight.grad,
            reference_embedding.word_embeddings.weight.grad,
            atol=1e-10,
            rtol=1e-10,
        )

        norm_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float64)
        head_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float64)
        reference_norm = _TokenShapePostProcess(norm_weight)
        reference_head = _TokenShapePostProcess(head_weight, returns_tuple=True)
        reference_post_input = reference_output.detach().clone().requires_grad_(True)
        reference_post_output, _ = reference_head(reference_norm(reference_post_input))
        (reference_post_output * reference_mask).square().sum().backward()

        norm = _TokenShapePostProcess(norm_weight)
        head = _TokenShapePostProcess(head_weight, returns_tuple=True)
        post_root = _PostProcessRoot(layer, norm, head)
        local_post_input = output.detach().clone().requires_grad_(True)
        original_norm_forward = norm.forward
        original_head_forward = head.forward
        with _p3o_full_sequence_post_process(_p3o_args(scope), post_root, packed_seq_params):
            local_post_output, _ = head(norm(local_post_input))
        assert norm.forward == original_norm_forward
        assert head.forward == original_head_forward
        assert packed_seq_params._relax_p3o_canonical_full_logits.dtype == torch.float32
        assert packed_seq_params._relax_p3o_canonical_full_logits.shape[:2] == (1, len(canonical_hidden))
        assert norm.seen_token_counts == [len(canonical_hidden)]
        assert head.seen_token_counts == [len(canonical_hidden)]
        (local_post_output * local_loss_mask).square().sum().backward()

        expected_full_post_input_grad = torch.zeros_like(full_hidden)
        source_offset = 0
        canonical_offset = 0
        for total_length, padded_length in zip(total_lengths, cp_padded_lengths, strict=True):
            expected_full_post_input_grad[source_offset : source_offset + total_length] = reference_post_input.grad[
                canonical_offset : canonical_offset + total_length
            ]
            source_offset += padded_length
            canonical_offset += total_length
        torch.testing.assert_close(
            local_post_input.grad,
            gdn_cp_slice(expected_full_post_input_grad, cu_seqlens, world_size, rank),
            atol=1e-10,
            rtol=1e-10,
        )
        for parameter, reference_parameter in (
            (norm.weight, reference_norm.weight),
            (head.weight, reference_head.weight),
        ):
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=1e-10, rtol=1e-10)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scope", ["micro-batch", "step"])
def test_p3o_full_sequence_attention_cpu_cp2_matches_cp1(scope: str) -> None:
    mp.spawn(_cp2_parity_worker, args=(2, _free_port(), scope), nprocs=2, join=True)


def test_p3o_full_sequence_attention_gate_is_p3o_cp_only() -> None:
    layer = TransformerLayer(
        SimpleNamespace(size=lambda: 1, rank=lambda: 0),
        torch.eye(2),
        torch.eye(2),
    )
    root = _LayerRoot(layer)
    original_forward = layer.forward

    assert _install_p3o_full_sequence_attention(_p3o_args("micro-batch", context_parallel_size=1), root) == 0
    assert layer.forward == original_forward
    assert (
        _install_p3o_full_sequence_attention(
            _p3o_args("micro-batch", advantage_estimator="grpo", context_parallel_size=2), root
        )
        == 0
    )
    assert layer.forward == original_forward
    assert _install_p3o_full_sequence_attention(_p3o_args("micro-batch"), root) == 1
    installed_forward = layer.forward
    assert _install_p3o_full_sequence_attention(_p3o_args("micro-batch"), root) == 0
    assert layer.forward == installed_forward


def test_p3o_full_sequence_attention_oom_refuses_native_cp_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from relax.backends.megatron import model as model_module

    cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    layer = TransformerLayer(cp_group, torch.eye(2), torch.eye(2))
    leaked_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)

    def raise_oom_after_group_switch(*args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
        del args, kwargs
        layer.self_attention.pg_collection.cp = leaked_group
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    layer.forward = raise_oom_after_group_switch
    root = _LayerRoot(layer)
    assert _install_p3o_full_sequence_attention(_p3o_args("micro-batch"), root) == 1

    monkeypatch.setattr(model_module, "_p3o_cp_gather_full", lambda hidden, *args: torch.cat([hidden, hidden]))
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 8], dtype=torch.int32),
        local_cp_size=None,
        cp_group=None,
        _relax_total_lengths=[5],
        _relax_attention_pad_multiple=8,
        _relax_cu_seqlens_cpu=[0, 8],
    )
    with pytest.raises(RuntimeError, match=P3O_CP_ATTENTION_OOM_ERROR):
        layer(torch.randn(4, 1, 2), packed_seq_params=packed_seq_params)
    assert layer.self_attention.pg_collection.cp is cp_group
