# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for optimizer-step P3O stats synchronization."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.backends.megatron._megatron_stub import stubbed_megatron_modules


with stubbed_megatron_modules(("megatron", "ray", "tensordict")):
    from relax.backends.megatron import cp_utils, p3o_step
    from relax.backends.megatron.p3o_step import synchronize_p3o_stats

from relax.utils.training.p3o_utils import P3OSufficientStats


@pytest.fixture(autouse=True)
def _stub_cp_world_size(monkeypatch):
    monkeypatch.setattr(
        cp_utils,
        "mpu",
        SimpleNamespace(get_context_parallel_world_size=lambda: 1),
    )
    # The target SIF has a real Megatron installation, whereas lightweight
    # developer environments use the module stubs above. Keep these unit tests
    # hermetic in both cases instead of querying an uninitialized PP group.
    monkeypatch.setattr(p3o_step.mpu, "is_pipeline_last_stage", lambda ignore_virtual=False: True)


def _stats(values: tuple[float, float, float]) -> P3OSufficientStats:
    vector = torch.tensor(values, dtype=torch.float64)
    return P3OSufficientStats.from_vector(vector)


def test_p3o_step_single_pipeline_stage_preserves_stats(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    stats = _stats((7.5, 21.25, 4.0))

    synchronized = synchronize_p3o_stats(
        stats,
        torch.zeros((), dtype=torch.float64),
        dp_cp_group=None,
        pp_group=None,
        is_pipeline_last_stage=True,
    )

    torch.testing.assert_close(synchronized.as_vector(), stats.as_vector(), rtol=0.0, atol=0.0)


def test_p3o_step_non_last_stage_receives_pipeline_last_stats(monkeypatch):
    expected = torch.tensor([7.5, 21.25, 4.0, 0.0], dtype=torch.float64)
    pp_group = object()

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def fail_if_reduced(*args, **kwargs):
        raise AssertionError("a non-last PP stage must not reduce token stats over DP x CP")

    def broadcast_from_last(vector, *, group, group_src):
        assert group is pp_group
        assert group_src == 1
        vector.copy_(expected)

    monkeypatch.setattr(torch.distributed, "all_reduce", fail_if_reduced)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast_from_last)

    synchronized = synchronize_p3o_stats(
        P3OSufficientStats.zeros(),
        torch.zeros((), dtype=torch.float64),
        dp_cp_group=None,
        pp_group=pp_group,
        is_pipeline_last_stage=False,
    )

    torch.testing.assert_close(synchronized.as_vector(), expected[:3], rtol=0.0, atol=0.0)


def test_p3o_step_raises_only_after_global_invalid_flag_is_visible(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    dp_cp_group = object()

    def all_reduce(vector, *, op, group):
        vector[3] = 1.0

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    with pytest.raises(ValueError, match="non-finite importance ratio"):
        synchronize_p3o_stats(
            _stats((1.0, 1.0, 1.0)),
            torch.zeros((), dtype=torch.float64),
            dp_cp_group=dp_cp_group,
            pp_group=None,
            is_pipeline_last_stage=True,
        )


@pytest.mark.parametrize("moment_index", range(3), ids=("s1", "s2", "n"))
@pytest.mark.parametrize("nonfinite", [float("inf"), float("nan")], ids=("inf", "nan"))
def test_p3o_step_raises_when_collective_produces_nonfinite_moment(monkeypatch, moment_index, nonfinite):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    dp_cp_group = object()

    def all_reduce(vector, *, op, group):
        assert group is dp_cp_group
        vector[moment_index] = nonfinite

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    with pytest.raises(ValueError, match="unrepresentable squared ratio"):
        synchronize_p3o_stats(
            _stats((1.0, 1.0, 1.0)),
            torch.zeros((), dtype=torch.float64),
            dp_cp_group=dp_cp_group,
            pp_group=None,
            is_pipeline_last_stage=True,
        )


def test_compute_p3o_step_context_plain_text_forward_kwargs(monkeypatch):
    """ESS pre-pass forward_step must use tokens+packed_seq_params for plain
    text."""
    from argparse import Namespace

    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 1, 768)

    def fake_get_batch(iterator, keys, *_args, **_kwargs):
        return {
            "tokens": torch.zeros(4, dtype=torch.long),
            "packed_seq_params": "packed_sentinel",
            "total_lengths": [4],
            "response_lengths": [2],
            "loss_masks": [torch.ones(4)],
            "rollout_log_probs": [torch.zeros(4)],
            "full_loss_masks": torch.ones(4),
            "unconcat_tokens": [torch.zeros(4, dtype=torch.long)],
        }

    def fake_forward_backward(forward_step_func, data_iterator, model, **_kwargs):
        # Call forward_step once to trigger kwarg capture (avoid calling collect callback)
        forward_step_func(data_iterator[0], model[0])
        return None

    # Prevent the lazy `from .loss import get_log_probs_and_entropy` from executing
    # by ensuring the forward_backward func never calls the collect callback
    monkeypatch.setattr(p3o_step, "get_batch", fake_get_batch)
    monkeypatch.setattr(p3o_step, "get_forward_backward_func", lambda: fake_forward_backward)
    monkeypatch.setattr(p3o_step, "synchronize_p3o_stats", lambda *_, **__: _stats((7.5, 21.25, 4.0)))
    monkeypatch.setattr(
        p3o_step,
        "finalize_p3o_step_context",
        lambda s: p3o_step.P3OStepContext(
            normalized_ess=torch.tensor(0.66),
            adaptive_cap=torch.tensor(0.66),
            valid_token_count=torch.tensor(4.0),
            ratio_mean=torch.tensor(1.875),
            ratio_std=torch.tensor(0.5),
            clamp_events=0,
        ),
    )
    monkeypatch.setattr(torch, "no_grad", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_iterator_positions", lambda _: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_rng_state", lambda: __import__("contextlib").nullcontext())

    args = Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        seq_length=512,
        micro_batch_size=1,
        decoder_seq_length=None,
    )
    # loss.py is a lazy import inside compute_p3o_step_context (line 140 of p3o_step.py).
    # It fires after the stubbed_megatron_modules context has already exited, so we must
    # inject a mock for loss before the function is called.
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.loss", MagicMock())
    p3o_step.compute_p3o_step_context(args, [iter([None])], [fake_model], num_microbatches=1)

    assert captured["input_ids"] is not None
    assert str(captured["input_ids"].dtype) == "torch.int64"
    assert captured["packed_seq_params"] == "packed_sentinel"
    assert captured["loss_mask"] is not None


def test_compute_p3o_step_context_matches_training_multimodal_kwarg_gate(monkeypatch):
    """ESS pre-pass must not pass multimodal kwargs that training omits."""
    from argparse import Namespace

    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 1, 768)

    def fake_get_batch(iterator, keys, *_args, **_kwargs):
        return {
            "tokens": torch.zeros(4, dtype=torch.long),
            "packed_seq_params": "packed_sentinel",
            "multimodal_train_inputs": {"pixel_values": torch.ones(1)},
            "total_lengths": [4],
            "response_lengths": [2],
            "loss_masks": [torch.ones(4)],
            "rollout_log_probs": [torch.zeros(4)],
            "full_loss_masks": torch.ones(4),
            "unconcat_tokens": [torch.zeros(4, dtype=torch.long)],
        }

    def fake_forward_backward(forward_step_func, data_iterator, model, **_kwargs):
        forward_step_func(data_iterator[0], model[0])
        return None

    monkeypatch.setattr(p3o_step, "get_batch", fake_get_batch)
    monkeypatch.setattr(p3o_step, "get_forward_backward_func", lambda: fake_forward_backward)
    monkeypatch.setattr(p3o_step, "synchronize_p3o_stats", lambda *_, **__: _stats((7.5, 21.25, 4.0)))
    monkeypatch.setattr(
        p3o_step,
        "finalize_p3o_step_context",
        lambda _: p3o_step.P3OStepContext(
            normalized_ess=torch.tensor(0.66),
            adaptive_cap=torch.tensor(0.66),
            valid_token_count=torch.tensor(4.0),
            ratio_mean=torch.tensor(1.875),
            ratio_std=torch.tensor(0.5),
        ),
    )
    monkeypatch.setattr(torch, "no_grad", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_iterator_positions", lambda _: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_rng_state", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.loss", MagicMock())

    args = Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        is_vl_model=False,
        seq_length=512,
        micro_batch_size=1,
        decoder_seq_length=None,
    )
    p3o_step.compute_p3o_step_context(args, [iter([None])], [fake_model], num_microbatches=1)

    assert "pixel_values" not in captured


def test_compute_p3o_step_context_vl_unsplit_forward_kwargs(monkeypatch):
    """ESS pre-pass forward_step must use unsplit_tokens for VL models."""
    from argparse import Namespace

    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 1, 768)

    def fake_get_batch(iterator, keys, *_args, **_kwargs):
        return {
            "tokens": torch.zeros(4, dtype=torch.long),
            "unsplit_tokens": torch.zeros(8, dtype=torch.long),  # VL path
            "packed_seq_params": "packed_sentinel",
            "total_lengths": [4],
            "response_lengths": [2],
            "loss_masks": [torch.ones(4)],
            "rollout_log_probs": [torch.zeros(4)],
            "full_loss_masks": torch.ones(4),
            "unconcat_tokens": [torch.zeros(4, dtype=torch.long)],
        }

    def fake_forward_backward(forward_step_func, data_iterator, model, **_kwargs):
        output_tensor, _ = forward_step_func(data_iterator[0], model[0])
        return None

    monkeypatch.setattr(p3o_step, "get_batch", fake_get_batch)
    monkeypatch.setattr(p3o_step, "get_forward_backward_func", lambda: fake_forward_backward)
    monkeypatch.setattr(p3o_step, "synchronize_p3o_stats", lambda *_, **__: _stats((7.5, 21.25, 4.0)))
    monkeypatch.setattr(
        p3o_step,
        "finalize_p3o_step_context",
        lambda s: p3o_step.P3OStepContext(
            normalized_ess=torch.tensor(0.66),
            adaptive_cap=torch.tensor(0.66),
            valid_token_count=torch.tensor(4.0),
            ratio_mean=torch.tensor(1.875),
            ratio_std=torch.tensor(0.5),
            clamp_events=0,
        ),
    )
    monkeypatch.setattr(torch, "no_grad", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_iterator_positions", lambda _: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_rng_state", lambda: __import__("contextlib").nullcontext())

    args = Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        is_vl_model=True,
        seq_length=512,
        micro_batch_size=1,
        decoder_seq_length=None,
    )
    # cp_utils.maybe_padded_total_lengths queries mpu for the CP world size; this
    # test is single-process, so report CP=1 instead of a bare MagicMock.
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.loss", MagicMock())
    p3o_step.compute_p3o_step_context(args, [iter([None])], [fake_model], num_microbatches=1)

    # VL path: should use unsplit_tokens, packed_seq_params=None
    assert captured["input_ids"].shape == (8,), "VL path must use unsplit_tokens"
    assert captured["packed_seq_params"] is None, "VL path sets packed_seq_params=None"
    assert captured["loss_mask"] is not None


def test_compute_p3o_step_context_vl_thd_bridge_forward_kwargs(monkeypatch):
    """ESS pre-pass forward_step must use thd bridge path
    (vlm_packed_seq_params, loss_mask=None)."""
    from argparse import Namespace

    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 1, 768)

    def fake_get_batch(iterator, keys, *_args, **_kwargs):
        return {
            "tokens": torch.zeros(4, dtype=torch.long),
            "unsplit_tokens": torch.zeros(8, dtype=torch.long),
            "unsplit_attention_mask": torch.ones(8),
            "vlm_packed_seq_params": "vlm_packed_sentinel",  # thd bridge marker
            "packed_seq_params": "packed_sentinel",
            "total_lengths": [4],
            "response_lengths": [2],
            "loss_masks": [torch.ones(4)],
            "rollout_log_probs": [torch.zeros(4)],
            "full_loss_masks": torch.ones(4),
            "unconcat_tokens": [torch.zeros(4, dtype=torch.long)],
        }

    def fake_forward_backward(forward_step_func, data_iterator, model, **_kwargs):
        output_tensor, _ = forward_step_func(data_iterator[0], model[0])
        return None

    monkeypatch.setattr(p3o_step, "get_batch", fake_get_batch)
    monkeypatch.setattr(p3o_step, "get_forward_backward_func", lambda: fake_forward_backward)
    monkeypatch.setattr(p3o_step, "synchronize_p3o_stats", lambda *_, **__: _stats((7.5, 21.25, 4.0)))
    monkeypatch.setattr(
        p3o_step,
        "finalize_p3o_step_context",
        lambda s: p3o_step.P3OStepContext(
            normalized_ess=torch.tensor(0.66),
            adaptive_cap=torch.tensor(0.66),
            valid_token_count=torch.tensor(4.0),
            ratio_mean=torch.tensor(1.875),
            ratio_std=torch.tensor(0.5),
            clamp_events=0,
        ),
    )
    monkeypatch.setattr(torch, "no_grad", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_iterator_positions", lambda _: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_rng_state", lambda: __import__("contextlib").nullcontext())

    args = Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        is_vl_model=True,
        seq_length=512,
        micro_batch_size=1,
        decoder_seq_length=None,
    )
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.loss", MagicMock())
    p3o_step.compute_p3o_step_context(args, [iter([None])], [fake_model], num_microbatches=1)

    # thd bridge path: unsplit_tokens, vlm_packed_seq_params, unsplit_attention_mask, loss_mask=None
    assert captured["input_ids"].shape == (8,), "thd bridge must use unsplit_tokens"
    assert captured["packed_seq_params"] == "vlm_packed_sentinel", "thd bridge uses vlm_packed_seq_params"
    assert captured["attention_mask"] is not None, "thd bridge requires attention_mask"
    assert captured["loss_mask"] is None, "thd bridge sets loss_mask=None"


def test_compute_p3o_step_context_dynamic_cp_group_switching(monkeypatch):
    """ESS pre-pass forward_step must switch pg_collection.cp for dynamic
    CP."""
    from argparse import Namespace

    captured_pg = []
    orig_cp_group = object()
    dynamic_cp_group = object()

    class FakePGCollection:
        def __init__(self):
            self.cp = orig_cp_group

    class FakeInner:
        def __init__(self):
            self.pg_collection = FakePGCollection()

    class FakeModel:
        def __init__(self):
            self.module = FakeInner()

        def __call__(self, **kwargs):
            captured_pg.append(self.module.pg_collection.cp)
            return torch.zeros(1, 1, 768)

    fake_model = FakeModel()

    batch = {
        "tokens": torch.zeros(4, dtype=torch.long),
        "unsplit_tokens": torch.zeros(8, dtype=torch.long),
        "packed_seq_params": "packed_sentinel",
        "dynamic_cp_size": 2,  # trigger dynamic CP path
        "padded_total_lengths": [8],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.ones(4)],
        "rollout_log_probs": [torch.zeros(4)],
        "full_loss_masks": torch.ones(4),
        "unconcat_tokens": [torch.zeros(4, dtype=torch.long)],
    }

    def fake_get_batch(iterator, keys, *_args, **_kwargs):
        return batch

    def fake_forward_backward(forward_step_func, data_iterator, model, **_kwargs):
        output_tensor, _ = forward_step_func(data_iterator[0], model[0])
        return None

    monkeypatch.setattr(p3o_step.mpu, "get_dynamic_data_context_parallel_groups", lambda group_size: dynamic_cp_group)
    monkeypatch.setattr(p3o_step, "get_batch", fake_get_batch)
    monkeypatch.setattr(p3o_step, "get_forward_backward_func", lambda: fake_forward_backward)
    monkeypatch.setattr(p3o_step, "synchronize_p3o_stats", lambda *_, **__: _stats((7.5, 21.25, 4.0)))
    monkeypatch.setattr(
        p3o_step,
        "finalize_p3o_step_context",
        lambda s: p3o_step.P3OStepContext(
            normalized_ess=torch.tensor(0.66),
            adaptive_cap=torch.tensor(0.66),
            valid_token_count=torch.tensor(4.0),
            ratio_mean=torch.tensor(1.875),
            ratio_std=torch.tensor(0.5),
            clamp_events=0,
        ),
    )
    monkeypatch.setattr(torch, "no_grad", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_iterator_positions", lambda _: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_rng_state", lambda: __import__("contextlib").nullcontext())

    args = Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        is_vl_model=True,
        seq_length=512,
        micro_batch_size=1,
        decoder_seq_length=None,
    )
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.loss", MagicMock())
    p3o_step.compute_p3o_step_context(args, [iter([None])], [fake_model], num_microbatches=1)

    # The forward should have been called with dynamic_cp_group active
    assert len(captured_pg) == 1, "forward_step should call model_chunk once"
    assert captured_pg[0] is dynamic_cp_group, "pg_collection.cp must switch to dynamic group during forward"
    # After forward, it should be restored (verify via the finally block's side effect)
    assert fake_model.module.pg_collection.cp is orig_cp_group, "pg_collection.cp must be restored after forward"
    assert batch["padded_total_lengths"] == [8], "dynamic-CP padding metadata must not be overwritten"


def test_compute_p3o_step_context_dynamic_cp_one_does_not_add_static_padding(monkeypatch):
    """Dynamic CP size one must not inherit padding from the static CP
    group."""
    from argparse import Namespace

    batch = {
        "tokens": torch.zeros(4, dtype=torch.long),
        "unsplit_tokens": torch.zeros(4, dtype=torch.long),
        "packed_seq_params": "packed_sentinel",
        "dynamic_cp_size": 1,
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.ones(4)],
        "rollout_log_probs": [torch.zeros(4)],
        "full_loss_masks": torch.ones(4),
        "unconcat_tokens": [torch.zeros(4, dtype=torch.long)],
    }

    class FakePGCollection:
        cp = object()

    class FakeInner:
        pg_collection = FakePGCollection()

    class FakeModel:
        module = FakeInner()

        def __call__(self, **kwargs):
            return torch.zeros(1, 1, 768)

    def fake_forward_backward(forward_step_func, data_iterator, model, **_kwargs):
        forward_step_func(data_iterator[0], model[0])
        return None

    monkeypatch.setattr(p3o_step, "get_batch", lambda *args, **kwargs: batch)
    monkeypatch.setattr(p3o_step, "get_forward_backward_func", lambda: fake_forward_backward)
    monkeypatch.setattr(p3o_step.mpu, "get_dynamic_data_context_parallel_groups", lambda group_size: object())
    monkeypatch.setattr(p3o_step, "synchronize_p3o_stats", lambda *_, **__: _stats((7.5, 21.25, 4.0)))
    monkeypatch.setattr(
        p3o_step,
        "finalize_p3o_step_context",
        lambda _: p3o_step.P3OStepContext(
            normalized_ess=torch.tensor(0.66),
            adaptive_cap=torch.tensor(0.66),
            valid_token_count=torch.tensor(4.0),
            ratio_mean=torch.tensor(1.875),
            ratio_std=torch.tensor(0.5),
        ),
    )
    monkeypatch.setattr(torch, "no_grad", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_iterator_positions", lambda _: __import__("contextlib").nullcontext())
    monkeypatch.setattr(p3o_step, "preserved_rng_state", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.loss", MagicMock())
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 4)

    args = Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        is_vl_model=True,
        seq_length=512,
        micro_batch_size=1,
        decoder_seq_length=None,
    )
    p3o_step.compute_p3o_step_context(args, [iter([None])], [FakeModel()], num_microbatches=1)

    assert "padded_total_lengths" not in batch
