# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for SFT producer component (loop-only, no Ray runtime)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from relax.engine.sft.dataset.streaming import ProcessedSample


def _make_processed(idx: int = 0, n_tokens: int = 8) -> ProcessedSample:
    return ProcessedSample(
        tokens=torch.arange(n_tokens, dtype=torch.long),
        loss_mask=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        total_length=n_tokens,
        multimodal_train_inputs=None,
        source_idx=idx,
    )


def _make_args(global_batch_size=4, max_tokens_per_gpu=128, num_rollout=1):
    return SimpleNamespace(
        global_batch_size=global_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        context_parallel_size=1,
        num_rollout=num_rollout,
        prompt_data="/fake/train.jsonl",
        input_key="messages",
        label_key=None,
        multimodal_keys=None,
        metadata_key="metadata",
        tool_key=None,
        system_prompt=None,
        eval_prompt_data=None,
        eval_size=None,
        sft_prefetch_buffer_size=0,
        sft_prefetch_chunk_size=32,
        sft_prefetch_num_workers=4,
        loss_type="sft",
        tq_config=SimpleNamespace(),
        hf_checkpoint="/fake/model",
        start_rollout_id=0,
        seed=42,
        max_staleness=0,
    )


def _patch_pipeline_dependencies(monkeypatch, n_samples: int = 8):
    fake_processed = [_make_processed(i) for i in range(n_samples)]

    fake_ds = MagicMock()
    fake_ds.__len__ = MagicMock(return_value=len(fake_processed))
    fake_ds.shuffle = MagicMock(return_value=None)
    fake_ds.stop = MagicMock(return_value=None)
    fake_ds.index_manager = SimpleNamespace(current_epoch=0)

    async def _get_batch_async(n):
        return fake_processed[:n], False

    fake_ds.get_batch_async = AsyncMock(side_effect=_get_batch_async)
    fake_ds.get_batch_in_order = MagicMock(side_effect=lambda start, n: fake_processed[start : start + n])

    fake_tok = MagicMock()
    fake_tok.chat_template = "{% generation %}assistant{% endgeneration %}"

    monkeypatch.setattr("relax.components.sft.SFTStreamingDataset", lambda **kw: fake_ds)
    monkeypatch.setattr("relax.components.sft.AutoTokenizer.from_pretrained", lambda *a, **kw: fake_tok)
    monkeypatch.setattr("relax.components.sft.ProcessorPool", MagicMock())
    monkeypatch.setattr("relax.components.sft._resolve_pad_token_ids_from_config", lambda *a, **kw: frozenset())
    monkeypatch.setattr("relax.components.sft.print_first_sample", lambda **kw: None)
    return fake_ds, fake_tok


def test_sft_component_imports_without_ray():
    from relax.components.sft import SFT  # noqa: F401


@pytest.mark.asyncio
async def test_sft_step_pushes_one_batch_to_tq(monkeypatch):
    from relax.components.sft import SFT

    _patch_pipeline_dependencies(monkeypatch)

    fake_client = MagicMock()
    fake_client.async_put = AsyncMock(return_value=None)
    monkeypatch.setattr("relax.components.sft.tq.init", lambda *a, **kw: None)
    monkeypatch.setattr("relax.components.sft.tq.get_client", lambda: fake_client)

    args = _make_args(global_batch_size=4)
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft._healthy = True
    sft.step = 0
    sft.data_system_client = fake_client
    sft._dataset = None
    sft._eval_dataset = None
    sft._eval_indices = None
    sft._train_size = 0
    sft._tokenizer = None
    sft._processor_pool = None
    sft._logger_instance = None
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)

    sft._init_data_pipeline()
    await sft._produce_one_step()
    assert fake_client.async_put.await_count == 1
    args_call, kwargs_call = fake_client.async_put.call_args
    pushed_data = kwargs_call.get("data")
    assert "tokens" in pushed_data
    assert "loss_masks" in pushed_data
    assert "total_lengths" in pushed_data
    assert "response_lengths" in pushed_data
    assert kwargs_call.get("partition_id") == "sft_0"


@pytest.mark.asyncio
async def test_sft_loop_advances_step(monkeypatch):
    from relax.components.sft import SFT

    _patch_pipeline_dependencies(monkeypatch)

    fake_client = MagicMock()
    fake_client.async_put = AsyncMock(return_value=None)
    fake_client.async_get_partition_list = AsyncMock(return_value=[])
    monkeypatch.setattr("relax.components.sft.tq.init", lambda *a, **kw: None)
    monkeypatch.setattr("relax.components.sft.tq.get_client", lambda: fake_client)

    args = _make_args(global_batch_size=2, num_rollout=3)
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft._healthy = True
    sft.step = 0
    sft.data_system_client = fake_client
    sft._dataset = None
    sft._eval_dataset = None
    sft._eval_indices = None
    sft._train_size = 0
    sft._tokenizer = None
    sft._processor_pool = None
    sft._logger_instance = None
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)
    sft._init_data_pipeline()

    for _ in range(3):
        await sft._produce_one_step()
    assert sft.step == 3
    assert fake_client.async_put.await_count == 3
    seen_partitions = [c.kwargs.get("partition_id") for c in fake_client.async_put.call_args_list]
    assert seen_partitions == ["sft_0", "sft_1", "sft_2"]
