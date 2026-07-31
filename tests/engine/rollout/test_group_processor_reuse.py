# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import pytest


try:
    from relax.engine.rollout import sglang_rollout

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


class _State:
    def __init__(self) -> None:
        self.aborted = False
        self.processor = object()
        self.semaphore = asyncio.Semaphore(1)
        self.group_processor_semaphore = asyncio.Semaphore(1)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        custom_generate_function_path=None,
        group_rm=False,
        mm_processor_group_dedup=True,
        sglang_enable_deterministic_inference=False,
    )


def _samples() -> list[SimpleNamespace]:
    media = {"images": [object()], "videos": [], "audio": []}
    return [
        SimpleNamespace(
            generate_function_path=None,
            index=index,
            multimodal_inputs=media,
            prompt="question",
            session_id=None,
        )
        for index in range(8)
    ]


def test_generate_state_rejects_non_atomic_dp_group_transport():
    args = SimpleNamespace(
        context_parallel_size=2,
        hybrid=True,
        mm_processor_group_dedup=True,
        pipeline_model_parallel_size=1,
        resource={"actor": [1, 8]},
        tensor_model_parallel_size=2,
    )
    state = object.__new__(sglang_rollout.GenerateState)

    with pytest.raises(ValueError, match="requires actor data-parallel size 1"):
        sglang_rollout.GenerateState.__init__(state, args)


def test_transfer_batch_sequence_is_unique_per_partition():
    state = object.__new__(sglang_rollout.GenerateState)
    state.transfer_batch_sequences = {}

    assert state.next_transfer_batch_sequence(7) == 0
    assert state.next_transfer_batch_sequence(7) == 1
    assert state.next_transfer_batch_sequence(8) == 0
    assert state.next_transfer_batch_sequence(7) == 2


async def test_group_processor_runs_once_without_holding_inference_permit(monkeypatch):
    state = _State()
    records = {"processor_calls": 0, "generated": 0, "owner_flags": []}

    async def fake_processor(_state, _args, _prompt, _media):
        records["processor_calls"] += 1
        assert state.group_processor_semaphore.locked()
        assert not state.semaphore.locked()
        return [1, 2, 3], {"pixel_values": object()}, 0.25

    async def fake_encode(_media):
        return {}, 0.0

    async def fake_generate(_args, sample, _sampling_params, evaluation=False):
        records["generated"] += 1
        records["owner_flags"].append((sample.index, sample._group_processor_output[3]))
        del sample._group_processor_output
        del sample._pre_encoded_mm
        del sample._pre_encoded_mm_elapsed
        return sample

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "_run_image_processor", fake_processor)
    monkeypatch.setattr(sglang_rollout, "_encode_multimodal_inputs", fake_encode)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate)
    samples = _samples()

    result = await sglang_rollout.generate_and_rm_group(_args(), samples, {"max_new_tokens": 8})

    assert result == samples
    assert records["processor_calls"] == 1
    assert records["generated"] == 8
    assert sorted(records["owner_flags"]) == [
        (0, True),
        (1, False),
        (2, False),
        (3, False),
        (4, False),
        (5, False),
        (6, False),
        (7, False),
    ]
    assert not state.semaphore.locked()
    assert all(not hasattr(sample, "_group_processor_output") for sample in samples)


async def test_group_processor_cleans_cached_attributes_after_generate_failure(monkeypatch):
    state = _State()
    never_finishes = asyncio.Event()

    async def fake_processor(_state, _args, _prompt, _media):
        return [1, 2, 3], {"pixel_values": object()}, 0.25

    async def fake_encode(_media):
        return {}, 0.0

    async def fake_generate(_args, sample, _sampling_params, evaluation=False):
        if sample.index == 0:
            await asyncio.sleep(0)
            raise RuntimeError("generate failed")
        await never_finishes.wait()
        return sample

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "_run_image_processor", fake_processor)
    monkeypatch.setattr(sglang_rollout, "_encode_multimodal_inputs", fake_encode)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate)
    samples = _samples()

    with pytest.raises(RuntimeError, match="generate failed"):
        await sglang_rollout.generate_and_rm_group(_args(), samples, {"max_new_tokens": 8})

    for sample in samples:
        assert not hasattr(sample, "_group_processor_output")
        assert not hasattr(sample, "_pre_encoded_mm")
        assert not hasattr(sample, "_pre_encoded_mm_elapsed")
    assert not state.semaphore.locked()
