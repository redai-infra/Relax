# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# CPU async tests for Deepeyes request-aware multiturn scheduling.

from __future__ import annotations

import asyncio
from argparse import Namespace
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from examples.deepeyes import rollout as deepeyes_rollout
from relax.engine.rollout import sglang_rollout as rollout_mod
from relax.engine.rollout.sglang_rollout import ModelRequestScheduler, RolloutRequestAborted, request_model_aware
from relax.utils.types import Sample


class _FakeEnv:
    def __init__(self) -> None:
        self.turn = 0
        self.current_image = None

    def reset(self):
        return None

    def close(self):
        return None


class _FakeGenState:
    def __init__(self, args):
        self.args = args
        self.aborted = False
        self.opd_manager = None
        self.model_request_scheduler = ModelRequestScheduler(self, capacity=2)

    def dp_rank_context(self):
        @contextmanager
        def _ctx():
            yield 0

        return _ctx()


def _deepeyes_args(*, partial_rollout: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        partial_rollout=partial_rollout,
        mask_offpolicy_in_partial_rollout=False,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
    )


def _generate_and_rm_args(*, custom_generate_function_path: str = "x.custom") -> Namespace:
    return Namespace(
        group_rm=False,
        custom_generate_function_path=custom_generate_function_path,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
    )


def test_deepeyes_sample_status_helpers() -> None:
    sample = Sample(prompt="p", tokens=[1, 2, 3], response="", response_length=0)
    sample.status = None  # type: ignore[assignment]
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded"

    deepeyes_rollout._sync_sample_outputs(
        sample,
        tokenizer=tokenizer,
        response_tokens=[2, 3],
        multimodal_train_inputs_buffer=[],
    )
    assert sample.response == "decoded"
    assert sample.response_length == 2
    assert sample.status is None

    tokenizer.decode.return_value = "done"
    deepeyes_rollout._finalize_sample(sample, tokenizer, [1], [])
    assert sample.status == Sample.Status.COMPLETED


@pytest.mark.asyncio
async def test_deepeyes_abort_before_http_uses_turn_idx_and_no_trace() -> None:
    sample = Sample(prompt="hello")
    sample.tokens = [10, 11]
    sample.metadata = {}

    fake_tokenizer = MagicMock()
    fake_tokenizer.decode.return_value = ""
    fake_state = SimpleNamespace(tokenizer=fake_tokenizer, processor=None)

    async def aborting_request_model(url, payload, *, headers=None):
        raise RolloutRequestAborted("aborted")

    with (
        patch.object(
            deepeyes_rollout,
            "_initialize_resources",
            return_value=(_FakeEnv(), SimpleNamespace(), {"max_turns": 3}, fake_state, "http://x/generate"),
        ),
        patch.object(
            deepeyes_rollout,
            "_prepare_start_state",
            new=AsyncMock(return_value=(None, [], None, None, [])),
        ),
    ):
        with pytest.raises(RolloutRequestAborted):
            await deepeyes_rollout.generate(
                _deepeyes_args(partial_rollout=True),
                sample,
                {"max_new_tokens": 16},
                request_model=aborting_request_model,
            )

    assert sample.metadata["rollout_turns"] == 0
    assert sample.metadata["rollout_stop_reason"] == "rollout_abort"
    assert sample.metadata.get("rollout_traces") == []
    assert sample.status is not Sample.Status.COMPLETED


@pytest.mark.asyncio
async def test_deepeyes_env_phase_allows_short_request_interleave() -> None:
    scheduler = ModelRequestScheduler(SimpleNamespace(aborted=False), capacity=1)
    turn1_done = asyncio.Event()
    env_gate = asyncio.Event()
    short_done = asyncio.Event()
    order: list[str] = []
    call_n = 0

    async def fake_post(url, payload, headers=None):
        nonlocal call_n
        call_n += 1
        label = payload.get("label", f"call-{call_n}")
        order.append(label)
        if label == "long-1":
            turn1_done.set()
        if label == "short":
            short_done.set()
        return {
            "text": "resp",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[0.0, 42]],
            },
        }

    sample = Sample(prompt="hello")
    sample.tokens = [1]
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.metadata = {}

    fake_tokenizer = MagicMock()
    fake_tokenizer.decode.return_value = "decoded"
    fake_state = SimpleNamespace(tokenizer=fake_tokenizer, processor=None)
    request_model = scheduler.request
    inference_calls = 0
    real_run_inference = deepeyes_rollout._run_inference_step

    async def inference_with_labels(*args, **kwargs):
        nonlocal inference_calls
        inference_calls += 1
        label = f"long-{inference_calls}"
        rm = kwargs["request_model"]

        async def labeled(url, payload, *, headers=None):
            payload = dict(payload)
            payload["label"] = label
            return await rm(url, payload, headers=headers)

        kwargs = dict(kwargs)
        kwargs["request_model"] = labeled
        return await real_run_inference(*args, **kwargs)

    async def env_step_mock(*args, **kwargs):
        await env_gate.wait()
        return None, None, None, None, True, {}

    with (
        patch("relax.engine.rollout.sglang_rollout.post", side_effect=fake_post),
        patch.object(
            deepeyes_rollout,
            "_initialize_resources",
            return_value=(_FakeEnv(), SimpleNamespace(), {"max_turns": 2}, fake_state, "http://x/generate"),
        ),
        patch.object(
            deepeyes_rollout,
            "_prepare_start_state",
            new=AsyncMock(return_value=(None, [], None, None, [])),
        ),
        patch.object(deepeyes_rollout, "_run_inference_step", side_effect=inference_with_labels),
        patch.object(deepeyes_rollout, "_process_env_step", new=env_step_mock),
    ):
        long_task = asyncio.create_task(
            deepeyes_rollout.generate(
                _deepeyes_args(),
                sample,
                {"max_new_tokens": 8},
                request_model=request_model,
            )
        )
        await asyncio.wait_for(turn1_done.wait(), timeout=1.0)
        short_task = asyncio.create_task(request_model("http://x", {"label": "short"}))
        await asyncio.wait_for(short_done.wait(), timeout=1.0)
        assert order == ["long-1", "short"]
        env_gate.set()
        await long_task
        await short_task


@pytest.mark.asyncio
async def test_generate_and_rm_legacy_vs_request_aware_dispatch() -> None:
    calls: list[str] = []

    async def legacy_custom(args, sample, sampling_params):
        calls.append("legacy")
        sample.status = Sample.Status.COMPLETED
        sample.reward = 0.0
        return sample

    @request_model_aware
    async def aware_custom(args, sample, sampling_params, *, request_model):
        calls.append("aware")
        await request_model("http://x", {"n": 1})
        sample.status = Sample.Status.COMPLETED
        sample.reward = 0.0
        return sample

    async def fake_post(url, payload, headers=None):
        return {"text": "t", "meta_info": {"finish_reason": {"type": "stop"}}}

    args = _generate_and_rm_args(custom_generate_function_path="mod.legacy_custom")
    with (
        patch.object(rollout_mod, "GenerateState", _FakeGenState),
        patch.object(rollout_mod, "load_function", return_value=legacy_custom),
        patch.object(rollout_mod, "async_rm", new=AsyncMock(return_value=0.0)),
        patch.object(rollout_mod, "post", side_effect=fake_post),
    ):
        out = await rollout_mod.generate_and_rm(args, Sample(prompt="p"), {}, evaluation=False)
        assert out.status == Sample.Status.COMPLETED
        assert calls == ["legacy"]

    calls.clear()
    args.custom_generate_function_path = "mod.aware_custom"
    with (
        patch.object(rollout_mod, "GenerateState", _FakeGenState),
        patch.object(rollout_mod, "load_function", return_value=aware_custom),
        patch.object(rollout_mod, "async_rm", new=AsyncMock(return_value=0.0)),
        patch.object(rollout_mod, "post", side_effect=fake_post),
    ):
        out = await rollout_mod.generate_and_rm(args, Sample(prompt="p"), {}, evaluation=False)
        assert out.status == Sample.Status.COMPLETED
        assert calls == ["aware"]


@pytest.mark.asyncio
async def test_generate_and_rm_bad_marker_signature_raises_before_run() -> None:
    @request_model_aware
    async def bad(args, sample, sampling_params):
        raise AssertionError("should not run")

    with (
        patch.object(rollout_mod, "GenerateState", _FakeGenState),
        patch.object(rollout_mod, "load_function", return_value=bad),
    ):
        with pytest.raises(TypeError, match="request_model"):
            await rollout_mod.generate_and_rm(
                _generate_and_rm_args(),
                Sample(prompt="p"),
                {},
                evaluation=False,
            )


@pytest.mark.asyncio
async def test_generate_and_rm_evaluation_respects_aborted_state() -> None:
    ran = False

    @request_model_aware
    async def aware_custom(args, sample, sampling_params, *, request_model):
        nonlocal ran
        ran = True
        await request_model("http://x", {"n": 1})
        sample.status = Sample.Status.COMPLETED
        sample.reward = 0.0
        return sample

    async def fake_post(url, payload, headers=None):
        return {"text": "t", "meta_info": {"finish_reason": {"type": "stop"}}}

    class AbortedState(_FakeGenState):
        def __init__(self, args):
            super().__init__(args)
            self.aborted = True

    with (
        patch.object(rollout_mod, "GenerateState", AbortedState),
        patch.object(rollout_mod, "load_function", return_value=aware_custom),
        patch.object(rollout_mod, "async_rm", new=AsyncMock(return_value=0.0)),
        patch.object(rollout_mod, "post", side_effect=fake_post),
    ):
        out_train = await rollout_mod.generate_and_rm(
            _generate_and_rm_args(),
            Sample(prompt="p"),
            {},
            evaluation=False,
        )
        assert out_train.status == Sample.Status.ABORTED
        assert ran is False

        out_eval = await rollout_mod.generate_and_rm(
            _generate_and_rm_args(),
            Sample(prompt="p"),
            {},
            evaluation=True,
        )
        assert ran is False
        assert out_eval.status == Sample.Status.ABORTED
