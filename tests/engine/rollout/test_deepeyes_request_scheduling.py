# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
from argparse import Namespace
from typing import Any

import pytest

from examples.deepeyes import rollout as deepeyes_rollout
from relax.engine.rollout import sglang_rollout
from relax.utils.types import Sample


_TIMEOUT = 2.0


class _Tokenizer:
    bos_token_id = None

    def encode(self, text: Any, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [1]

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(str(token) for token in tokens)


def _make_args() -> Namespace:
    return Namespace(
        hf_checkpoint="unused-task20",
        mm_processor_pool_size=0,
        sglang_server_concurrency=1,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=8,
        rollout_max_context_len=None,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        sglang_dp_size=1,
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
        group_rm=True,
        custom_generate_function_path="examples.deepeyes.rollout.generate",
        use_rollout_routing_replay=False,
        num_layers=1,
        moe_router_topk=1,
    )


async def _acquire_once(state: sglang_rollout.GenerateState) -> None:
    async with state.request_permit():
        pass


async def test_deepeyes_queued_abort_finalizes_and_resumes_partial_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(sglang_rollout.opd, "is_opd_enabled", lambda args: False)
    sglang_rollout.GenerateState.clear_instances()

    args = _make_args()
    state = sglang_rollout.GenerateState(args)
    sample = Sample(prompt="task20")
    env_entered = asyncio.Event()
    release_env = asyncio.Event()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()
    second_inference_attempted = asyncio.Event()
    envs: list[Any] = []
    request_input_ids: list[list[int]] = []
    inference_attempts = 0
    finalize_count = 0

    class FakeEnv:
        def __init__(self, run_index: int) -> None:
            self.run_index = run_index
            self.turn = 0
            self.current_image = "initial-image"
            self.turn_seen: int | None = None
            self.image_seen: str | None = None
            self.closed = False

        def reset(self) -> None:
            return None

        async def run_step(self) -> tuple[Any, ...]:
            self.turn_seen = self.turn
            self.image_seen = self.current_image
            if self.run_index == 0:
                self.current_image = "image-after-turn-0"
                env_entered.set()
                await release_env.wait()
                self.turn += 1
                return [20], None, None, None, False, {}

            self.turn += 1
            return None, None, None, None, True, {}

        def close(self) -> None:
            self.closed = True

    def fake_initialize(current_args: Namespace, current_sample: Sample) -> tuple[Any, ...]:
        del current_args, current_sample
        env = FakeEnv(len(envs))
        envs.append(env)
        return env, None, {"max_turns": 2}, state, "http://unused"

    async def fake_env_step(env: FakeEnv, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        del args, kwargs
        return await env.run_step()

    async def fake_post(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del url, headers
        request_index = len(request_input_ids)
        request_input_ids.append(list(payload["input_ids"]))
        token = 10 + request_index
        return {
            "text": f"response-{request_index}",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, token]],
            },
        }

    original_inference_step = deepeyes_rollout._run_inference_step

    async def observed_inference_step(*step_args: Any, **step_kwargs: Any) -> tuple[Any, ...]:
        nonlocal inference_attempts
        inference_attempts += 1
        if inference_attempts == 2:
            second_inference_attempted.set()
        return await original_inference_step(*step_args, **step_kwargs)

    original_finalize = deepeyes_rollout._finalize_sample

    def observed_finalize(*finalize_args: Any, **finalize_kwargs: Any) -> Sample:
        nonlocal finalize_count
        finalize_count += 1
        return original_finalize(*finalize_args, **finalize_kwargs)

    async def hold_request_permit() -> None:
        async with state.request_permit():
            holder_entered.set()
            await release_holder.wait()

    monkeypatch.setattr(deepeyes_rollout, "_initialize_resources", fake_initialize)
    monkeypatch.setattr(deepeyes_rollout, "_process_env_step", fake_env_step)
    monkeypatch.setattr(deepeyes_rollout, "_run_inference_step", observed_inference_step)
    monkeypatch.setattr(deepeyes_rollout, "_finalize_sample", observed_finalize)
    monkeypatch.setattr(deepeyes_rollout, "post", fake_post)

    rollout_task: asyncio.Task[Sample] | None = None
    holder_task: asyncio.Task[None] | None = None
    try:
        rollout_task = asyncio.create_task(
            sglang_rollout.generate_and_rm(
                args,
                sample,
                sampling_params={"max_new_tokens": args.rollout_max_response_len},
            ),
            name="deepeyes-abort",
        )
        await asyncio.wait_for(env_entered.wait(), timeout=_TIMEOUT)

        holder_task = asyncio.create_task(hold_request_permit(), name="unrelated-holder")
        await asyncio.wait_for(holder_entered.wait(), timeout=_TIMEOUT)
        release_env.set()
        await asyncio.wait_for(second_inference_attempted.wait(), timeout=_TIMEOUT)

        state.aborted = True
        release_holder.set()
        await asyncio.wait_for(holder_task, timeout=_TIMEOUT)
        aborted = await asyncio.wait_for(rollout_task, timeout=_TIMEOUT)

        state.aborted = False
        await asyncio.wait_for(_acquire_once(state), timeout=_TIMEOUT)

        assert aborted is sample
        assert aborted.status == Sample.Status.ABORTED
        assert request_input_ids == [[1]]
        assert inference_attempts == 2
        assert envs[0].closed
        assert finalize_count == 1
        assert aborted.metadata["rollout_turns"] == 1
        assert aborted.metadata["rollout_stop_reason"] == "finish_abort"
        assert [trace["turn_index"] for trace in aborted.metadata["rollout_traces"]] == [0]
        assert aborted.response_length == 2
        assert aborted.response == "10 20"
        assert aborted.loss_mask == [1, 0]
        assert aborted.metadata["_current_turn_response_start"] == 2

        resumed = await asyncio.wait_for(
            sglang_rollout.generate_and_rm(
                args,
                sample,
                sampling_params={"max_new_tokens": args.rollout_max_response_len},
            ),
            timeout=_TIMEOUT,
        )

        assert resumed is sample
        assert resumed.status == Sample.Status.COMPLETED
        assert request_input_ids == [[1], [1, 10, 20]]
        assert inference_attempts == 3
        assert finalize_count == 2
        assert resumed.metadata["rollout_turns"] == 2
        assert resumed.metadata["rollout_stop_reason"] == "env_done"
        assert [trace["turn_index"] for trace in resumed.metadata["rollout_traces"]] == [0, 1]
        assert resumed.response_length == 3
        assert resumed.response == "10 20 11"
        assert resumed.loss_mask == [0, 0, 1]
        assert len(envs) == 2
        assert envs[1].turn_seen == 1
        assert envs[1].image_seen == "image-after-turn-0"
        assert envs[1].closed
        await asyncio.wait_for(_acquire_once(state), timeout=_TIMEOUT)
    finally:
        state.aborted = False
        release_env.set()
        release_holder.set()
        pending_tasks = [task for task in (rollout_task, holder_task) if task is not None and not task.done()]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        sglang_rollout.GenerateState.clear_instances()
