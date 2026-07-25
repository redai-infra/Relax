# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

from examples.deepeyes import rollout as deepeyes_rollout
from relax.engine.rollout import sglang_rollout
from relax.utils.types import Sample


Runner = Callable[[Any, Sample], Awaitable[None]]


class _Tokenizer:
    bos_token_id = None

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(str(token) for token in tokens)


class _RequestProbe:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.in_flight = 0
        self.peak_in_flight = 0
        self.start_order: list[str] = []
        self.capacity_reached = asyncio.Event()
        self._started: dict[str, asyncio.Event] = {}

    def started(self, label: str) -> asyncio.Event:
        return self._started.setdefault(label, asyncio.Event())

    async def execute(self, label: str, gate: asyncio.Event | None = None) -> None:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        self.start_order.append(label)
        self.started(label).set()
        if self.in_flight == self.capacity:
            self.capacity_reached.set()
        try:
            if gate is not None:
                await gate.wait()
        finally:
            self.in_flight -= 1


def _make_state(capacity: int) -> Any:
    state = object.__new__(sglang_rollout.GenerateState)
    state.semaphore = asyncio.Semaphore(capacity)
    state.aborted = False
    state.dp_counts = [0]
    state.dp_rank = 0
    state.opd_manager = None
    state.tokenizer = _Tokenizer()
    state.processor = None
    return state


def _make_args(custom_generate_function_path: str | None = None) -> Namespace:
    return Namespace(
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=True,
        custom_generate_function_path=custom_generate_function_path,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        num_layers=1,
        moe_router_topk=1,
    )


def _function_path(function: Callable[..., Any]) -> str:
    return f"{function.__module__}.{function.__name__}"


def _make_sample(runner: Runner | None = None, *, path: str | None = None) -> Sample:
    metadata = {"runner": runner} if runner is not None else {}
    return Sample(prompt="task20", metadata=metadata, generate_function_path=path)


async def _run_sample(args: Namespace, sample: Sample) -> Sample:
    result = await sglang_rollout.generate_and_rm(
        args,
        sample,
        sampling_params={"max_new_tokens": args.rollout_max_response_len},
    )
    assert isinstance(result, Sample)
    return result


async def _acquire_request_permit(state: Any) -> None:
    async with state.request_permit():
        pass


async def _acquire_semaphore(semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        pass


async def _permit_aware_generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    del sampling_params, evaluation
    state = sglang_rollout.GenerateState(args)
    await sample.metadata["runner"](state, sample)
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    return sample


_permit_aware_generate.manages_inference_permit = True


async def _legacy_generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    del args, sampling_params, evaluation
    await sample.metadata["runner"](None, sample)
    sample.status = Sample.Status.COMPLETED
    return sample


async def _unmarked_permit_generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    del sampling_params
    state = sglang_rollout.GenerateState(args)
    async with state.request_permit():
        sample.status = Sample.Status.COMPLETED
    return sample


async def test_request_permit_recovers_after_exception_and_cancellation() -> None:
    state = _make_state(capacity=1)
    assert callable(getattr(state, "request_permit", None))

    with pytest.raises(RuntimeError, match="request failed"):
        async with state.request_permit():
            raise RuntimeError("request failed")

    holder_entered = asyncio.Event()
    hold = asyncio.Event()

    async def hold_permit() -> None:
        async with state.request_permit():
            holder_entered.set()
            await hold.wait()

    holder = asyncio.create_task(hold_permit())
    await asyncio.wait_for(holder_entered.wait(), timeout=1.0)
    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder

    holder_entered.clear()
    second_holder = asyncio.create_task(hold_permit())
    await asyncio.wait_for(holder_entered.wait(), timeout=1.0)
    waiter_attempted = asyncio.Event()
    waiter_entered = asyncio.Event()

    async def wait_for_permit() -> None:
        waiter_attempted.set()
        async with state.request_permit():
            waiter_entered.set()

    waiter = asyncio.create_task(wait_for_permit())
    await asyncio.wait_for(waiter_attempted.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert not waiter_entered.is_set()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    hold.set()
    await second_holder
    await asyncio.wait_for(_acquire_request_permit(state), timeout=1.0)


async def test_generate_and_rm_keeps_legacy_custom_generator_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    capacity = 2
    state = _make_state(capacity)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    args = _make_args(_function_path(_legacy_generate))
    release = asyncio.Event()
    probe = _RequestProbe(capacity)
    tasks = []

    for index in range(8):

        async def runner(current_state: Any, sample: Sample, label: str = f"legacy-{index}") -> None:
            del current_state, sample
            await probe.execute(label, release)

        tasks.append(asyncio.create_task(_run_sample(args, _make_sample(runner))))

    try:
        await asyncio.wait_for(probe.capacity_reached.wait(), timeout=1.0)
        for _ in range(20):
            await asyncio.sleep(0)
        assert probe.peak_in_flight == capacity
    finally:
        release.set()
        await asyncio.gather(*tasks)

    assert probe.in_flight == 0


async def test_queued_permit_aware_request_stops_after_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_state(capacity=1)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    args = _make_args(_function_path(_permit_aware_generate))
    release = asyncio.Event()
    probe = _RequestProbe(capacity=1)

    async def holder_runner(current_state: Any, sample: Sample) -> None:
        del sample
        async with current_state.request_permit():
            await probe.execute("holder", release)

    queued_request_started = asyncio.Event()

    async def queued_runner(current_state: Any, sample: Sample) -> None:
        del sample
        async with current_state.request_permit():
            queued_request_started.set()

    holder = asyncio.create_task(_run_sample(args, _make_sample(holder_runner)))
    await asyncio.wait_for(probe.capacity_reached.wait(), timeout=1.0)
    queued = asyncio.create_task(_run_sample(args, _make_sample(queued_runner)))
    await asyncio.sleep(0)
    assert not queued.done()

    state.aborted = True
    release.set()
    holder_result, queued_result = await asyncio.wait_for(asyncio.gather(holder, queued), timeout=2.0)

    assert holder_result.status == Sample.Status.COMPLETED
    assert queued_result.status == Sample.Status.ABORTED
    assert not queued_request_started.is_set()
    assert probe.in_flight == 0

    state.aborted = False
    await asyncio.wait_for(_acquire_request_permit(state), timeout=1.0)


async def test_unmarked_custom_generator_cannot_nest_request_permit(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_state(capacity=1)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    args = _make_args(_function_path(_unmarked_permit_generate))

    with pytest.raises(RuntimeError, match="manages_inference_permit"):
        await asyncio.wait_for(_run_sample(args, _make_sample()), timeout=1.0)

    await asyncio.wait_for(_acquire_semaphore(state.semaphore), timeout=1.0)


async def test_deepeyes_dispatch_releases_permit_during_environment_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state(capacity=1)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    args = _make_args("examples.deepeyes.rollout.generate")
    env_entered = asyncio.Event()
    release_env = asyncio.Event()
    probe = _RequestProbe(capacity=1)
    env_calls = 0
    model_calls = 0

    class FakeEnv:
        def __init__(self) -> None:
            self.closed = False

        def reset(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    env = FakeEnv()

    async def fake_prepare(
        sample: Sample,
        current_state: Any,
        current_args: Namespace,
        sampling_params: dict[str, Any],
        is_resuming: bool = False,
    ) -> tuple[None, list[int], None, None, list[Any]]:
        del current_state, current_args, sampling_params, is_resuming
        sample.tokens = [1]
        sample.loss_mask = []
        sample.rollout_log_probs = []
        return None, [], None, None, []

    async def fake_post(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        nonlocal model_calls
        del url, payload, headers
        label = f"deepeyes-{model_calls}"
        model_calls += 1
        await probe.execute(label)
        return {
            "text": "response",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 10]],
            },
        }

    async def fake_env_step(*step_args: Any, **step_kwargs: Any) -> tuple[Any, ...]:
        nonlocal env_calls
        del step_args, step_kwargs
        env_calls += 1
        if env_calls == 1:
            env_entered.set()
            await release_env.wait()
            return [20], None, None, None, False, {}
        return None, None, None, None, True, {}

    monkeypatch.setattr(
        deepeyes_rollout,
        "_initialize_resources",
        lambda current_args, sample: (env, SimpleNamespace(), {"max_turns": 2}, state, "http://unused"),
    )
    monkeypatch.setattr(deepeyes_rollout, "_prepare_start_state", fake_prepare)
    monkeypatch.setattr(deepeyes_rollout, "_process_env_step", fake_env_step)
    monkeypatch.setattr(deepeyes_rollout, "post", fake_post)

    long_task = asyncio.create_task(_run_sample(args, _make_sample()), name="long-deepeyes")
    short_task = None
    interleaved = False
    long_waited_in_env = False
    try:
        await asyncio.wait_for(env_entered.wait(), timeout=2.0)

        async def short_runner(current_state: Any, sample: Sample) -> None:
            del sample
            async with current_state.request_permit():
                await probe.execute("short")

        short_sample = _make_sample(short_runner, path=_function_path(_permit_aware_generate))
        short_task = asyncio.create_task(_run_sample(args, short_sample), name="short-request")
        for _ in range(50):
            if probe.started("short").is_set():
                break
            await asyncio.sleep(0)
        interleaved = probe.started("short").is_set()
        long_waited_in_env = not long_task.done()
    finally:
        release_env.set()

    outcomes = await asyncio.wait_for(asyncio.gather(long_task, short_task, return_exceptions=True), timeout=2.0)
    assert interleaved
    assert long_waited_in_env
    assert all(not isinstance(outcome, BaseException) for outcome in outcomes)
    long_result, short_result = outcomes

    assert isinstance(long_result, Sample)
    assert isinstance(short_result, Sample)
    assert long_result.status == Sample.Status.COMPLETED
    assert short_result.status == Sample.Status.COMPLETED
    assert probe.peak_in_flight == 1
    assert probe.in_flight == 0
    assert probe.start_order.index("short") < probe.start_order.index("deepeyes-1")
    assert env.closed
