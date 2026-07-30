# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU-only async tests for per-model-request concurrency permits.

These validate ``GenerateState.model_request_permit``, the ``generate_and_rm``
legacy/opt-in dispatch boundary, Deepeyes integration, and the queued-request
abort race introduced by request-level concurrency. No GPU, SGLang server, or
model weights are required.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.utils.types import Sample


# The GitHub CPU test environment intentionally omits SGLang. Importing
# sglang_rollout normally pulls in relax.distributed.ray.rollout, whose runtime
# engine imports sglang.srt. Stub only that unrelated logging dependency while
# loading the module under test, then restore sys.modules immediately.
_DISTRIBUTED_ROLLOUT_MODULE = "relax.distributed.ray.rollout"
_installed_rollout_stub = _DISTRIBUTED_ROLLOUT_MODULE not in sys.modules and importlib.util.find_spec("sglang") is None
if _installed_rollout_stub:
    _rollout_stub = ModuleType(_DISTRIBUTED_ROLLOUT_MODULE)
    _rollout_stub._log_rollout_data = Mock()
    sys.modules[_DISTRIBUTED_ROLLOUT_MODULE] = _rollout_stub

try:
    sglang_rollout = importlib.import_module("relax.engine.rollout.sglang_rollout")
finally:
    if _installed_rollout_stub:
        sys.modules.pop(_DISTRIBUTED_ROLLOUT_MODULE, None)

GenerateState = sglang_rollout.GenerateState
request_model_aware = sglang_rollout.request_model_aware


class _PermitState:
    """Minimal stand-in that reuses the *real* shipped permit implementation.

    ``model_request_permit`` only touches ``self.semaphore``, so binding the
    unbound method here exercises production code without constructing a full
    ``GenerateState`` (which would load a tokenizer / processor).
    """

    def __init__(self, limit: int) -> None:
        self.semaphore = asyncio.Semaphore(limit)
        self.aborted = False

    # Reuse the exact method under test.
    model_request_permit = GenerateState.model_request_permit
    request_model = GenerateState.request_model

    def available(self) -> int:
        return self.semaphore._value


class _DispatchState(_PermitState):
    """State used to exercise the real ``generate_and_rm`` dispatch."""

    def __init__(self, limit: int) -> None:
        super().__init__(limit)

    @contextmanager
    def dp_rank_context(self):
        yield 0


def _dispatch_args(custom_path: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=True,
        custom_generate_function_path=custom_path,
    )


# ---------------------------------------------------------------------------
# 1. Concurrency limit is enforced and never exceeded.
# ---------------------------------------------------------------------------
async def test_permit_never_exceeds_limit() -> None:
    limit = 3
    st = _PermitState(limit)
    inflight = 0
    max_inflight = 0

    async def worker() -> None:
        nonlocal inflight, max_inflight
        async with st.model_request_permit():
            inflight += 1
            max_inflight = max(max_inflight, inflight)
            await asyncio.sleep(0.01)
            inflight -= 1

    await asyncio.gather(*(worker() for _ in range(24)))

    assert max_inflight == limit, f"observed {max_inflight} concurrent permits, limit was {limit}"
    assert st.available() == limit, "all permits should be returned after completion"


# ---------------------------------------------------------------------------
# 2. Fairness: with limit=1, a short single-turn request must NOT wait for a
#    long multi-turn session to finish entirely. It should slip in while the
#    multi-turn session is running its (permit-free) environment step.
#    This is the core behaviour Task 20 unlocks.
# ---------------------------------------------------------------------------
async def test_short_request_not_blocked_by_multiturn_env() -> None:
    st = _PermitState(1)
    finish_order: list[str] = []

    async def long_multiturn() -> None:
        for _ in range(3):
            # Short model request while holding the permit ...
            async with st.model_request_permit():
                await asyncio.sleep(0.02)
            # ... then a long environment / tool step WITHOUT the permit.
            await asyncio.sleep(0.10)
        finish_order.append("long")

    async def short_single_turn() -> None:
        # Arrive shortly after the long session enters its first env step.
        await asyncio.sleep(0.03)
        async with st.model_request_permit():
            await asyncio.sleep(0.02)
        finish_order.append("short")

    await asyncio.gather(long_multiturn(), short_single_turn())

    # The short request finishes long before the multi-turn session does,
    # proving it was not head-of-line blocked by the whole session.
    assert finish_order == ["short", "long"], finish_order


# ---------------------------------------------------------------------------
# 3. Dispatch integration: legacy custom generators retain the session cap,
#    while explicit opt-in and default generation use the intended boundaries.
# ---------------------------------------------------------------------------
async def test_generate_and_rm_keeps_session_permit_for_legacy_custom(monkeypatch) -> None:
    state = _DispatchState(limit=1)
    permit_was_held: list[bool] = []

    async def legacy_generate(args, sample, sampling_params):  # noqa: ANN001
        permit_was_held.append(state.semaphore.locked())
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: legacy_generate)

    result = await sglang_rollout.generate_and_rm(
        _dispatch_args("legacy.generate"), Sample(prompt="legacy"), {}, evaluation=False
    )

    assert permit_was_held == [True], "undecorated custom generators must retain the legacy session cap"
    assert result.status == Sample.Status.COMPLETED
    assert state.available() == 1


async def test_generate_and_rm_skips_outer_permit_for_explicit_opt_in(monkeypatch) -> None:
    state = _DispatchState(limit=1)
    outer_permit_was_free: list[bool] = []
    request_permit_was_held: list[bool] = []

    @request_model_aware
    async def request_managed_generate(args, sample, sampling_params, *, request_model):  # noqa: ANN001
        outer_permit_was_free.append(not state.semaphore.locked())
        await request_model("http://model", {})
        sample.status = Sample.Status.COMPLETED
        return sample

    async def fake_post(url, payload, headers=None):  # noqa: ANN001
        request_permit_was_held.append(state.semaphore.locked())
        return {}

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: request_managed_generate)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    result = await sglang_rollout.generate_and_rm(
        _dispatch_args("request_managed.generate"), Sample(prompt="request managed"), {}, evaluation=False
    )

    assert outer_permit_was_free == [True], "opted-in custom generators must not receive an outer session permit"
    assert request_permit_was_held == [True]
    assert result.status == Sample.Status.COMPLETED
    assert state.available() == 1


async def test_generate_and_rm_default_path_holds_permit(monkeypatch) -> None:
    state = _DispatchState(limit=1)
    permit_was_held: list[bool] = []

    async def fake_default_generate(args, sample, sampling_params, evaluation=False):  # noqa: ANN001
        permit_was_held.append(state.semaphore.locked())
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "generate", fake_default_generate)

    result = await sglang_rollout.generate_and_rm(
        _dispatch_args(None), Sample(prompt="single turn"), {}, evaluation=False
    )

    assert permit_was_held == [True], "default generation must remain concurrency-limited"
    assert result.status == Sample.Status.COMPLETED
    assert state.available() == 1


# ---------------------------------------------------------------------------
# 4. Exception-safe release: an error inside the permit still returns it.
# ---------------------------------------------------------------------------
async def test_permit_released_on_exception() -> None:
    st = _PermitState(1)

    with pytest.raises(ValueError):
        async with st.model_request_permit():
            assert st.available() == 0
            raise ValueError("boom")

    assert st.available() == 1, "permit must be released after an exception"

    # And the slot is genuinely reusable.
    async with st.model_request_permit():
        assert st.available() == 0


# ---------------------------------------------------------------------------
# 5. Cancellation-safe release: cancelling a task holding the permit returns it.
# ---------------------------------------------------------------------------
async def test_permit_released_on_cancellation() -> None:
    st = _PermitState(1)
    holding = asyncio.Event()

    async def holder() -> None:
        async with st.model_request_permit():
            holding.set()
            await asyncio.sleep(10)  # will be cancelled

    task = asyncio.create_task(holder())
    await holding.wait()
    assert st.available() == 0, "permit should be held while the request is in flight"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert st.available() == 1, "permit must be released after cancellation"


# ---------------------------------------------------------------------------
# 6. Failure-safe release: a queued waiter proceeds after the holder fails.
# ---------------------------------------------------------------------------
class _RequestFailure(Exception):
    pass


async def test_permit_released_on_failure_and_waiter_proceeds() -> None:
    st = _PermitState(1)
    waiter_ran: list[bool] = []
    failing_request_has_permit = asyncio.Event()

    async def failing_request() -> None:
        async with st.model_request_permit():
            failing_request_has_permit.set()
            await asyncio.sleep(0.02)  # hold the permit while the waiter queues
            raise _RequestFailure

    async def waiter() -> None:
        async with st.model_request_permit():
            waiter_ran.append(True)

    t_failure = asyncio.create_task(failing_request())
    await failing_request_has_permit.wait()

    t_wait = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # let the waiter block on acquire()
    assert st.available() == 0, "failing request should still hold the only permit"

    with pytest.raises(_RequestFailure):
        await t_failure
    await t_wait

    assert waiter_ran == [True], "waiter must acquire the permit released by the failed request"
    assert st.available() == 1, "no permit should be leaked after request failure"


# ---------------------------------------------------------------------------
# 7. Integration: the real deepeyes multi-turn rollout must hold the permit
#    only around the model request and release it during env / tool execution.
# ---------------------------------------------------------------------------
class _StubTokenizer:
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):  # noqa: ANN001
        return [1, 2, 3]

    def decode(self, tokens, skip_special_tokens=False):  # noqa: ANN001
        return "decoded"


class _IntegState:
    """Minimal GenerateState substitute for the deepeyes rollout path."""

    def __init__(self, limit: int) -> None:
        self.semaphore = asyncio.Semaphore(limit)
        self.tokenizer = _StubTokenizer()
        self.processor = None
        self.aborted = False

    model_request_permit = GenerateState.model_request_permit
    request_model = GenerateState.request_model

    def available(self) -> int:
        return self.semaphore._value


class _TwoTurnEnv:
    """Fake environment: one intermediate turn, then done.

    Records whether the permit is free while the (permit-less) environment step
    runs.
    """

    def __init__(self, state: _IntegState, records: dict) -> None:
        self._state = state
        self._records = records
        self.turn = 0
        self._steps = 0

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def step(self, response_text):  # noqa: ANN001
        # The permit must have been released before env execution begins.
        self._records["locked_during_env"].append(self._state.semaphore.locked())
        self._steps += 1
        done = self._steps >= 2
        return "observation", done, {}

    def format_observation(self, observation):  # noqa: ANN001
        return {"role": "user", "content": observation}


async def test_deepeyes_releases_permit_during_env(monkeypatch) -> None:
    import examples.deepeyes.rollout as de

    state = _IntegState(limit=1)
    records = {
        "locked_during_infer": [],
        "locked_during_env": [],
        "inflight": 0,
        "max_inflight": 0,
    }
    env = _TwoTurnEnv(state, records)

    async def fake_post(url, payload, headers=None):  # noqa: ANN001
        # The permit must be held while the model request is in flight.
        records["locked_during_infer"].append(state.semaphore.locked())
        records["inflight"] += 1
        records["max_inflight"] = max(records["max_inflight"], records["inflight"])
        await asyncio.sleep(0.01)
        records["inflight"] -= 1
        return {}

    async def fake_infer(  # noqa: ANN001
        url, tokens, sampling_params, image_data, tokenizer, args=None, *, request_model
    ):
        await request_model(url, {})
        return "resp", [10, 11], [-0.1, -0.2], "stop", {}

    monkeypatch.setattr(de, "GenerateState", lambda args: state)
    monkeypatch.setattr(de, "_load_env_module", lambda path: SimpleNamespace(build_env=lambda sample, args: env))
    monkeypatch.setattr(de, "_run_inference_step", fake_infer)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    args = SimpleNamespace(
        rollout_interaction_env_path="dummy",
        max_turns=2,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        rollout_max_context_len=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        use_rollout_routing_replay=False,
    )
    sample = Sample(prompt="hello world")

    result = await de.generate(args, sample, dict(max_new_tokens=8), request_model=state.request_model)

    assert records["locked_during_infer"] == [True, True], "permit must be held during every model request"
    assert records["locked_during_env"] == [False, False], "permit must be released during env / tool execution"
    assert records["max_inflight"] == 1, "concurrent model requests must not exceed the limit"
    assert result.status == Sample.Status.COMPLETED
    assert state.available() == 1, "permit must not be leaked after the rollout finishes"


# ---------------------------------------------------------------------------
# 8. Real abort race: a Deepeyes turn queued on the permit must recheck abort
#    after acquire and must not start a request after abort_all has run.
# ---------------------------------------------------------------------------
class _OneTurnEnv:
    def __init__(self) -> None:
        self.turn = 0

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def step(self, response_text):  # noqa: ANN001
        self.turn += 1
        return "done", True, {}

    def format_observation(self, observation):  # noqa: ANN001
        return {"role": "user", "content": observation}


async def test_deepeyes_queued_turn_does_not_request_after_abort(monkeypatch) -> None:
    import examples.deepeyes.rollout as de

    state = _IntegState(limit=1)
    first_infer_started = asyncio.Event()
    release_first_infer = asyncio.Event()
    waiter_queued = asyncio.Event()
    http_calls = 0

    original_acquire = state.semaphore.acquire

    async def tracked_acquire() -> bool:
        if state.semaphore.locked():
            waiter_queued.set()
        return await original_acquire()

    state.semaphore.acquire = tracked_acquire

    async def fake_post(url, payload, headers=None):  # noqa: ANN001
        nonlocal http_calls
        http_calls += 1
        if http_calls == 1:
            first_infer_started.set()
            await release_first_infer.wait()
        return {}

    async def fake_infer(  # noqa: ANN001
        url, tokens, sampling_params, image_data, tokenizer, args=None, *, request_model
    ):
        await request_model(url, {})
        return "resp", [10], [-0.1], "stop", {}

    monkeypatch.setattr(de, "GenerateState", lambda args: state)
    monkeypatch.setattr(
        de,
        "_load_env_module",
        lambda path: SimpleNamespace(build_env=lambda sample, args: _OneTurnEnv()),
    )
    monkeypatch.setattr(de, "_run_inference_step", fake_infer)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    args = SimpleNamespace(
        rollout_interaction_env_path="dummy",
        max_turns=1,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        rollout_max_context_len=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        use_rollout_routing_replay=False,
    )

    first = asyncio.create_task(
        de.generate(args, Sample(prompt="first"), dict(max_new_tokens=8), request_model=state.request_model)
    )
    await first_infer_started.wait()

    queued = asyncio.create_task(
        de.generate(args, Sample(prompt="queued"), dict(max_new_tokens=8), request_model=state.request_model)
    )
    await asyncio.wait_for(waiter_queued.wait(), timeout=1)

    # Models abort() setting state.aborted and issuing its one-shot abort_all
    # while the second turn is still queued for the only permit.
    state.aborted = True
    release_first_infer.set()

    first_result, queued_result = await asyncio.gather(first, queued)

    assert http_calls == 1, "the queued turn must not start a fresh request after abort"
    assert first_result.status == Sample.Status.COMPLETED
    assert queued_result.status == Sample.Status.ABORTED
    assert queued_result.metadata["rollout_stop_reason"] == "abort_before_inference"
    assert state.available() == 1
