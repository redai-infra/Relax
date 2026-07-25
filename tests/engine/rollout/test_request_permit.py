# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU-only async tests for per-model-request concurrency permits.

These validate ``GenerateState.model_request_permit`` — the fine-grained
concurrency unit introduced for Task 20 (shifting rollout concurrency control
from "session-level" to "single model request-level"). No GPU, no SGLang, and
no model weights are required: the real context-manager method is exercised on
a minimal object that only carries an ``asyncio.Semaphore``.
"""

from __future__ import annotations

import asyncio

import pytest

from relax.engine.rollout.sglang_rollout import GenerateState


class _PermitState:
    """Minimal stand-in that reuses the *real* shipped permit implementation.

    ``model_request_permit`` only touches ``self.semaphore``, so binding the
    unbound method here exercises production code without constructing a full
    ``GenerateState`` (which would load a tokenizer / processor).
    """

    def __init__(self, limit: int) -> None:
        self.semaphore = asyncio.Semaphore(limit)

    # Reuse the exact method under test.
    model_request_permit = GenerateState.model_request_permit

    def available(self) -> int:
        return self.semaphore._value


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
# 3. Exception-safe release: an error inside the permit still returns it.
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
# 4. Cancellation-safe release: cancelling a task holding the permit returns it.
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
# 5. Abort-safe release + a queued waiter can then proceed.
#    Models SGLang aborting an in-flight request mid-session.
# ---------------------------------------------------------------------------
class _AbortSignal(Exception):
    pass


async def test_permit_released_on_abort_and_waiter_proceeds() -> None:
    st = _PermitState(1)
    waiter_ran: list[bool] = []
    aborter_has_permit = asyncio.Event()

    async def aborting_request() -> None:
        async with st.model_request_permit():
            aborter_has_permit.set()
            await asyncio.sleep(0.02)  # hold the permit while the waiter queues
            raise _AbortSignal

    async def waiter() -> None:
        async with st.model_request_permit():
            waiter_ran.append(True)

    t_abort = asyncio.create_task(aborting_request())
    await aborter_has_permit.wait()

    t_wait = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # let the waiter block on acquire()
    assert st.available() == 0, "aborting request should still hold the only permit"

    with pytest.raises(_AbortSignal):
        await t_abort
    await t_wait

    assert waiter_ran == [True], "waiter must acquire the permit released by the abort"
    assert st.available() == 1, "no permit should be leaked after abort"


# ---------------------------------------------------------------------------
# 6. Integration: the real deepeyes multi-turn rollout must hold the permit
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

    model_request_permit = GenerateState.model_request_permit

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
    from types import SimpleNamespace

    import examples.deepeyes.rollout as de
    from relax.utils.types import Sample

    state = _IntegState(limit=1)
    records = {
        "locked_during_infer": [],
        "locked_during_env": [],
        "inflight": 0,
        "max_inflight": 0,
    }
    env = _TwoTurnEnv(state, records)

    async def fake_infer(url, tokens, sampling_params, image_data, tokenizer, args=None):  # noqa: ANN001
        # The permit must be held while the model request is in flight.
        records["locked_during_infer"].append(state.semaphore.locked())
        records["inflight"] += 1
        records["max_inflight"] = max(records["max_inflight"], records["inflight"])
        await asyncio.sleep(0.01)
        records["inflight"] -= 1
        return "resp", [10, 11], [-0.1, -0.2], "stop", {}

    monkeypatch.setattr(de, "GenerateState", lambda args: state)
    monkeypatch.setattr(de, "_load_env_module", lambda path: SimpleNamespace(build_env=lambda sample, args: env))
    monkeypatch.setattr(de, "_run_inference_step", fake_infer)

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

    result = await de.generate(args, sample, dict(max_new_tokens=8))

    assert records["locked_during_infer"] == [True, True], "permit must be held during every model request"
    assert records["locked_during_env"] == [False, False], "permit must be released during env / tool execution"
    assert records["max_inflight"] == 1, "concurrent model requests must not exceed the limit"
    assert result.status == Sample.Status.COMPLETED
    assert state.available() == 1, "permit must not be leaked after the rollout finishes"
