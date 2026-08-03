# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU async unit tests for per-request inference permit scheduling.

Task 20 - claim issue #120 (task spec: #86 section 3-20). Covers acceptance V1-V6:
- T1-T8 exercise ``InferencePermitManager`` directly (no ``GenerateState``, no GPU/network).
- T9-T11 exercise ``_dispatch_generate`` with a lightweight stub state (duck-typed:
  only ``.aborted`` / ``.dp_rank_context`` / ``.semaphore``).
- T12-T13 run the real deepeyes ``generate`` loop (only the network ``post`` is stubbed).

Timing-sensitive cases use ``asyncio.Event`` for deterministic synchronization
instead of wall-clock sleeps.

The module is skipped when the inference stack (ray / sglang_router) is missing,
e.g. on the CPU-only GitHub runner; it is covered by the internal nightly GPU run.
"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from relax.engine.rollout.request_permit import GenerationAborted, InferencePermitManager
from relax.utils.types import Sample


try:
    from relax.engine.rollout import sglang_rollout
    from relax.engine.rollout.sglang_rollout import (
        _dispatch_generate,
        _ensure_not_holding_session_lock,
        _holding_session_lock,
        generate_and_rm_group,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


# --------------------------------------------------------------------------- #
# Stubs for _dispatch_generate (T9-T11): duck-typed state / args / sample.
# --------------------------------------------------------------------------- #
class _StubState:
    def __init__(self, capacity: int = 1) -> None:
        self.aborted = False
        self.semaphore = asyncio.Semaphore(capacity)

    @contextlib.contextmanager
    def dp_rank_context(self):
        yield 0


class _StubArgs:
    def __init__(self, custom_generate_function_path=None) -> None:
        self.custom_generate_function_path = custom_generate_function_path


class _StubSample:
    def __init__(self) -> None:
        self.status = Sample.Status.PENDING
        self.generate_function_path = None


def _completed_sample() -> SimpleNamespace:
    return SimpleNamespace(
        status=Sample.Status.COMPLETED,
        response="done",
        response_length=1,
        reward=1.0,
        loss_mask=None,
        session_id=None,
        metadata={},
    )


async def test_group_rm_waits_for_cross_version_strict_retry(monkeypatch) -> None:
    completed = _completed_sample()
    aborted = SimpleNamespace(
        status=Sample.Status.ABORTED,
        response="partial",
        response_length=1,
        reward=None,
        loss_mask=None,
        session_id=None,
        metadata={},
        generate_function_path=None,
    )
    state = SimpleNamespace(aborted=False, opd_manager=None)
    reward_calls: list[list[SimpleNamespace]] = []
    dispatch_calls: list[SimpleNamespace] = []

    async def fake_dispatch(_state, _args, sample, _sampling_params, **_kwargs):
        dispatch_calls.append(sample)
        sample.status = Sample.Status.COMPLETED
        sample.response = "resumed"
        return sample

    async def fake_group_rm(_args, group):
        reward_calls.append(list(group))
        return [1.0] * len(group)

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "_dispatch_generate", fake_dispatch)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", fake_group_rm)
    args = SimpleNamespace(
        enable_cross_version_kv_continuation=True,
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
        group_rm=True,
        sglang_enable_deterministic_inference=False,
    )

    original_group = [completed, aborted]
    result = await generate_and_rm_group(args, original_group, sampling_params={})

    assert result[0] is completed
    assert result[1] is aborted
    assert dispatch_calls == [aborted]
    assert reward_calls == [original_group]
    assert completed.reward == 1.0
    assert aborted.reward == 1.0


async def test_group_rm_skips_mixed_terminal_group(monkeypatch) -> None:
    completed = _completed_sample()
    aborted = SimpleNamespace(
        status=Sample.Status.ABORTED,
        response="partial",
        response_length=1,
        reward=None,
        loss_mask=None,
        session_id=None,
        metadata={},
    )
    state = SimpleNamespace(aborted=False, opd_manager=None)
    reward_called = False

    async def fake_generate_and_rm(_args, sample, _sampling_params, **_kwargs):
        return sample

    async def fake_group_rm(_args, _group):
        nonlocal reward_called
        reward_called = True
        return [1.0, 1.0]

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate_and_rm)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", fake_group_rm)
    args = SimpleNamespace(
        enable_cross_version_kv_continuation=True,
        group_rm=True,
        sglang_enable_deterministic_inference=False,
    )

    result = await generate_and_rm_group(args, [completed, aborted], sampling_params={})

    assert result == [completed, aborted]
    assert not reward_called


@pytest.mark.parametrize(
    ("a3_enabled", "evaluation"),
    [(False, False), (True, True)],
)
async def test_group_rm_preserves_default_and_evaluation_behavior(
    monkeypatch,
    a3_enabled: bool,
    evaluation: bool,
) -> None:
    completed = _completed_sample()
    aborted = SimpleNamespace(status=Sample.Status.ABORTED, session_id=None)
    state = SimpleNamespace(aborted=False, opd_manager=None)
    reward_calls = 0

    async def fake_generate_and_rm(_args, sample, _sampling_params, **_kwargs):
        return sample

    async def fake_group_rm(_args, group):
        nonlocal reward_calls
        reward_calls += 1
        return [1.0] * len(group)

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate_and_rm)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", fake_group_rm)
    args = SimpleNamespace(
        enable_cross_version_kv_continuation=a3_enabled,
        group_rm=True,
        sglang_enable_deterministic_inference=False,
    )

    await generate_and_rm_group(args, [completed, aborted], sampling_params={}, evaluation=evaluation)

    assert reward_calls == 1


# --------------------------------------------------------------------------- #
# T1-T8: InferencePermitManager
# --------------------------------------------------------------------------- #
def test_capacity_must_be_positive() -> None:
    """Capacity < 1 (e.g. from integer division) fails loudly, not a silent
    hang."""
    with pytest.raises(ValueError):
        InferencePermitManager(capacity=0)


async def test_permit_bounds_concurrency() -> None:
    """V3: peak in-flight requests never exceed the configured capacity."""
    mgr = InferencePermitManager(capacity=4)
    in_flight = 0
    peak = 0

    async def one_turn() -> None:
        nonlocal in_flight, peak
        async with mgr.permit():
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1

    async def session() -> None:
        for _ in range(3):
            await one_turn()

    await asyncio.gather(*[session() for _ in range(16)])
    assert peak <= 4
    assert in_flight == 0
    assert not mgr.semaphore.locked()


async def test_capacity_one_short_request_interleaves() -> None:
    """V5: with capacity=1 a short request runs between the long session's turns
    (FIFO fairness), rather than waiting for all long turns to finish."""
    mgr = InferencePermitManager(capacity=1)
    order: list[str] = []
    long_holds_turn0 = asyncio.Event()
    short_enqueued = asyncio.Event()

    async def long_session() -> None:
        for turn in range(3):
            async with mgr.permit():
                order.append(f"long{turn}")
                if turn == 0:
                    long_holds_turn0.set()
                    # Release turn 0 only after the short request is queued on the
                    # semaphore, so FIFO hands the freed permit to it first.
                    await short_enqueued.wait()
            await asyncio.sleep(0)  # yield so the freed permit goes to the FIFO head

    async def short_request() -> None:
        await long_holds_turn0.wait()
        short_enqueued.set()
        async with mgr.permit():
            order.append("short")

    await asyncio.wait_for(asyncio.gather(long_session(), short_request()), timeout=5.0)
    assert order == ["long0", "short", "long1", "long2"]


async def test_permit_released_on_success() -> None:
    """V4: permit is released after a normal exit."""
    mgr = InferencePermitManager(capacity=1)
    for _ in range(3):
        async with mgr.permit():
            pass
    assert not mgr.semaphore.locked()


async def test_permit_released_on_exception() -> None:
    """V4: permit is released when the body raises."""
    mgr = InferencePermitManager(capacity=1)
    with pytest.raises(RuntimeError):
        async with mgr.permit():
            raise RuntimeError("boom")
    assert not mgr.semaphore.locked()


async def test_permit_released_on_cancel_while_holding() -> None:
    """V4: permit is released when the holding task is cancelled."""
    mgr = InferencePermitManager(capacity=1)
    started = asyncio.Event()

    async def holder() -> None:
        async with mgr.permit():
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(holder())
    await started.wait()
    assert mgr.semaphore.locked()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not mgr.semaphore.locked()


async def test_cancel_while_waiting_does_not_leak() -> None:
    """V4: cancelling a task that is *waiting* to acquire does not corrupt the
    permit count; a later acquire still succeeds promptly."""
    mgr = InferencePermitManager(capacity=1)
    holder_started = asyncio.Event()
    release_holder = asyncio.Event()
    waiter_ready = asyncio.Event()

    async def holder() -> None:
        async with mgr.permit():
            holder_started.set()
            await release_holder.wait()

    async def waiter() -> None:
        waiter_ready.set()
        async with mgr.permit():
            pass

    h = asyncio.create_task(holder())
    await holder_started.wait()
    w = asyncio.create_task(waiter())
    await waiter_ready.wait()
    await asyncio.sleep(0)  # let the waiter reach acquire() and block
    w.cancel()
    with pytest.raises(asyncio.CancelledError):
        await w
    release_holder.set()
    await h

    async def _acquire_once() -> None:
        async with mgr.permit():
            pass

    await asyncio.wait_for(_acquire_once(), timeout=1.0)
    assert not mgr.semaphore.locked()


async def test_abort_check_raises_and_releases() -> None:
    """V4: abort_check True right after acquire raises GenerationAborted and the
    permit is still released."""
    mgr = InferencePermitManager(capacity=1)
    with pytest.raises(GenerationAborted):
        async with mgr.permit(abort_check=lambda: True):
            pass
    assert not mgr.semaphore.locked()


async def test_multiturn_loop_abort_between_turns() -> None:
    """V2/V4 (mechanism): a multi-turn loop stops with ABORTED when abort is
    signalled between turns, having issued exactly one request."""
    mgr = InferencePermitManager(capacity=1)
    aborted = False
    requests = 0
    status = None

    for turn in range(3):
        try:
            async with mgr.permit(abort_check=lambda: aborted):
                requests += 1
        except GenerationAborted:
            status = Sample.Status.ABORTED
            break
        if turn == 0:  # abort arrives during the env step after turn 0
            aborted = True

    assert status == Sample.Status.ABORTED
    assert requests == 1
    assert not mgr.semaphore.locked()


async def test_abort_retries_until_late_pending_task_is_terminal(monkeypatch) -> None:
    release_pending = asyncio.Event()
    sample = SimpleNamespace(
        status=Sample.Status.ABORTED,
        abort_count=0,
        response="partial",
        metadata={},
    )

    async def pending_group():
        await release_pending.wait()
        return [sample]

    task = asyncio.create_task(pending_group())
    state = SimpleNamespace(
        aborted=False,
        evaluating=0,
        protected_pendings=set(),
        pendings={task},
    )
    attempts = 0

    async def fake_get(_url):
        return {"urls": ["http://engine"], "workers": [{"url": "http://engine"}]}

    async def fake_post(_url, _payload):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            release_pending.set()
        return {}

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "get", fake_get)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    args = SimpleNamespace(
        use_slime_router=False,
        partial_rollout=True,
        sglang_router_ip="router",
        sglang_router_port=3000,
    )

    aborted, protected = await sglang_rollout.abort(
        args,
        rollout_id=3,
        retry_interval_seconds=0.01,
        timeout_seconds=1.0,
    )

    assert attempts == 2
    assert aborted == [[sample]]
    assert protected == []
    assert sample.abort_count == 1
    assert state.aborted
    assert not state.pendings


async def test_abort_retry_timeout_fails_closed(monkeypatch) -> None:
    never = asyncio.Event()

    async def pending_group():
        await never.wait()
        return []

    task = asyncio.create_task(pending_group())
    state = SimpleNamespace(
        aborted=False,
        evaluating=0,
        protected_pendings=set(),
        pendings={task},
    )

    async def fake_get(_url):
        return {"urls": ["http://engine"], "workers": [{"url": "http://engine"}]}

    async def fake_post(_url, _payload):
        return {}

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "get", fake_get)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    args = SimpleNamespace(
        use_slime_router=False,
        partial_rollout=True,
        sglang_router_ip="router",
        sglang_router_port=3000,
    )

    with pytest.raises(RuntimeError, match="Abort drain timed out"):
        await sglang_rollout.abort(
            args,
            rollout_id=4,
            retry_interval_seconds=0.01,
            timeout_seconds=0.03,
        )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_protected_abort_drain_timeout_fails_closed(monkeypatch) -> None:
    never = asyncio.Event()

    async def protected_group():
        await never.wait()
        return []

    task = asyncio.create_task(protected_group())
    state = SimpleNamespace(
        aborted=False,
        evaluating=0,
        protected_pendings={task},
        pendings=set(),
    )
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    args = SimpleNamespace(
        use_slime_router=False,
        partial_rollout=True,
        sglang_router_ip="router",
        sglang_router_port=3000,
    )

    with pytest.raises(RuntimeError, match="Protected abort drain timed out"):
        await sglang_rollout.abort(
            args,
            rollout_id=4,
            retry_interval_seconds=0.01,
            timeout_seconds=1.0,
            protected_timeout_seconds=0.01,
        )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --------------------------------------------------------------------------- #
# T9-T11: _dispatch_generate dispatch / lock scope / containment / misuse guard
# --------------------------------------------------------------------------- #
async def test_dispatch_optin_vs_legacy_lock_scope(monkeypatch) -> None:
    """Compat: a legacy function holds the session lock for its whole execution;
    an opt-in function does not (per-request acquisition is transient)."""
    # Legacy: session-level lock held across the entire (multi-step) call.
    legacy_state = _StubState(capacity=1)
    observed: dict[str, bool] = {}

    async def legacy_func(args, sample, sampling_params):
        observed["locked_step0"] = legacy_state.semaphore.locked()
        await asyncio.sleep(0)
        observed["locked_step1"] = legacy_state.semaphore.locked()
        return sample

    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: legacy_func)
    await _dispatch_generate(legacy_state, _StubArgs("x"), _StubSample(), {})
    assert observed == {"locked_step0": True, "locked_step1": True}
    assert not legacy_state.semaphore.locked()

    # Opt-in: no session lock; the function acquires a per-request permit per turn.
    optin_state = _StubState(capacity=1)
    free_between_turns: list[bool] = []

    async def optin_func(args, sample, sampling_params):
        for _ in range(2):
            async with optin_state.semaphore:  # simulate one per-request permit
                pass
            free_between_turns.append(optin_state.semaphore.locked())
        return sample

    optin_func.manages_inference_permit = True
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: optin_func)
    await _dispatch_generate(optin_state, _StubArgs("x"), _StubSample(), {})
    assert free_between_turns == [False, False]
    assert not optin_state.semaphore.locked()


async def test_group_dispatch_barrier_releases_for_completed_partial_samples() -> None:
    args = SimpleNamespace(
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
        group_rm=False,
        sglang_enable_deterministic_inference=False,
    )
    state = SimpleNamespace(aborted=False, opd_manager=None)
    submitted = asyncio.Event()
    group = [_completed_sample(), _completed_sample()]

    with patch("relax.engine.rollout.sglang_rollout.GenerateState", return_value=state):
        result = await asyncio.wait_for(
            generate_and_rm_group(args, group, {}, submitted_event=submitted),
            timeout=1.0,
        )

    assert result == group
    assert submitted.is_set()


async def test_group_dispatch_barrier_releases_when_rollout_already_aborted() -> None:
    args = SimpleNamespace()
    state = SimpleNamespace(aborted=True)
    submitted = asyncio.Event()
    group = [_completed_sample()]

    with patch("relax.engine.rollout.sglang_rollout.GenerateState", return_value=state):
        result = await asyncio.wait_for(
            generate_and_rm_group(args, group, {}, submitted_event=submitted),
            timeout=1.0,
        )

    assert result == group
    assert submitted.is_set()


async def test_uncaught_abort_contained_in_dispatch(monkeypatch) -> None:
    """V4/compat: GenerationAborted raised inside an opt-in function (and NOT
    caught by it) is contained by _dispatch_generate, which returns an ABORTED
    sample instead of letting the exception escape to asyncio.gather."""
    state = _StubState(capacity=1)

    async def optin_func(args, sample, sampling_params):
        raise GenerationAborted()  # simulates post_generate detecting abort, uncaught

    optin_func.manages_inference_permit = True
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: optin_func)

    sample = _StubSample()
    result = await _dispatch_generate(state, _StubArgs("x"), sample, {})
    assert result is sample
    assert result.status == Sample.Status.ABORTED
    assert not state.semaphore.locked()


async def test_misuse_permit_under_session_lock_raises(monkeypatch) -> None:
    """R10 guard: while a legacy function runs (under the session lock) the
    misuse ContextVar is set and the guard raises; for an opt-in function it is
    unset and the guard is a no-op."""
    legacy_state = _StubState(capacity=1)
    legacy_probe: dict[str, bool] = {}

    async def legacy_func(args, sample, sampling_params):
        legacy_probe["flag"] = _holding_session_lock.get()
        try:
            _ensure_not_holding_session_lock()
            legacy_probe["raised"] = False
        except RuntimeError:
            legacy_probe["raised"] = True
        return sample

    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: legacy_func)
    await _dispatch_generate(legacy_state, _StubArgs("x"), _StubSample(), {})
    assert legacy_probe == {"flag": True, "raised": True}

    optin_state = _StubState(capacity=1)
    optin_probe: dict[str, bool] = {}

    async def optin_func(args, sample, sampling_params):
        optin_probe["flag"] = _holding_session_lock.get()
        _ensure_not_holding_session_lock()  # must not raise
        optin_probe["ok"] = True
        return sample

    optin_func.manages_inference_permit = True
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: optin_func)
    await _dispatch_generate(optin_state, _StubArgs("x"), _StubSample(), {})
    assert optin_probe == {"flag": False, "ok": True}

    # ContextVar must not leak out of dispatch.
    assert _holding_session_lock.get() is False


# --------------------------------------------------------------------------- #
# T12-T13: end-to-end deepeyes.generate wiring. Runs the *real* generate loop
# and the *real* _run_inference_step -> state.post_generate -> permit chain;
# only the network `post` is stubbed. This exercises the actual permit scope in
# the shipped multi-turn rollout (not a mechanism mock).
# --------------------------------------------------------------------------- #
class _StubTokenizer:
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [7, 8, 9]

    def decode(self, tokens, skip_special_tokens=False):
        return "decoded"


class _IntegState:
    """Tokenizer-free GenerateState stand-in that reuses the *real* shipped
    permit methods (attached below so production code runs without loading a
    tokenizer/processor)."""

    def __init__(self, capacity: int = 1) -> None:
        self.permit_manager = InferencePermitManager(capacity)
        self.semaphore = self.permit_manager.semaphore
        self.aborted = False
        self.tokenizer = _StubTokenizer()
        self.processor = None
        self.records = {"locked_during_post": [], "locked_during_env": [], "posts": 0}


# Attached outside the class body: a class-level reference would run at import
# time and defeat the HAS_DEPS guard above on the CPU-only runner.
if HAS_DEPS:
    _IntegState.inference_permit = sglang_rollout.GenerateState.inference_permit
    _IntegState.post_generate = sglang_rollout.GenerateState.post_generate


class _FakeEnv:
    """Two-turn fake env; records whether the permit is free during env
    steps."""

    def __init__(self, state, done_at_step, on_step=None) -> None:
        self._state = state
        self._done_at_step = done_at_step
        self._on_step = on_step
        self._steps = 0
        self.turn = 0

    def reset(self):
        pass

    def step(self, response_text):
        self._state.records["locked_during_env"].append(self._state.semaphore.locked())
        self._steps += 1
        if self._on_step is not None:
            self._on_step(self._steps)
        return "observation", self._steps >= self._done_at_step, {}

    def format_observation(self, observation):
        return {"role": "user", "content": observation}

    def close(self):
        pass


def _patch_deepeyes(monkeypatch, de, state, env) -> None:
    async def fake_post(url, payload, headers=None):
        state.records["locked_during_post"].append(state.semaphore.locked())
        state.records["posts"] += 1
        await asyncio.sleep(0)
        return {
            "text": "resp",
            "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 10], [-0.2, 11]]},
        }

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    monkeypatch.setattr(de, "GenerateState", lambda args: state)
    monkeypatch.setattr(de, "_load_env_module", lambda path: SimpleNamespace(build_env=lambda sample, args: env))


def _deepeyes_args() -> SimpleNamespace:
    return SimpleNamespace(
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


async def test_deepeyes_generate_holds_permit_only_during_request(monkeypatch) -> None:
    """V2 (real wiring): the shipped deepeyes rollout holds the permit only
    around each model request and releases it during env/tool execution."""
    import examples.deepeyes.rollout as de

    state = _IntegState(capacity=1)
    env = _FakeEnv(state, done_at_step=2)
    _patch_deepeyes(monkeypatch, de, state, env)

    result = await de.generate(_deepeyes_args(), Sample(prompt="hello world"), {"max_new_tokens": 8})

    assert state.records["posts"] == 2
    assert state.records["locked_during_post"] == [True, True]  # held during each request
    assert state.records["locked_during_env"] == [False, False]  # free during env execution
    assert result.status == Sample.Status.COMPLETED
    assert not state.semaphore.locked()


async def test_deepeyes_generate_aborts_between_turns_without_new_request(monkeypatch) -> None:
    """V4 (real wiring): when abort is signalled between turns, the next turn
    acquires a permit, sees the abort and raises GenerationAborted *before*
    issuing a new request.

    Exactly one request is sent and the loop ends ABORTED with finish_abort
    semantics (the acquire-then-check behavior that a permit without an abort
    check would lose).
    """
    import examples.deepeyes.rollout as de

    state = _IntegState(capacity=1)

    def _abort_after_turn0(step):
        if step == 1:  # abort arrives during the env step following turn 0
            state.aborted = True

    env = _FakeEnv(state, done_at_step=99, on_step=_abort_after_turn0)
    _patch_deepeyes(monkeypatch, de, state, env)

    result = await de.generate(_deepeyes_args(), Sample(prompt="hello world"), {"max_new_tokens": 8})

    assert state.records["posts"] == 1  # turn 1 issued NO new request after abort
    assert result.status == Sample.Status.ABORTED
    assert result.metadata.get("rollout_stop_reason") == "finish_abort"
    assert not state.semaphore.locked()
