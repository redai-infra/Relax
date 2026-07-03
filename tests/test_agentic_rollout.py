# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from relax.agentic.pipeline.runtime import (
    BackendGenerateResult,
    RuntimeGroup,
    _request_envelope_from_sample,
)
from relax.agentic.pipeline.transfer import TransferDomain
from relax.agentic.rollout import AgenticResidentPipeline, _AgenticStepHandle
from relax.agentic.session.service import (
    AgenticSessionShard,
    _decide_ir_release,
    _normalized_chat_request,
    _openai_token_logprobs_payload,
)
from relax.agentic.session.state import RequestKind, SessionForest, check_messages
from relax.utils.types import Sample


def _runtime_args(**overrides):
    base = {
        "agent_command": "python -c 'pass'",
        "agent_cwd": None,
        "agent_env": [],
        "agentic_prepare_pool_size": None,
        "hf_checkpoint": "/tmp/relax-test-model",
        "mm_processor_pool_size": 0,
        "rollout_batch_size": 2,
        "n_samples_per_prompt": 2,
        "over_sampling_batch_size": None,
        "rollout_max_context_len": 4096,
        "rollout_max_response_len": 128,
        "rollout_temperature": 1.0,
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
        "rollout_stop": None,
        "rollout_stop_token_ids": None,
        "rollout_skip_special_tokens": False,
        "group_rm": False,
        "reward_max_concurrency": None,
        "partial_rollout": False,
        "fully_async": False,
        "max_staleness": 0,
        "colocate": True,
        "global_batch_size": 2,
        "num_iters_per_train_update": 1,
    }
    base.update(overrides)
    if base["over_sampling_batch_size"] is None:
        base["over_sampling_batch_size"] = base["rollout_batch_size"]
    return SimpleNamespace(**base)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in str(text)]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _chars(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _forest_with_initial_obs(
    *,
    session_id: str,
    messages: list[dict[str, Any]],
    train_token_delta: list[int],
    rollout_token_delta: list[int],
    rollout_id: int = 0,
    metadata: dict[str, Any] | None = None,
    group_index: int | None = None,
    index: int | None = None,
    label: str | None = None,
    train_metadata: dict[str, Any] | None = None,
):
    forest = SessionForest.create_empty(
        session_id=session_id,
        group_index=group_index,
        index=index,
        label=label,
        train_metadata=train_metadata,
        metadata=metadata,
    )
    initial_obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=rollout_id,
        abort_count=0,
        messages_delta=check_messages(messages),
        train_token_delta=list(train_token_delta),
        rollout_token_delta=list(rollout_token_delta),
    )
    return forest, initial_obs


def _make_chat_test_shard(
    *,
    session_sampling_params: dict | None = None,
):
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = SimpleNamespace(
        partial_rollout=True,
        partial_rollout_max_aborted_count=2,
        fully_async=False,
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        rollout_max_response_len=8,
        rollout_max_context_len=64,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        rollout_seed=1,
    )
    shard.backend = SimpleNamespace(tokenizer=_FakeTokenizer())
    shard._session_records = {}
    shard._session_locks = {}
    shard._evaluating = 0
    shard._terminal_ir_gate_closed = False
    shard._sglang_request_semaphore = None
    shard._sglang_request_limiter = None
    forest, initial_obs = _forest_with_initial_obs(
        session_id="sess-chat",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        train_token_delta=_chars("hello"),
        rollout_token_delta=_chars("hello"),
        rollout_id=0,
        metadata={"template_kwargs": {}},
    )
    record = SimpleNamespace(
        forest=forest,
        next_ir_sequence=0,
        rollout_id=0,
        scope_id="train",
        group_id=None,
        group_generation=None,
        session_seed={"prompt": "hello", "metadata": {"template_kwargs": {}}},
        session_sampling_params=session_sampling_params or {"max_new_tokens": 8},
        resp_state_hash_by_request_id={},
        irs_by_id={},
        ir_queue=deque(),
        active_ir_runner_tasks={},
        pending_chat_waiters={},
        gate_reason=None,
        protected_until_finalize=False,
    )
    shard._session_records["sess-chat"] = record
    shard._session_locks["sess-chat"] = asyncio.Lock()

    async def _ensure_record(**kwargs):
        del kwargs
        return record, {}

    async def _append_observation_if_needed(**kwargs):
        del kwargs
        return initial_obs.state_hash, {}

    shard._ensure_record = _ensure_record
    shard._match_parent_state_hash = lambda **kwargs: (initial_obs.state_hash, [])
    shard._append_observation_if_needed = _append_observation_if_needed
    shard._budget_sampling_params = lambda **kwargs: dict(kwargs["sampling_params"])
    return shard_cls, shard, record, initial_obs


def _sample_group(name: str, *, group_index: int, rollout_id: int, count: int = 1) -> list[Sample]:
    return [
        Sample(
            group_index=group_index,
            index=i,
            session_id=f"{name}-{i}",
            reward=1.0,
            metadata={"start_rollout_id": rollout_id},
        )
        for i in range(count)
    ]


def _pipeline_with_transfer(args):
    runtime = SimpleNamespace(
        args=args,
        rollout_id=0,
        runtime_groups_by_key={},
        interrupted_current_groups=0,
        interrupted_previous_groups=0,
    )
    runtime.require_rollout_id = lambda: runtime.rollout_id
    runtime.resident_group_keys = lambda: set(runtime.runtime_groups_by_key)
    runtime.accounting_snapshot = lambda: {
        "resident_groups": len(runtime.runtime_groups_by_key),
        "interrupted_current_groups": runtime.interrupted_current_groups,
        "interrupted_previous_groups": runtime.interrupted_previous_groups,
    }
    runtime.interrupted_group_count_for_step = lambda *, rollout_id, previous: (
        runtime.interrupted_previous_groups if previous else runtime.interrupted_current_groups
    )

    async def _refresh_interrupted_close_accounting() -> dict[str, int]:
        return {
            "interrupted_current_groups": runtime.interrupted_current_groups,
            "interrupted_previous_groups": runtime.interrupted_previous_groups,
        }

    runtime.refresh_interrupted_close_accounting = _refresh_interrupted_close_accounting
    pipeline = AgenticResidentPipeline()
    pipeline.runtime_domain = runtime
    pipeline.prepare_domain = SimpleNamespace(accounting_snapshot=lambda: {"ready_groups": 0})
    pipeline.reward_domain = SimpleNamespace(
        accounting_snapshot=lambda: {"waiting_groups": 0, "completed_groups": 0, "ready_groups": 0},
        resident_group_keys=lambda: set(),
    )
    pipeline.transfer_domain = TransferDomain(args=args, data_system_client=None)
    return pipeline


def _set_runtime_resident_groups(pipeline, count: int, *, rollout_id: int = 0) -> None:
    pipeline.runtime_domain.runtime_groups_by_key = {
        (("resident", idx),): RuntimeGroup(
            group_key=(("resident", idx),),
            expected_count=1,
            admission_rollout_id=rollout_id,
        )
        for idx in range(count)
    }


def test_session_forest_build_sample_and_request_envelope() -> None:
    forest, initial_obs = _forest_with_initial_obs(
        session_id="sess-build",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        train_token_delta=_chars("hello"),
        rollout_token_delta=_chars("hello"),
        rollout_id=11,
        group_index=3,
        index=7,
        label="lab",
        train_metadata={"loss": "grpo"},
        metadata={"seed_stage": "bootstrap"},
    )
    response_kwargs = {
        "parent_state_hash": initial_obs.state_hash,
        "rollout_id": 11,
        "abort_count": 0,
        "messages_delta": [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]}],
        "train_token_delta": _chars("ok"),
        "rollout_token_delta": _chars("ok"),
        "logprob_delta": [-0.1, -0.2],
        "status": "completed",
        "reward": {"score": 0.5},
        "export_metadata_patch": {"request_id": "req-build", "base_state_hash": initial_obs.state_hash},
    }
    leaf = forest.append_resp(**response_kwargs)
    duplicate_leaf = forest.append_resp(**response_kwargs)
    assert duplicate_leaf.state_hash == leaf.state_hash
    assert forest.export_leaf_hashes() == [leaf.state_hash]
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_FakeTokenizer())
    assert (sample.prompt, sample.response, sample.group_index, sample.index) == ("hello", "ok", 3, 7)
    assert sample.train_metadata == {"loss": "grpo"}
    assert sample.metadata["agentic_trace"]["turn_count"] == 1
    envelope = _request_envelope_from_sample(sample, rollout_id=9, sampling_params={"temperature": 0.2})
    assert (envelope.rollout_id, envelope.session_id, envelope.seed.train_metadata) == (
        9,
        "sess-build",
        {"loss": "grpo"},
    )
    with pytest.raises(ValueError, match="group_index"):
        _request_envelope_from_sample(Sample(index=3, prompt="bad"), rollout_id=9)


def test_session_shard_prepare_gate_activation_and_logprobs() -> None:
    shard_cls, shard, record, _initial_obs = _make_chat_test_shard()
    record.group_id = "group-prepare"
    record.group_generation = 3
    record.scope_id = "train"
    record.gate_reason = "prepare"
    backend_calls = {"count": 0}
    backend_return_logprobs = []

    async def _generate(**kwargs):
        backend_return_logprobs.append(kwargs["return_logprob"])
        backend_calls["count"] += 1
        return BackendGenerateResult(
            new_tokens=_chars("ok"), new_log_probs=[-0.1, -0.2], finish_type="stop", meta_info={}, elapsed=0.1
        )

    shard.backend.generate = _generate

    async def _run():
        chat_task = asyncio.create_task(
            shard_cls.chat(
                shard,
                session_id="sess-chat",
                messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                tools=[],
                chat_template_kwargs=None,
                temperature=None,
                top_p=None,
                max_completion_tokens=None,
                stop=None,
                seed=None,
                logprobs=False,
            )
        )
        for _ in range(20):
            await asyncio.sleep(0)
            if record.ir_queue:
                break
        assert backend_calls["count"] == 0
        assert await shard_cls.prepare_group_status(shard, scope_id="train") == [
            {"group_id": "group-prepare", "group_generation": 3, "total_sessions": 1, "ready_sessions": 1}
        ]
        activation = await shard_cls.activate_group_sessions(
            shard,
            scope_id="train",
            groups=[{"group_id": "group-prepare", "group_generation": 3}],
            rollout_id=7,
        )
        assert activation == {"activated_sessions": 1, "started_sessions": 1}
        return await chat_task

    payload = asyncio.run(_run())
    assert payload["message"]["content"] == "ok"
    assert payload["logprobs"] is None
    assert backend_return_logprobs == [True]
    assert backend_calls["count"] == 1
    assert record.rollout_id == 7


def test_chat_request_validation_context_limit_and_logprob_payload() -> None:
    assert (
        _normalized_chat_request({"messages": [{"role": "user", "content": "hello"}], "logprobs": True})["logprobs"]
        is True
    )
    with pytest.raises(HTTPException, match="logprobs must be a boolean"):
        _normalized_chat_request({"messages": [{"role": "user", "content": "hello"}], "logprobs": "true"})
    with pytest.raises(HTTPException, match="top_logprobs is not supported"):
        _normalized_chat_request({"messages": [{"role": "user", "content": "hello"}], "top_logprobs": 1})
    assert (
        _openai_token_logprobs_payload(tokenizer=_FakeTokenizer(), token_ids=_chars("ok"), token_logprobs=[-0.1])[
            "content"
        ][1]["logprob"]
        == -9999.0
    )

    shard_cls, shard, record, _initial_obs = _make_chat_test_shard(session_sampling_params={"max_new_tokens": 0})
    shard.backend.generate = lambda **kwargs: (_ for _ in ()).throw(AssertionError("backend should not run"))

    async def _run() -> dict[str, Any]:
        return await shard_cls.chat(
            shard,
            session_id="sess-chat",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            tools=[],
            chat_template_kwargs=None,
            temperature=None,
            top_p=None,
            max_completion_tokens=None,
            stop=None,
            seed=None,
        )

    payload = asyncio.run(_run())
    assert payload["_http_status"] == 400
    assert record.pending_chat_waiters == record.irs_by_id == record.active_ir_runner_tasks == {}


@pytest.mark.parametrize(
    ("fully_async", "initial_abort_count", "expected_kind", "expected_protected", "expected_sibling_kind"),
    [
        (False, 0, RequestKind.RESUMED, False, RequestKind.FRESH),
        (False, 1, RequestKind.PROTECTED, True, RequestKind.PROTECTED),
        (True, 1, RequestKind.RESUMED, False, RequestKind.FRESH),
    ],
)
def test_ir_gate_and_requeue_policy(
    fully_async: bool,
    initial_abort_count: int,
    expected_kind: RequestKind,
    expected_protected: bool,
    expected_sibling_kind: RequestKind,
) -> None:
    shard_cls, shard, record, _initial_obs = _make_chat_test_shard()
    shard.args.fully_async = fully_async
    shard._release_ir_locked = lambda record, ir_id: record.irs_by_id.pop(ir_id, None)
    shard._enqueue_ir_locked = lambda record, ir: record.ir_queue.append(ir.request_id)
    ir = SimpleNamespace(request_id="req-1", abort_count=initial_abort_count, kind=None, pending_status=None)
    sibling = SimpleNamespace(request_id="req-sibling", kind=RequestKind.FRESH)
    record.irs_by_id = {ir.request_id: ir, sibling.request_id: sibling}
    assert _decide_ir_release(record=record).allow is True
    record.gate_reason = "prepare"
    assert _decide_ir_release(record=record).blocked_reason == "prepare_gate"
    record.gate_reason = None

    shard_cls._requeue_aborted_ir_locked(shard, record=record, ir_id=ir.request_id, ir=ir)
    assert (ir.kind, record.gate_reason, record.protected_until_finalize, sibling.kind) == (
        expected_kind,
        "partial_resume",
        expected_protected,
        expected_sibling_kind,
    )


def test_admission_quota_prepare_isolation_and_resident_tail_carry() -> None:
    args = _runtime_args(
        partial_rollout=True, rollout_batch_size=32, over_sampling_batch_size=48, n_samples_per_prompt=1
    )
    pipeline = _pipeline_with_transfer(args)
    pipeline.transfer_domain.rebind_step(rollout_id=1)
    pipeline.transfer_domain.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=32)
    snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    assert snapshot["current_partition_quota"] == 32
    assert pipeline._current_window_admission_counts(resident_group_count=0, transfer_snapshot=snapshot)[1:] == (
        0,
        0,
        48,
    )

    pipeline.prepare_domain = SimpleNamespace(accounting_snapshot=lambda: {"ready_groups": 99})
    assert pipeline._current_window_admission_counts(resident_group_count=0, transfer_snapshot=snapshot)[1:] == (
        0,
        0,
        48,
    )

    pipeline.transfer_domain._committed_current_group_count = 32
    snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    _set_runtime_resident_groups(pipeline, 16)
    step_handle = _AgenticStepHandle(rollout_id=1, required_group_count=32, terminal_step=False)
    assert pipeline._close_status(step_handle) is None
    assert pipeline._current_window_admission_counts(
        resident_group_count=pipeline.resident_group_count, transfer_snapshot=snapshot
    )[1:] == (16, 48, 0)

    pipeline.transfer_domain.rebind_step(rollout_id=2)
    pipeline.runtime_domain.rollout_id = 2
    pipeline.transfer_domain.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=32)
    snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    assert pipeline._current_window_admission_counts(
        resident_group_count=pipeline.resident_group_count, transfer_snapshot=snapshot
    )[1:] == (16, 16, 32)


@pytest.mark.parametrize(
    ("fully_async", "terminal_step", "resident_groups", "interrupted_current", "expected_finish", "status"),
    [
        (True, False, 2, 0, False, "committed_target"),
        (True, False, 2, 2, True, None),
        (False, False, 0, 0, False, "committed_target"),
        (True, True, 0, 2, False, "committed_target"),
    ],
)
def test_finish_eligibility_interrupted_current_policy(
    fully_async: bool,
    terminal_step: bool,
    resident_groups: int,
    interrupted_current: int,
    expected_finish: bool,
    status: str | None,
) -> None:
    args = _runtime_args(fully_async=fully_async, rollout_batch_size=2, n_samples_per_prompt=1)
    pipeline = _pipeline_with_transfer(args)
    pipeline.transfer_domain.rebind_step(rollout_id=3)
    pipeline.runtime_domain.rollout_id = 3
    if resident_groups:
        _set_runtime_resident_groups(pipeline, resident_groups)
    pipeline.transfer_domain.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=2)
    pipeline.runtime_domain.interrupted_current_groups = interrupted_current
    step_handle = _AgenticStepHandle(rollout_id=3, required_group_count=2, terminal_step=terminal_step)
    assert pipeline._close_status(step_handle) == status
    assert (status is None) is expected_finish


def test_transfer_fifo_routes_slots_by_arrival_ignoring_metadata(monkeypatch) -> None:
    recorded: list[tuple[int, list[str]]] = []
    recorded_is_last: list[tuple[int, bool]] = []

    async def _fake_transfer(args, batch_samples, batch_count, rollout_id, data_system_client, is_last=False):
        del args, batch_count, data_system_client
        recorded.append((int(rollout_id), [s.metadata["label"] for group in batch_samples for s in group]))
        recorded_is_last.append((int(rollout_id), bool(is_last)))

    monkeypatch.setattr("relax.agentic.pipeline.transfer._transfer_batch_to_data_system", _fake_transfer)
    args = _runtime_args(fully_async=True, rollout_batch_size=2, n_samples_per_prompt=1)
    transfer = TransferDomain(args=args, data_system_client=object())
    transfer.rebind_step(rollout_id=3)
    transfer.configure_transfer_quota(previous_partition_quota=2, current_partition_quota=2)
    labels = ["current-first", "old-second", "old-third", "current-fourth"]
    for idx, label in enumerate(labels):
        group = _sample_group(label, group_index=idx, rollout_id=3 if "current" in label else 2)
        for sample in group:
            sample.metadata.update(label=label, admission_rollout_id=3 if "current" in label else 2)
        transfer.enqueue_ready_groups([group])
    released_groups, released_count = asyncio.run(transfer.drain_ready_group_payloads())
    assert released_count == 4
    assert len(released_groups) == 4
    asyncio.run(transfer.wait_for_pending_transfers())
    assert recorded == [(2, ["current-first", "old-second"]), (3, ["old-third", "current-fourth"])]
    # Both partitions are fully filled this step (prev quota=2 backfilled, current
    # quota=2 met), so each partition's batch is marked is_last for end-of-stream.
    assert recorded_is_last == [(2, True), (3, True)]


def test_transfer_is_last_only_when_partition_target_met(monkeypatch) -> None:
    # Current partition NOT fully filled this step (quota=4, only 2 committed) -> its
    # batch is NOT marked is_last; the tail is backfilled next step (as the previous
    # partition), which is where is_last fires. Guards against premature end-of-stream.
    recorded_is_last: list[tuple[int, bool]] = []

    async def _fake_transfer(args, batch_samples, batch_count, rollout_id, data_system_client, is_last=False):
        del args, batch_samples, batch_count, data_system_client
        recorded_is_last.append((int(rollout_id), bool(is_last)))

    monkeypatch.setattr("relax.agentic.pipeline.transfer._transfer_batch_to_data_system", _fake_transfer)
    args = _runtime_args(fully_async=True, rollout_batch_size=4, over_sampling_batch_size=4, n_samples_per_prompt=1)
    transfer = TransferDomain(args=args, data_system_client=object())
    transfer.rebind_step(rollout_id=5)
    transfer.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=4)
    # Only 2 groups available this step -> current partition under-filled (deficit 2).
    for idx in range(2):
        transfer.enqueue_ready_groups([_sample_group(f"g{idx}", group_index=idx, rollout_id=5)])
    asyncio.run(transfer.drain_ready_group_payloads())
    asyncio.run(transfer.wait_for_pending_transfers())
    # No is_last: the current partition will be completed next step via backfill.
    assert recorded_is_last == [(5, False)]


def test_oversampling_surplus_retained_not_dropped(monkeypatch) -> None:
    # When over_sampling_batch_size > rollout_batch_size, completed groups beyond the
    # commit target (current_partition_quota) stay in ready_group_buffer (NOT dropped),
    # so the next step's current partition can re-commit them. We do NOT account them
    # separately — the next-step previous-partition debt is sized from the deficit, not
    # from the buffer (see test_previous_quota_is_current_partition_deficit).
    async def _fake_transfer(*args, **kwargs):
        del args, kwargs

    monkeypatch.setattr("relax.agentic.pipeline.transfer._transfer_batch_to_data_system", _fake_transfer)
    args = _runtime_args(fully_async=True, rollout_batch_size=2, over_sampling_batch_size=4, n_samples_per_prompt=1)
    transfer = TransferDomain(args=args, data_system_client=object())
    transfer.rebind_step(rollout_id=3)
    transfer.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=2)
    for idx in range(4):  # over-sample: 4 ready groups, commit target only 2
        transfer.enqueue_ready_groups([_sample_group(f"g{idx}", group_index=idx, rollout_id=3)])
    _released_groups, released_count = asyncio.run(transfer.drain_ready_group_payloads())

    assert released_count == 2  # only current_partition_quota committed
    assert len(transfer.ready_group_buffer) == 2  # surplus preserved, not dropped


def test_previous_quota_is_current_partition_deficit() -> None:
    # Core fix: previous_partition_quota (next-step backfill debt) equals how many groups
    # the previous step left short of its current-partition target (rollout_batch_size),
    # i.e. rollout_batch_size - committed_current. It is INDEPENDENT of any over-sampling
    # surplus still resident in the transfer ready buffer.
    args = _runtime_args(fully_async=True, rollout_batch_size=4, over_sampling_batch_size=6, n_samples_per_prompt=1)
    pipeline = _pipeline_with_transfer(args)

    # Case A: previous step met its target (committed_current == rollout_batch_size).
    # Even with surplus left in the buffer, the deficit (and thus next-step debt) is 0.
    pipeline.transfer_domain.rebind_step(rollout_id=0)
    pipeline.transfer_domain.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=4)
    pipeline.transfer_domain._committed_current_group_count = 4
    for idx in range(2):  # 2 surplus completed groups parked in the buffer
        pipeline.transfer_domain.enqueue_ready_groups([_sample_group(f"s{idx}", group_index=idx, rollout_id=0)])
    end_snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    required = 4  # required_group_count == current_partition_quota
    deficit = max(required - end_snapshot["committed_current_groups"], 0)
    assert deficit == 0  # met target → no debt, regardless of the 2 buffered surplus groups

    # Case B: previous step fell short by 1 (an aborted group never came back).
    pipeline.transfer_domain.rebind_step(rollout_id=1)
    pipeline.transfer_domain.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=4)
    pipeline.transfer_domain._committed_current_group_count = 3
    end_snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    deficit = max(required - end_snapshot["committed_current_groups"], 0)
    assert deficit == 1  # short by exactly 1 → next step backfills 1


def test_deficit_quota_keeps_admission_ledger_consistent() -> None:
    # With the deficit-sized previous quota, the admission ledger stays self-consistent
    # even when over-sampling surplus is resident: surplus is folded into the current
    # window (resident_current_window_groups), not the previous debt, and no RuntimeError
    # invariant fires in _current_window_admission_counts.
    args = _runtime_args(fully_async=True, rollout_batch_size=4, over_sampling_batch_size=6, n_samples_per_prompt=1)
    pipeline = _pipeline_with_transfer(args)
    pipeline.transfer_domain.rebind_step(rollout_id=1)
    pipeline.runtime_domain.rollout_id = 1

    # Previous step left a deficit of 1; pipeline holds 1 aborted group (runtime) plus
    # 2 over-sampling surplus groups (transfer ready buffer).
    pipeline._last_step_current_deficit = 1
    _set_runtime_resident_groups(pipeline, 1, rollout_id=0)
    for idx in range(2):
        pipeline.transfer_domain.enqueue_ready_groups([_sample_group(f"surplus{idx}", group_index=idx, rollout_id=1)])

    previous_partition_quota = pipeline._last_step_current_deficit
    pipeline.transfer_domain.configure_transfer_quota(
        previous_partition_quota=previous_partition_quota, current_partition_quota=4
    )
    resident_group_count = pipeline.resident_group_count
    assert resident_group_count == 3  # 1 abort + 2 surplus

    snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    remaining_previous_debt, resident_current_window_groups, _, current_window_slack = (
        pipeline._current_window_admission_counts(
            resident_group_count=resident_group_count, transfer_snapshot=snapshot
        )
    )
    assert remaining_previous_debt == 1  # only the genuine deficit is debt
    assert resident_current_window_groups == 2  # the 2 surplus folded into current window
    assert current_window_slack >= 0
    pipeline._assert_resident_group_count_invariant(context="test_deficit_ledger")
