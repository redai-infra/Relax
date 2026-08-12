# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for ``Rollout.can_do_update_weight_for_async`` transaction safety.

These exercise the real method on a shell instance (``object.__new__``) with the
collaborators it touches (``rollout_manager`` remote calls,
``_async_check_production_for_update_weight``) stubbed, so we test the
try/finally control flow in isolation -- without Ray Serve / GPUs.

Each pause attempt has a unique transaction and completion event. Compensating
end requests may cancel only their matching transaction and cannot resume a
newer pause.
"""

import asyncio
import logging

import pytest
from fastapi import HTTPException
from ray.exceptions import RayActorError

from relax.components import rollout as rollout_module
from relax.components.rollout import Rollout as RolloutDeployment


# ``Rollout`` is wrapped by ``@serve.deployment`` / ``@serve.ingress`` -- reach
# the underlying class so we can build a plain shell instance.
Rollout = RolloutDeployment.func_or_class


class _RemoteStub:
    """Mimics a Ray actor-handle method: ``.remote(...)`` returns an awaitable
    (or raises synchronously to simulate a dead handle)."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


async def _ok(*_args, **_kwargs):
    return None


def _raise_dead(*_args, **_kwargs):
    raise RayActorError()


class _ManagerStub:
    def __init__(self, set_weight_updating_fn=None):
        self.health_monitoring_pause = _RemoteStub(lambda *a, **k: _ok())
        self.health_monitoring_resume = _RemoteStub(lambda *a, **k: _ok())
        self.set_weight_updating = _RemoteStub(set_weight_updating_fn or (lambda *a, **k: _ok()))


def _make_rollout(*, can_update: bool, manager: _ManagerStub) -> "Rollout":
    shell = object.__new__(Rollout)
    # ``_logger`` is a lazy read-only property on Base; prime its backing field.
    shell._logger_instance = logging.getLogger("test.rollout")
    shell.step = 0
    shell.status = "running"
    shell._weight_update_transactions = {}
    shell._completed_weight_update_transaction_sequences = {}
    shell._weight_update_session_last_seen = {}
    shell._weight_update_prepare_lock = asyncio.Lock()
    shell._weight_update_idle = asyncio.Event()
    shell._weight_update_idle.set()
    shell._active_weight_update_transaction_id = None
    shell.rollout_manager = manager

    async def _check(_step):
        return can_update

    shell._async_check_production_for_update_weight = _check  # type: ignore[method-assign]
    return shell


def _transaction_id(sequence: int, session: str = "0" * 32) -> str:
    return f"relax-v1:{session}:{sequence}"


def test_can_do_update_weight_normal_path_sets_event_and_returns_1():
    shell = _make_rollout(can_update=True, manager=_ManagerStub())
    result = asyncio.run(shell.can_do_update_weight_for_async())
    assert result == 1
    assert shell.status == "paused"
    assert shell._weight_update_transactions["legacy"].ready.is_set()


def test_can_do_update_weight_cannot_update_returns_0():
    shell = _make_rollout(can_update=False, manager=_ManagerStub())
    result = asyncio.run(shell.can_do_update_weight_for_async())
    assert result == 0
    assert shell.status == "running"
    assert "legacy" not in shell._weight_update_transactions


def test_not_ready_transaction_can_be_polled_again_with_same_id():
    async def _can_update(_step):
        return True

    async def _scenario():
        shell = _make_rollout(can_update=False, manager=_ManagerStub())

        transaction_id = _transaction_id(7)
        assert await shell.can_do_update_weight_for_async(transaction_id) == 0
        shell._async_check_production_for_update_weight = _can_update  # type: ignore[method-assign]
        assert await shell.can_do_update_weight_for_async(transaction_id) == 1

        await shell.end_update_weight(transaction_id)

    asyncio.run(_scenario())


def test_can_do_update_weight_dead_engine_releases_gate_then_reraises():
    manager = _ManagerStub(set_weight_updating_fn=_raise_dead)
    shell = _make_rollout(can_update=True, manager=manager)

    with pytest.raises(RayActorError):
        asyncio.run(shell.can_do_update_weight_for_async())

    assert shell._weight_update_transactions["legacy"].recovery_required
    assert shell.status == "paused"


def test_end_update_weight_does_not_block_after_failed_can_do():
    async def _scenario():
        manager = _ManagerStub(set_weight_updating_fn=_raise_dead)
        shell = _make_rollout(can_update=True, manager=manager)

        with pytest.raises(RayActorError):
            await shell.can_do_update_weight_for_async()

        assert shell._weight_update_transactions["legacy"].recovery_required

        shell.rollout_manager = _ManagerStub()
        await asyncio.wait_for(shell.end_update_weight(), timeout=1)
        assert shell.status == "running"
        assert "legacy" not in shell._weight_update_transactions

    asyncio.run(_scenario())


def test_timed_out_pause_is_cancelled_before_it_can_pause_rollout():
    async def _scenario():
        production_check_started = asyncio.Event()
        finish_production_check = asyncio.Event()
        state_updates = []

        async def _check(_step):
            production_check_started.set()
            await finish_production_check.wait()
            return True

        async def _set_weight_updating(value):
            state_updates.append(value)

        shell = _make_rollout(can_update=True, manager=_ManagerStub(_set_weight_updating))
        shell._async_check_production_for_update_weight = _check  # type: ignore[method-assign]

        transaction_id = _transaction_id(7)
        pause_task = asyncio.create_task(shell.can_do_update_weight_for_async(transaction_id))
        await production_check_started.wait()
        assert not shell._weight_update_transactions[transaction_id].ready.is_set()

        resume_task = asyncio.create_task(shell.end_update_weight(transaction_id))
        await asyncio.sleep(0)
        assert not resume_task.done()

        finish_production_check.set()
        assert await pause_task == 0
        await asyncio.wait_for(resume_task, timeout=1)

        assert shell.status == "running"
        assert state_updates == []

    asyncio.run(_scenario())


def test_cancelled_pause_waits_for_rollback_before_propagating_cancellation():
    async def _scenario():
        pause_started = asyncio.Event()
        rollback_started = asyncio.Event()
        release_rollback = asyncio.Event()
        pause_blocker = asyncio.Event()
        state_updates = []
        health_updates = []

        async def _pause_health_monitoring():
            pause_started.set()
            await pause_blocker.wait()

        async def _resume_health_monitoring():
            health_updates.append("resumed")

        async def _set_weight_updating(value):
            state_updates.append(value)
            if not value:
                rollback_started.set()
                await release_rollback.wait()

        manager = _ManagerStub(_set_weight_updating)
        manager.health_monitoring_pause = _RemoteStub(_pause_health_monitoring)
        manager.health_monitoring_resume = _RemoteStub(_resume_health_monitoring)
        shell = _make_rollout(can_update=True, manager=manager)

        transaction_id = _transaction_id(7)
        pause_task = asyncio.create_task(shell.can_do_update_weight_for_async(transaction_id))
        await pause_started.wait()
        pause_task.cancel()
        await rollback_started.wait()

        pause_task.cancel()
        await asyncio.sleep(0)
        assert not pause_task.done()

        release_rollback.set()
        with pytest.raises(asyncio.CancelledError):
            await pause_task

        assert shell.status == "running"
        assert shell._active_weight_update_transaction_id is None
        assert state_updates == [False]
        assert health_updates == ["resumed"]
        assert transaction_id not in shell._weight_update_transactions

        await shell.end_update_weight(transaction_id)
        assert shell.status == "running"

    asyncio.run(_scenario())


def test_failed_prepare_rollback_is_retried_by_compensating_end():
    async def _scenario():
        state_updates = []
        outcomes = [
            RuntimeError("prepare failed"),
            RuntimeError("rollback failed"),
            RuntimeError("retry failed"),
            None,
        ]

        async def _set_weight_updating(value):
            state_updates.append(value)
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome

        shell = _make_rollout(can_update=True, manager=_ManagerStub(_set_weight_updating))

        with pytest.raises(RuntimeError, match="prepare failed"):
            await shell.can_do_update_weight_for_async(_transaction_id(7))

        transaction = shell._weight_update_transactions[_transaction_id(7)]
        assert transaction.recovery_required
        assert shell.status == "paused"

        with pytest.raises(RuntimeError, match="Failed to recover weight-update transaction"):
            await shell.end_update_weight(_transaction_id(7))

        assert transaction.recovery_required
        assert shell.status == "paused"

        await shell.end_update_weight(_transaction_id(7))
        assert not transaction.recovery_required
        assert shell.status == "running"
        assert state_updates == [True, False, False, False]

    asyncio.run(_scenario())


def test_end_before_pause_leaves_transaction_tombstone():
    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())

        transaction_id = _transaction_id(7)
        await shell.end_update_weight(transaction_id)
        result = await shell.can_do_update_weight_for_async(transaction_id)

        assert result == 0
        assert shell.status == "running"

    asyncio.run(_scenario())


def test_same_rollout_can_retry_with_a_new_transaction():
    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())

        first_transaction_id = _transaction_id(1)
        second_transaction_id = _transaction_id(2)
        await shell.end_update_weight(first_transaction_id)
        assert await shell.can_do_update_weight_for_async(first_transaction_id) == 0
        assert await shell.can_do_update_weight_for_async(second_transaction_id) == 1

        await shell.end_update_weight(second_transaction_id)
        assert shell.status == "running"

    asyncio.run(_scenario())


def test_late_old_end_does_not_resume_new_transaction():
    async def _scenario():
        state_updates = []

        async def _set_weight_updating(value):
            state_updates.append(value)

        shell = _make_rollout(can_update=True, manager=_ManagerStub(_set_weight_updating))
        first_transaction_id = _transaction_id(7)
        second_transaction_id = _transaction_id(8)
        assert await shell.can_do_update_weight_for_async(first_transaction_id) == 1
        await shell.end_update_weight(first_transaction_id)
        assert await shell.can_do_update_weight_for_async(second_transaction_id) == 1

        await shell.end_update_weight(first_transaction_id)

        assert shell.status == "paused"
        assert shell._active_weight_update_transaction_id == second_transaction_id
        assert state_updates == [True, False, True]

        await shell.end_update_weight(second_transaction_id)
        assert state_updates == [True, False, True, False]

    asyncio.run(_scenario())


def test_untagged_legacy_end_cannot_resume_uuid_transaction():
    async def _scenario():
        state_updates = []

        async def _set_weight_updating(value):
            state_updates.append(value)

        shell = _make_rollout(can_update=True, manager=_ManagerStub(_set_weight_updating))
        transaction_id = _transaction_id(0)
        assert await shell.can_do_update_weight_for_async(transaction_id) == 1

        await shell.end_update_weight()

        assert shell.status == "paused"
        assert shell._active_weight_update_transaction_id == transaction_id
        assert state_updates == [True]

        await shell.end_update_weight(transaction_id)
        assert state_updates == [True, False]

    asyncio.run(_scenario())


def test_duplicate_end_handlers_are_serialized_per_transaction():
    async def _scenario():
        first_resume_started = asyncio.Event()
        release_first_resume = asyncio.Event()
        resume_calls = 0

        async def _set_weight_updating(value):
            nonlocal resume_calls
            if value:
                return
            resume_calls += 1
            first_resume_started.set()
            await release_first_resume.wait()

        shell = _make_rollout(can_update=True, manager=_ManagerStub(_set_weight_updating))
        first_transaction_id = _transaction_id(7)
        assert await shell.can_do_update_weight_for_async(first_transaction_id) == 1

        first_end = asyncio.create_task(shell.end_update_weight(first_transaction_id))
        await first_resume_started.wait()
        duplicate_end = asyncio.create_task(shell.end_update_weight(first_transaction_id))
        await asyncio.sleep(0)

        assert resume_calls == 1
        assert not duplicate_end.done()

        release_first_resume.set()
        await asyncio.gather(first_end, duplicate_end)
        assert resume_calls == 1
        assert shell.status == "running"

        second_transaction_id = _transaction_id(8)
        assert await shell.can_do_update_weight_for_async(second_transaction_id) == 1
        await shell.end_update_weight(second_transaction_id)

    asyncio.run(_scenario())


def test_completed_transaction_replay_cannot_pause_again():
    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        transaction_id = _transaction_id(7)
        assert await shell.can_do_update_weight_for_async(transaction_id) == 1
        await shell.end_update_weight(transaction_id)

        assert await shell.can_do_update_weight_for_async(transaction_id) == 0
        assert shell.status == "running"

    asyncio.run(_scenario())


def test_queued_older_transaction_is_rechecked_after_newer_transaction_completes():
    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        newer_transaction_id = _transaction_id(8)
        older_transaction_id = _transaction_id(7)
        assert await shell.can_do_update_weight_for_async(newer_transaction_id) == 1

        older_pause = asyncio.create_task(shell.can_do_update_weight_for_async(older_transaction_id))
        await asyncio.sleep(0)
        assert not older_pause.done()

        await shell.end_update_weight(newer_transaction_id)
        assert await asyncio.wait_for(older_pause, timeout=1) == 0
        assert shell.status == "running"
        assert shell._active_weight_update_transaction_id is None

    asyncio.run(_scenario())


def test_older_transaction_is_rechecked_after_production_check():
    async def _scenario():
        production_check_started = asyncio.Event()
        release_production_check = asyncio.Event()

        async def _check(_step):
            production_check_started.set()
            await release_production_check.wait()
            return True

        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        shell._async_check_production_for_update_weight = _check  # type: ignore[method-assign]
        older_pause = asyncio.create_task(shell.can_do_update_weight_for_async(_transaction_id(7)))
        await production_check_started.wait()

        await shell.end_update_weight(_transaction_id(8))
        release_production_check.set()

        assert await older_pause == 0
        assert shell.status == "running"
        assert shell._active_weight_update_transaction_id is None

    asyncio.run(_scenario())


def test_superseded_transaction_rolls_back_remote_pause_before_returning():
    async def _scenario():
        remote_pause_started = asyncio.Event()
        release_remote_pause = asyncio.Event()
        state_updates = []

        async def _set_weight_updating(value):
            state_updates.append(value)
            if value:
                remote_pause_started.set()
                await release_remote_pause.wait()

        shell = _make_rollout(can_update=True, manager=_ManagerStub(_set_weight_updating))
        older_pause = asyncio.create_task(shell.can_do_update_weight_for_async(_transaction_id(7)))
        await remote_pause_started.wait()

        await shell.end_update_weight(_transaction_id(8))
        release_remote_pause.set()

        assert await older_pause == 0
        assert state_updates == [True, False]
        assert shell.status == "running"
        assert shell._active_weight_update_transaction_id is None

    asyncio.run(_scenario())


def test_legacy_untagged_handshake_can_run_multiple_cycles():
    async def _scenario():
        state_updates = []

        async def _set_weight_updating(value):
            state_updates.append(value)

        shell = _make_rollout(can_update=True, manager=_ManagerStub(_set_weight_updating))

        assert await shell.can_do_update_weight_for_async() == 1
        await shell.end_update_weight()
        assert "legacy" not in shell._weight_update_transactions

        assert await shell.can_do_update_weight_for_async() == 1
        await shell.end_update_weight()
        assert state_updates == [True, False, True, False]

    asyncio.run(_scenario())


def test_legacy_end_before_pause_cancels_one_late_pause_without_blocking_future_cycles():
    async def _scenario():
        state_updates = []

        async def _set_weight_updating(value):
            state_updates.append(value)

        shell = _make_rollout(can_update=True, manager=_ManagerStub(_set_weight_updating))

        await shell.end_update_weight()
        await shell.end_update_weight()
        assert shell._weight_update_transactions["legacy"].cancel_requested

        assert await shell.can_do_update_weight_for_async() == 0
        assert shell.status == "running"
        assert shell._active_weight_update_transaction_id is None
        assert state_updates == []

        assert await shell.can_do_update_weight_for_async() == 1
        await shell.end_update_weight()
        assert state_updates == [True, False]

    asyncio.run(_scenario())


def test_completed_transaction_tombstones_are_bounded():
    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        for transaction_index in range(1025):
            await shell.end_update_weight(_transaction_id(transaction_index))

        assert len(shell._weight_update_transactions) == 1024
        assert _transaction_id(0) not in shell._weight_update_transactions
        assert shell._weight_update_transactions[_transaction_id(1024)].completed

        assert await shell.can_do_update_weight_for_async(_transaction_id(0)) == 0
        assert shell.status == "running"

    asyncio.run(_scenario())


def test_completed_transaction_session_watermarks_are_bounded():
    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        for session_index in range(1100):
            session_id = f"{session_index:032x}"
            if session_index < 1024:
                await shell.end_update_weight(_transaction_id(0, session_id))
            else:
                with pytest.raises(HTTPException) as error:
                    await shell.end_update_weight(_transaction_id(0, session_id))
                assert error.value.status_code == 429

        assert len(shell._weight_update_transactions) == 1024
        assert len(shell._completed_weight_update_transaction_sequences) == 1024
        assert f"{0:032x}" in shell._completed_weight_update_transaction_sequences
        assert f"{1099:032x}" not in shell._completed_weight_update_transaction_sequences

    asyncio.run(_scenario())


def test_expired_completed_sessions_are_reclaimed_at_capacity(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(rollout_module.time, "monotonic", lambda: clock[0])

    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        for session_index in range(1024):
            await shell.end_update_weight(_transaction_id(0, f"{session_index:032x}"))

        clock[0] = rollout_module._WEIGHT_UPDATE_SESSION_RETENTION_SECONDS + 1
        new_session_id = f"{1024:032x}"
        await shell.end_update_weight(_transaction_id(0, new_session_id))

        assert len(shell._weight_update_transactions) == 1024
        assert len(shell._completed_weight_update_transaction_sequences) == 1024
        assert f"{0:032x}" not in shell._completed_weight_update_transaction_sequences
        assert new_session_id in shell._completed_weight_update_transaction_sequences

    asyncio.run(_scenario())


def test_stale_duplicate_end_cannot_resurrect_reclaimed_session(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(rollout_module.time, "monotonic", lambda: clock[0])

    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        for session_index in range(1024):
            await shell.end_update_weight(_transaction_id(0, f"{session_index:032x}"))

        old_session_id = f"{0:032x}"
        old_transaction_id = _transaction_id(0, old_session_id)
        old_transaction = shell._weight_update_transactions[old_transaction_id]
        await old_transaction.end_lock.acquire()
        duplicate_end = asyncio.create_task(shell.end_update_weight(old_transaction_id))
        await asyncio.sleep(0)
        assert not duplicate_end.done()

        clock[0] = rollout_module._WEIGHT_UPDATE_SESSION_RETENTION_SECONDS + 1
        new_session_id = f"{1024:032x}"
        await shell.end_update_weight(_transaction_id(0, new_session_id))
        old_transaction.end_lock.release()
        await duplicate_end

        assert len(shell._completed_weight_update_transaction_sequences) == 1024
        assert len(shell._weight_update_session_last_seen) == 1024
        assert old_session_id not in shell._completed_weight_update_transaction_sequences
        assert old_session_id not in shell._weight_update_session_last_seen
        assert new_session_id in shell._completed_weight_update_transaction_sequences

    asyncio.run(_scenario())


def test_unrecoverable_transactions_enforce_hard_transaction_limit():
    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        for transaction_index in range(1024):
            transaction = rollout_module._WeightUpdateTransaction()
            transaction.completed = True
            transaction.recovery_required = True
            shell._weight_update_transactions[_transaction_id(transaction_index)] = transaction

        with pytest.raises(HTTPException) as error:
            await shell.can_do_update_weight_for_async(_transaction_id(1024))

        assert error.value.status_code == 429
        assert len(shell._weight_update_transactions) == 1024

    asyncio.run(_scenario())


def test_unstructured_transaction_ids_are_rejected_without_tombstones():
    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        for transaction_index in range(1100):
            with pytest.raises(HTTPException) as error:
                await shell.end_update_weight(f"raw-uuid-{transaction_index}")
            assert error.value.status_code == 400

        assert shell._weight_update_transactions == {}
        assert shell._completed_weight_update_transaction_sequences == {}
        with pytest.raises(HTTPException) as error:
            await shell.can_do_update_weight_for_async("raw-uuid-0")
        assert error.value.status_code == 400
        for sequence in ("+1", "01", " 1", "-0", "\u0661"):
            with pytest.raises(HTTPException) as error:
                await shell.can_do_update_weight_for_async(f"relax-v1:{'0' * 32}:{sequence}")
            assert error.value.status_code == 400
        assert shell.status == "running"

    asyncio.run(_scenario())


def test_new_transaction_waits_for_active_transaction_to_end():
    async def _scenario():
        shell = _make_rollout(can_update=True, manager=_ManagerStub())
        first_transaction_id = _transaction_id(7)
        second_transaction_id = _transaction_id(8)
        assert await shell.can_do_update_weight_for_async(first_transaction_id) == 1

        next_pause = asyncio.create_task(shell.can_do_update_weight_for_async(second_transaction_id))
        await asyncio.sleep(0)
        assert not next_pause.done()

        await shell.end_update_weight(first_transaction_id)
        assert await asyncio.wait_for(next_pause, timeout=1) == 1
        assert shell._active_weight_update_transaction_id == second_transaction_id

        await shell.end_update_weight(second_transaction_id)

    asyncio.run(_scenario())
