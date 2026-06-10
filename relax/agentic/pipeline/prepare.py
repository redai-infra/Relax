# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from relax.agentic import format_agentic_event
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)


@dataclass(frozen=True)
class PrepareRequestHandle:
    slot_idx: int
    managed_session_handle: Any
    managed_session_submitted_at: float


@dataclass
class PrepareGroupState:
    group_id: str
    group_generation: int
    sample_group: list[Sample]
    request_group: list[Any]
    request_handles: list[PrepareRequestHandle]
    status: str = "warming"


@dataclass(frozen=True)
class ExecutionBatchInput:
    rollout_id: int
    leased_group_states: list[PrepareGroupState]
    leased_group_ids: list[str]
    leased_groups: list[tuple[str, int]] = field(default_factory=list)


class PrepareSourceExhaustedError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchBatch:
    sample_groups: list[list[Sample]]


@dataclass(frozen=True)
class PrepareGroupSpec:
    sample_group: list[Sample]
    group_id: str
    group_generation: int


class PrepareDomain:
    """Long-lived prepare domain that outlives individual rollout steps.

    Lifecycle::

        pool = PrepareDomain(...)
        pool.start()

        # per step
        pool.configure(...)
        ...  # lease / query / dispatch

        pool.close()
    """

    # Construction

    def __init__(
        self,
        *,
        scope_id: str,
        data_source,
        prefetch_concurrency: int = 1,
        pool_target_group_count: int = 0,
    ) -> None:
        if not isinstance(scope_id, str) or not scope_id:
            raise RuntimeError("PrepareDomain requires a non-empty scope_id.")
        self.scope_id = scope_id
        self.prefetch_concurrency = prefetch_concurrency
        self.configured_pool_target_group_count = pool_target_group_count
        if self.configured_pool_target_group_count < 0:
            raise RuntimeError(
                f"pool_target_group_count must be non-negative, got {self.configured_pool_target_group_count}."
            )
        self.data_source = data_source
        self.pool_target_group_count = self.configured_pool_target_group_count
        self.runtime_driver = None

        # Resident mutable state
        self.pending_prepare_jobs: deque[PrepareGroupSpec] = deque()
        self.prepare_groups_by_id: dict[str, Any] = {}
        self.warming_group_ids: deque[str] = deque()
        self.ready_group_ids: deque[str] = deque()
        self.next_prepare_group_seq = 0
        self._ready_batches: deque[FetchBatch] = deque()
        self._fetch_task: asyncio.Task[None] | None = None  # runs on background loop
        self._fetch_submitted = False  # set in lock before run_coroutine_threadsafe, cleared in _bg_fetch finally
        self._fetch_inflight_group_count = 0
        self._launching_group_count = 0
        self._launch_tasks: set[asyncio.Task[None]] = set()
        self._launch_error: BaseException | None = None
        self._last_logged_pool_target: int | None = None

        # Event signalling that a fetch cycle completed (success, error, or cancel).
        # Waited on by the step thread, set by the background fetch coroutine.
        self._fetch_ready_event = threading.Event()
        self._fetch_error: BaseException | None = None
        self._source_exhausted = False

        # Background worker
        self._worker_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._lock = threading.Lock()

    # Lifecycle

    def start(self) -> None:
        """Start the background worker thread.

        Idempotent.
        """
        if self._worker_thread is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="resident-prepare-pool",
            daemon=True,
        )
        self._worker_thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def close(self) -> None:
        """Permanently shut down the pool."""
        if self._closed:
            return
        self._closed = True
        # Pump runs on the step event-loop; setting _closed is enough to
        # stop it (the loop checks ``not self._closed`` each iteration).
        loop = self._loop
        if loop is not None and loop.is_running():
            # Cancel fetch if running (fetch runs on the background loop).
            fetch_task = self._fetch_task
            if fetch_task is not None and not fetch_task.done():
                loop.call_soon_threadsafe(fetch_task.cancel)

                async def _drain_and_stop() -> None:
                    fetch_results = await asyncio.gather(fetch_task, return_exceptions=True)
                    for result in fetch_results:
                        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                            raise RuntimeError("PrepareDomain fetch cleanup failed") from result
                    loop.stop()

                loop.call_soon_threadsafe(asyncio.ensure_future, _drain_and_stop())
            else:
                loop.call_soon_threadsafe(loop.stop)
        # Unblock any step thread waiting on fetch.
        self._fetch_ready_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=10)
            self._worker_thread = None
        if loop is not None and not loop.is_closed():
            loop.close()
            self._loop = None

    async def shutdown(self) -> None:
        """Stop prepare work and discard sessions still owned by the prepare
        pool."""
        self.close()
        launch_tasks = list(self._launch_tasks)
        if launch_tasks:
            await asyncio.gather(*launch_tasks, return_exceptions=True)
        discarded_groups, discarded_sessions = await self.discard_resident_groups()
        with self._lock:
            self.pending_prepare_jobs.clear()
            self._ready_batches.clear()
        if discarded_groups:
            self._log_pool_state(
                "discard_resident_groups",
                discarded_groups=discarded_groups,
                discarded_sessions=discarded_sessions,
            )

    async def discard_resident_groups(self) -> tuple[int, int]:
        runtime_driver = self.runtime_driver
        with self._lock:
            group_states = list(self.prepare_groups_by_id.values())
        if not group_states:
            return 0, 0
        if runtime_driver is None:
            raise RuntimeError("PrepareDomain cannot discard resident groups before runtime driver is bound.")
        discarded_sessions = 0
        for group_state in group_states:
            discarded_sessions += await runtime_driver.discard_prepare_group(group_state=group_state)
            self._forget_prepare_group(group_state)
        return len(group_states), discarded_sessions

    def _forget_prepare_group(self, group_state: PrepareGroupState) -> None:
        group_id = group_state.group_id
        with self._lock:
            self.prepare_groups_by_id.pop(group_id, None)
            self.warming_group_ids = deque(gid for gid in self.warming_group_ids if gid != group_id)
            self.ready_group_ids = deque(gid for gid in self.ready_group_ids if gid != group_id)

    def configure(
        self,
        *,
        runtime_driver,
        pool_target_group_count: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("PrepareDomain is closed")
        if pool_target_group_count < 0:
            raise RuntimeError(f"pool_target_group_count must be non-negative, got {pool_target_group_count}.")
        runtime_scope_id = getattr(runtime_driver, "scope_id", None)
        if runtime_scope_id != self.scope_id:
            raise RuntimeError(
                f"PrepareDomain scope mismatch: prepare_scope={self.scope_id!r} runtime_scope={runtime_scope_id!r}."
            )
        self.start()
        with self._lock:
            if self._fetch_task is not None and self._fetch_task.done():
                self._fetch_task = None
                self._fetch_submitted = False
            self.runtime_driver = runtime_driver
        self.set_pool_target(pool_target_group_count)

    # Pool counts (thread-safe via _lock)

    def _resident_group_count(self) -> int:
        return len(self.warming_group_ids) + len(self.ready_group_ids)

    def _prepare_owned_group_count(self) -> int:
        return self._resident_group_count() + self._launching_group_count

    def _buffered_prepare_group_count(self) -> int:
        return sum(len(batch.sample_groups) for batch in self._ready_batches)

    def _pool_group_count_locked(self) -> int:
        return (
            len(self.pending_prepare_jobs) + self._buffered_prepare_group_count() + self._prepare_owned_group_count()
        )

    def _pool_counts(self) -> dict[str, int]:
        with self._lock:
            pool_count = self._pool_group_count_locked()
            return {
                "pool": pool_count,
                "pending": len(self.pending_prepare_jobs),
                "launching": self._launching_group_count,
                "warming": len(self.warming_group_ids),
                "ready": len(self.ready_group_ids),
                "resident": self._resident_group_count(),
                "pool_target": self.pool_target_group_count,
                "pool_cap": self.configured_pool_target_group_count,
                "fetch_ready_batches": len(self._ready_batches),
                "buffered_groups": self._buffered_prepare_group_count(),
                "fetch_inflight": self._fetch_task is not None or self._fetch_submitted,
            }

    def accounting_snapshot(self) -> dict[str, int]:
        counts = self._pool_counts()
        return {
            "pool_groups": counts["pool"],
            "pending_prepare_groups": counts["pending"],
            "warming_groups": counts["warming"],
            "ready_groups": counts["ready"],
            "pool_target_groups": counts["pool_target"],
            "fetch_inflight_groups": self._fetch_inflight_group_count if counts["fetch_inflight"] else 0,
        }

    def _log_pool_state(self, event: str, *, level: str = "info", **extra: Any) -> None:
        counts = self._pool_counts()
        log_fn = logger.debug if level == "debug" else logger.info
        fields = {**counts, **{key: value for key, value in extra.items() if value is not None}}
        log_fn(format_agentic_event("PREPARE", event, **fields))

    # Pool target

    def set_pool_target(self, target_groups: int) -> None:
        requested_target = target_groups
        if requested_target < 0:
            raise RuntimeError(f"target_groups must be non-negative, got {requested_target}.")
        capped_target = min(self.configured_pool_target_group_count, requested_target)
        self.pool_target_group_count = capped_target
        if self._last_logged_pool_target != capped_target:
            self._last_logged_pool_target = capped_target
            self._log_pool_state("target_update")

    # Prepare accept

    async def accept_prepare(
        self,
        sample_groups: list[list[Sample]],
    ) -> int:
        """Enqueue prepare jobs and submit launch tasks.

        Returns the number of groups whose launch tasks were submitted.
        """
        with self._lock:
            for sample_group in sample_groups:
                if not sample_group:
                    continue
                group_id, group_generation = self._next_prepare_group_label()
                self.pending_prepare_jobs.append(
                    PrepareGroupSpec(
                        sample_group=sample_group,
                        group_id=group_id,
                        group_generation=group_generation,
                    )
                )
        if sample_groups:
            self._log_pool_state("accept_prepare", accepted_groups=len(sample_groups))
        return await self.launch_pending()

    def has_pending_prepare(self) -> bool:
        with self._lock:
            return bool(self.pending_prepare_jobs)

    # Ready / warming queries

    def has_ready_groups(self) -> bool:
        with self._lock:
            return bool(self.ready_group_ids)

    def has_warming_groups(self) -> bool:
        with self._lock:
            return bool(self.warming_group_ids)

    # Group labeling

    def _next_prepare_group_label(self) -> tuple[str, int]:
        group_generation = self.next_prepare_group_seq
        self.next_prepare_group_seq += 1
        group_id = f"prepare_group_{group_generation}"
        return group_id, group_generation

    # Refresh warming → ready

    async def refresh_ready_groups(self, *, status_fetcher) -> int:
        self._raise_launch_error_if_any()
        if not self.warming_group_ids:
            return 0
        with self._lock:
            runtime_driver = self.runtime_driver
        if runtime_driver is None:
            raise RuntimeError("PrepareDomain cannot refresh warming groups before runtime driver is bound.")
        snapshots = await status_fetcher()
        ready_count = 0
        status_by_key = {(str(item["group_id"]), int(item["group_generation"])): item for item in snapshots}
        # Take a snapshot of current warming ids under lock, then work on
        # the snapshot.  The pump thread may append new ids concurrently;
        # those will remain in the deque and be picked up next refresh.
        with self._lock:
            snapshot_len = len(self.warming_group_ids)
        still_warming: list[str] = []
        for _ in range(snapshot_len):
            with self._lock:
                if not self.warming_group_ids:
                    break
                group_id = self.warming_group_ids.popleft()
                group_state = self.prepare_groups_by_id.get(group_id)
            if group_state is None or group_state.status != "warming":
                continue
            snapshot = status_by_key.get((group_state.group_id, group_state.group_generation))
            expected_sessions = len(group_state.request_handles)
            ready_sessions = int(snapshot.get("ready_sessions") or 0) if snapshot else 0
            total_sessions = int(snapshot.get("total_sessions") or 0) if snapshot else 0
            if total_sessions >= expected_sessions and ready_sessions >= expected_sessions:
                with self._lock:
                    group_state.status = "ready"
                    self.ready_group_ids.append(group_id)
                ready_count += 1
                continue
            runtime_driver.raise_if_prepare_group_completed_before_ready(
                group_state=group_state,
                total_sessions=total_sessions,
                ready_sessions=ready_sessions,
            )
            still_warming.append(group_id)
        # Re-insert still-warming ids at the front (before any newly appended
        # ids from the pump thread) so they are checked first next time.
        if still_warming:
            with self._lock:
                for gid in reversed(still_warming):
                    self.warming_group_ids.appendleft(gid)
        if ready_count > 0:
            self._log_pool_state("warming_to_ready", promoted_groups=ready_count)
        return ready_count

    async def lease_ready_groups(
        self,
        *,
        quota_group_count: int,
        rollout_id: int,
    ) -> ExecutionBatchInput | None:
        with self._lock:
            if not self.ready_group_ids:
                return None
            remaining_quota_groups = quota_group_count
            leased_group_ids: list[str] = []
            leased_groups: list[tuple[str, int]] = []
            leased_group_states: list[Any] = []
            retained_group_ids: deque[str] = deque()
            while self.ready_group_ids:
                group_id = self.ready_group_ids.popleft()
                group_state = self.prepare_groups_by_id.get(group_id)
                if group_state is None or group_state.status != "ready":
                    continue
                if remaining_quota_groups <= 0:
                    retained_group_ids.append(group_id)
                    continue
                self.prepare_groups_by_id.pop(group_id, None)
                remaining_quota_groups -= 1
                leased_group_ids.append(group_id)
                leased_groups.append((group_state.group_id, group_state.group_generation))
                leased_group_states.append(group_state)
            self.ready_group_ids = retained_group_ids
        if not leased_group_ids:
            return None
        self._log_pool_state("lease_ready_groups", leased_groups=len(leased_group_ids))
        return ExecutionBatchInput(
            rollout_id=rollout_id,
            leased_group_states=leased_group_states,
            leased_group_ids=leased_group_ids,
            leased_groups=leased_groups,
        )

    # Fetch

    async def _fetch_batch(self, requested_group_count: int) -> FetchBatch:
        import ray

        # Resident agentic work stays inside the pipeline; data source fetches step input groups.
        fetch_ref = self.data_source.get_samples.remote(requested_group_count, False)
        sample_groups = await asyncio.to_thread(ray.get, fetch_ref)
        if not sample_groups:
            raise PrepareSourceExhaustedError("data source returned no sample groups")
        self._assert_sample_groups(sample_groups)
        if len(sample_groups) > requested_group_count:
            raise RuntimeError(
                "Agentic prepare fetch returned more groups than requested; data buffer carry-over must remain "
                "inside the resident pipeline. "
                f"returned={len(sample_groups)}, requested={requested_group_count}."
            )
        self._log_pool_state(
            "fetch_batch_ready",
            fetched_groups=len(sample_groups),
            fetched_ready_groups=len(sample_groups),
        )
        return FetchBatch(sample_groups=sample_groups)

    def start_fetch(self) -> bool:
        """Kick a fetch on the background loop.

        Non-blocking, thread-safe.

        Hard constraints:
        - Refuses when the domain is closed.
        - Refuses when no data source is configured; eval callers can feed
          prepared batches directly.
        - Refuses when ``pool_target_group_count`` is 0.  That state is
          reserved for explicit outer-loop shutdown, not ordinary step
          progression.
        - Measures resident capacity with prepared groups owned by this domain.
        - Requests exactly the current prepare-pool gap.
        - Caps ``_ready_batches`` at ``prefetch_concurrency`` (typically 1).
        """
        with self._lock:
            if self._closed:
                return False
            if self.data_source is None:
                return False
            if self.pool_target_group_count <= 0:
                return False
            if self._source_exhausted:
                return False
            if self._pool_group_count_locked() >= self.pool_target_group_count:
                return False
            if self._fetch_task is not None or self._fetch_submitted:
                return False
            if len(self._ready_batches) >= self.prefetch_concurrency:
                return False
            pool_group_count = self._pool_group_count_locked()
            remaining_group_count = self.pool_target_group_count - pool_group_count
            if remaining_group_count <= 0:
                return False
            requested_group_count = remaining_group_count
            if requested_group_count <= 0:
                raise RuntimeError(
                    f"PrepareDomain requested group count must be positive when data_source is configured, "
                    f"got {requested_group_count}."
                )
            # Mark submitted inside the lock so that no concurrent
            # start_fetch() can slip through before _bg_fetch() sets
            # _fetch_task.
            self._fetch_submitted = True
            self._fetch_inflight_group_count = requested_group_count
        # Check for a cached error from a previous fetch cycle.
        err = self._fetch_error
        if err is not None:
            self._fetch_error = None
            self._fetch_submitted = False
            self._fetch_inflight_group_count = 0
            if isinstance(err, PrepareSourceExhaustedError):
                self._source_exhausted = True
                return False
            raise err
        self._fetch_ready_event.clear()
        loop = self._loop
        if loop is None or loop.is_closed():
            self._fetch_submitted = False
            self._fetch_inflight_group_count = 0
            return False
        asyncio.run_coroutine_threadsafe(self._bg_fetch(requested_group_count=requested_group_count), loop)
        self._log_pool_state("fetch_started", requested_groups=requested_group_count)
        return True

    async def _bg_fetch(self, requested_group_count: int) -> None:
        """Fetch coroutine that runs on the background event loop.

        A successfully submitted fetch runs to completion unless the domain is
        closed.  Normal rollout-step teardown preserves any inflight fetch or
        buffered output for natural resident prepare progress in a later step.

        The ``_fetch_submitted`` flag (set by ``start_fetch()`` inside
        the lock) prevents the race where multiple coroutines are queued
        before any of them can set ``_fetch_task``.
        """
        self._fetch_task = asyncio.current_task()
        try:
            with self._lock:
                if self._closed:
                    return
            batch = await self._fetch_batch(requested_group_count=requested_group_count)
            with self._lock:
                if not self._closed:
                    self._ready_batches.append(batch)
            self._log_pool_state(
                "fetch_batch_buffered",
                buffered_groups=len(batch.sample_groups),
            )
        except asyncio.CancelledError:
            pass
        except PrepareSourceExhaustedError:
            self._source_exhausted = True
        except Exception as exc:
            self._fetch_error = exc
        finally:
            with self._lock:
                self._fetch_task = None
                self._fetch_submitted = False
                self._fetch_inflight_group_count = 0
            self._fetch_ready_event.set()

    def has_ready_output(self) -> bool:
        with self._lock:
            return bool(self._ready_batches)

    def has_inflight_work(self) -> bool:
        with self._lock:
            return (
                self._fetch_task is not None
                or self._fetch_submitted
                or self._launching_group_count > 0
                or self._launch_error is not None
            )

    async def accept_fetched_batch(self) -> bool:
        if not self.has_ready_output():
            return False
        with self._lock:
            if self._closed or self.pool_target_group_count <= 0:
                return False
            fetch_output = self._ready_batches.popleft()
        await self.accept_prepare(fetch_output.sample_groups)
        return bool(fetch_output.sample_groups)

    # Launch (immediate — called from accept_prepare and flush)

    async def launch_pending(self) -> int:
        """Launch pending prepare jobs that fit within the target budget.

        Called from :meth:`accept_prepare` and after lease/consume
        operations that free up budget.

        Returns the number of groups whose launch tasks were submitted.
        """
        runtime_driver = self.runtime_driver
        self._raise_launch_error_if_any()
        if runtime_driver is None or self._closed:
            return 0
        pending = self.pending_prepare_jobs
        if not pending:
            return 0
        remaining_pool_budget = self.pool_target_group_count - self._prepare_owned_group_count()
        if remaining_pool_budget <= 0:
            return 0
        first_job = pending[0]
        runner_pool = runtime_driver.ensure_session_runner_pool(total_requests=len(first_job.sample_group))
        available_session_slots = runner_pool.available_launch_slots()
        if available_session_slots <= 0:
            return 0
        selected_prepare_groups: list[PrepareGroupSpec] = []
        retained_prepare_groups: deque[PrepareGroupSpec] = deque()
        used_session_slots = 0
        used_groups = 0
        while pending:
            prepare_group = pending.popleft()
            required_session_slots = len(prepare_group.sample_group)
            if used_session_slots + required_session_slots > available_session_slots:
                retained_prepare_groups.append(prepare_group)
                retained_prepare_groups.extend(pending)
                pending.clear()
                break
            if used_groups >= remaining_pool_budget:
                retained_prepare_groups.append(prepare_group)
                continue
            selected_prepare_groups.append(prepare_group)
            used_session_slots += required_session_slots
            used_groups += 1
        pending.extend(retained_prepare_groups)
        if not selected_prepare_groups:
            return 0
        runner_pool.reserve_launch_slots(used_session_slots)
        return self._start_launches(
            runtime_driver=runtime_driver,
            runner_pool=runner_pool,
            prepare_groups=selected_prepare_groups,
        )

    def _start_launches(self, *, runtime_driver, runner_pool, prepare_groups: list[PrepareGroupSpec]) -> int:
        launched_group_count = len(prepare_groups)
        reserved_session_slots = sum(len(prepare_group.sample_group) for prepare_group in prepare_groups)
        with self._lock:
            self._launching_group_count += launched_group_count
        task = asyncio.create_task(
            self._launch_and_publish(
                prepare_groups=prepare_groups,
                runtime_driver=runtime_driver,
                runner_pool=runner_pool,
                reserved_session_slots=reserved_session_slots,
            )
        )
        self._launch_tasks.add(task)
        task.add_done_callback(self._launch_tasks.discard)
        self._log_pool_state("launch_submitted", submitted_groups=len(prepare_groups))
        return len(prepare_groups)

    async def _launch_and_publish(
        self,
        *,
        prepare_groups: list[PrepareGroupSpec],
        runtime_driver,
        runner_pool,
        reserved_session_slots: int,
    ) -> None:
        try:
            group_states = await runtime_driver.start_prepare_group_sessions(
                prepare_groups=prepare_groups,
                runner_pool=runner_pool,
            )
            if len(group_states) != len(prepare_groups):
                raise RuntimeError(
                    "RuntimeDomain launched a mismatched number of prepare groups: "
                    f"expected={len(prepare_groups)}, got={len(group_states)}."
                )
            if self._closed:
                for group_state in group_states:
                    await runtime_driver.discard_prepare_group(group_state=group_state)
                    self._forget_prepare_group(group_state)
                return
            with self._lock:
                for group_state in group_states:
                    self.prepare_groups_by_id[group_state.group_id] = group_state
                    self.warming_group_ids.append(group_state.group_id)
            self._log_pool_state(
                "launch_prepare_group",
                level="debug",
                launched_groups=len(group_states),
                launched_sessions=sum(len(group_state.request_handles) for group_state in group_states),
            )
        except Exception as exc:
            with self._lock:
                self._launch_error = exc
        finally:
            runner_pool.release_launch_slots(reserved_session_slots)
            with self._lock:
                next_launching_count = self._launching_group_count - len(prepare_groups)
                if next_launching_count < 0:
                    raise RuntimeError("PrepareDomain launching group counter underflow")
                self._launching_group_count = next_launching_count

    def _raise_launch_error_if_any(self) -> None:
        with self._lock:
            exc = self._launch_error
            self._launch_error = None
        if exc is not None:
            raise RuntimeError("prepare launch failed") from exc

    # Utilities

    @staticmethod
    def _assert_sample_groups(sample_groups) -> None:
        for group in sample_groups:
            for sample in group:
                if sample.status == sample.Status.ABORTED:
                    raise RuntimeError(
                        "Agentic rollout no longer accepts aborted samples from the data source. "
                        "Partial rollout state must remain internal to runtime."
                    )
