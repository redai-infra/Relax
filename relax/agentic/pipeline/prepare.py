# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from relax.agentic import format_agentic_event
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)


# Minimum wall-clock gap between two `prepare_group_status` RPCs from
# `refresh_ready_groups`. The resident dataflow pump ticks at ~20Hz and calls
# refresh from two sites per tick, which otherwise fires 20-40 status RPCs/s per
# rollout replica (flooding the Serve access log and burning serialization).
# warming->ready is a sandbox-startup event on the seconds-to-minutes scale, so
# polling it faster than ~0.5s buys nothing; refresh skips the RPC when the last
# one was more recent than this and reuses the current warming set unchanged.
_STATUS_POLL_THROTTLE_S = 0.5


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

        # per step
        pool.configure(...)
        ...  # lease / query / dispatch

        await pool.shutdown()
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
        self.data_source = data_source
        self.pool_target_group_count = pool_target_group_count
        self.runtime_driver = None

        # Resident mutable state
        self.pending_prepare_jobs: deque[PrepareGroupSpec] = deque()
        self.prepare_groups_by_id: dict[str, Any] = {}
        self.warming_group_ids: deque[str] = deque()
        self.ready_group_ids: deque[str] = deque()
        # Monotonic timestamp of the last prepare_group_status RPC issued by
        # refresh_ready_groups; used to throttle it to _STATUS_POLL_THROTTLE_S.
        self._last_status_fetch_monotonic: float = 0.0
        self.next_prepare_group_seq = 0
        self._ready_batches: deque[FetchBatch] = deque()
        self._fetch_task: asyncio.Task[None] | None = None
        self._fetch_wakeup = asyncio.Event()
        self._fetch_inflight_group_count = 0
        self._launching_group_count = 0
        self._launch_tasks: set[asyncio.Task[None]] = set()
        self._launch_error: BaseException | None = None
        self._fetch_error: BaseException | None = None
        self._source_exhausted = False
        self._closed = False

    # Lifecycle

    async def shutdown(self) -> None:
        """Stop prepare work and discard sessions still owned by the prepare
        pool."""
        if self._closed:
            return
        self._closed = True
        if self._fetch_task is not None:
            self._fetch_task.cancel()
            await asyncio.gather(self._fetch_task, return_exceptions=True)
            self._fetch_task = None
        self._fetch_inflight_group_count = 0
        launch_tasks = list(self._launch_tasks)
        if launch_tasks:
            await asyncio.gather(*launch_tasks, return_exceptions=True)
        discarded_groups, discarded_sessions = await self.discard_resident_groups()
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
        runtime_scope_id = getattr(runtime_driver, "scope_id", None)
        if runtime_scope_id != self.scope_id:
            raise RuntimeError(
                f"PrepareDomain scope mismatch: prepare_scope={self.scope_id!r} runtime_scope={runtime_scope_id!r}."
            )
        self.runtime_driver = runtime_driver
        self.pool_target_group_count = pool_target_group_count

    def _resident_group_count(self) -> int:
        return len(self.warming_group_ids) + len(self.ready_group_ids)

    def _prepare_owned_group_count(self) -> int:
        return self._resident_group_count() + self._launching_group_count

    def _buffered_prepare_group_count(self) -> int:
        return sum(len(batch.sample_groups) for batch in self._ready_batches)

    def _pool_group_count(self) -> int:
        return (
            len(self.pending_prepare_jobs) + self._buffered_prepare_group_count() + self._prepare_owned_group_count()
        )

    def _pool_counts(self) -> dict[str, int]:
        pool_count = self._pool_group_count()
        return {
            "pool": pool_count,
            "pending": len(self.pending_prepare_jobs),
            "launching": self._launching_group_count,
            "warming": len(self.warming_group_ids),
            "ready": len(self.ready_group_ids),
            "resident": self._resident_group_count(),
            "pool_target": self.pool_target_group_count,
            "fetch_ready_batches": len(self._ready_batches),
            "buffered_groups": self._buffered_prepare_group_count(),
            "fetch_inflight": self._fetch_inflight_group_count > 0,
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

    # Prepare accept

    async def accept_prepare(
        self,
        sample_groups: list[list[Sample]],
    ) -> int:
        """Enqueue prepare jobs and submit launch tasks.

        Returns the number of groups whose launch tasks were submitted.
        """
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
        return bool(self.pending_prepare_jobs)

    # Ready / warming queries

    def has_ready_groups(self) -> bool:
        return bool(self.ready_group_ids)

    def has_warming_groups(self) -> bool:
        return bool(self.warming_group_ids)

    # Group labeling

    def _next_prepare_group_label(self) -> tuple[str, int]:
        group_generation = self.next_prepare_group_seq
        self.next_prepare_group_seq += 1
        group_id = f"prepare_group_{group_generation}"
        return group_id, group_generation

    # Refresh warming → ready

    async def refresh_ready_groups(self, *, status_fetcher, drop_completed_before_ready: bool = False) -> int:
        self._raise_launch_error_if_any()
        if not self.warming_group_ids:
            return 0
        # Throttle the status RPC: the pump calls this at ~20Hz from two sites,
        # but warming->ready is a seconds-to-minutes sandbox-startup event, so
        # polling faster than _STATUS_POLL_THROTTLE_S only floods the Serve
        # access log. Skip this tick if the last fetch was too recent; warming
        # groups stay queued and are re-checked on the next allowed refresh.
        now = time.monotonic()
        if now - self._last_status_fetch_monotonic < _STATUS_POLL_THROTTLE_S:
            return 0
        self._last_status_fetch_monotonic = now
        runtime_driver = self.runtime_driver
        if runtime_driver is None:
            raise RuntimeError("PrepareDomain cannot refresh warming groups before runtime driver is bound.")
        snapshots = await status_fetcher()
        ready_count = 0
        status_by_key = {(str(item["group_id"]), int(item["group_generation"])): item for item in snapshots}
        # Launch tasks may append new ids while this coroutine awaits status.
        # Only process the ids that existed when the status snapshot returned.
        snapshot_len = len(self.warming_group_ids)
        still_warming: list[str] = []
        for _ in range(snapshot_len):
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
            if ready_sessions == expected_sessions:
                group_state.status = "ready"
                self.ready_group_ids.append(group_id)
                ready_count += 1
                continue
            if drop_completed_before_ready:
                # A managed session that finished without producing a chat IR
                # (e.g. an apptainer instance that failed to start, or an
                # upstream LLM that returned null content) must not crash the
                # whole rollout service. Drop the group and keep going; the
                # over-sampling prepare pool + transfer quota gate refill the
                # dropped group, so the committed batch size is preserved. Used
                # by both the train and eval rollout loops.
                completed_requests = runtime_driver.prepare_group_completed_before_ready(group_state=group_state)
                if completed_requests:
                    logger.warning(
                        "Prepare-owned managed agent session completed before producing a chat IR; "
                        "dropping group (will be refilled by the over-sampling pool): "
                        f"group_id={group_state.group_id}, group_generation={group_state.group_generation}, "
                        f"expected_sessions={expected_sessions}, total_sessions={total_sessions}, "
                        f"ready_sessions={ready_sessions}, completed_requests={completed_requests[:8]}."
                    )
                    await runtime_driver.discard_prepare_group(group_state=group_state)
                    self._forget_prepare_group(group_state)
                    # Do NOT re-queue: leaving it warming would deadlock the loop.
                    continue
            else:
                runtime_driver.raise_if_prepare_group_completed_before_ready(
                    group_state=group_state,
                    total_sessions=total_sessions,
                    ready_sessions=ready_sessions,
                )
            still_warming.append(group_id)
        # Re-insert still-warming ids before any ids appended by launch tasks.
        if still_warming:
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
        # Resident agentic work stays inside the pipeline; data source fetches step input groups.
        sample_groups = await self.data_source.get_samples.remote(requested_group_count)
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
        """Wake the resident fetch task without blocking the caller.

        Hard constraints:
        - Refuses when the domain is closed.
        - Measures resident capacity with prepared groups owned by this domain.
        - Requests exactly the current prepare-pool gap.
        - Caps ``_ready_batches`` at ``prefetch_concurrency`` (typically 1).
        """
        if self._closed or self.data_source is None:
            return False
        # Check for a cached error from a previous fetch cycle.
        err = self._fetch_error
        if err is not None:
            self._fetch_error = None
            raise err
        if (
            self._source_exhausted
            or self._fetch_inflight_group_count > 0
            or len(self._ready_batches) >= self.prefetch_concurrency
        ):
            return False
        prepare_pool_gap = self.pool_target_group_count - self._pool_group_count()
        if prepare_pool_gap <= 0:
            return False
        self._fetch_inflight_group_count = prepare_pool_gap
        if self._fetch_task is None:
            self._fetch_task = asyncio.create_task(
                self._fetch_loop(),
                name=f"agentic-prepare-fetch:{self.scope_id}",
            )
        self._fetch_wakeup.set()
        self._log_pool_state("fetch_started", requested_groups=prepare_pool_gap)
        return True

    async def _fetch_loop(self) -> None:
        """Keep fetching on the resident loop whenever the pipeline requests
        it."""
        while True:
            await self._fetch_wakeup.wait()
            self._fetch_wakeup.clear()
            requested_group_count = self._fetch_inflight_group_count
            try:
                batch = await self._fetch_batch(requested_group_count=requested_group_count)
                if not self._closed:
                    self._ready_batches.append(batch)
                self._log_pool_state(
                    "fetch_batch_buffered",
                    buffered_groups=len(batch.sample_groups),
                )
            except asyncio.CancelledError:
                raise
            except PrepareSourceExhaustedError:
                self._source_exhausted = True
            except Exception as exc:
                self._fetch_error = exc
            finally:
                self._fetch_inflight_group_count = 0

    def has_ready_output(self) -> bool:
        return bool(self._ready_batches)

    def has_inflight_work(self) -> bool:
        return (
            self._fetch_inflight_group_count > 0 or self._launching_group_count > 0 or self._launch_error is not None
        )

    async def accept_fetched_batch(self) -> bool:
        if not self.has_ready_output():
            return False
        if self._closed:
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
            self._launch_error = exc
        finally:
            runner_pool.release_launch_slots(reserved_session_slots)
            next_launching_count = self._launching_group_count - len(prepare_groups)
            if next_launching_count < 0:
                raise RuntimeError("PrepareDomain launching group counter underflow")
            self._launching_group_count = next_launching_count

    def _raise_launch_error_if_any(self) -> None:
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
