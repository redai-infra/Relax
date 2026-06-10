# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import contextlib
import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from relax.agentic import format_agentic_event
from relax.agentic.pipeline import GroupKey
from relax.agentic.pipeline.runtime import (
    RuntimeDomain,
    agentic_prepare_pool_target_from_args,
    get_agentic_runtime_resources,
)
from relax.agentic.profile import TRACE_KEY
from relax.engine.filters.base_types import MetricGatherer, call_dynamic_filter
from relax.engine.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import (
    compute_rollout_explicit_reward_metrics,
    compute_rollout_step,
    compute_statistics,
    finalize_rollout_explicit_metric_values,
    has_repetition,
)
from relax.utils.misc import group_by
from relax.utils.profile_utils import start_sglang_profile, stop_sglang_profile
from relax.utils.training.eval_config import EvalDatasetConfig
from relax.utils.training.train_dump_utils import save_debug_rollout_data, save_rollout_result_jsonl
from relax.utils.types import Sample

from .pipeline.prepare import PrepareDomain
from .pipeline.reward import RewardDomain
from .pipeline.transfer import TransferDomain


logger = get_logger(__name__)
_AGENT_METADATA_INTERNAL_KEYS = {TRACE_KEY}
_IDLE_HEARTBEAT_INTERVAL_S = 30.0
_BACKGROUND_POLL_INTERVAL_S = 0.05


_RESIDENT_PIPELINE: "AgenticResidentPipeline | None" = None
_RESIDENT_PIPELINE_LOCK = threading.Lock()
_RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID: int | None = None
_RESIDENT_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
_RESIDENT_ASYNC_THREAD: threading.Thread | None = None
_RESIDENT_ASYNC_LOCK = threading.Lock()


def _eval_scope_id(*, dataset_name: str, rollout_id: int) -> str:
    return f"eval:{dataset_name}:{rollout_id}"


def _post_train_eval_expected(args, rollout_id: int, data_source) -> bool:
    eval_interval = getattr(args, "eval_interval", None)
    if eval_interval is None or getattr(args, "eval_prompt_data", None) is None:
        return False
    step = rollout_id + 1
    if step % eval_interval == 0:
        return True
    if not getattr(args, "rollout_global_dataset", False):
        return False

    import ray

    num_rollout_per_epoch = ray.get(data_source.lengths.remote()) // args.rollout_batch_size
    return step % num_rollout_per_epoch == 0


def _resident_async_loop_main(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _shutdown_resident_async_loop() -> None:
    global _RESIDENT_ASYNC_LOOP, _RESIDENT_ASYNC_THREAD
    with _RESIDENT_ASYNC_LOCK:
        loop = _RESIDENT_ASYNC_LOOP
        thread = _RESIDENT_ASYNC_THREAD
        _RESIDENT_ASYNC_LOOP = None
        _RESIDENT_ASYNC_THREAD = None
    if loop is None:
        return
    if not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5)
    if not loop.is_closed() and (thread is None or thread is not threading.current_thread()):
        loop.close()


def _run_on_resident_async_loop(coro):
    global _RESIDENT_ASYNC_LOOP, _RESIDENT_ASYNC_THREAD
    with _RESIDENT_ASYNC_LOCK:
        loop = _RESIDENT_ASYNC_LOOP
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=_resident_async_loop_main, args=(loop,), name="agentic-resident-loop", daemon=True
            )
            thread.start()
            _RESIDENT_ASYNC_LOOP = loop
            _RESIDENT_ASYNC_THREAD = thread
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _shutdown_resident_pipeline_after_deferred_eval(rollout_id: int) -> None:
    global _RESIDENT_PIPELINE, _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID
    with _RESIDENT_PIPELINE_LOCK:
        if _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID != rollout_id:
            return
        _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID = None
        resident_pipeline = _RESIDENT_PIPELINE
        _RESIDENT_PIPELINE = None
    if resident_pipeline is not None:
        _run_on_resident_async_loop(resident_pipeline.shutdown())
    _shutdown_resident_async_loop()


class RolloutProgress:
    def __init__(self, *, total_sessions: int) -> None:
        if total_sessions <= 0:
            raise RuntimeError(f"RolloutProgress requires a positive total_sessions, got {total_sessions}.")
        self.total_sessions = total_sessions
        self._materialized_sessions = 0
        self._committed_sessions = 0
        self._scored_samples = 0
        self._bar = tqdm(
            total=self.total_sessions,
            desc="Rollout generation",
            unit="session",
            disable=False,
            mininterval=0.0,
            miniters=1,
        )
        self._set_bar_postfix(refresh=True)

    def _postfix_str(self) -> str:
        return f"scored={self._scored_samples}"

    def _set_bar_postfix(self, *, refresh: bool) -> None:
        self._bar.set_postfix_str(self._postfix_str(), refresh=refresh)

    def update_counts(
        self,
        *,
        materialized_sessions: int,
        committed_sessions: int,
        scored_samples: int,
    ) -> None:
        if materialized_sessions < self._materialized_sessions:
            raise RuntimeError(
                "RolloutProgress materialized session count regressed: "
                f"{materialized_sessions} < {self._materialized_sessions}."
            )
        if committed_sessions < self._committed_sessions:
            raise RuntimeError(
                "RolloutProgress committed session count regressed: "
                f"{committed_sessions} < {self._committed_sessions}."
            )
        if scored_samples < self._scored_samples:
            raise RuntimeError(
                f"RolloutProgress scored sample count regressed: {scored_samples} < {self._scored_samples}."
            )
        if (
            materialized_sessions == self._materialized_sessions
            and committed_sessions == self._committed_sessions
            and scored_samples == self._scored_samples
        ):
            return
        old_bar_count = min(self._materialized_sessions, self.total_sessions)
        new_bar_count = min(materialized_sessions, self.total_sessions)
        if new_bar_count > old_bar_count:
            self._bar.update(new_bar_count - old_bar_count)
        self._materialized_sessions = materialized_sessions
        self._committed_sessions = committed_sessions
        self._scored_samples = scored_samples
        self._set_bar_postfix(refresh=True)

    def update_total_sessions(self, total_sessions: int) -> None:
        if total_sessions < 0:
            raise RuntimeError(f"RolloutProgress total session count must be non-negative, got {total_sessions}.")
        if total_sessions == self.total_sessions:
            return
        self.total_sessions = total_sessions
        self._bar.total = total_sessions
        self._bar.refresh()

    def close(self) -> None:
        self._bar.close()

    def snapshot(self) -> dict[str, int]:
        return {
            "materialized_sessions": self._materialized_sessions,
            "committed_sessions": self._committed_sessions,
            "scored_samples": self._scored_samples,
        }


class AgenticResidentPipeline:
    def __init__(self) -> None:
        self.prepare_domain: PrepareDomain | None = None
        self.runtime_domain: RuntimeDomain | None = None
        self.transfer_domain: TransferDomain | None = None
        self.reward_domain: RewardDomain | None = None
        self.resident_dataflow_task: asyncio.Task | None = None
        self.resident_dataflow_progress_event: asyncio.Event | None = None
        self.resident_dataflow_lock: asyncio.Lock | None = None
        self.resident_dataflow_error: BaseException | None = None
        self.step_admission_closed = False
        self._step_get_samples_wait_started_at: float | None = None
        self._step_get_samples_times: list[float] = []
        self._active_step_handle: _AgenticStepHandle | None = None

    def _dataflow_lock(self) -> asyncio.Lock:
        lock = self.resident_dataflow_lock
        if lock is None:
            lock = asyncio.Lock()
            self.resident_dataflow_lock = lock
        return lock

    async def _pump_once(self) -> bool:
        async with self._dataflow_lock():
            self._assert_resident_group_count_invariant(context="pump_start")
            progressed = False
            while True:
                tick_progressed = False
                if await self._pump_prepare_once():
                    tick_progressed = True
                if await self._pump_admission_once():
                    tick_progressed = True
                if await self._pump_runtime_to_reward_once():
                    tick_progressed = True
                if await self._pump_reward_to_transfer_once():
                    tick_progressed = True
                if await self._pump_transfer_once():
                    tick_progressed = True
                self._assert_resident_group_count_invariant(context="pump_tick")
                if not tick_progressed:
                    return progressed
                progressed = True

    @property
    def resident_group_count(self) -> int:
        return self._resident_group_count_projection()[0]

    def _resident_group_count_projection(self) -> tuple[int, dict[str, int]]:
        if (
            self.runtime_domain is None
            or self.reward_domain is None
            or self.transfer_domain is None
            or not hasattr(self.runtime_domain, "runtime_groups_by_key")
        ):
            return 0, {}
        runtime_snapshot = dict(self.runtime_domain.accounting_snapshot())
        reward_snapshot = dict(self.reward_domain.accounting_snapshot())
        transfer_snapshot = dict(self.transfer_domain.accounting_snapshot())
        runtime_keys = set(self.runtime_domain.resident_group_keys())
        reward_keys = set(self.reward_domain.resident_group_keys())
        transfer_keys = set(self.transfer_domain.resident_group_keys())
        parts = {
            "runtime_resident_groups": int(runtime_snapshot.get("resident_groups", 0) or 0),
            "reward_waiting_groups": int(reward_snapshot.get("waiting_groups", 0) or 0),
            "reward_completed_groups": int(reward_snapshot.get("completed_groups", 0) or 0),
            "reward_ready_groups": int(reward_snapshot.get("ready_groups", 0) or 0),
            "reward_inflight_group_rewards": int(reward_snapshot.get("inflight_group_rewards", 0) or 0),
            "transfer_ready_groups": int(transfer_snapshot.get("ready_groups", 0) or 0),
        }
        resident_keys = runtime_keys | reward_keys | transfer_keys
        parts["distinct_resident_groups"] = len(resident_keys)
        parts["duplicate_domain_group_refs"] = sum(parts.values()) - len(resident_keys)
        return len(resident_keys), parts

    def _assert_resident_group_count_invariant(self, *, context: str) -> None:
        projected_count, parts = self._resident_group_count_projection()
        if not parts:
            return
        if projected_count != parts["distinct_resident_groups"]:
            raise RuntimeError(
                "Agentic resident_group_count projection diverged from distinct resident keys: "
                f"context={context}, projected={projected_count}, parts={parts}."
            )

    async def _pump_prepare_once(self) -> bool:
        prepare_domain = self.prepare_domain
        runtime_domain = self.runtime_domain
        if prepare_domain is None or runtime_domain is None:
            raise RuntimeError("Agentic prepare pump requires initialized prepare and runtime domains.")
        progressed = False
        if prepare_domain.has_ready_output():
            if await prepare_domain.accept_fetched_batch():
                progressed = True
        should_refresh_prepare = (
            prepare_domain.has_pending_prepare()
            or prepare_domain.has_inflight_work()
            or prepare_domain.has_ready_output()
            or prepare_domain.has_warming_groups()
        )
        if should_refresh_prepare:
            if await prepare_domain.refresh_ready_groups(status_fetcher=runtime_domain.prepare_group_status):
                progressed = True
        if prepare_domain.has_pending_prepare():
            if await prepare_domain.launch_pending():
                progressed = True
        if prepare_domain.start_fetch():
            progressed = True
        return progressed

    async def _pump_admission_once(self) -> bool:
        prepare_domain = self.prepare_domain
        runtime_domain = self.runtime_domain
        transfer_domain = self.transfer_domain
        reward_domain = self.reward_domain
        if prepare_domain is None or runtime_domain is None:
            raise RuntimeError("Agentic admission pump requires initialized prepare and runtime domains.")
        if self.step_admission_closed:
            return False
        if transfer_domain is None or reward_domain is None:
            raise RuntimeError("Agentic admission pump requires initialized transfer and reward domains.")

        progressed = False
        transfer_snapshot = dict(transfer_domain.accounting_snapshot())
        _, _, _, current_window_slack = self._current_window_admission_counts(
            resident_group_count=self.resident_group_count,
            transfer_snapshot=transfer_snapshot,
        )
        if current_window_slack == 0:
            self._step_get_samples_wait_started_at = None
            return progressed
        if self._step_get_samples_wait_started_at is None:
            self._step_get_samples_wait_started_at = time.monotonic()
        if prepare_domain.has_warming_groups():
            await prepare_domain.refresh_ready_groups(status_fetcher=runtime_domain.prepare_group_status)
        if not prepare_domain.has_ready_groups():
            return progressed
        batch_input = await prepare_domain.lease_ready_groups(
            quota_group_count=min(transfer_domain.over_sampling_batch_size, current_window_slack),
            rollout_id=runtime_domain.require_rollout_id(),
        )
        if batch_input is None:
            return progressed
        prepare_domain.start_fetch()
        if self._step_get_samples_wait_started_at is not None:
            self._step_get_samples_times.append(time.monotonic() - self._step_get_samples_wait_started_at)
            self._step_get_samples_wait_started_at = None
        admitted_batch_groups = await runtime_domain.start_batch(batch_input=batch_input)
        if admitted_batch_groups <= 0:
            raise RuntimeError("RuntimeDomain accepted an admission batch without owning any groups.")
        return True

    async def _pump_runtime_to_reward_once(self) -> bool:
        runtime_domain = self.runtime_domain
        reward_domain = self.reward_domain
        if runtime_domain is None or reward_domain is None:
            raise RuntimeError("Agentic runtime-to-reward pump requires initialized runtime and reward domains.")
        progressed = False
        if runtime_domain.has_pending_runtime_work():
            batch_progress = await runtime_domain.wait_for_next_runtime_slot(timeout_s=0.0)
            if batch_progress is not None:
                progressed = True
        discarded_group_keys = runtime_domain.drain_discarded_group_keys()
        if discarded_group_keys:
            await self._discard_terminal_runtime_groups(
                reward_domain=reward_domain,
                discarded_group_keys=discarded_group_keys,
            )
            progressed = True
        runtime_dispatch = runtime_domain.drain_ready_execution()
        if runtime_dispatch.materialized_batches:
            if not reward_domain.group_rm:
                reward_domain.accept_session_materializations(runtime_dispatch.materialized_batches)
            progressed = True
        for group in runtime_dispatch.ready_groups:
            await reward_domain.ingest_groups([group])
            progressed = True
        return progressed

    async def _discard_terminal_runtime_groups(
        self,
        *,
        reward_domain: RewardDomain,
        discarded_group_keys: set[GroupKey],
    ) -> int:
        discarded_group_count = len(discarded_group_keys)
        if discarded_group_count == 0:
            return 0
        await reward_domain.drop_waiting_groups_by_key(discarded_group_keys)
        return discarded_group_count

    async def _pump_reward_to_transfer_once(self) -> bool:
        reward_domain = self.reward_domain
        transfer_domain = self.transfer_domain
        if reward_domain is None:
            raise RuntimeError("Agentic reward-to-transfer pump requires initialized reward domain.")
        progressed = await reward_domain.step_once()
        if transfer_domain is None:
            return progressed

        ready_reward_groups = reward_domain.drain_ready_dispatch(max_groups=transfer_domain.remaining_ready_capacity())
        if ready_reward_groups:
            for group in ready_reward_groups:
                transfer_domain.enqueue_ready_groups([group])
            progressed = True
        return progressed

    async def _pump_transfer_once(self) -> bool:
        transfer_domain = self.transfer_domain
        if transfer_domain is None:
            raise RuntimeError("Agentic transfer pump requires initialized transfer domain.")
        released_groups, committed_group_count = await transfer_domain.drain_ready_group_payloads()
        if committed_group_count <= 0:
            return False
        self._log_first_rollout_sample_for_active_step(released_groups)
        return True

    def _log_first_rollout_sample_for_active_step(self, released_groups: list[list]) -> None:
        step_handle = self._active_step_handle
        if step_handle is None or step_handle.first_rollout_sample_logged or not released_groups:
            return
        sample = released_groups[0][0][0] if isinstance(released_groups[0][0], list) else released_groups[0][0]
        logger.info(
            "First rollout sample: %s, label: %s, reward: %s",
            [str(sample.prompt) + sample.response],
            str(sample.label)[:100],
            sample.reward,
        )
        step_handle.first_rollout_sample_logged = True

    def _raise_resident_dataflow_error(self) -> None:
        error = self.resident_dataflow_error
        if error is None:
            return
        self.resident_dataflow_error = None
        raise RuntimeError("agentic resident dataflow loop failed") from error

    async def _wait_for_resident_dataflow_progress(self) -> bool:
        self._raise_resident_dataflow_error()
        progress_event = self.resident_dataflow_progress_event
        if progress_event is None:
            await asyncio.sleep(_BACKGROUND_POLL_INTERVAL_S)
            self._raise_resident_dataflow_error()
            return False
        if progress_event.is_set():
            self._raise_resident_dataflow_error()
            progress_event.clear()
            return True
        try:
            await asyncio.wait_for(progress_event.wait(), timeout=_BACKGROUND_POLL_INTERVAL_S)
        except asyncio.TimeoutError:
            self._raise_resident_dataflow_error()
            return False
        self._raise_resident_dataflow_error()
        if not progress_event.is_set():
            return False
        progress_event.clear()
        return True

    async def _wait_for_step_event(self) -> bool:
        task = self.resident_dataflow_task
        if task is not None and not task.done():
            return await self._wait_for_resident_dataflow_progress()
        await asyncio.sleep(_BACKGROUND_POLL_INTERVAL_S)
        return False

    async def _resident_dataflow_loop(self) -> None:
        while True:
            try:
                progressed = await self._pump_once()
                if progressed:
                    if self.resident_dataflow_progress_event is not None:
                        self.resident_dataflow_progress_event.set()
                    continue
                await asyncio.sleep(_BACKGROUND_POLL_INTERVAL_S)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.resident_dataflow_error = exc
                if self.resident_dataflow_progress_event is not None:
                    self.resident_dataflow_progress_event.set()
                logger.exception("Agentic resident pipeline dataflow loop failed")
                return

    async def start_resident_dataflow(self) -> None:
        if self.resident_dataflow_task is not None and not self.resident_dataflow_task.done():
            return
        self.resident_dataflow_error = None
        self.resident_dataflow_progress_event = asyncio.Event()
        self.resident_dataflow_task = asyncio.create_task(self._resident_dataflow_loop())

    async def stop_resident_dataflow(self) -> None:
        task = self.resident_dataflow_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.resident_dataflow_task = None
        self.resident_dataflow_progress_event = None
        self.resident_dataflow_error = None

    async def shutdown(self) -> None:
        await self.stop_resident_dataflow()
        domains = (
            ("prepare_domain", self.prepare_domain),
            ("runtime_domain", self.runtime_domain),
            ("transfer_domain", self.transfer_domain),
            ("reward_domain", self.reward_domain),
        )
        for attr_name, domain in domains:
            if domain is None:
                continue
            await domain.shutdown()
            setattr(self, attr_name, None)
        self.step_admission_closed = False

    async def init_pipeline(
        self,
        *,
        args,
        data_source,
        data_system_client,
    ) -> None:
        pool_target_group_count = agentic_prepare_pool_target_from_args(args)
        async with self._dataflow_lock():
            prepare_domain = self.prepare_domain
            if prepare_domain is None:
                prepare_domain = PrepareDomain(
                    scope_id="train",
                    data_source=data_source,
                    prefetch_concurrency=1,
                    pool_target_group_count=pool_target_group_count,
                )
                self.prepare_domain = prepare_domain
            elif prepare_domain.configured_pool_target_group_count != pool_target_group_count:
                raise RuntimeError(
                    "PrepareDomain configuration changed after initialization: "
                    f"pool_target_group_count {prepare_domain.configured_pool_target_group_count} "
                    f"-> {pool_target_group_count}."
                )
            runtime_domain = self.runtime_domain
            if runtime_domain is None:
                runtime_domain = RuntimeDomain(args=args, scope_id="train")
                self.runtime_domain = runtime_domain
            if self.reward_domain is None:
                self.reward_domain = RewardDomain(
                    args=args,
                    group_filter=None,
                )
            if self.transfer_domain is None:
                self.transfer_domain = TransferDomain(
                    args=args,
                    data_system_client=data_system_client,
                )
            prepare_domain.configure(
                runtime_driver=runtime_domain,
                pool_target_group_count=pool_target_group_count,
            )
            self.step_admission_closed = True
            prepare_domain.start_fetch()
        await self.start_resident_dataflow()

    async def open_step(
        self,
        *,
        args,
        rollout_id: int,
        group_filter,
        defer_terminal_shutdown_for_eval: bool = False,
    ) -> "_AgenticStepHandle":
        fully_async = args.fully_async
        current_partition_quota = args.rollout_batch_size
        num_rollout = getattr(args, "num_rollout", None)
        terminal_step = num_rollout is not None and rollout_id + 1 >= num_rollout
        prepare_pool_target = agentic_prepare_pool_target_from_args(args)
        runtime_resources = get_agentic_runtime_resources(args)
        async with self._dataflow_lock():
            prepare_domain = self.prepare_domain
            runtime_domain = self.runtime_domain
            reward_domain = self.reward_domain
            transfer_domain = self.transfer_domain
            if None in (prepare_domain, runtime_domain, reward_domain, transfer_domain):
                raise RuntimeError("Agentic resident pipeline must be initialized before opening a rollout step.")
            if prepare_domain.configured_pool_target_group_count != prepare_pool_target:
                raise RuntimeError(
                    f"PrepareDomain pool target changed: "
                    f"{prepare_domain.configured_pool_target_group_count} -> {prepare_pool_target}."
                )
            runtime_domain.rebind_step(
                args=args,
                rollout_id=rollout_id,
            )
            reward_domain.rebind_step(
                group_filter=group_filter,
            )
            transfer_domain.rebind_step(
                rollout_id=rollout_id,
            )
            self._step_get_samples_wait_started_at = None
            self._step_get_samples_times = []
            inflight_groups_at_open = self.resident_group_count
            previous_partition_quota = inflight_groups_at_open if fully_async else 0
            transfer_domain.configure_transfer_quota(
                previous_partition_quota=previous_partition_quota,
                current_partition_quota=current_partition_quota,
            )
            self.step_admission_closed = False
            runtime_domain.ensure_session_runner_pool(total_requests=runtime_resources.step_session_capacity())
            prepare_domain.configure(
                runtime_driver=runtime_domain,
                pool_target_group_count=prepare_pool_target,
            )
            await runtime_domain.release_partial_resume_gate()
            transfer_start_snapshot = transfer_domain.accounting_snapshot()
        progress = RolloutProgress(
            total_sessions=transfer_domain.target_group_count() * transfer_start_snapshot["group_size"]
        )
        step_handle = _AgenticStepHandle(
            rollout_id=rollout_id,
            required_group_count=current_partition_quota,
            terminal_step=terminal_step,
            defer_terminal_shutdown_for_eval=defer_terminal_shutdown_for_eval,
            progress=progress,
        )
        self._active_step_handle = step_handle
        await self.start_resident_dataflow()
        return step_handle

    def _committed_target_count(self) -> int:
        transfer_snapshot = dict(self.transfer_domain.accounting_snapshot())
        return transfer_snapshot["committed_current_groups"]

    def _interrupted_group_count(self, *, step_handle: "_AgenticStepHandle", previous: bool) -> int:
        counter = getattr(self.runtime_domain, "interrupted_group_count_for_step", None)
        if not callable(counter):
            return 0
        return int(counter(rollout_id=step_handle.rollout_id, previous=previous) or 0)

    def _interrupted_close_accounting_enabled(self, step_handle: "_AgenticStepHandle") -> bool:
        return bool(self.runtime_domain.args.fully_async and not step_handle.terminal_step)

    async def _refresh_close_accounting(self) -> None:
        refresher = getattr(self.runtime_domain, "refresh_interrupted_close_accounting", None)
        if callable(refresher):
            await refresher()

    def _previous_partition_debt_satisfied(self, step_handle: "_AgenticStepHandle") -> bool:
        transfer_snapshot = dict(self.transfer_domain.accounting_snapshot())
        previous_count = transfer_snapshot["committed_previous_groups"]
        if self._interrupted_close_accounting_enabled(step_handle):
            previous_count += self._interrupted_group_count(step_handle=step_handle, previous=True)
        return previous_count >= transfer_snapshot["previous_partition_quota"]

    def _committed_target_satisfied(self, step_handle: "_AgenticStepHandle") -> bool:
        current_count = self._committed_target_count()
        if self._interrupted_close_accounting_enabled(step_handle):
            current_count += self._interrupted_group_count(step_handle=step_handle, previous=False)
        return current_count >= step_handle.required_group_count

    def _remaining_previous_debt(self, transfer_snapshot: dict[str, int]) -> int:
        committed_previous_groups = transfer_snapshot["committed_previous_groups"]
        previous_partition_quota = transfer_snapshot["previous_partition_quota"]
        remaining_previous_debt = previous_partition_quota - committed_previous_groups
        if remaining_previous_debt < 0:
            raise RuntimeError(
                "Agentic previous partition accounting is inconsistent: "
                f"quota={previous_partition_quota}, committed={committed_previous_groups}."
            )
        return remaining_previous_debt

    def _current_window_admission_counts(
        self,
        *,
        resident_group_count: int,
        transfer_snapshot: dict[str, int],
    ) -> tuple[int, int, int, int]:
        current_admission_quota = self._current_window_admission_quota(transfer_snapshot)
        remaining_previous_debt = self._remaining_previous_debt(transfer_snapshot)
        if resident_group_count < remaining_previous_debt:
            raise RuntimeError(
                "Agentic resident accounting cannot cover remaining previous partition debt: "
                f"resident={resident_group_count}, remaining_previous_debt={remaining_previous_debt}."
            )
        resident_current_window_groups = resident_group_count - remaining_previous_debt
        current_window_admitted_groups = transfer_snapshot["committed_current_groups"] + resident_current_window_groups
        current_window_slack = current_admission_quota - current_window_admitted_groups
        if current_window_slack < 0:
            raise RuntimeError(
                "Agentic current window admission exceeded current admission quota: "
                f"admission_accounted={current_window_admitted_groups}, "
                f"current_admission_quota={current_admission_quota}, "
                f"current_partition_quota={transfer_snapshot['current_partition_quota']}."
            )
        return (
            remaining_previous_debt,
            resident_current_window_groups,
            current_window_admitted_groups,
            current_window_slack,
        )

    def _current_window_admission_quota(self, transfer_snapshot: dict[str, int]) -> int:
        if self.transfer_domain is None:
            return transfer_snapshot["current_partition_quota"]
        return int(
            getattr(self.transfer_domain, "over_sampling_batch_size", transfer_snapshot["current_partition_quota"])
        )

    def _close_status(self, step_handle: "_AgenticStepHandle") -> str | None:
        if not self._previous_partition_debt_satisfied(step_handle):
            return "previous_partition_debt"
        if not self._committed_target_satisfied(step_handle):
            return "committed_target"
        return None

    def _finish_eligible(self, step_handle: "_AgenticStepHandle") -> bool:
        return self._close_status(step_handle) is None

    def _refresh_progress(self, step_handle: "_AgenticStepHandle") -> None:
        progress = step_handle.progress
        if progress is None:
            return
        reward_snapshot = dict(self.reward_domain.accounting_snapshot())
        transfer_snapshot = dict(self.transfer_domain.accounting_snapshot())
        group_size = transfer_snapshot["group_size"]
        previous_partition_quota = transfer_snapshot["previous_partition_quota"]
        progress.update_total_sessions((step_handle.required_group_count + previous_partition_quota) * group_size)
        complete_groups = (
            transfer_snapshot["committed_previous_groups"]
            + transfer_snapshot["committed_current_groups"]
            + transfer_snapshot["ready_groups"]
            + reward_snapshot["ready_groups"]
        )
        complete_group_sessions = min(
            progress.total_sessions,
            complete_groups * group_size,
        )
        progress_snapshot = progress.snapshot()
        committed_sessions = (
            transfer_snapshot["committed_previous_groups"] + transfer_snapshot["committed_current_groups"]
        ) * group_size
        progress.update_counts(
            materialized_sessions=max(progress_snapshot["materialized_sessions"], complete_group_sessions),
            committed_sessions=max(progress_snapshot["committed_sessions"], committed_sessions),
            scored_samples=reward_snapshot["scored_samples_ready"],
        )

    def _compact_accounting_snapshot(self, step_handle: "_AgenticStepHandle", *, phase: str) -> dict[str, object]:
        self._assert_resident_group_count_invariant(context=phase)
        transfer_snapshot = dict(self.transfer_domain.accounting_snapshot())
        prepare_snapshot = dict(self.prepare_domain.accounting_snapshot())
        reward_snapshot = dict(self.reward_domain.accounting_snapshot())
        resident_group_count = self.resident_group_count
        previous_partition_quota = transfer_snapshot["previous_partition_quota"]
        (
            remaining_previous_debt,
            resident_current_window_groups,
            admission_current_window_groups,
            admission_current_slack,
        ) = self._current_window_admission_counts(
            resident_group_count=resident_group_count,
            transfer_snapshot=transfer_snapshot,
        )
        target_data_size = previous_partition_quota + transfer_snapshot["current_partition_quota"]
        admission_current_quota = self._current_window_admission_quota(transfer_snapshot)
        interrupted_previous_groups = self._interrupted_group_count(
            step_handle=step_handle,
            previous=True,
        )
        interrupted_current_groups = self._interrupted_group_count(
            step_handle=step_handle,
            previous=False,
        )
        return {
            "phase": phase,
            "rollout_id": step_handle.rollout_id,
            "train_target_groups": step_handle.required_group_count,
            "target_data_size": target_data_size,
            "resident_group_count": resident_group_count,
            "previous_partition_quota": previous_partition_quota,
            "remaining_previous_debt": remaining_previous_debt,
            "resident_current_window_groups": resident_current_window_groups,
            "admission_current_window_groups": admission_current_window_groups,
            "admission_current_slack": admission_current_slack,
            "admission_current_quota": admission_current_quota,
            "interrupted_previous_groups": interrupted_previous_groups,
            "interrupted_current_groups": interrupted_current_groups,
            "close_accounted_previous_groups": transfer_snapshot["committed_previous_groups"]
            + (interrupted_previous_groups if self._interrupted_close_accounting_enabled(step_handle) else 0),
            "close_accounted_current_groups": transfer_snapshot["committed_current_groups"]
            + (interrupted_current_groups if self._interrupted_close_accounting_enabled(step_handle) else 0),
            "finish_eligible_mode": "committed_plus_interrupted"
            if self._interrupted_close_accounting_enabled(step_handle)
            else "committed_current",
            "prepare_pool_groups": prepare_snapshot["pool_groups"],
            "prepare_pool_target_groups": prepare_snapshot["pool_target_groups"],
            "prepare_pending_groups": prepare_snapshot["pending_prepare_groups"],
            "prepare_ready_groups": prepare_snapshot["ready_groups"],
            "prepare_warming_groups": prepare_snapshot["warming_groups"],
            "reward_waiting_groups": reward_snapshot["waiting_groups"],
            "reward_waiting_records": reward_snapshot["waiting_records"],
            "reward_ready_groups": reward_snapshot["ready_groups"],
            "transfer_committed_current_groups": transfer_snapshot["committed_current_groups"],
            "transfer_committed_previous_groups": transfer_snapshot["committed_previous_groups"],
            "transfer_current_quota": transfer_snapshot["current_partition_quota"],
            "transfer_previous_quota": transfer_snapshot["previous_partition_quota"],
            "finish_eligible_blocked_by": self._close_status(step_handle),
            "transfer_ready_groups": transfer_snapshot["ready_groups"],
        }

    async def _wait_step_target(self, step_handle: "_AgenticStepHandle") -> None:
        while True:
            self._raise_resident_dataflow_error()
            if self._interrupted_close_accounting_enabled(step_handle):
                await self._refresh_close_accounting()
            self._refresh_progress(step_handle)
            async with self._dataflow_lock():
                if self._finish_eligible(step_handle):
                    self.step_admission_closed = True
                    self._refresh_progress(step_handle)
                    return
            progressed = await self._wait_for_step_event()
            now = time.time()
            if progressed:
                self._refresh_progress(step_handle)
                step_handle.last_progress_at = now
                continue
            if (
                now - step_handle.last_progress_at >= _IDLE_HEARTBEAT_INTERVAL_S
                and now - step_handle.last_idle_heartbeat_at >= _IDLE_HEARTBEAT_INTERVAL_S
            ):
                step_handle.last_idle_heartbeat_at = now
                runtime_snapshot = self.runtime_domain.debug_snapshot()
                logger.info(
                    "Agentic rollout=%s idle_heartbeat idle_for=%.1fs committed_target=%s required=%s "
                    "step_accounting_snapshot=%s runtime=%s",
                    step_handle.rollout_id,
                    now - step_handle.last_progress_at,
                    self._committed_target_count(),
                    step_handle.required_group_count,
                    self._compact_accounting_snapshot(step_handle, phase="idle_heartbeat"),
                    runtime_snapshot,
                )

    async def _gate_step_irs_once(
        self,
        step_handle: "_AgenticStepHandle",
        *,
        shutdown_all_irs: bool = False,
    ) -> int:
        """Apply the step-close IR gate once.

        Partial close uses a partial-resume gate. Non-partial close gates the
        current rollout for discard. Terminal/error cleanup gates all train
        IRs.
        """
        if step_handle.rollout_irs_gated and not shutdown_all_irs:
            return 0
        if shutdown_all_irs:
            locked = await self.runtime_domain.gate_all_irs_for_shutdown()
        elif self.runtime_domain.args.partial_rollout or self.runtime_domain.args.fully_async:
            locked = await self.runtime_domain.gate_rollout_irs_for_partial_resume()
        else:
            locked = await self.runtime_domain.gate_rollout_irs_for_discard()
        step_handle.rollout_irs_gated = True
        return locked

    async def _wait_for_active_request_counts(
        self,
        *,
        require_no_protected: bool,
        require_no_abortable: bool,
    ) -> dict[str, int]:
        while True:
            self._raise_resident_dataflow_error()
            counts = await self.runtime_domain.active_rollout_request_counts()
            protected_active = counts.get("protected_active", 0) or 0
            abortable_active = counts.get("abortable_active", 0) or 0
            evaluating = counts.get("evaluating", 0) or 0
            if evaluating == 0:
                protected_done = (not require_no_protected) or protected_active == 0
                abortable_done = (not require_no_abortable) or abortable_active == 0
                if protected_done and abortable_done:
                    return counts
            await self._wait_for_step_event()

    def _drop_completed_surplus_groups_locked(self) -> int:
        dropped_completed_groups = self.reward_domain.drop_completed_groups()
        return dropped_completed_groups

    async def _seal_step(
        self,
        step_handle: "_AgenticStepHandle",
        *,
        shutdown_all_irs: bool = False,
    ) -> None:
        if step_handle.active_backend_requests_sealed:
            return
        tail_carry_enabled = bool(
            (self.runtime_domain.args.partial_rollout or self.runtime_domain.args.fully_async) and not shutdown_all_irs
        )
        if tail_carry_enabled:
            await self._wait_for_active_request_counts(
                require_no_protected=True,
                require_no_abortable=False,
            )
        await self._gate_step_irs_once(step_handle, shutdown_all_irs=shutdown_all_irs)
        abort_result = await self.runtime_domain.abort_rollout_requests()
        logger.info(
            "Agentic rollout=%s abort_all requested_workers=%s failed_workers=%s abortable_active=%s",
            step_handle.rollout_id,
            abort_result.get("abort_requested_workers", 0) or 0,
            abort_result.get("abort_failed_workers", 0) or 0,
            abort_result.get("abortable_active", 0) or 0,
        )
        await self._wait_for_active_request_counts(
            require_no_protected=tail_carry_enabled,
            require_no_abortable=True,
        )
        step_handle.active_backend_requests_sealed = True

    async def _discard_resident_tail(self) -> None:
        async with self._dataflow_lock():
            await self.runtime_domain.drop_resident_results()
            await self.reward_domain.drop_resident_groups()
            self.transfer_domain.drop_ready_groups()

    async def close_step(
        self,
        *,
        step_handle: "_AgenticStepHandle",
        cleanup_only: bool = False,
    ) -> "_AgenticClosedStep | None":
        terminal_step = step_handle.terminal_step
        if cleanup_only:
            async with self._dataflow_lock():
                self.step_admission_closed = True
            await self._seal_step(step_handle, shutdown_all_irs=True)
            await self.stop_resident_dataflow()
            await self._discard_resident_tail()
            await self.runtime_domain.trim_agentic_session_shards(reason="cleanup")
            if step_handle.progress is not None:
                step_handle.progress.close()
                step_handle.progress = None
            return None
        await self._wait_step_target(step_handle)
        async with self._dataflow_lock():
            self.transfer_domain.close_output_window()
            output = await self.transfer_domain.build_output(extra_metrics=None)
            end_snapshot = self.transfer_domain.accounting_snapshot()
            committed_groups = self.transfer_domain.committed_transfer_groups_snapshot()
            progress_snapshot = step_handle.progress.snapshot() if step_handle.progress is not None else {}
            get_samples_times = list(self._step_get_samples_times)
            self._drop_completed_surplus_groups_locked()
            self.transfer_domain.release_step_output_payloads()
            if step_handle.progress is not None:
                step_handle.progress.close()
                step_handle.progress = None
        await self.transfer_domain.wait_for_pending_transfers()
        shutdown_all_irs = terminal_step and not step_handle.defer_terminal_shutdown_for_eval
        await self._seal_step(step_handle, shutdown_all_irs=shutdown_all_irs)
        if terminal_step or not (self.runtime_domain.args.partial_rollout or self.runtime_domain.args.fully_async):
            await self._discard_resident_tail()
        await self.runtime_domain.trim_agentic_session_shards(reason=f"close_step_{step_handle.rollout_id}")
        async with self._dataflow_lock():
            accounting_end_snapshot = self._compact_accounting_snapshot(step_handle, phase="accounting_end")
        runtime_snapshot = self.runtime_domain.debug_snapshot()
        return _AgenticClosedStep(
            output=output,
            end_snapshot=end_snapshot,
            runtime_snapshot=runtime_snapshot,
            committed_groups=committed_groups,
            progress_snapshot=progress_snapshot,
            get_samples_times=get_samples_times,
            accounting_end_snapshot=accounting_end_snapshot,
        )

    async def run_step(
        self,
        *,
        args,
        rollout_id,
        defer_terminal_shutdown_for_eval: bool = False,
    ) -> RolloutFnTrainOutput:
        rollout_started_at = time.monotonic()
        await start_sglang_profile(args, rollout_id)
        sglang_profile_stopped = False
        rollout_batch_size = args.rollout_batch_size
        metric_gatherer = MetricGatherer()
        filter_path = args.dynamic_sampling_filter_path
        if filter_path:
            from relax.utils.utils import load_function

            filter_fn = load_function(filter_path)

            def group_filter(group) -> bool:
                output = call_dynamic_filter(filter_fn, args, group)
                keep = output.keep
                if not keep:
                    metric_gatherer.on_dynamic_filter_drop(reason=getattr(output, "reason", None))
                return keep

        else:
            group_filter = None
        step_handle = await self.open_step(
            args=args,
            rollout_id=rollout_id,
            group_filter=group_filter,
            defer_terminal_shutdown_for_eval=defer_terminal_shutdown_for_eval,
        )
        terminal_step = step_handle.terminal_step
        transfer_start_snapshot = self.transfer_domain.accounting_snapshot()
        accounting_start_snapshot = self._compact_accounting_snapshot(step_handle, phase="accounting_start")
        logger.info(
            format_agentic_event(
                "ROLLOUT",
                "accounting_start",
                rollout=rollout_id,
                group_size=transfer_start_snapshot["group_size"],
                target_groups=rollout_batch_size,
                terminal_step=terminal_step,
                ready_groups=transfer_start_snapshot["ready_groups"],
            )
        )
        logger.info(
            format_agentic_event(
                "ROLLOUT",
                "accounting_snapshot",
                **accounting_start_snapshot,
            )
        )
        step_sealed = False
        try:
            # Resident flow: keep moving data until previous/current transfer slots are filled.
            # Protected sessions finish before the IR release gate catches the remaining abortable request tail.
            closed_step = await self.close_step(step_handle=step_handle)
            if closed_step is None:
                raise RuntimeError("Agentic pipeline close_step returned no output outside cleanup mode.")
            step_sealed = True
            output = closed_step.output
            end_snapshot = closed_step.end_snapshot
            runtime_snapshot = closed_step.runtime_snapshot
            committed_groups = closed_step.committed_groups
            progress_snapshot = closed_step.progress_snapshot
            logger.info(
                format_agentic_event(
                    "ROLLOUT",
                    "accounting_end",
                    rollout=rollout_id,
                    group_size=end_snapshot["group_size"],
                    committed_previous_groups=end_snapshot["committed_previous_groups"],
                    committed_current_groups=end_snapshot["committed_current_groups"],
                    materialized_sessions=progress_snapshot["materialized_sessions"],
                    scored_samples=progress_snapshot["scored_samples"],
                    accounting_snapshot=closed_step.accounting_end_snapshot,
                )
            )
            resident_groups = int(closed_step.accounting_end_snapshot.get("resident_group_count", 0) or 0)
            if resident_groups:
                logger.info(
                    f"Rollout not completed for rollout_id: {rollout_id}, have {resident_groups} samples aborted."
                )
            else:
                logger.info(f"Rollout fully completed for rollout_id: {rollout_id}.")
            logger.info(
                "Agentic runtime volatile resource snapshot rollout=%s emitted=%s drained=%s "
                "waiting_materialized_records=%s prepare_gate_blocked_irs=%s "
                "partial_resume_gate_blocked_irs=%s session_debug_totals=%s",
                rollout_id,
                runtime_snapshot.get("emitted_materialized_session_count_total"),
                runtime_snapshot.get("drained_materialized_session_count_total"),
                runtime_snapshot.get("waiting_materialized_records"),
                runtime_snapshot.get("prepare_gate_blocked_ir_count"),
                runtime_snapshot.get("partial_resume_gate_blocked_ir_count"),
                runtime_snapshot.get("session_debug_totals"),
            )
            committed_samples = (
                _assert_and_flatten_agentic_export_samples(committed_groups) if committed_groups else []
            )
            agentic_metrics = dict(output.metrics or {})
            agentic_metrics.update(
                _aggregate_rollout_timing_from_agentic_trace(
                    samples=committed_samples,
                    get_samples_times=closed_step.get_samples_times,
                )
            )
            agentic_metrics.update(metric_gatherer.collect())
            _merge_non_conflicting_metrics(
                agentic_metrics, compute_rollout_explicit_reward_metrics(args, committed_samples)
            )
            _merge_non_conflicting_metrics(agentic_metrics, collect_agentic_metadata_metrics(committed_samples))
            await stop_sglang_profile(args, rollout_id)
            sglang_profile_stopped = True
            if committed_samples:
                last_sample = committed_samples[-1]
                logger.info(
                    "Finish rollout: %s, label: %s, reward: %s",
                    [str(last_sample.prompt) + last_sample.response],
                    str(last_sample.label)[:100],
                    last_sample.reward,
                )
                rollout_time = time.monotonic() - rollout_started_at
                if args.save_debug_rollout_data is not None:
                    save_debug_rollout_data(
                        args,
                        committed_samples,
                        rollout_id=rollout_id,
                        evaluation=False,
                        tokenizer=get_agentic_runtime_resources(args).compiler.tokenizer,
                    )
                _log_rollout_data(
                    rollout_id,
                    args,
                    committed_samples,
                    agentic_metrics,
                    rollout_time,
                )
                if args.debug_rollout_only:
                    if self.transfer_domain is None or self.transfer_domain.data_system_client is None:
                        raise RuntimeError("Agentic debug rollout cleanup requires an initialized TransferDomain.")
                    logger.info("Debug rollout only mode - data system cleanup")
                    await self.transfer_domain.data_system_client.async_clear_partition(
                        partition_id=f"train_{rollout_id}"
                    )
            output.metrics = agentic_metrics
            flat_samples = _assert_and_flatten_agentic_export_samples(output.samples)
            output.samples = flat_samples
            return output
        finally:
            if not sglang_profile_stopped:
                await stop_sglang_profile(args, rollout_id)
            if not step_sealed:
                await self.close_step(step_handle=step_handle, cleanup_only=True)
            if terminal_step and not step_handle.defer_terminal_shutdown_for_eval:
                await self.shutdown()
            if self._active_step_handle is step_handle:
                self._active_step_handle = None


def get_agentic_resident_pipeline() -> AgenticResidentPipeline:
    with _RESIDENT_PIPELINE_LOCK:
        if _RESIDENT_PIPELINE is None:
            raise RuntimeError(
                "Agentic resident pipeline has not been initialized. "
                "Call init_agentic_resident_pipeline before generate_rollout."
            )
        return _RESIDENT_PIPELINE


def init_agentic_resident_pipeline(args, data_source, data_system_client) -> None:
    global _RESIDENT_PIPELINE
    with _RESIDENT_PIPELINE_LOCK:
        if _RESIDENT_PIPELINE is None:
            _RESIDENT_PIPELINE = AgenticResidentPipeline()
        resident_pipeline = _RESIDENT_PIPELINE
    _run_on_resident_async_loop(
        resident_pipeline.init_pipeline(
            args=args,
            data_source=data_source,
            data_system_client=data_system_client,
        )
    )


def _assert_and_flatten_agentic_export_samples(groups: list[list[Sample]]) -> list[Sample]:
    flat_samples: list[Sample] = []
    seen_session_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, list):
            raise TypeError(f"Agentic export expected list[list[Sample]], got group type {type(group)}")
        for sample in group:
            session_id = sample.session_id
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError("Agentic export expects every sample to carry a non-empty session_id.")
            if session_id in seen_session_ids:
                raise RuntimeError(f"Agentic export expects one sample per session, got duplicate {session_id}.")
            seen_session_ids.add(session_id)
            flat_samples.append(sample)
    return flat_samples


@dataclass(frozen=True)
class _AgenticClosedStep:
    output: RolloutFnTrainOutput
    end_snapshot: dict[str, Any]
    runtime_snapshot: dict[str, Any]
    committed_groups: list[list[Sample]]
    progress_snapshot: dict[str, int]
    get_samples_times: list[float]
    accounting_end_snapshot: dict[str, object]


@dataclass
class _AgenticStepHandle:
    """Step boundary handle for a resident agentic pipeline.

    Dataflow and lifecycle control stay in AgenticResidentPipeline; this object
    carries step-scoped domains, progress, and accounting state.
    """

    rollout_id: int
    required_group_count: int
    terminal_step: bool = False
    defer_terminal_shutdown_for_eval: bool = False
    progress: RolloutProgress | None = None
    rollout_irs_gated: bool = False
    active_backend_requests_sealed: bool = False
    first_rollout_sample_logged: bool = False
    last_progress_at: float = field(default_factory=time.time)
    last_idle_heartbeat_at: float = 0.0


def _aggregate_rollout_timing_from_agentic_trace(
    *,
    samples: list[Sample],
    get_samples_times: list[float],
) -> dict[str, float]:
    generate_values: list[float] = []
    post_generate_values: list[float] = []
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        agentic_trace = metadata.get(TRACE_KEY)
        if not isinstance(agentic_trace, dict):
            continue
        turns = agentic_trace.get("turns")
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            generation_elapsed_s = turn.get("generation_elapsed_s")
            wall_elapsed_s = turn.get("wall_elapsed_s")
            if isinstance(generation_elapsed_s, (int, float)):
                generate_values.append(float(generation_elapsed_s))
            if (
                isinstance(wall_elapsed_s, (int, float))
                and isinstance(generation_elapsed_s, (int, float))
                and wall_elapsed_s >= generation_elapsed_s
            ):
                post_generate_values.append(float(wall_elapsed_s) - float(generation_elapsed_s))

    metrics: dict[str, float] = {}
    if generate_values:
        metrics["perf_detail/rollout/generate_time/mean"] = sum(generate_values) / len(generate_values)
        metrics["perf_detail/rollout/generate_time/max"] = max(generate_values)
    if post_generate_values:
        metrics["perf_detail/rollout/post_generate_time/mean"] = sum(post_generate_values) / len(post_generate_values)
        metrics["perf_detail/rollout/post_generate_time/max"] = max(post_generate_values)
    if get_samples_times:
        metrics["perf_detail/rollout/get_samples_time/total"] = sum(get_samples_times)
        metrics["perf_detail/rollout/get_samples_time/mean"] = sum(get_samples_times) / len(get_samples_times)
    return metrics


def _merge_non_conflicting_metrics(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key not in target:
            target[key] = value


def collect_agentic_metadata_metrics(samples: list[Sample]) -> dict[str, float]:
    metric_values: dict[str, list[float]] = {}
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else None
        if not metadata:
            continue
        for key, value in metadata.items():
            if not isinstance(key, str) or not key or key.startswith("_") or key in _AGENT_METADATA_INTERNAL_KEYS:
                continue
            if not isinstance(value, (int, float)):
                continue
            metric_values.setdefault(key, []).append(float(value))
    return finalize_rollout_explicit_metric_values(metric_values)


def _dict_add_prefix(d, prefix):
    return {f"{prefix}{k}": v for k, v in d.items()}


def _compute_zero_std_metrics(args, all_samples: list[Sample]) -> dict[str, float]:
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]) -> bool:
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == reward for reward in rewards)

    all_sample_groups = group_by(all_samples, lambda sample: sample.group_index)
    interesting_sample_groups = [group for group in all_sample_groups.values() if _is_zero_std(group)]
    interesting_rewards = [str(round(group[0].get_reward_value(args), 1)) for group in interesting_sample_groups]
    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_reward_cat_metrics(args, all_samples: list[Sample]) -> dict[str, float]:
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}
    samples_of_reward_cat = group_by(all_samples, lambda sample: sample.reward[reward_cat_key])
    return {
        f"error_cat/{reward_cat}": len(samples) / len(all_samples)
        for reward_cat, samples in samples_of_reward_cat.items()
    }


def _compute_rollout_metrics_from_samples(args, samples: list[Sample]) -> dict[str, float]:
    response_lengths = [sample.effective_response_length for sample in samples]
    log_dict: dict[str, float] = {}
    log_dict |= _dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([has_repetition(sample.response) for sample in samples]).item()
    log_dict["truncated_ratio"] = np.mean([sample.status == Sample.Status.TRUNCATED for sample in samples]).item()
    return log_dict


def _compute_rollout_perf_metrics_from_samples(args, samples: list[Sample], rollout_time: float) -> dict[str, float]:
    non_generation_time = [sample.non_generation_time for sample in samples]
    log_dict: dict[str, float] = {"rollout_time": rollout_time}
    if max(non_generation_time) > 0:
        log_dict |= _dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths: list[int], non_generation_time_values: list[float], key: str = "") -> None:
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time_values) == 0:
            return

        aligned_non_generation_time = [
            value
            for value, length in zip(non_generation_time_values, response_lengths, strict=True)
            if length == max_response_length
        ]
        mean_non_generation_time = sum(aligned_non_generation_time) / len(aligned_non_generation_time)
        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")
    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> None:
    from relax.utils import tracking_utils

    save_rollout_result_jsonl(args, rollout_id, samples)

    if args.custom_rollout_log_function_path is not None:
        from relax.utils.misc import load_function

        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= _dict_add_prefix(_compute_rollout_metrics_from_samples(args, samples), "rollout/")
    log_dict |= _dict_add_prefix(_compute_rollout_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")


def _load_eval_dataset(args, dataset_cfg: EvalDatasetConfig):
    from relax.utils.data.data import Dataset

    resources = get_agentic_runtime_resources(args).compiler
    return Dataset(
        path=dataset_cfg.path,
        tokenizer=resources.tokenizer,
        processor=resources.processor,
        max_length=args.eval_max_prompt_len,
        prompt_key=dataset_cfg.input_key,
        label_key=dataset_cfg.label_key,
        multimodal_keys=args.multimodal_keys,
        metadata_key=dataset_cfg.metadata_key,
        tool_key=dataset_cfg.tool_key,
        apply_chat_template=args.apply_chat_template,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        use_audio_in_video=args.use_audio_in_video,
        system_prompt=args.system_prompt,
    )


def _eval_sampling_params(args, dataset_cfg: EvalDatasetConfig, *, sample_slot_idx: int) -> dict[str, Any]:
    sampling_params = {
        "temperature": dataset_cfg.temperature,
        "top_p": dataset_cfg.top_p,
        "top_k": dataset_cfg.top_k,
        "max_new_tokens": dataset_cfg.max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    if args.sglang_enable_deterministic_inference:
        sampling_params["sampling_seed"] = args.rollout_seed + sample_slot_idx
    return sampling_params


def _build_eval_samples(args, dataset_cfg: EvalDatasetConfig) -> list[Sample]:
    dataset = _load_eval_dataset(args, dataset_cfg)
    samples: list[Sample] = []
    sample_index = 0
    group_index = 0
    for prompt_sample in dataset.samples:
        for sample_slot_idx in range(dataset_cfg.n_samples_per_eval_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample.group_index = group_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(sample.metadata)
            samples.append(sample)
        group_index += 1
    return samples


async def _advance_eval_outputs(
    *,
    runtime_domain,
    reward_domain,
    completed_samples: list[Sample],
    pbar,
    do_print: bool,
) -> tuple[bool, bool, int]:
    progressed = False
    runtime_materialized_group_count = 0
    runtime_dispatch = runtime_domain.drain_ready_execution()
    if runtime_dispatch.materialized_batches or runtime_dispatch.ready_groups:
        if runtime_dispatch.materialized_batches and not reward_domain.group_rm:
            reward_domain.accept_session_materializations(runtime_dispatch.materialized_batches)
            runtime_materialized_group_count += len(
                {record["group_key"] for batch in runtime_dispatch.materialized_batches for record in batch}
            )
        for group in runtime_dispatch.ready_groups:
            await reward_domain.ingest_groups([group])
        if reward_domain.group_rm:
            runtime_materialized_group_count += len(runtime_dispatch.ready_groups)
        progressed = True

    reward_progressed = await reward_domain.step_once()
    if reward_progressed:
        ready_reward_groups = reward_domain.drain_ready_dispatch()
        for group in ready_reward_groups:
            if do_print and group:
                sample = group[0]
                logger.info(
                    "eval_rollout_single_dataset example data: "
                    f"{[str(sample.prompt) + sample.response]} "
                    f"reward={sample.reward}"
                )
                do_print = False
            completed_samples.extend(group)
            pbar.update(len(group))
        progressed = True
    return progressed, do_print, runtime_materialized_group_count


async def _run_eval_samples(
    args, rollout_id: int, dataset_cfg: EvalDatasetConfig, eval_samples: list[Sample]
) -> list[Sample]:
    if not eval_samples:
        return []

    completed_samples: list[Sample] = []
    do_print = True
    eval_sample_count = len(eval_samples)
    if args.group_rm:
        eval_prepare_groups: list[list[Sample]] = []
        current_group_index = object()
        for sample in eval_samples:
            if sample.group_index != current_group_index:
                eval_prepare_groups.append([])
                current_group_index = sample.group_index
            eval_prepare_groups[-1].append(sample)
    else:
        eval_prepare_groups = [[sample] for sample in eval_samples]
    total_group_count = len(eval_prepare_groups)
    eval_group_size = dataset_cfg.n_samples_per_eval_prompt if args.group_rm else 1
    train_prepare_pool_target_group_count = agentic_prepare_pool_target_from_args(args)
    train_prepare_pool_target_session_count = train_prepare_pool_target_group_count * args.n_samples_per_prompt
    eval_prepare_pool_target_group_count = min(
        total_group_count,
        (train_prepare_pool_target_session_count + eval_group_size - 1) // eval_group_size,
    )
    eval_runtime_admission_group_count = total_group_count
    pbar = tqdm(total=eval_sample_count, desc=f"Eval {dataset_cfg.name}", unit="sample")
    logger.info(
        "Agentic eval dataset=%s total_samples=%s total_group_count=%s "
        "prepare_pool_target_group_count=%s runtime_admission_group_count=%s",
        dataset_cfg.name,
        eval_sample_count,
        total_group_count,
        eval_prepare_pool_target_group_count,
        eval_runtime_admission_group_count,
    )
    eval_namespace = str(dataset_cfg.name).replace("/", "_").replace(" ", "_")
    for sample in eval_samples:
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["eval_dataset"] = dataset_cfg.name
        sample.session_id = f"eval_session_{eval_namespace}_{rollout_id}_{sample.index}"

    prepare_domain = PrepareDomain(
        scope_id=_eval_scope_id(dataset_name=str(dataset_cfg.name), rollout_id=rollout_id),
        data_source=None,
        pool_target_group_count=eval_prepare_pool_target_group_count,
    )
    runtime_domain = RuntimeDomain(
        args=args,
        scope_id=prepare_domain.scope_id,
    )
    runtime_domain.rebind_step(
        args=args,
        rollout_id=rollout_id,
    )
    reward_domain = RewardDomain(
        args=args,
        group_filter=None,
    )
    prepare_domain.configure(
        runtime_driver=runtime_domain,
        pool_target_group_count=eval_prepare_pool_target_group_count,
    )

    try:
        await runtime_domain.enter_eval()
        runtime_domain.ensure_session_runner_pool(total_requests=eval_sample_count)
        next_eval_group_idx = 0
        started_group_count = 0
        runtime_materialized_group_count = 0

        def runtime_resident_group_count() -> int:
            return started_group_count - runtime_materialized_group_count

        async def refresh_eval_ready_groups(*, wait_for_progress: bool) -> int:
            if not prepare_domain.has_warming_groups():
                return 0
            if wait_for_progress:
                await asyncio.sleep(0.05)
            return await prepare_domain.refresh_ready_groups(status_fetcher=runtime_domain.prepare_group_status)

        async def fill_prepare_pool() -> int:
            nonlocal next_eval_group_idx
            if next_eval_group_idx >= len(eval_prepare_groups):
                return 0
            snapshot = prepare_domain.accounting_snapshot()
            prepare_capacity = int(snapshot["pool_target_groups"]) - int(snapshot["pool_groups"])
            if prepare_capacity <= 0:
                return 0
            select_count = min(prepare_capacity, len(eval_prepare_groups) - next_eval_group_idx)
            selected_groups = eval_prepare_groups[next_eval_group_idx : next_eval_group_idx + select_count]
            for group in selected_groups:
                for sample_slot_idx, sample in enumerate(group):
                    eval_sample_slot_idx = (
                        sample_slot_idx if args.group_rm else sample.index % dataset_cfg.n_samples_per_eval_prompt
                    )
                    sample.sampling_params = _eval_sampling_params(
                        args,
                        dataset_cfg,
                        sample_slot_idx=eval_sample_slot_idx,
                    )
            next_eval_group_idx += len(selected_groups)
            await prepare_domain.accept_prepare(selected_groups)
            return len(selected_groups)

        prepare_domain.set_pool_target(eval_prepare_pool_target_group_count)
        await fill_prepare_pool()
        while True:
            launched_count = await prepare_domain.launch_pending()
            runtime_capacity = eval_runtime_admission_group_count - runtime_resident_group_count()
            if runtime_capacity > 0:
                if not prepare_domain.has_ready_groups():
                    if prepare_domain.has_pending_prepare():
                        await prepare_domain.launch_pending()
                    if prepare_domain.has_warming_groups() or prepare_domain.has_inflight_work():
                        await refresh_eval_ready_groups(wait_for_progress=True)
                elif prepare_domain.has_warming_groups():
                    await refresh_eval_ready_groups(wait_for_progress=False)
                batch_input = await prepare_domain.lease_ready_groups(
                    quota_group_count=runtime_capacity,
                    rollout_id=rollout_id,
                )
                if batch_input is not None:
                    started_group_count += await runtime_domain.start_batch(batch_input=batch_input)
                    await fill_prepare_pool()
                    (
                        _progressed_after_start,
                        do_print,
                        materialized_groups,
                    ) = await _advance_eval_outputs(
                        runtime_domain=runtime_domain,
                        reward_domain=reward_domain,
                        completed_samples=completed_samples,
                        pbar=pbar,
                        do_print=do_print,
                    )
                    runtime_materialized_group_count += materialized_groups
                    if _progressed_after_start:
                        await fill_prepare_pool()
                    continue

            progressed, do_print, materialized_groups = await _advance_eval_outputs(
                runtime_domain=runtime_domain,
                reward_domain=reward_domain,
                completed_samples=completed_samples,
                pbar=pbar,
                do_print=do_print,
            )
            runtime_materialized_group_count += materialized_groups
            if progressed:
                await fill_prepare_pool()
                continue
            if runtime_domain.has_pending_runtime_work():
                runtime_progress = await runtime_domain.wait_for_next_runtime_slot()
                if runtime_progress is not None:
                    (
                        _progressed_after_wait,
                        do_print,
                        materialized_groups,
                    ) = await _advance_eval_outputs(
                        runtime_domain=runtime_domain,
                        reward_domain=reward_domain,
                        completed_samples=completed_samples,
                        pbar=pbar,
                        do_print=do_print,
                    )
                    runtime_materialized_group_count += materialized_groups
                    await fill_prepare_pool()
                    continue
            elif reward_domain.has_inflight_work():
                reward_completed = await reward_domain.wait_for_next_completion()
                if reward_completed:
                    await fill_prepare_pool()
                    continue

            has_prepare_work = (
                prepare_domain.has_pending_prepare()
                or prepare_domain.has_inflight_work()
                or prepare_domain.has_warming_groups()
                or prepare_domain.has_ready_groups()
            )
            if (
                next_eval_group_idx >= len(eval_prepare_groups)
                and not has_prepare_work
                and not runtime_domain.has_inflight_work()
                and not runtime_domain.has_ready_output()
                and not reward_domain.has_inflight_work()
                and not reward_domain.has_pending_submission_work()
                and not reward_domain.has_ready_output()
            ):
                break
            if (
                prepare_domain.has_pending_prepare()
                and launched_count == 0
                and not prepare_domain.has_warming_groups()
                and not prepare_domain.has_ready_groups()
            ):
                raise RuntimeError(
                    "Agentic eval prepare queue is stuck with pending groups but no runtime driver launch."
                )
            await asyncio.sleep(_BACKGROUND_POLL_INTERVAL_S)
    finally:
        pbar.close()
        await reward_domain.shutdown()
        await prepare_domain.shutdown()
        await runtime_domain.exit_eval()
        await runtime_domain.shutdown()

    completed_samples.sort(key=lambda sample: sample.index)
    return completed_samples


async def eval_rollout(args, rollout_id) -> RolloutFnEvalOutput:
    try:
        results = {}
        reward_key = args.eval_reward_key or args.reward_key
        for dataset_cfg in args.eval_datasets:
            data = await _run_eval_samples(args, rollout_id, dataset_cfg, _build_eval_samples(args, dataset_cfg))
            results[dataset_cfg.name] = {
                "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
                "samples": data,
            }
        return RolloutFnEvalOutput(data=results)
    finally:
        await asyncio.to_thread(_shutdown_resident_pipeline_after_deferred_eval, rollout_id)


def generate_rollout(args, rollout_id, data_source, data_system_client=None, evaluation=False):
    global _RESIDENT_PIPELINE, _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID
    if evaluation:
        return asyncio.run(eval_rollout(args=args, rollout_id=rollout_id))
    resident_pipeline = get_agentic_resident_pipeline()
    num_rollout = args.num_rollout
    terminal_step = num_rollout is not None and rollout_id + 1 >= num_rollout
    defer_terminal_shutdown_for_eval = terminal_step and _post_train_eval_expected(args, rollout_id, data_source)
    completed = False
    try:
        output = _run_on_resident_async_loop(
            resident_pipeline.run_step(
                args=args,
                rollout_id=rollout_id,
                defer_terminal_shutdown_for_eval=defer_terminal_shutdown_for_eval,
            )
        )
        completed = True
        if defer_terminal_shutdown_for_eval:
            with _RESIDENT_PIPELINE_LOCK:
                _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID = rollout_id
        return output
    finally:
        if terminal_step and (not defer_terminal_shutdown_for_eval or not completed):
            with _RESIDENT_PIPELINE_LOCK:
                _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID = None
                if _RESIDENT_PIPELINE is resident_pipeline:
                    _RESIDENT_PIPELINE = None
            _shutdown_resident_async_loop()
