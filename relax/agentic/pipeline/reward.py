# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from relax.agentic.pipeline import ExportMode, GroupKey, PendingExportUnit, SampleKey, sample_group_key, sample_key
from relax.agentic.profile import mark_metadata_agentic_event
from relax.utils.misc import load_function


def _sample_needs_reward(sample: Any) -> bool:
    return sample.reward is None


def _effective_reward_concurrency(args, *, explicit_limit: int | None = None) -> int | None:
    if explicit_limit is not None:
        if explicit_limit <= 0:
            raise RuntimeError(f"reward concurrency limit must be positive when set, got {explicit_limit}.")
        return explicit_limit
    configured = args.reward_max_concurrency
    if configured is None:
        return None
    if configured <= 0:
        raise RuntimeError(f"reward_max_concurrency must be positive when set, got {configured}.")
    return configured


async def _async_rm(args, sample):
    custom_rm_path = args.custom_rm_path
    if custom_rm_path:
        from relax.utils.utils import load_function

        rm_function = load_function(custom_rm_path)
        return await rm_function(args, sample)
    from relax.engine.rewards import async_rm

    return await async_rm(args, sample)


async def _batched_async_rm(args, samples):
    custom_rm_path = args.custom_rm_path
    if custom_rm_path:
        from relax.utils.utils import load_function

        rm_function = load_function(custom_rm_path)
        return await rm_function(args, samples)
    from relax.engine.rewards import batched_async_rm

    return await batched_async_rm(args, samples)


@dataclass
class RewardWaitingGroup:
    expected_count: int
    units_by_slot: dict[int, list[PendingExportUnit]] = field(default_factory=dict)

    def add_slot(self, *, slot_idx: int, units: list[PendingExportUnit]) -> None:
        self.units_by_slot[slot_idx] = units

    def is_complete(self) -> bool:
        return len(self.units_by_slot) >= self.expected_count

    def materialized_units(self) -> list[PendingExportUnit]:
        units: list[PendingExportUnit] = []
        for idx in range(self.expected_count):
            units.extend(self.units_by_slot.get(idx, []))
        return units

    def materialized_group(self) -> list[Any]:
        return [unit.sample for unit in self.materialized_units()]

    def metadata_by_slot(self) -> list[dict[Any, dict[str, Any]]]:
        payload: list[dict[Any, dict[str, Any]]] = []
        for idx in range(self.expected_count):
            slot_payload: dict[Any, dict[str, Any]] = {}
            for unit in self.units_by_slot.get(idx, []):
                slot_payload[unit.name] = copy.deepcopy(unit.sample.metadata)
            payload.append(slot_payload)
        return payload

    def materialized_slot_count(self) -> int:
        return len(self.units_by_slot)


class RewardDomain:
    def __init__(
        self,
        *,
        args,
        group_filter: Callable[[list[Any]], bool] | None,
        max_submissions_per_step: int | None = None,
        use_custom_advantage: bool = True,
    ) -> None:
        self.args = args
        self.group_rm = args.group_rm
        self.group_filter = group_filter
        self.max_submissions_per_step = _effective_reward_concurrency(args, explicit_limit=max_submissions_per_step)
        custom_advantage_path = getattr(args, "agentic_custom_advantage_path", None) if use_custom_advantage else None
        self.custom_advantage_func = (
            load_function(custom_advantage_path) if custom_advantage_path is not None else None
        )
        self._waiting_groups: dict[GroupKey, RewardWaitingGroup] = {}
        self._inflight_sample_tasks: dict[SampleKey, asyncio.Task] = {}
        self._inflight_group_tasks: dict[GroupKey, asyncio.Task] = {}
        self._completed_samples: dict[SampleKey, Any] = {}
        self._completed_group_results: dict[GroupKey, list[Any]] = {}
        self._ready_dispatches: deque[tuple[GroupKey, list[Any]]] = deque()
        self._ready_scored_sample_count = 0
        self._progress_counted_samples: set[SampleKey] = set()
        self._reward_semaphore = (
            asyncio.Semaphore(self.max_submissions_per_step) if self.max_submissions_per_step is not None else None
        )

    def rebind_step(
        self,
        *,
        group_filter: Callable[[list[Any]], bool] | None,
    ) -> None:
        self.group_filter = group_filter
        self._ready_scored_sample_count = 0
        self._progress_counted_samples.clear()

    async def drop_waiting_groups_by_key(self, group_keys: set[GroupKey]) -> int:
        if not group_keys:
            return 0
        dropped_group_keys: set[GroupKey] = set()
        cancelled_tasks: list[asyncio.Task] = []
        for key in group_keys:
            if key in self._completed_group_results:
                raise RuntimeError(f"RewardDomain cannot drop completed group after runtime discarded it: {key!r}")
        for key, _group in self._ready_dispatches:
            if key in group_keys:
                raise RuntimeError(f"RewardDomain cannot drop ready group after runtime discarded it: {key!r}")
        for key in group_keys:
            group_task = self._inflight_group_tasks.pop(key, None)
            if group_task is not None:
                group_task.cancel()
                cancelled_tasks.append(group_task)
                dropped_group_keys.add(key)
            waiting_group = self._waiting_groups.pop(key, None)
            if waiting_group is None:
                continue
            materialized_group = waiting_group.materialized_group()
            self._drop_scored_progress_for_group(materialized_group)
            cancelled_tasks.extend(self._release_group_sample_reward_cache(materialized_group))
            dropped_group_keys.add(key)
        await self._drain_cancelled_tasks(cancelled_tasks)
        return len(dropped_group_keys)

    def _cancel_inflight_sample_tasks(self) -> list[asyncio.Task]:
        inflight_sample_tasks = list(self._inflight_sample_tasks.values())
        self._inflight_sample_tasks.clear()
        for task in inflight_sample_tasks:
            task.cancel()
        return inflight_sample_tasks

    def _cancel_inflight_group_tasks(self) -> list[asyncio.Task]:
        inflight_group_tasks = list(self._inflight_group_tasks.values())
        self._inflight_group_tasks.clear()
        for task in inflight_group_tasks:
            task.cancel()
        return inflight_group_tasks

    async def _drain_cancelled_tasks(self, tasks: list[asyncio.Task]) -> None:
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _clear_resident_state(self) -> None:
        self._waiting_groups.clear()
        self._completed_samples.clear()
        self._completed_group_results.clear()
        self._ready_dispatches.clear()
        self._ready_scored_sample_count = 0
        self._progress_counted_samples.clear()

    @staticmethod
    def _unit_from_sample(sample: Any) -> PendingExportUnit:
        return PendingExportUnit(
            name=None,
            sample=sample,
            assistant_token_spans=[],
            mode=ExportMode.IMPLICIT,
        )

    async def shutdown(self) -> None:
        inflight_sample_tasks = self._cancel_inflight_sample_tasks()
        inflight_group_tasks = self._cancel_inflight_group_tasks()
        self._clear_resident_state()
        inflight_tasks = inflight_sample_tasks + inflight_group_tasks
        await self._drain_cancelled_tasks(inflight_tasks)

    def _release_group_sample_reward_cache(self, group) -> list[asyncio.Task]:
        cancelled_tasks: list[asyncio.Task] = []
        group_task = self._inflight_group_tasks.pop(sample_group_key(group), None)
        if group_task is not None:
            group_task.cancel()
            cancelled_tasks.append(group_task)
        for sample in group:
            key = sample_key(sample)
            task = self._inflight_sample_tasks.pop(key, None)
            if task is not None:
                task.cancel()
                cancelled_tasks.append(task)
            self._completed_samples.pop(key, None)
        return cancelled_tasks

    def _drop_scored_progress_for_group(self, group) -> None:
        if self.group_rm:
            return
        for sample in group:
            key = sample_key(sample)
            if key in self._progress_counted_samples:
                self._progress_counted_samples.remove(key)
                self._ready_scored_sample_count -= 1

    def accept_session_materializations(self, outputs) -> None:
        for batch in outputs:
            for record in batch:
                group_wait_key = record["group_key"]
                waiting_group = self._waiting_groups.get(group_wait_key)
                if waiting_group is None:
                    waiting_group = RewardWaitingGroup(
                        expected_count=record["expected_count"],
                    )
                    self._waiting_groups[group_wait_key] = waiting_group
                pending_units = record["pending_units"]
                waiting_group.add_slot(
                    slot_idx=record["slot_idx"],
                    units=pending_units,
                )
                for sample in (unit.sample for unit in pending_units):
                    mark_metadata_agentic_event(sample.metadata, "reward_arrive_at")
                    self._note_scored_sample(sample)
                if self.custom_advantage_func is None:
                    self._start_missing_sample_rewards([unit.sample for unit in pending_units])

    async def ingest_groups(self, groups) -> None:
        for group in groups:
            await self._enqueue_group(group=group)

    async def _enqueue_group(self, *, group) -> None:
        if self.group_rm:
            key = sample_group_key(group)
            if key not in self._completed_group_results and key not in self._inflight_group_tasks:
                self._inflight_group_tasks[key] = asyncio.create_task(self._run_group_reward(copy.deepcopy(group)))
            return
        group_wait_key = sample_group_key(group)
        self._start_missing_sample_rewards(group)
        if self._all_group_rewards_ready(group):
            await self._finalize_group(key=group_wait_key, group=group)
            return
        self._waiting_groups[group_wait_key] = RewardWaitingGroup(
            expected_count=len(group),
            units_by_slot={idx: [self._unit_from_sample(sample)] for idx, sample in enumerate(group)},
        )

    def _start_missing_sample_rewards(self, group) -> int:
        started = 0
        for sample in group:
            key = sample_key(sample)
            if (
                key in self._completed_samples
                or key in self._inflight_sample_tasks
                or not _sample_needs_reward(sample)
            ):
                continue
            self._inflight_sample_tasks[key] = asyncio.create_task(self._run_sample_reward(sample))
            started += 1
        return started

    def _note_scored_sample(self, sample) -> None:
        if self.group_rm:
            return
        if _sample_needs_reward(sample):
            return
        key = sample_key(sample)
        if key in self._progress_counted_samples:
            return
        self._progress_counted_samples.add(key)
        self._ready_scored_sample_count += 1

    async def _run_sample_reward(self, sample):
        if self._reward_semaphore is None:
            mark_metadata_agentic_event(sample.metadata, "reward_start_at")
            sample.reward = await _async_rm(self.args, sample)
            mark_metadata_agentic_event(sample.metadata, "reward_end_at")
            self._note_scored_sample(sample)
            return sample
        async with self._reward_semaphore:
            mark_metadata_agentic_event(sample.metadata, "reward_start_at")
            sample.reward = await _async_rm(self.args, sample)
            mark_metadata_agentic_event(sample.metadata, "reward_end_at")
        self._note_scored_sample(sample)
        return sample

    async def _run_group_reward(self, group):
        if self._reward_semaphore is None:
            group_started_at = time.time()
            for sample in group:
                mark_metadata_agentic_event(sample.metadata, "reward_start_at", group_started_at)
            rewards = await _batched_async_rm(self.args, group)
            for sample, reward in zip(group, rewards, strict=True):
                sample.reward = reward
                mark_metadata_agentic_event(sample.metadata, "reward_end_at")
            return group
        async with self._reward_semaphore:
            group_started_at = time.time()
            for sample in group:
                mark_metadata_agentic_event(sample.metadata, "reward_start_at", group_started_at)
            rewards = await _batched_async_rm(self.args, group)
            for sample, reward in zip(group, rewards, strict=True):
                sample.reward = reward
                mark_metadata_agentic_event(sample.metadata, "reward_end_at")
        return group

    def _rewarded_sample(self, sample):
        key = sample_key(sample)
        task = self._inflight_sample_tasks.get(key)
        if task is not None and task.done():
            self._inflight_sample_tasks.pop(key, None)
            self._completed_samples[key] = task.result()
        rewarded = self._completed_samples.get(key)
        if rewarded is not None:
            sample.reward = copy.deepcopy(getattr(rewarded, "reward", None))
            if getattr(rewarded, "metadata", None):
                sample.metadata.update(copy.deepcopy(rewarded.metadata))
            self._note_scored_sample(sample)
        return rewarded

    def _all_group_rewards_ready(self, group) -> bool:
        for sample in group:
            self._rewarded_sample(sample)
            if _sample_needs_reward(sample):
                return False
        return True

    @staticmethod
    def _apply_custom_advantage(value: Any, unit: PendingExportUnit) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            unit.sample.custom_advantage = float(value)
            return
        # TODO(agentic): support assistant-turn-level advantages from list[float].
        # TODO(agentic): support assistant-token-level advantages from list[list[float]].
        raise TypeError("custom advantage output must be a number in this version.")

    async def _run_custom_advantage(self, waiting_group: RewardWaitingGroup) -> list[Any] | None:
        result = self.custom_advantage_func(waiting_group.metadata_by_slot())
        if asyncio.iscoroutine(result):
            result = await result
        if result is None:
            # None is the group-level drop signal from custom advantage.
            return None
        for slot_idx in range(waiting_group.expected_count):
            slot_values = result[slot_idx]
            for unit in waiting_group.units_by_slot.get(slot_idx, []):
                self._apply_custom_advantage(slot_values[unit.name], unit)
                self._note_scored_sample(unit.sample)
        return waiting_group.materialized_group()

    def _harvest_group_reward_completions(self) -> int:
        done_keys = [key for key, task in self._inflight_group_tasks.items() if task.done()]
        for key in done_keys:
            task = self._inflight_group_tasks.pop(key)
            self._completed_group_results[key] = task.result()
        return len(done_keys)

    async def _harvest_sample_reward_completions(self) -> int:
        done_keys = [key for key, task in self._inflight_sample_tasks.items() if task.done()]
        for key in done_keys:
            task = self._inflight_sample_tasks.pop(key)
            self._completed_samples[key] = task.result()
        for waiting_group in self._waiting_groups.values():
            if waiting_group.is_complete() and self.custom_advantage_func is None:
                self._start_missing_sample_rewards(waiting_group.materialized_group())
        ready_groups: list[tuple[GroupKey, list[Any]]] = []
        dropped_keys: list[GroupKey] = []
        for key, waiting_group in self._waiting_groups.items():
            if key in self._completed_group_results:
                continue
            if not waiting_group.is_complete():
                continue
            if self.custom_advantage_func is not None:
                group = await self._run_custom_advantage(waiting_group)
                if group is None:
                    group = waiting_group.materialized_group()
                    self._drop_scored_progress_for_group(group)
                    self._release_group_sample_reward_cache(group)
                    dropped_keys.append(key)
                    continue
                ready_groups.append((key, group))
                continue
            group = waiting_group.materialized_group()
            if not self._all_group_rewards_ready(group):
                continue
            ready_groups.append((key, group))
        for key, group in ready_groups:
            self._waiting_groups.pop(key, None)
            self._completed_group_results[key] = group
        for key in dropped_keys:
            self._waiting_groups.pop(key, None)
        return len(done_keys) + len(ready_groups) + len(dropped_keys)

    async def precompute_once(self) -> bool:
        progressed = False
        if await self._harvest_sample_reward_completions():
            progressed = True
        if self._harvest_group_reward_completions():
            progressed = True
        return progressed

    async def finalize_ready_for_step(self) -> bool:
        progressed = False
        for key in list(self._completed_group_results.keys()):
            completed_group = self._completed_group_results.pop(key, None)
            if completed_group is None:
                continue
            await self._finalize_group(key=key, group=completed_group)
            progressed = True
        return progressed

    async def _finalize_group(self, *, key: GroupKey, group) -> None:
        group_finalize_started_at = time.time()
        for sample in group:
            mark_metadata_agentic_event(sample.metadata, "group_finalize_start_at", group_finalize_started_at)
        if self.group_filter is not None and not self.group_filter(group):
            for sample in group:
                mark_metadata_agentic_event(sample.metadata, "group_finalize_end_at")
            self._drop_scored_progress_for_group(group)
            self._release_group_sample_reward_cache(group)
            return
        self._ready_dispatches.append((key, group))
        group_finalize_ended_at = time.time()
        for sample in group:
            mark_metadata_agentic_event(sample.metadata, "group_finalize_end_at", group_finalize_ended_at)
        self._release_group_sample_reward_cache(group)

    async def step_once(self) -> bool:
        progressed = False
        if await self.precompute_once():
            progressed = True
        if await self.finalize_ready_for_step():
            progressed = True
        return progressed

    async def wait_for_next_completion(self) -> bool:
        wait_set = set(self._inflight_sample_tasks.values()) | set(self._inflight_group_tasks.values())
        if wait_set:
            done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
            return bool(done)
        return False

    def drain_ready_dispatch(self, *, max_groups: int | None = None) -> list[list[Any]]:
        ready_groups: list[list[Any]] = []
        retained_groups: list[list[Any]] = []
        if max_groups is None:
            remaining_groups = None
        else:
            remaining_groups = max_groups
        while self._ready_dispatches:
            key, group = self._ready_dispatches.popleft()
            if remaining_groups is None or remaining_groups > 0:
                ready_groups.append(group)
                if remaining_groups is not None:
                    remaining_groups -= 1
                continue
            retained_groups.append((key, group))
        for item in reversed(retained_groups):
            self._ready_dispatches.appendleft(item)
        return ready_groups

    def drop_completed_groups(self) -> int:
        dropped_groups = 0
        while self._ready_dispatches:
            self._ready_dispatches.popleft()
            dropped_groups += 1
        for key, completed_group in list(self._completed_group_results.items()):
            self._completed_group_results.pop(key, None)
            self._release_group_sample_reward_cache(completed_group)
            dropped_groups += 1
        return dropped_groups

    async def drop_resident_groups(self) -> int:
        dropped_groups = self.drop_completed_groups()
        cancelled_tasks = self._cancel_inflight_group_tasks()
        dropped_groups += len(cancelled_tasks)
        for key, waiting_group in list(self._waiting_groups.items()):
            self._waiting_groups.pop(key, None)
            materialized_group = waiting_group.materialized_group()
            self._drop_scored_progress_for_group(materialized_group)
            cancelled_tasks.extend(self._release_group_sample_reward_cache(materialized_group))
            dropped_groups += 1
        await self._drain_cancelled_tasks(cancelled_tasks)
        return dropped_groups

    def accounting_snapshot(self) -> dict[str, int]:
        waiting_groups = len(self._waiting_groups)
        waiting_records = sum(
            waiting_group.materialized_slot_count() for waiting_group in self._waiting_groups.values()
        )
        ready_groups = len(self._ready_dispatches)
        completed_groups = len(self._completed_group_results)
        inflight_sample_rewards = len(self._inflight_sample_tasks)
        inflight_group_rewards = len(self._inflight_group_tasks)
        return {
            "waiting_groups": waiting_groups,
            "waiting_records": waiting_records,
            "ready_groups": ready_groups,
            "completed_groups": completed_groups,
            "inflight_sample_rewards": inflight_sample_rewards,
            "inflight_group_rewards": inflight_group_rewards,
            "scored_samples_ready": self._ready_scored_sample_count,
        }

    def resident_group_keys(self) -> set[GroupKey]:
        keys = set(self._waiting_groups)
        keys.update(self._completed_group_results)
        keys.update(self._inflight_group_tasks)
        keys.update(key for key, _group in self._ready_dispatches)
        return keys

    def has_inflight_work(self) -> bool:
        return bool(self._inflight_group_tasks or self._inflight_sample_tasks)

    def has_pending_submission_work(self) -> bool:
        return bool(self._waiting_groups or self._inflight_group_tasks)

    def has_ready_output(self) -> bool:
        return bool(self._ready_dispatches)
