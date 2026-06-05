# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from relax.agentic.pipeline import GroupKey, SampleKey, sample_group_key, sample_key
from relax.agentic.profile import mark_metadata_agentic_event


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
    samples_by_slot: dict[int, list[Any]] = field(default_factory=dict)

    def add_slot(self, *, slot_idx: int, samples: list[Any]) -> None:
        self.samples_by_slot[slot_idx] = samples

    def is_complete(self) -> bool:
        return len(self.samples_by_slot) >= self.expected_count

    def materialized_group(self) -> list[Any]:
        group: list[Any] = []
        for idx in range(self.expected_count):
            group.extend(self.samples_by_slot.get(idx, []))
        return group

    def materialized_slot_count(self) -> int:
        return len(self.samples_by_slot)


class RewardDomain:
    def __init__(
        self,
        *,
        args,
        group_filter: Callable[[list[Any]], bool] | None,
        max_submissions_per_step: int | None = None,
    ) -> None:
        self.args = args
        self.group_rm = args.group_rm
        self.group_filter = group_filter
        self.max_submissions_per_step = _effective_reward_concurrency(args, explicit_limit=max_submissions_per_step)
        self._waiting_groups: dict[GroupKey, RewardWaitingGroup] = {}
        self._inflight_sample_tasks: dict[SampleKey, asyncio.Task] = {}
        self._inflight_group_tasks: dict[GroupKey, asyncio.Task] = {}
        self._completed_samples: dict[SampleKey, Any] = {}
        self._completed_group_results: dict[GroupKey, list[Any]] = {}
        self._ready_dispatches: deque[list[Any]] = deque()
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
        for group in self._ready_dispatches:
            key = sample_group_key(group)
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
            cancelled_tasks.extend(self._release_group_sample_reward_cache(waiting_group.materialized_group()))
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
            self._progress_counted_samples.discard(key)
        return cancelled_tasks

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
                waiting_group.add_slot(
                    slot_idx=record["slot_idx"],
                    samples=record["samples"],
                )
                for sample in record["samples"]:
                    mark_metadata_agentic_event(sample.metadata, "reward_arrive_at")
                    self._note_scored_sample(sample)
                self._start_missing_sample_rewards(record["samples"])

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
            await self._finalize_group(group=group)
            return
        self._waiting_groups[group_wait_key] = RewardWaitingGroup(
            expected_count=len(group),
            samples_by_slot={idx: [sample] for idx, sample in enumerate(group)},
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
            rewarded = self._rewarded_sample(sample)
            if rewarded is not None and rewarded is not sample:
                sample.reward = copy.deepcopy(getattr(rewarded, "reward", None))
                if getattr(rewarded, "metadata", None):
                    sample.metadata.update(copy.deepcopy(rewarded.metadata))
            if _sample_needs_reward(sample):
                return False
        return True

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
            if waiting_group.is_complete():
                self._start_missing_sample_rewards(waiting_group.materialized_group())
        ready_groups: list[tuple[GroupKey, list[Any]]] = []
        for key, waiting_group in self._waiting_groups.items():
            if key in self._completed_group_results:
                continue
            if not waiting_group.is_complete():
                continue
            group = waiting_group.materialized_group()
            if not self._all_group_rewards_ready(group):
                continue
            ready_groups.append((key, group))
        for key, group in ready_groups:
            self._waiting_groups.pop(key, None)
            self._completed_group_results[key] = group
        return len(done_keys) + len(ready_groups)

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
            await self._finalize_group(group=completed_group)
            progressed = True
        return progressed

    async def _finalize_group(self, *, group) -> None:
        group_finalize_started_at = time.time()
        for sample in group:
            mark_metadata_agentic_event(sample.metadata, "group_finalize_start_at", group_finalize_started_at)
        if self.group_filter is not None and not self.group_filter(group):
            for sample in group:
                mark_metadata_agentic_event(sample.metadata, "group_finalize_end_at")
            self._release_group_sample_reward_cache(group)
            return
        self._ready_dispatches.append(group)
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
            group = self._ready_dispatches.popleft()
            if remaining_groups is None or remaining_groups > 0:
                ready_groups.append(group)
                if remaining_groups is not None:
                    remaining_groups -= 1
                continue
            retained_groups.append(group)
        for item in reversed(retained_groups):
            self._ready_dispatches.appendleft(item)
        return ready_groups

    def _drop_ready_dispatches(self) -> int:
        dropped_groups = 0
        while self._ready_dispatches:
            self._ready_dispatches.popleft()
            dropped_groups += 1
        return dropped_groups

    def drop_completed_groups(self) -> int:
        dropped_groups = self._drop_ready_dispatches()
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
            cancelled_tasks.extend(self._release_group_sample_reward_cache(waiting_group.materialized_group()))
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
        keys.update(sample_group_key(group) for group in self._ready_dispatches)
        return keys

    def has_inflight_work(self) -> bool:
        return bool(self._inflight_group_tasks or self._inflight_sample_tasks)

    def has_pending_submission_work(self) -> bool:
        return bool(self._waiting_groups or self._inflight_group_tasks)

    def has_ready_output(self) -> bool:
        return bool(self._ready_dispatches)
