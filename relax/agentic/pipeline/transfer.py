# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import time
from collections import deque

from relax.agentic.pipeline import GroupKey, sample_group_key
from relax.agentic.profile import (
    mark_sample_agentic_event,
    mark_sample_agentic_event_once,
)
from relax.engine.rollout.base_types import RolloutFnTrainOutput
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


async def _transfer_batch_to_data_system(
    *,
    args,
    batch_samples: list,
    batch_count: int,
    rollout_id: int,
    data_system_client,
    metadata=None,
    is_last: bool = False,
) -> list[str]:
    from relax.utils.utils import convert_samples_to_train_data

    if batch_samples:
        enqueue_at = time.time()
        flat_samples = batch_samples
        while flat_samples and isinstance(flat_samples[0], list):
            flat_samples = sum(flat_samples, [])
        for sample in flat_samples:
            mark_sample_agentic_event(sample, "transfer_enqueue_at", enqueue_at)
    else:
        logger.warning(
            "transfer_batch_to_data_system called with empty batch_samples for rollout_id=%s, batch_count=%s",
            rollout_id,
            batch_count,
        )
        return []
    batch_samples = sorted(
        batch_samples, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )
    while isinstance(batch_samples[0], list):
        batch_samples = sum(batch_samples, [])
    rollout_batch = convert_samples_to_train_data(args, batch_samples)
    logger.info("Prepared rollout batch %s with %s samples for transfer", batch_count, rollout_batch.numel())
    logger.info("Transferring batch rollout_batch: %s", rollout_batch)
    if metadata is None:
        metadata = await data_system_client.async_put(
            data=rollout_batch, partition_id=f"train_{rollout_id}", is_last=is_last
        )
    else:
        # is_last only applies to the insert path; a backfill put with explicit
        # metadata writes into existing samples and must not announce end-of-stream.
        metadata = await data_system_client.async_put(data=rollout_batch, metadata=metadata)
    if metadata and metadata.size > 0:
        total_lengths = rollout_batch.get("total_lengths", None)
        if total_lengths is not None:
            custom_meta = [{"total_lengths": int(tl)} for tl in total_lengths]
            metadata.update_custom_meta(custom_meta)
            await data_system_client.async_set_custom_meta(metadata)
    logger.info("Batch %s transferred successfully for rollout_id: %s", batch_count, rollout_id)
    return list(rollout_batch.keys())


class TransferDomain:
    def __init__(
        self,
        *,
        args,
        data_system_client,
    ) -> None:
        self.args = args
        self.rollout_id: int | None = None
        self.data_system_client = data_system_client
        self.ready_group_buffer: deque[list] = deque()
        self.rollout_batch_size = args.rollout_batch_size
        self.over_sampling_batch_size = args.over_sampling_batch_size
        self.n_samples_per_prompt = args.n_samples_per_prompt
        self.transfer_batch_group_count = (
            args.rollout_batch_size
            if args.colocate
            else args.global_batch_size // args.num_iters_per_train_update // args.n_samples_per_prompt
        )
        self._transfer_buffer: deque[list] = deque()
        self._transfer_tasks: list[asyncio.Task] = []
        self._reset_step_partition_state()

    def _reset_step_partition_state(self) -> None:
        self._step_output_groups: list[list] = []
        self._previous_partition_quota = 0
        self._current_partition_quota = self.rollout_batch_size
        self._output_window_closed = False
        self._committed_previous_group_count = 0
        self._committed_current_group_count = 0
        self._dispatched_previous_group_count = 0
        self._dispatched_current_group_count = 0

    def rebind_step(
        self,
        *,
        rollout_id: int,
    ) -> None:
        self._reap_completed_transfer_tasks()
        if self._transfer_buffer or self._transfer_tasks:
            raise RuntimeError(
                "TransferDomain cannot rebind step with pending transfer state: "
                f"rollout_id={self.rollout_id}, next_rollout_id={rollout_id}, "
                f"buffer_groups={len(self._transfer_buffer)}, pending_tasks={len(self._transfer_tasks)}."
            )
        self.rollout_id = rollout_id
        self._reset_step_partition_state()

    def _reap_completed_transfer_tasks(self) -> None:
        if not self._transfer_tasks:
            return
        completed_tasks: list[asyncio.Task] = []
        pending_tasks: list[asyncio.Task] = []
        for task in self._transfer_tasks:
            if task.done():
                completed_tasks.append(task)
            else:
                pending_tasks.append(task)
        self._transfer_tasks = pending_tasks
        for task in completed_tasks:
            task.result()

    def _buffer_transfer_group(self, group) -> None:
        buffered_at = time.time()
        self._transfer_buffer.append(group)
        for sample in group:
            mark_sample_agentic_event_once(sample, "transfer_buffer_enter_at", buffered_at)

    async def _dispatch_transfer_batch(self, *, groups, partition_rollout_id: int, is_last: bool = False) -> None:
        if self.data_system_client is None:
            return
        if partition_rollout_id < self.rollout_id:
            yielded_groups = self._committed_previous_group_count
            target_groups = self._previous_partition_quota
        else:
            yielded_groups = self._committed_current_group_count
            target_groups = self._current_partition_quota
        logger.info("Total yielded: %s/%s for step: %s", yielded_groups, target_groups, partition_rollout_id)
        release_started_at = time.time()
        for group in groups:
            for sample in group:
                mark_sample_agentic_event(sample, "transfer_release_start_at", release_started_at)
        await _transfer_batch_to_data_system(
            args=self.args,
            batch_samples=groups,
            batch_count=len(groups),
            rollout_id=partition_rollout_id,
            data_system_client=self.data_system_client,
            is_last=is_last,
        )
        release_ended_at = time.time()
        for group in groups:
            for sample in group:
                mark_sample_agentic_event(sample, "transfer_release_end_at", release_ended_at)

    def _spawn_transfer(self, *, force: bool = False) -> int:
        self._reap_completed_transfer_tasks()
        if not self._transfer_buffer:
            return 0
        if not force and len(self._transfer_buffer) < self.transfer_batch_group_count:
            return 0
        if self._dispatched_previous_group_count < self._committed_previous_group_count:
            partition_rollout_id = self.rollout_id - 1
            take_count = min(
                len(self._transfer_buffer),
                self._committed_previous_group_count - self._dispatched_previous_group_count,
            )
            self._dispatched_previous_group_count += take_count
            # The previous partition's quota is closed entirely within this step's
            # backfill, so reaching the quota marks its final batch.
            is_last = self._previous_partition_quota > 0 and (
                self._dispatched_previous_group_count >= self._previous_partition_quota
            )
        elif self._dispatched_current_group_count < self._committed_current_group_count:
            partition_rollout_id = self.rollout_id
            take_count = min(
                len(self._transfer_buffer),
                self._committed_current_group_count - self._dispatched_current_group_count,
            )
            self._dispatched_current_group_count += take_count
            # The current partition is only fully filled within this step (no tail
            # backfilled next step) when its committed count reached the full quota.
            # _committed_current_group_count can only reach _current_partition_quota
            # when this step met the target, so dispatching up to the quota == last.
            is_last = (
                self._current_partition_quota > 0
                and self._committed_current_group_count >= self._current_partition_quota
                and self._dispatched_current_group_count >= self._current_partition_quota
            )
        else:
            return 0
        groups = []
        for _ in range(take_count):
            groups.append(self._transfer_buffer.popleft())
        if self.data_system_client is None:
            return len(groups)
        self._transfer_tasks.append(
            asyncio.create_task(
                self._dispatch_transfer_batch(
                    groups=groups, partition_rollout_id=partition_rollout_id, is_last=is_last
                )
            )
        )
        return len(groups)

    def _spawn_ready_transfers(self) -> int:
        spawned_group_count = 0
        while True:
            spawned = self._spawn_transfer()
            if spawned <= 0:
                break
            spawned_group_count += spawned
        return spawned_group_count

    def configure_transfer_quota(
        self,
        *,
        previous_partition_quota: int,
        current_partition_quota: int,
    ) -> None:
        self._previous_partition_quota = previous_partition_quota
        self._current_partition_quota = current_partition_quota

    def close_output_window(self) -> None:
        if self.ready_group_buffer:
            logger.info(
                "Agentic transfer closing output window with resident ready groups preserved for next step: "
                "rollout_id=%s ready_groups=%s committed_previous=%s committed_current=%s "
                "previous_quota=%s current_quota=%s",
                self.rollout_id,
                len(self.ready_group_buffer),
                self._committed_previous_group_count,
                self._committed_current_group_count,
                self._previous_partition_quota,
                self._current_partition_quota,
            )
        self._output_window_closed = True

    def target_group_count(self) -> int:
        return self._previous_partition_quota + self._current_partition_quota

    def committed_group_count(self) -> int:
        return self._committed_previous_group_count + self._committed_current_group_count

    def remaining_ready_capacity(self) -> int:
        return self.target_group_count() - self.committed_group_count() - len(self.ready_group_buffer)

    def accounting_snapshot(self) -> dict[str, int]:
        group_size = self.n_samples_per_prompt
        return {
            "group_size": group_size,
            "ready_groups": len(self.ready_group_buffer),
            "transfer_buffer_groups": len(self._transfer_buffer),
            "transfer_tasks": len(self._transfer_tasks),
            "committed_previous_groups": self._committed_previous_group_count,
            "committed_current_groups": self._committed_current_group_count,
            "previous_partition_quota": self._previous_partition_quota,
            "current_partition_quota": self._current_partition_quota,
        }

    def resident_group_keys(self) -> set[GroupKey]:
        return {sample_group_key(group) for group in self.ready_group_buffer}

    def committed_transfer_groups_snapshot(self) -> list[list]:
        return list(self._step_output_groups)

    def release_step_output_payloads(self) -> None:
        self._step_output_groups.clear()

    def enqueue_ready_groups(self, groups) -> None:
        for group in groups:
            if not all(sample.reward is not None for sample in group):
                raise RuntimeError("TransferDomain received unrewarded group.")
            self.ready_group_buffer.append(group)

    def drop_ready_groups(self) -> int:
        if not self.ready_group_buffer:
            return 0
        dropped_count = len(self.ready_group_buffer)
        self.ready_group_buffer.clear()
        return dropped_count

    async def discard_pending_transfers(self) -> tuple[int, int]:
        dropped_buffer_groups = len(self._transfer_buffer)
        self._transfer_buffer.clear()
        tasks = list(self._transfer_tasks)
        self._transfer_tasks.clear()
        cancelled_tasks = 0
        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled_tasks += 1
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return dropped_buffer_groups, cancelled_tasks

    async def drain_ready_group_payloads(self) -> tuple[list[list], int]:
        if self._output_window_closed or not self.ready_group_buffer:
            return [], 0
        released_groups: list[list] = []
        next_previous_count = self._committed_previous_group_count
        next_current_count = self._committed_current_group_count
        if next_previous_count > self._previous_partition_quota:
            raise RuntimeError(
                "TransferDomain committed count exceeds previous partition quota: "
                f"quota={self._previous_partition_quota}, committed={next_previous_count}."
            )
        if next_current_count > self._current_partition_quota:
            raise RuntimeError(
                "TransferDomain committed count exceeds current partition quota: "
                f"quota={self._current_partition_quota}, committed={next_current_count}."
            )
        for group in self.ready_group_buffer:
            if next_previous_count < self._previous_partition_quota:
                next_previous_count += 1
            elif next_current_count < self._current_partition_quota:
                next_current_count += 1
            else:
                break
            released_groups.append(group)
        if not released_groups:
            return [], 0
        for _group in released_groups:
            self.ready_group_buffer.popleft()
        self._step_output_groups.extend(released_groups)
        self._committed_previous_group_count = next_previous_count
        self._committed_current_group_count = next_current_count
        for group in released_groups:
            self._buffer_transfer_group(group)
        self._spawn_ready_transfers()
        return released_groups, len(released_groups)

    async def build_output(self):
        export_groups = list(self._step_output_groups)
        return RolloutFnTrainOutput(
            samples=export_groups,
            metrics={},
        )

    async def wait_for_pending_transfers(self) -> None:
        while self._transfer_buffer:
            spawned = self._spawn_transfer(force=True)
            if spawned <= 0:
                raise RuntimeError(
                    "TransferDomain cannot drain transfer buffer without dispatch progress: "
                    f"rollout_id={self.rollout_id}, buffer_groups={len(self._transfer_buffer)}, "
                    f"committed_previous={self._committed_previous_group_count}, "
                    f"dispatched_previous={self._dispatched_previous_group_count}, "
                    f"committed_current={self._committed_current_group_count}, "
                    f"dispatched_current={self._dispatched_current_group_count}."
                )
        self._reap_completed_transfer_tasks()
        if not self._transfer_tasks:
            return
        await asyncio.gather(*list(self._transfer_tasks))
        self._reap_completed_transfer_tasks()

    async def shutdown(self) -> None:
        await self.wait_for_pending_transfers()
        self._transfer_buffer.clear()
        self._transfer_tasks.clear()
        self.ready_group_buffer.clear()
        self._step_output_groups.clear()
        self._committed_previous_group_count = 0
        self._committed_current_group_count = 0
        self._dispatched_previous_group_count = 0
        self._dispatched_current_group_count = 0
