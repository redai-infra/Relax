# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Sync-intent-aware rollout loop used only when the policy is enabled."""

from __future__ import annotations

import asyncio
import statistics
from argparse import Namespace
from collections.abc import Callable
from time import monotonic
from typing import Any

import numpy as np
import ray
from tqdm import tqdm

from relax.distributed.ray.rollout import _log_rollout_data
from relax.engine.filters.base_types import MetricGatherer, call_dynamic_filter
from relax.engine.rollout.base_types import RolloutFnTrainOutput
from relax.engine.rollout.sglang_rollout import (
    GenerateState,
    _aggregate_rollout_timing,
    abort,
    begin_permit_observability_rollout,
    export_terminal_permit_observability_rows,
    generate_and_rm_group,
    permit_observability_dir,
)
from relax.engine.rollout.sync_intent import (
    get_sync_intent,
    mark_work_origin,
    plan_adaptive_window_fetch,
    plan_dp_aligned_extra_groups,
    plan_intent_guard_fetch,
    sync_intent_abort_retry_interval_seconds,
    sync_intent_abort_timeout_seconds,
    sync_intent_protected_drain_timeout_seconds,
    validate_disjoint_rollout_groups,
    wait_for_sync_intent_end,
)
from relax.utils.cross_version_kv import (
    clear_cross_version_kv_progress_hedge_marker,
    clear_cross_version_kv_task_markers,
    count_cross_version_kv_progress_hedge_groups,
    cross_version_kv_enabled,
    estimate_cross_version_kv_group_remaining_tokens,
    mark_cross_version_kv_carry,
    plan_cross_version_kv_progress_hedge,
)
from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.profile_utils import start_sglang_profile, stop_sglang_profile
from relax.utils.timer import Timer
from relax.utils.training.train_dump_utils import save_debug_rollout_data
from relax.utils.types import Sample
from relax.utils.utils import CURRENT_ROLLOUT_BATCH, compute_dp_size, transfer_batch_to_data_system


logger = get_logger(__name__)


def _submit_generate_tasks(
    state: GenerateState,
    args: Namespace,
    samples: list[list[Sample]],
    task_started_at: dict[asyncio.Task[Any], float],
    task_groups: dict[asyncio.Task[Any], list[Sample]],
    submitted_events: list[asyncio.Event] | None = None,
) -> None:
    if submitted_events is not None and len(submitted_events) != len(samples):
        raise ValueError("submitted_events must match samples")
    max_aborted_count = getattr(args, "partial_rollout_max_aborted_count", None)
    for group_index, group in enumerate(samples):
        event = submitted_events[group_index] if submitted_events else None
        task = asyncio.create_task(
            generate_and_rm_group(
                args,
                group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
                submitted_event=event,
            )
        )
        task_started_at[task] = monotonic()
        task_groups[task] = group
        if event is not None:
            task.add_done_callback(lambda _task, event=event: event.set())
        if max_aborted_count is not None and any(sample.abort_count >= max_aborted_count for sample in group):
            state.protected_pendings.add(task)
        else:
            state.pendings.add(task)
    state.remaining_batch_size += len(samples)


async def _submit_generate_tasks_debt_first(
    state: GenerateState,
    args: Namespace,
    samples: list[list[Sample]],
    debt_groups: int,
    task_started_at: dict[asyncio.Task[Any], float],
    task_groups: dict[asyncio.Task[Any], list[Sample]],
) -> None:
    debt = samples[:debt_groups]
    fresh = samples[debt_groups:]
    if debt:
        submitted_events = [asyncio.Event() for _ in debt]
        _submit_generate_tasks(state, args, debt, task_started_at, task_groups, submitted_events)
        await asyncio.gather(*(event.wait() for event in submitted_events))
    if fresh:
        _submit_generate_tasks(state, args, fresh, task_started_at, task_groups)


def _adopt_cross_version_tasks(
    state: GenerateState,
    args: Namespace,
    num_old_samples: int,
    task_started_at: dict[asyncio.Task[Any], float],
    task_groups: dict[asyncio.Task[Any], list[Sample]],
) -> int:
    tasks = list(state.cross_version_kv_tasks)
    groups = state.cross_version_kv_task_groups
    started_at = state.cross_version_kv_task_started_at
    protected = set(state.cross_version_kv_protected_tasks)
    state.cross_version_kv_tasks = []
    state.cross_version_kv_task_groups = {}
    state.cross_version_kv_task_started_at = {}
    state.cross_version_kv_protected_tasks = set()

    task_group_list = [groups[task] for task in tasks]
    adopted_debt = mark_cross_version_kv_carry(
        task_group_list,
        num_old_samples=num_old_samples,
    )
    for task in tasks:
        group = groups[task]
        if task in protected:
            state.protected_pendings.add(task)
        else:
            state.pendings.add(task)
        task_started_at[task] = started_at.get(task, monotonic())
        task_groups[task] = group

    state.remaining_batch_size += len(tasks)
    if tasks:
        logger.info(
            "TASK22_A3 event=adopt carried_groups=%s debt_groups=%s fresh_groups=%s",
            len(tasks),
            adopted_debt,
            len(tasks) - adopted_debt,
        )
    return adopted_debt


def _persist_cross_version_tasks(
    state: GenerateState,
    task_started_at: dict[asyncio.Task[Any], float],
    task_groups: dict[asyncio.Task[Any], list[Sample]],
) -> int:
    pending = state.pendings | state.protected_pendings
    ordered_tasks = [task for task in task_groups if task in pending]
    state.cross_version_kv_tasks = ordered_tasks
    state.cross_version_kv_task_groups = {task: task_groups[task] for task in ordered_tasks}
    state.cross_version_kv_task_started_at = {task: task_started_at.get(task, monotonic()) for task in ordered_tasks}
    state.cross_version_kv_protected_tasks = {task for task in ordered_tasks if task in state.protected_pendings}
    return len(ordered_tasks)


async def generate_rollout_async_with_sync_intent(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    data_system_client: Any,
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    timer = Timer()
    timer.start("rollout")
    assert args.rollout_global_dataset

    state = GenerateState(args)
    begin_permit_observability_rollout(state, rollout_id)
    if not hasattr(state, "recent_group_latency_seconds"):
        state.recent_group_latency_seconds: list[float] = []
    if not hasattr(state, "recent_completed_response_lengths"):
        state.recent_completed_response_lengths: list[int] = []
    await start_sglang_profile(args, rollout_id)

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    num_old_samples = state.last_step_current_deficit if args.fully_async else 0
    is_final_backfill = args.fully_async and rollout_id >= args.num_rollout
    target_data_size = num_old_samples if is_final_backfill else args.rollout_batch_size + num_old_samples
    if target_data_size <= 0:
        raise RuntimeError(f"Final rollout backfill requested for rollout_id={rollout_id} without pending deficit")

    data: list[list[Sample]] = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc=f"Rollout {rollout_id} generation")
    transfer_tasks: list[asyncio.Task[Any]] = []
    debt_batch_to_transfer: list[list[Sample]] = []
    fresh_batch_to_transfer: list[list[Sample]] = []
    aborted_samples: list[list[Sample]] = []
    oversample_surplus: list[list[Sample]] = []
    accepted_debt_groups = 0
    accepted_fresh_groups = 0
    progressed_fresh_groups = 0
    get_samples_times: list[float] = []
    intent_debt_groups_inflight = 0
    original_hedge_groups = 0 if is_final_backfill else max(args.over_sampling_batch_size - args.rollout_batch_size, 0)
    candidate_window_initialized = False
    task_started_at: dict[asyncio.Task[Any], float] = {}
    task_groups: dict[asyncio.Task[Any], list[Sample]] = {}
    adopted_debt_groups = 0
    adopted_cross_version_groups = 0
    strict_fallback_groups = 0
    strict_fallback_prefix_tokens = 0
    completed_cross_version_groups = 0
    a3_progress_hedge_inflight = 0
    a3_progress_hedge_groups = 0
    a3_estimated_remaining_tokens = 0
    strict_retry_pending = False
    if cross_version_kv_enabled(args):
        adopted_debt_groups = _adopt_cross_version_tasks(
            state,
            args,
            num_old_samples,
            task_started_at,
            task_groups,
        )
        adopted_cross_version_groups = len(task_groups)
        a3_progress_hedge_inflight = count_cross_version_kv_progress_hedge_groups(task_groups.values())
        intent_debt_groups_inflight = adopted_debt_groups
        candidate_window_initialized = bool(task_groups)
        strict_retry_pending = any(
            sample.status == Sample.Status.ABORTED for group in task_groups.values() for sample in group
        )
        adopted_fresh_groups = list(task_groups.values())[adopted_debt_groups:]
        a3_estimated_remaining_tokens = sum(
            estimate_cross_version_kv_group_remaining_tokens(
                group,
                recent_completed_response_lengths=state.recent_completed_response_lengths,
                max_response_length=args.rollout_max_response_len,
            )
            for group in adopted_fresh_groups
        )
    committed_prev = 0
    prev_target = num_old_samples
    curr_target = 0 if is_final_backfill else args.rollout_batch_size
    loop = asyncio.get_running_loop()
    completed_buffer_groups = await loop.run_in_executor(
        None,
        ray.get,
        data_source.get_completed_buffer_group_count.remote(target_data_size),
    )
    completed_buffer_groups = int(completed_buffer_groups)

    if is_final_backfill:
        logger.info(f"Starting final rollout backfill step {rollout_id}: target(prev)={target_data_size}")
    else:
        logger.info(
            f"Starting rollout step {rollout_id}: target(commit)={target_data_size} "
            f"(rollout_batch={args.rollout_batch_size} + old={num_old_samples}) "
            f"completed_buffer={completed_buffer_groups}"
        )

    def target_reached() -> bool:
        return accepted_debt_groups >= prev_target and progressed_fresh_groups >= curr_target

    while not target_reached():
        while True:
            intent_snapshot = get_sync_intent()
            baseline_fetch_groups = args.over_sampling_batch_size + num_old_samples
            remaining_commit_groups = max(prev_target - accepted_debt_groups, 0) + max(
                curr_target - progressed_fresh_groups,
                0,
            )
            missing_debt_groups = max(
                num_old_samples - accepted_debt_groups - intent_debt_groups_inflight,
                0,
            )
            default_fetch_groups = plan_adaptive_window_fetch(
                resident_groups=state.remaining_batch_size,
                baseline_fetch_groups=baseline_fetch_groups,
                remaining_commit_groups=remaining_commit_groups,
                hedge_groups=original_hedge_groups,
                window_initialized=candidate_window_initialized,
                completed_buffer_groups=completed_buffer_groups,
            )
            default_fetch_groups = max(default_fetch_groups, missing_debt_groups)
            using_a3_progress_hedge = False
            if default_fetch_groups == 0 and missing_debt_groups == 0:
                live_hedge = plan_cross_version_kv_progress_hedge(
                    adopted_groups=adopted_cross_version_groups,
                    adopted_debt_groups=adopted_debt_groups,
                    resident_groups=state.remaining_batch_size,
                    remaining_fresh_groups=max(curr_target - progressed_fresh_groups, 0),
                    rollout_batch_size=args.rollout_batch_size,
                    strict_retry_pending=strict_retry_pending,
                    estimated_remaining_tokens=a3_estimated_remaining_tokens,
                    max_response_length=args.rollout_max_response_len,
                )
                default_fetch_groups = live_hedge
                using_a3_progress_hedge = default_fetch_groups > 0
            if default_fetch_groups == 0:
                break

            observed_group_latency_seconds = (
                statistics.median(state.recent_group_latency_seconds) if state.recent_group_latency_seconds else None
            )
            fetch_groups = plan_intent_guard_fetch(
                snapshot=intent_snapshot,
                physical_rollout_id=rollout_id,
                old_debt_groups=num_old_samples,
                completed_debt_groups=accepted_debt_groups,
                inflight_debt_groups=intent_debt_groups_inflight,
                observed_group_latency_seconds=observed_group_latency_seconds,
                default_fetch_groups=default_fetch_groups,
            )
            if fetch_groups == 0:
                if state.pendings or state.protected_pendings:
                    break
                assert intent_snapshot.sync_id is not None
                await asyncio.to_thread(wait_for_sync_intent_end, intent_snapshot.sync_id)
                continue
            get_samples_started_at = monotonic()
            use_prefetched = state.prefetched_samples_ref is not None and fetch_groups == default_fetch_groups
            if use_prefetched:
                ref = state.prefetched_samples_ref
                state.prefetched_samples_ref = None
                logger.info(f"Rollout step {rollout_id}: using pre-fetched data from previous step")
            else:
                ref = data_source.get_samples.remote(fetch_groups)
            samples = await loop.run_in_executor(None, ray.get, ref)
            get_samples_times.append(monotonic() - get_samples_started_at)
            if using_a3_progress_hedge:
                for group in samples:
                    for sample in group:
                        sample.metadata["a3_progress_hedge"] = True
                a3_progress_hedge_inflight += len(samples)
                a3_progress_hedge_groups += len(samples)
                logger.info(
                    "TASK22_A3 event=progress_hedge rollout_id=%s groups=%s "
                    "resident_groups=%s remaining_fresh_groups=%s",
                    rollout_id,
                    len(samples),
                    state.remaining_batch_size,
                    max(curr_target - progressed_fresh_groups, 0),
                )

            active_guard = intent_snapshot.active and (
                intent_snapshot.actor_rollout_id is None or rollout_id > intent_snapshot.actor_rollout_id
            )
            old_debt_in_fetch = min(missing_debt_groups, len(samples)) if num_old_samples > 0 else 0
            speculative_groups_in_fetch = len(samples) - old_debt_in_fetch if num_old_samples and active_guard else 0
            mark_work_origin(
                samples,
                old_debt_in_fetch,
                fresh_origin="speculative_fresh" if speculative_groups_in_fetch else "fresh",
            )
            intent_debt_groups_inflight += old_debt_in_fetch
            await _submit_generate_tasks_debt_first(
                state,
                args,
                samples,
                old_debt_in_fetch,
                task_started_at,
                task_groups,
            )
            candidate_window_initialized = True

        all_pendings = state.pendings | state.protected_pendings
        done, remaining = await asyncio.wait(all_pendings, return_when=asyncio.FIRST_COMPLETED)
        state.pendings &= remaining
        state.protected_pendings &= remaining
        for task in done:
            started_at = task_started_at.pop(task, None)
            source_group = task_groups.pop(task, None)
            state.remaining_batch_size -= 1
            if state.remaining_batch_size < 0:
                raise RuntimeError("Rollout resident group accounting became negative")
            group: list[Sample] = task.result()
            source_group_is_hedge = source_group is not None and clear_cross_version_kv_progress_hedge_marker(
                source_group
            )
            if source_group_is_hedge:
                a3_progress_hedge_inflight -= 1
                if a3_progress_hedge_inflight < 0:
                    raise RuntimeError("A3 progress hedge inflight accounting became negative")
            if source_group is not None and (
                len(group) != len(source_group)
                or any(result is not source for result, source in zip(group, source_group, strict=True))
            ):
                raise RuntimeError("Cross-version task returned different sample identities")
            group_is_debt = bool(group) and all(sample.metadata.get("work_origin") == "old_debt" for sample in group)
            if group_is_debt:
                if intent_debt_groups_inflight <= 0:
                    raise RuntimeError("Rollout debt inflight accounting became negative")
                intent_debt_groups_inflight -= 1

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                if permit_observability_dir() is not None:
                    logger.info(
                        "First rollout sample: prompt_response_redacted=true response_tokens=%s "
                        "group_index=%s sample_index=%s status=%s",
                        sample.response_length,
                        sample.group_index,
                        sample.index,
                        sample.status,
                    )
                else:
                    logger.info(
                        f"First rollout sample: {[str(sample.prompt) + sample.response]}, "
                        f"label: {str(sample.label)[:100]}, reward: {sample.reward}",
                    )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            group_aborted = any(sample.status == Sample.Status.ABORTED for sample in group)
            cross_version_fallback = (
                group_aborted
                and cross_version_kv_enabled(args)
                and any(sample.metadata.get("cross_version_kv_carried") for sample in group)
            )
            if cross_version_fallback:
                # Fail closed for the rest of this physical rollout: do not
                # overlap fresh hedge admission with any strict rebuild wave.
                strict_retry_pending = True
                prefix_tokens = sum(sample.response_length for sample in group)
                strict_fallback_groups += 1
                strict_fallback_prefix_tokens += prefix_tokens
                for sample in group:
                    if sample.status == Sample.Status.ABORTED:
                        sample.abort_count += 1
                    sample.metadata["cross_version_kv_carried"] = False
                    sample.metadata["cross_version_kv_carryovers"] = 0
                    sample.metadata["cross_version_kv_fallbacks"] = (
                        int(sample.metadata.get("cross_version_kv_fallbacks", 0)) + 1
                    )
                await _submit_generate_tasks_debt_first(
                    state,
                    args,
                    [group],
                    1 if group_is_debt else 0,
                    task_started_at,
                    task_groups,
                )
                if group_is_debt:
                    intent_debt_groups_inflight += 1
                logger.info(
                    "TASK22_A3 event=strict_fallback rollout_id=%s origin=%s groups=1 retained_prefix_tokens=%s",
                    rollout_id,
                    "old_debt" if group_is_debt else "fresh",
                    prefix_tokens,
                )
                continue

            if not group_aborted:
                state.recent_completed_response_lengths.extend(sample.response_length for sample in group)
                state.recent_completed_response_lengths = state.recent_completed_response_lengths[-512:]
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                clear_cross_version_kv_task_markers(group)
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue

            if started_at is not None and not group_aborted:
                state.recent_group_latency_seconds.append(monotonic() - started_at)
                state.recent_group_latency_seconds = state.recent_group_latency_seconds[-64:]
            if group_aborted:
                if any(sample.metadata.get("cross_version_kv_carried") for sample in group):
                    completed_cross_version_groups += 1
                clear_cross_version_kv_task_markers(group)
                for sample in group:
                    if sample.response and "start_rollout_id" not in sample.metadata:
                        sample.metadata["start_rollout_id"] = rollout_id
                aborted_samples.append(group)
                if not group_is_debt and progressed_fresh_groups < curr_target:
                    # Lifecycle progress lets this physical rollout finish after
                    # publication aborts, but aborted work is never train data.
                    progressed_fresh_groups += 1
                    pbar.update(args.n_samples_per_prompt)
            elif group_is_debt and accepted_debt_groups < prev_target:
                if any(sample.metadata.get("cross_version_kv_carried") for sample in group):
                    completed_cross_version_groups += 1
                clear_cross_version_kv_task_markers(group)
                debt_batch_to_transfer.append(group)
                accepted_debt_groups += 1
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
            elif not group_is_debt and progressed_fresh_groups < curr_target:
                if any(sample.metadata.get("cross_version_kv_carried") for sample in group):
                    completed_cross_version_groups += 1
                clear_cross_version_kv_task_markers(group)
                fresh_batch_to_transfer.append(group)
                accepted_fresh_groups += 1
                progressed_fresh_groups += 1
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
            else:
                if any(sample.metadata.get("cross_version_kv_carried") for sample in group):
                    completed_cross_version_groups += 1
                clear_cross_version_kv_task_markers(group)
                oversample_surplus.append(group)

        transfer_batch_size = (
            args.global_batch_size // args.num_iters_per_train_update // args.n_samples_per_prompt
            if args.fully_async
            else args.rollout_batch_size
        )
        if debt_batch_to_transfer and accepted_debt_groups >= prev_target and committed_prev < prev_target:
            debt_tail = len(debt_batch_to_transfer)
            transfer_tasks.append(
                asyncio.create_task(
                    transfer_batch_to_data_system(
                        args,
                        debt_batch_to_transfer,
                        debt_tail,
                        rollout_id - 1,
                        data_system_client,
                        is_last=True,
                    )
                )
            )
            committed_prev += debt_tail
            debt_batch_to_transfer = []
            logger.info(
                f"{num_old_samples} old samples completed! Total yielded: "
                f"{committed_prev}/{num_old_samples} for step: {rollout_id - 1}"
            )
        if committed_prev >= prev_target and len(fresh_batch_to_transfer) >= transfer_batch_size:
            transfer_groups = fresh_batch_to_transfer[:transfer_batch_size]
            fresh_batch_to_transfer = fresh_batch_to_transfer[transfer_batch_size:]
            curr_is_last = args.fully_async and accepted_fresh_groups >= curr_target and not fresh_batch_to_transfer
            transfer_tasks.append(
                asyncio.create_task(
                    transfer_batch_to_data_system(
                        args,
                        transfer_groups,
                        transfer_batch_size,
                        rollout_id,
                        data_system_client,
                        is_last=curr_is_last,
                    )
                )
            )
            logger.info(f"Total yielded: {accepted_fresh_groups}/{args.rollout_batch_size} for step: {rollout_id}")
    if debt_batch_to_transfer:
        n = len(debt_batch_to_transfer)
        transfer_tasks.append(
            asyncio.create_task(
                transfer_batch_to_data_system(
                    args,
                    debt_batch_to_transfer,
                    n,
                    rollout_id - 1,
                    data_system_client,
                    is_last=args.fully_async and accepted_debt_groups >= prev_target,
                )
            )
        )
        committed_prev += n
        logger.info(f"Total yielded: {committed_prev}/{num_old_samples} for step: {rollout_id - 1}")
    if fresh_batch_to_transfer:
        n = len(fresh_batch_to_transfer)
        transfer_tasks.append(
            asyncio.create_task(
                transfer_batch_to_data_system(
                    args,
                    fresh_batch_to_transfer,
                    n,
                    rollout_id,
                    data_system_client,
                    is_last=args.fully_async and accepted_fresh_groups >= curr_target,
                )
            )
        )
        logger.info(f"Total yielded: {accepted_fresh_groups}/{args.rollout_batch_size} for step: {rollout_id}")
    logger.info(f"Generator exhausted. Waiting for {len(transfer_tasks)} transfer tasks to complete...")
    if transfer_tasks:
        await asyncio.gather(*transfer_tasks)
    pbar.close()
    await stop_sglang_profile(args, rollout_id)

    rollout_time = timer.end("rollout")
    if data:
        sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
        if permit_observability_dir() is not None:
            logger.info(
                "Finish rollout: prompt_response_redacted=true response_tokens=%s group_index=%s sample_index=%s",
                sample.response_length,
                sample.group_index,
                sample.index,
            )
        else:
            logger.info(
                f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
                f"label: {str(sample.label)[:100]}, reward: {sample.reward}",
            )
    else:
        logger.info("Finish rollout with no accepted groups; all lifecycle progress is buffered for retry.")
    all_samples = [sample for group in data for sample in group]
    timing_metrics = _aggregate_rollout_timing(all_samples, get_samples_times)

    can_carry_tasks = cross_version_kv_enabled(args) and not is_final_backfill and rollout_id + 1 < args.num_rollout
    carried_groups = 0
    if can_carry_tasks:
        carried_groups = _persist_cross_version_tasks(state, task_started_at, task_groups)
        logger.info(
            "TASK22_A3 event=carry rollout_id=%s carried_groups=%s current_weight_version=%s max_gap=%s",
            rollout_id,
            carried_groups,
            getattr(args, "cross_version_kv_weight_version", "actor-managed"),
            args.cross_version_kv_max_gap,
        )
    else:
        new_aborted, completed_protected = await abort(
            args,
            rollout_id,
            retry_interval_seconds=sync_intent_abort_retry_interval_seconds(),
            timeout_seconds=sync_intent_abort_timeout_seconds(),
            protected_timeout_seconds=sync_intent_protected_drain_timeout_seconds(),
        )
        aborted_samples.extend(new_aborted)
        aborted_samples.extend(completed_protected)
        for group in new_aborted:
            clear_cross_version_kv_task_markers(group)
        for group in completed_protected:
            clear_cross_version_kv_task_markers(group)
    if aborted_samples:
        logger.info(
            f"Rollout not completed for rollout_id: {rollout_id}, have {len(aborted_samples)} samples aborted."
        )
    elif carried_groups:
        logger.info(
            "Rollout preserved %s in-flight groups for cross-version KV continuation at rollout_id=%s.",
            carried_groups,
            rollout_id,
        )
    else:
        logger.info(f"Rollout fully completed for rollout_id: {rollout_id}.")
    exported_permit_counts = export_terminal_permit_observability_rows(state)
    if exported_permit_counts:
        logger.info(
            "TASK22_CALIBRATION_PERMIT rollout_id=%s terminal_rows=%s exported=%s",
            rollout_id,
            exported_permit_counts.get(rollout_id, 0),
            exported_permit_counts,
        )
    timing_metrics.update(
        {
            "rollout/a3/adopted_groups": adopted_cross_version_groups,
            "rollout/a3/adopted_debt_groups": adopted_debt_groups,
            "rollout/a3/completed_groups": completed_cross_version_groups,
            "rollout/a3/carried_groups": carried_groups,
            "rollout/a3/estimated_remaining_tokens": a3_estimated_remaining_tokens,
            "rollout/a3/progress_hedge_groups": a3_progress_hedge_groups,
            "rollout/a3/strict_fallback_groups": strict_fallback_groups,
            "rollout/a3/strict_fallback_prefix_tokens": strict_fallback_prefix_tokens,
        }
    )

    committed_current = 0 if is_final_backfill else accepted_fresh_groups
    if is_final_backfill:
        state.last_step_current_deficit = 0
    elif args.fully_async:
        state.last_step_current_deficit = max(args.rollout_batch_size - committed_current, 0)
    else:
        state.last_step_current_deficit = 0
    aborted_samples.extend(oversample_surplus)
    logger.info(
        f"Rollout step {rollout_id} carry-over: committed_current={committed_current} "
        f"next_step_deficit={state.last_step_current_deficit} "
        f"oversample_surplus={len(oversample_surplus)} aborted={len(aborted_samples) - len(oversample_surplus)}"
    )

    if args.partial_rollout and args.use_dynamic_global_batch_size:
        extra_completed = [
            group
            for group in aborted_samples
            if all(sample.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) for sample in group)
        ]
        if extra_completed:
            aborted_samples = [
                group
                for group in aborted_samples
                if not all(sample.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) for sample in group)
            ]
            dp_size = compute_dp_size(args)
            max_extra = plan_dp_aligned_extra_groups(
                current_groups=len(data),
                available_extra_groups=len(extra_completed),
                dp_size=dp_size,
            )
            accepted = extra_completed[:max_extra]
            aborted_samples.extend(extra_completed[max_extra:])
            data.extend(accepted)
            if accepted:
                await transfer_batch_to_data_system(args, accepted, len(accepted), rollout_id, data_system_client)

    validate_disjoint_rollout_groups(data, aborted_samples)

    global CURRENT_ROLLOUT_BATCH
    if CURRENT_ROLLOUT_BATCH:
        save_debug_rollout_data(
            args,
            CURRENT_ROLLOUT_BATCH,
            rollout_id=rollout_id,
            evaluation=False,
            tokenizer=state.tokenizer,
        )
        rollout_metrics = dict(timing_metrics)
        if args.partial_rollout and not args.fully_async:
            assert len(CURRENT_ROLLOUT_BATCH) == len(data) * args.n_samples_per_prompt
            staleness_gaps = [
                rollout_id - sample.metadata.get("start_rollout_id", rollout_id) for group in data for sample in group
            ]
            rollout_metrics["rollout/staleness/avg"] = np.mean(staleness_gaps).item()
            rollout_metrics["rollout/staleness/max"] = np.max(staleness_gaps).item()
            rollout_metrics["rollout/staleness/min"] = np.min(staleness_gaps).item()
            rollout_metrics["rollout/global_batch_size"] = len(data) * args.n_samples_per_prompt
        _log_rollout_data(rollout_id, args, CURRENT_ROLLOUT_BATCH, rollout_metrics, rollout_time)
        if args.debug_rollout_only:
            await data_system_client.async_clear_partition(partition_id=f"train_{rollout_id}")
        CURRENT_ROLLOUT_BATCH.clear()

    state.reset()
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
