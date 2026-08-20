# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

from relax.agentic.pipeline import GroupKey, sample_group_key
from relax.agentic.profile import (
    mark_sample_agentic_event,
    mark_sample_agentic_event_once,
)
from relax.engine.rollout.base_types import RolloutFnTrainOutput
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

AGENTIC_VARIABLE_ROW_PADDING_KEY = "agentic_variable_row_padding"
AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV = "RELAX_AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE"
AGENTIC_ROW_IDENTITY_KEY = "agentic_row_identity"
AGENTIC_ROW_IDENTITY_TAG_KEY = "agentic_row_identity_tag"
AGENTIC_ROW_IDENTITY_TAGS_FIELD = "agentic_row_identity_tags"
AGENTIC_ROW_IDENTITY_SCHEMA_VERSION = 1
_AGENTIC_TRANSFER_IDENTITY_ATTR = "_agentic_transfer_identity"


def _agentic_max_exported_rows_per_sample() -> int | None:
    raw_max_rows = os.environ.get(AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV)
    if raw_max_rows is None or not raw_max_rows.strip():
        return None
    normalized_max_rows = raw_max_rows.strip()
    if not normalized_max_rows.isascii() or not normalized_max_rows.isdigit() or int(normalized_max_rows) <= 0:
        raise ValueError(
            f"{AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV} must be a positive integer, got {raw_max_rows!r}."
        )
    return int(normalized_max_rows)


def use_agentic_variable_row_mode(args) -> bool:
    """Return whether variable explicit-row transfer is enabled.

    The feature remains default-off and is limited to synchronous training.
    Async and hybrid consumers have different lifecycle and accounting
    contracts and must not silently enter this path.
    """

    base_enabled = bool(
        getattr(args, "group_rm", False)
        and getattr(args, "agentic_custom_advantage_path", None)
        and getattr(args, "use_dynamic_batch_size", False)
    )
    if not base_enabled:
        return False

    max_rows_per_sample = _agentic_max_exported_rows_per_sample()
    if max_rows_per_sample is None:
        return False

    if getattr(args, "fully_async", False) or getattr(args, "hybrid", False):
        raise ValueError("Agentic variable-row transfer supports synchronous training only.")
    return True


def _flatten_transfer_samples(batch_samples: list) -> list:
    flat_samples: list = []
    pending = deque(batch_samples)
    while pending:
        item = pending.popleft()
        if isinstance(item, list):
            pending.extendleft(reversed(item))
        else:
            flat_samples.append(item)
    return flat_samples


def _identity_scalar(name: str, value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise RuntimeError(f"Agentic row {name} must be a non-empty string or integer, got {value!r}.")
    return str(value)


def _captured_policy_version(sample: Any, metadata: dict[str, Any]) -> str:
    weight_versions = getattr(sample, "weight_versions", None)
    if weight_versions is None:
        weight_versions = []
    if not isinstance(weight_versions, (list, tuple)):
        raise RuntimeError("Sample.weight_versions must be a list or tuple for agentic row transfer.")
    normalized_versions = {_identity_scalar("weight_version", value) for value in weight_versions}
    if len(normalized_versions) > 1:
        raise RuntimeError(
            "Agentic row spans multiple policy versions before transfer: "
            f"weight_versions={sorted(normalized_versions)!r}."
        )
    if normalized_versions:
        captured = next(iter(normalized_versions))
    else:
        captured = _identity_scalar("start_rollout_id", metadata.get("start_rollout_id"))

    declared = metadata.get("policy_version")
    if declared is not None and _identity_scalar("policy_version", declared) != captured:
        raise RuntimeError(
            "Agentic row policy_version conflicts with Relax's captured version: "
            f"declared={declared!r}, captured={captured!r}."
        )
    if declared is None:
        metadata["policy_version"] = captured
    return captured


def _agentic_row_identity(sample: Any) -> dict[str, Any]:
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise RuntimeError("Agentic variable-row samples must carry dict metadata.")
    row_id = metadata.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise RuntimeError(f"Agentic row_id must be a non-empty string, got {row_id!r}.")
    rollout_group_id = _identity_scalar("rollout_group_id", metadata.get("rollout_group_id"))
    sample_group_id = _identity_scalar("Sample.group_index", getattr(sample, "group_index", None))
    if rollout_group_id != sample_group_id:
        raise RuntimeError(
            "Agentic row rollout_group_id does not match Sample.group_index: "
            f"metadata={rollout_group_id!r}, sample={sample_group_id!r}."
        )
    policy_version = _captured_policy_version(sample, metadata)
    task_id = metadata.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, (str, int)) or not str(task_id):
        raise RuntimeError(f"Agentic row task_id must be a non-empty string or integer, got {task_id!r}.")
    trajectory_id = metadata.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise RuntimeError(f"Agentic row trajectory_id must be a non-empty string, got {trajectory_id!r}.")
    if getattr(sample, "session_id", None) != trajectory_id:
        raise RuntimeError(
            "Agentic row trajectory_id does not match Sample.session_id: "
            f"metadata={trajectory_id!r}, sample={getattr(sample, 'session_id', None)!r}."
        )
    turn_index = metadata.get("turn_index")
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise RuntimeError(f"Agentic row turn_index must be a non-negative integer, got {turn_index!r}.")
    turn_id = metadata.get("turn_id")
    expected_turn_id = f"turn_{turn_index:03d}"
    if turn_id != expected_turn_id or getattr(sample, "_agentic_export_name", None) != expected_turn_id:
        raise RuntimeError(
            "Agentic row turn identity mismatch: "
            f"turn_id={turn_id!r}, export_name={getattr(sample, '_agentic_export_name', None)!r}, "
            f"expected={expected_turn_id!r}."
        )
    sample_index = getattr(sample, "index", None)
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise RuntimeError(f"Agentic row Sample.index must be a non-negative integer, got {sample_index!r}.")
    terminal = metadata.get("terminal")
    truncated = metadata.get("truncated")
    if type(terminal) is not bool or type(truncated) is not bool:
        raise RuntimeError("Agentic row terminal and truncated markers must be bool values.")

    response_length = getattr(sample, "response_length", None)
    if isinstance(response_length, bool) or not isinstance(response_length, int) or response_length <= 0:
        raise RuntimeError(f"Agentic row response_length must be positive, got {response_length!r}.")
    tokens = getattr(sample, "tokens", None)
    if not isinstance(tokens, list) or len(tokens) < response_length:
        raise RuntimeError("Agentic row tokens must contain the complete response token suffix.")
    loss_mask = getattr(sample, "loss_mask", None)
    if not isinstance(loss_mask, list) or len(loss_mask) != response_length:
        raise RuntimeError(
            "Agentic row loss_mask must contain one entry per response token: "
            f"response_length={response_length}, mask_length={len(loss_mask) if isinstance(loss_mask, list) else None}."
        )
    if any(type(value) is not int or value not in (0, 1) for value in loss_mask):
        raise RuntimeError("Agentic row loss_mask must contain only integer 0/1 values.")
    action_token_count = sum(loss_mask)
    if action_token_count <= 0:
        raise RuntimeError("Agentic row must contain at least one trainable assistant action token.")
    rollout_log_probs = getattr(sample, "rollout_log_probs", None)
    if not isinstance(rollout_log_probs, list) or len(rollout_log_probs) != response_length:
        raise RuntimeError(
            "Agentic row rollout_log_probs must contain one entry per response token: "
            f"response_length={response_length}, "
            f"log_prob_length={len(rollout_log_probs) if isinstance(rollout_log_probs, list) else None}."
        )

    return {
        "schema_version": AGENTIC_ROW_IDENTITY_SCHEMA_VERSION,
        "padding": False,
        "row_id": row_id,
        "rollout_group_id": rollout_group_id,
        "policy_version": policy_version,
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "turn_id": turn_id,
        "turn_index": turn_index,
        "sample_index": sample_index,
        "terminal": terminal,
        "truncated": truncated,
        "total_length": len(tokens),
        "response_length": response_length,
        "action_token_count": action_token_count,
    }


def _row_id_digest(row_ids: list[str]) -> str:
    encoded = "\n".join(sorted(row_ids)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_identity_tag(row_id: str) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def _padding_row_identity(
    *,
    rollout_id: int,
    row_ordinal: int,
    policy_version: str,
    total_length: int,
    response_length: int,
    expected_group_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": AGENTIC_ROW_IDENTITY_SCHEMA_VERSION,
        "padding": True,
        "row_id": f"agentic-padding-v1:{rollout_id}:{row_ordinal}",
        "rollout_group_id": None,
        "policy_version": policy_version,
        "task_id": None,
        "trajectory_id": None,
        "turn_id": None,
        "turn_index": None,
        "sample_index": None,
        "terminal": False,
        "truncated": False,
        "total_length": total_length,
        "response_length": response_length,
        "action_token_count": 0,
        "group_row_count": 0,
        "group_trajectory_count": 0,
        "group_row_ids_sha256": None,
        "partition_expected_group_count": expected_group_count,
    }


def _validate_agentic_materialized_group(
    *, args: Any, group: list
) -> tuple[list[tuple[Any, dict[str, Any]]], dict[str, Any]]:
    if not isinstance(group, list) or not group or any(isinstance(sample, list) for sample in group):
        raise RuntimeError("Agentic variable-row transfer requires a non-empty flat materialized group.")
    identities = [(sample, _agentic_row_identity(sample)) for sample in group]
    row_ids = [identity["row_id"] for _, identity in identities]
    if len(set(row_ids)) != len(row_ids):
        raise RuntimeError("Agentic materialized group contains a duplicate row_id.")

    group_ids = {identity["rollout_group_id"] for _, identity in identities}
    policy_versions = {identity["policy_version"] for _, identity in identities}
    task_ids = {str(identity["task_id"]) for _, identity in identities}
    if len(group_ids) != 1 or len(policy_versions) != 1 or len(task_ids) != 1:
        raise RuntimeError(
            "Agentic materialized group mixes rollout group, policy version, or task identity: "
            f"groups={sorted(group_ids)!r}, policies={sorted(policy_versions)!r}, tasks={sorted(task_ids)!r}."
        )

    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    trajectory_by_sample_index: dict[int, str] = {}
    for _, identity in identities:
        trajectory_id = identity["trajectory_id"]
        sample_index = identity["sample_index"]
        previous = trajectory_by_sample_index.setdefault(sample_index, trajectory_id)
        if previous != trajectory_id:
            raise RuntimeError(f"Sample.index {sample_index} maps to multiple trajectory_id values.")
        by_trajectory.setdefault(trajectory_id, []).append(identity)
    if len(trajectory_by_sample_index) != len(by_trajectory):
        raise RuntimeError("Multiple trajectory_id values share one Sample.index in an agentic group.")
    expected_trajectories = int(args.n_samples_per_prompt)
    if len(by_trajectory) != expected_trajectories:
        raise RuntimeError(
            "Agentic materialized group is incomplete: "
            f"expected_trajectories={expected_trajectories}, got={len(by_trajectory)}."
        )
    for trajectory_id, trajectory in by_trajectory.items():
        ordered = sorted(trajectory, key=lambda identity: identity["turn_index"])
        actual_turns = [identity["turn_index"] for identity in ordered]
        if actual_turns != list(range(len(ordered))):
            raise RuntimeError(
                f"Agentic trajectory {trajectory_id!r} has missing or duplicate turn rows: {actual_turns!r}."
            )
        for identity in ordered[:-1]:
            if identity["terminal"] or identity["truncated"]:
                raise RuntimeError(f"Agentic trajectory {trajectory_id!r} terminates before its final row.")
        final = ordered[-1]
        if final["terminal"] == final["truncated"]:
            raise RuntimeError(
                f"Agentic trajectory {trajectory_id!r} final row must be exactly one of terminal or truncated."
            )

    manifest = {
        "group_row_count": len(identities),
        "group_trajectory_count": len(by_trajectory),
        "group_row_ids_sha256": _row_id_digest(row_ids),
        "partition_expected_group_count": int(args.rollout_batch_size),
    }
    return identities, manifest


def _variable_row_padding_sample(*, args, template):
    sample = copy.deepcopy(template)
    sample.remove_sample = True
    sample.loss_mask = None
    sample.custom_advantage = 0.0
    reward_key = getattr(args, "reward_key", None)
    sample.reward = {reward_key: 0.0} if reward_key else 0.0
    sample.metadata = copy.deepcopy(sample.metadata) if isinstance(sample.metadata, dict) else {}
    sample.metadata.pop("raw_reward", None)
    for key in (
        "row_id",
        "rollout_group_id",
        "policy_version",
        "task_id",
        "trajectory_id",
        "turn_id",
        "turn_index",
        "terminal",
        "truncated",
    ):
        sample.metadata.pop(key, None)
    sample.metadata[AGENTIC_VARIABLE_ROW_PADDING_KEY] = True
    if hasattr(sample, _AGENTIC_TRANSFER_IDENTITY_ATTR):
        delattr(sample, _AGENTIC_TRANSFER_IDENTITY_ATTR)
    return sample


async def _transfer_batch_to_data_system(
    *,
    args,
    batch_samples: list,
    batch_count: int,
    rollout_id: int,
    data_system_client,
    is_last: bool = False,
    timing_sink=None,
) -> list[str]:
    from relax.utils.utils import convert_samples_to_train_data

    if not batch_samples:
        logger.warning(
            "transfer_batch_to_data_system called with empty batch_samples for rollout_id=%s, batch_count=%s",
            rollout_id,
            batch_count,
        )
        return []

    variable_row_mode = use_agentic_variable_row_mode(args)
    timing_record = {
        "rollout_id": rollout_id,
        "batch_count": batch_count,
        "is_last": is_last,
        "row_count": 0,
        "reorder_ms": 0.0,
        "identity_validation_ms": 0.0,
        "serialization_ms": 0.0,
        "queue_transfer_ms": 0.0,
        "ok": False,
    }

    reorder_started_at = time.perf_counter()
    flat_samples = _flatten_transfer_samples(batch_samples)
    enqueue_at = time.time()
    for sample in flat_samples:
        mark_sample_agentic_event(sample, "transfer_enqueue_at", enqueue_at)
    ordered_groups = sorted(
        batch_samples, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )
    ordered_samples = _flatten_transfer_samples(ordered_groups)
    timing_record["reorder_ms"] = (time.perf_counter() - reorder_started_at) * 1000.0

    transfer_samples = []
    for sample in ordered_samples:
        if sample.reward is None:
            sample = copy.copy(sample)
            # Fall back to custom advantage when neither JSON reward nor custom RM supplies a reward.
            # The transferred raw_reward is then unsuitable for raw-reward statistics; rollout metrics
            # still use the original sample and ignore this fallback.
            sample.reward = {args.reward_key: sample.custom_advantage} if args.reward_key else sample.custom_advantage
        transfer_samples.append(sample)

    identities: list[dict[str, Any]] = []
    if variable_row_mode:
        identity_started_at = time.perf_counter()
        actual_policy_versions: set[str] = set()
        for sample in transfer_samples:
            is_padding = bool(
                sample.remove_sample
                and isinstance(sample.metadata, dict)
                and sample.metadata.get(AGENTIC_VARIABLE_ROW_PADDING_KEY) is True
            )
            if is_padding:
                identities.append({})
                continue
            cached_identity = getattr(sample, _AGENTIC_TRANSFER_IDENTITY_ATTR, None)
            if not isinstance(cached_identity, Mapping):
                raise RuntimeError("Agentic row reached transfer without a validated group identity manifest.")
            current_identity = _agentic_row_identity(sample)
            for key, value in current_identity.items():
                if cached_identity.get(key) != value:
                    raise RuntimeError(
                        "Agentic row identity changed after group validation: "
                        f"row_id={current_identity['row_id']!r}, field={key!r}."
                    )
            identity = copy.deepcopy(dict(cached_identity))
            identities.append(identity)
            actual_policy_versions.add(identity["policy_version"])
        if len(actual_policy_versions) != 1:
            raise RuntimeError(
                "One TransferQueue batch must contain exactly one policy version: "
                f"versions={sorted(actual_policy_versions)!r}."
            )
        policy_version = next(iter(actual_policy_versions))
        for row_ordinal, (sample, identity) in enumerate(zip(transfer_samples, identities, strict=True)):
            if identity:
                continue
            identity.update(
                _padding_row_identity(
                    rollout_id=rollout_id,
                    row_ordinal=row_ordinal,
                    policy_version=policy_version,
                    total_length=len(sample.tokens),
                    response_length=sample.response_length,
                    expected_group_count=int(args.rollout_batch_size),
                )
            )
            sample.metadata.update(
                {
                    "row_id": identity["row_id"],
                    "policy_version": policy_version,
                }
            )
        if len({identity["row_id"] for identity in identities}) != len(identities):
            raise RuntimeError("TransferQueue batch contains duplicate physical row identities.")
        timing_record["identity_validation_ms"] = (time.perf_counter() - identity_started_at) * 1000.0

    serialization_started_at = time.perf_counter()
    rollout_batch = convert_samples_to_train_data(args, transfer_samples)
    total_lengths = [int(length) for length in rollout_batch["total_lengths"]]
    if len(total_lengths) != len(transfer_samples):
        raise RuntimeError(
            "Converted rollout batch lost row alignment: "
            f"samples={len(transfer_samples)}, lengths={len(total_lengths)}."
        )
    if variable_row_mode:
        import torch

        for identity, total_length in zip(identities, total_lengths, strict=True):
            if identity["total_length"] != total_length:
                raise RuntimeError(
                    "Converted rollout row length does not match its identity: "
                    f"row_id={identity['row_id']!r}, identity={identity['total_length']}, batch={total_length}."
                )
        rollout_batch[AGENTIC_ROW_IDENTITY_TAGS_FIELD] = torch.tensor(
            [_row_identity_tag(identity["row_id"]) for identity in identities],
            dtype=torch.int64,
        )
    timing_record["serialization_ms"] = (time.perf_counter() - serialization_started_at) * 1000.0
    timing_record["row_count"] = len(transfer_samples)
    logger.info("Prepared rollout batch %s with %s samples for transfer", batch_count, rollout_batch.numel())
    logger.info("Transferring batch rollout_batch: %s", rollout_batch)
    if variable_row_mode:
        custom_meta = [
            {
                "total_lengths": total_length,
                AGENTIC_VARIABLE_ROW_PADDING_KEY: identity["padding"],
                AGENTIC_ROW_IDENTITY_KEY: copy.deepcopy(identity),
                AGENTIC_ROW_IDENTITY_TAG_KEY: _row_identity_tag(identity["row_id"]),
            }
            for identity, total_length in zip(identities, total_lengths, strict=True)
        ]
    else:
        custom_meta = [{"total_lengths": length} for length in total_lengths]
    queue_started_at = time.perf_counter()
    try:
        await data_system_client.async_put(
            data=rollout_batch,
            partition_id=f"train_{rollout_id}",
            custom_meta=custom_meta,
            is_last=is_last,
        )
        timing_record["ok"] = True
    finally:
        timing_record["queue_transfer_ms"] = (time.perf_counter() - queue_started_at) * 1000.0
        if variable_row_mode and timing_sink is not None:
            timing_sink(copy.deepcopy(timing_record))
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
        self._variable_row_mode = use_agentic_variable_row_mode(args)
        self._partition_actual_rows: dict[int, int] = {}
        self._partition_identity_state: dict[int, dict[str, Any]] = {}
        self._transfer_timing_records: deque[dict[str, Any]] = deque(maxlen=4096)
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
        stale_identity_partitions = [
            partition_id
            for partition_id in self._partition_identity_state
            if partition_id < rollout_id - 1 or partition_id > rollout_id
        ]
        if stale_identity_partitions:
            raise RuntimeError(
                "TransferDomain cannot rebind with stale or future agentic identity state: "
                f"next_rollout_id={rollout_id}, partitions={sorted(stale_identity_partitions)!r}."
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
            timing_sink=self._transfer_timing_records.append if self._variable_row_mode else None,
        )
        if self._variable_row_mode and is_last:
            self._partition_actual_rows.pop(partition_rollout_id, None)
            self._partition_identity_state.pop(partition_rollout_id, None)
        release_ended_at = time.time()
        for group in groups:
            for sample in group:
                mark_sample_agentic_event(sample, "transfer_release_end_at", release_ended_at)

    async def _dispatch_transfer_batch_after(
        self,
        *,
        predecessor: asyncio.Task | None,
        groups,
        partition_rollout_id: int,
        is_last: bool,
    ) -> None:
        if predecessor is not None:
            await predecessor
        await self._dispatch_transfer_batch(
            groups=groups,
            partition_rollout_id=partition_rollout_id,
            is_last=is_last,
        )

    def _prepare_variable_row_groups(
        self,
        *,
        groups: list[list],
        partition_rollout_id: int,
        is_last: bool,
    ) -> list[list]:
        actual_samples = _flatten_transfer_samples(groups)
        if not actual_samples:
            raise RuntimeError(
                "Agentic variable-row transfer cannot dispatch an empty materialized batch: "
                f"partition_rollout_id={partition_rollout_id}."
            )
        validated_groups = [_validate_agentic_materialized_group(args=self.args, group=group) for group in groups]
        candidate_row_ids: set[str] = set()
        candidate_group_manifests: dict[str, dict[str, Any]] = {}
        candidate_policy_versions: set[str] = set()
        for identities, manifest in validated_groups:
            group_id = identities[0][1]["rollout_group_id"]
            if group_id in candidate_group_manifests:
                raise RuntimeError(f"Agentic transfer batch repeats rollout_group_id {group_id!r}.")
            candidate_group_manifests[group_id] = manifest
            for _, identity in identities:
                row_id = identity["row_id"]
                if row_id in candidate_row_ids:
                    raise RuntimeError(f"Agentic transfer batch repeats row_id {row_id!r}.")
                candidate_row_ids.add(row_id)
                candidate_policy_versions.add(identity["policy_version"])
        if len(candidate_policy_versions) != 1:
            raise RuntimeError(
                f"Agentic transfer batch mixes policy versions: versions={sorted(candidate_policy_versions)!r}."
            )

        previous_identity_state = self._partition_identity_state.get(partition_rollout_id)
        existing_row_ids = set(previous_identity_state["row_ids"]) if previous_identity_state else set()
        existing_group_manifests = (
            copy.deepcopy(previous_identity_state["group_manifests"]) if previous_identity_state else {}
        )
        existing_policy_version = previous_identity_state["policy_version"] if previous_identity_state else None
        duplicate_row_ids = existing_row_ids.intersection(candidate_row_ids)
        if duplicate_row_ids:
            raise RuntimeError(
                "Agentic partition received duplicate row_id values across retries or batches: "
                f"duplicates={sorted(duplicate_row_ids)!r}."
            )
        duplicate_group_ids = set(existing_group_manifests).intersection(candidate_group_manifests)
        if duplicate_group_ids:
            raise RuntimeError(
                f"Agentic partition received a rollout group more than once: groups={sorted(duplicate_group_ids)!r}."
            )
        candidate_policy_version = next(iter(candidate_policy_versions))
        if existing_policy_version is not None and existing_policy_version != candidate_policy_version:
            raise RuntimeError(
                "Agentic partition mixes policy versions across transfer batches: "
                f"existing={existing_policy_version!r}, candidate={candidate_policy_version!r}."
            )
        next_group_manifests = {**existing_group_manifests, **candidate_group_manifests}
        if len(next_group_manifests) > int(self.args.rollout_batch_size):
            raise RuntimeError(
                "Agentic partition contains more rollout groups than configured: "
                f"groups={len(next_group_manifests)}, rollout_batch_size={self.args.rollout_batch_size}."
            )
        if is_last and len(next_group_manifests) != int(self.args.rollout_batch_size):
            raise RuntimeError(
                "Agentic partition closed with a residual or missing rollout group: "
                f"expected={self.args.rollout_batch_size}, got={len(next_group_manifests)}."
            )

        partition_actual_rows = self._partition_actual_rows.get(partition_rollout_id, 0) + len(actual_samples)
        max_rows_per_sample = _agentic_max_exported_rows_per_sample()
        if max_rows_per_sample is None:
            raise RuntimeError(
                f"{AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV} disappeared while "
                "agentic variable-row transfer was active."
            )
        max_partition_actual_rows = (
            int(self.args.rollout_batch_size) * int(self.args.n_samples_per_prompt) * max_rows_per_sample
        )
        if partition_actual_rows > max_partition_actual_rows:
            raise RuntimeError(
                "Agentic variable-row export exceeded the configured partition bound: "
                f"partition_rollout_id={partition_rollout_id}, "
                f"actual_rows={partition_actual_rows}, "
                f"max_rows={max_partition_actual_rows}."
            )
        for identities, manifest in validated_groups:
            for sample, identity in identities:
                setattr(
                    sample,
                    _AGENTIC_TRANSFER_IDENTITY_ATTR,
                    {**copy.deepcopy(identity), **copy.deepcopy(manifest)},
                )
        self._partition_actual_rows[partition_rollout_id] = partition_actual_rows
        self._partition_identity_state[partition_rollout_id] = {
            "row_ids": existing_row_ids.union(candidate_row_ids),
            "group_manifests": next_group_manifests,
            "policy_version": existing_policy_version or candidate_policy_version,
        }
        if not is_last:
            return groups

        padding_rows = (-partition_actual_rows) % self.args.global_batch_size
        if padding_rows == 0:
            return groups
        padding_group = [
            _variable_row_padding_sample(args=self.args, template=actual_samples[-1]) for _ in range(padding_rows)
        ]
        return [*groups, padding_group]

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
        transferred_group_count = len(groups)
        if self._variable_row_mode:
            groups = self._prepare_variable_row_groups(
                groups=groups,
                partition_rollout_id=partition_rollout_id,
                is_last=is_last,
            )
            if self.data_system_client is None:
                if is_last:
                    self._partition_actual_rows.pop(partition_rollout_id, None)
                    self._partition_identity_state.pop(partition_rollout_id, None)
                return transferred_group_count
            predecessor = self._transfer_tasks[-1] if self._transfer_tasks else None
            task = asyncio.create_task(
                self._dispatch_transfer_batch_after(
                    predecessor=predecessor,
                    groups=groups,
                    partition_rollout_id=partition_rollout_id,
                    is_last=is_last,
                )
            )
        else:
            if self.data_system_client is None:
                return transferred_group_count
            task = asyncio.create_task(
                self._dispatch_transfer_batch(
                    groups=groups,
                    partition_rollout_id=partition_rollout_id,
                    is_last=is_last,
                )
            )
        self._transfer_tasks.append(task)
        return transferred_group_count

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

    def transfer_timing_snapshot(self, *, clear: bool = False) -> list[dict[str, Any]]:
        """Return raw per-batch transfer timings for median/p95 reporting."""

        snapshot = [copy.deepcopy(record) for record in self._transfer_timing_records]
        if clear:
            self._transfer_timing_records.clear()
        return snapshot

    def resident_group_keys(self) -> set[GroupKey]:
        return {sample_group_key(group) for group in self.ready_group_buffer}

    def committed_transfer_groups_snapshot(self) -> list[list]:
        return list(self._step_output_groups)

    def release_step_output_payloads(self) -> None:
        self._step_output_groups.clear()

    def enqueue_ready_groups(self, groups) -> None:
        for group in groups:
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
        self._partition_actual_rows.clear()
        self._partition_identity_state.clear()
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
        self._partition_actual_rows.clear()
        self._partition_identity_state.clear()
