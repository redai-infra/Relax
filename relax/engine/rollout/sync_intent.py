# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Synchronization-aware rollout admission policy."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Literal


SYNC_INTENT_POLICY_ENV = "RELAX_SYNC_INTENT_POLICY"
SYNC_INTENT_ADMISSION_POLICY_ENV = "RELAX_SYNC_INTENT_ADMISSION_POLICY"
SYNC_INTENT_TTL_ENV = "RELAX_SYNC_INTENT_TTL_SECONDS"
SYNC_INTENT_WINDOW_GROUPS_ENV = "RELAX_SYNC_INTENT_WINDOW_GROUPS"
SYNC_INTENT_QUIESCE_MULTIPLIER_ENV = "RELAX_SYNC_INTENT_QUIESCE_MULTIPLIER"
SYNC_INTENT_QUIESCE_FLOOR_ENV = "RELAX_SYNC_INTENT_QUIESCE_FLOOR_SECONDS"
SYNC_INTENT_ABORT_RETRY_INTERVAL_ENV = "RELAX_SYNC_INTENT_ABORT_RETRY_INTERVAL_SECONDS"
SYNC_INTENT_ABORT_TIMEOUT_ENV = "RELAX_SYNC_INTENT_ABORT_TIMEOUT_SECONDS"
SYNC_INTENT_PROTECTED_DRAIN_TIMEOUT_ENV = "RELAX_SYNC_INTENT_PROTECTED_DRAIN_TIMEOUT_SECONDS"
DEFAULT_ROLLOUT_REQUEST_PRIORITY = 0
CARRYOVER_RESUME_REQUEST_PRIORITY = 1
OLD_DEBT_REQUEST_PRIORITY = 2
SyncIntentPhase = Literal["data_wait", "actor_train", "quiesce", "weight_sync"]
_PHASE_ORDER: dict[SyncIntentPhase, int] = {
    "data_wait": 0,
    "actor_train": 1,
    "quiesce": 2,
    "weight_sync": 3,
}


def sync_intent_policy_enabled() -> bool:
    return os.environ.get(SYNC_INTENT_POLICY_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def sync_intent_admission_policy_enabled() -> bool:
    raw_value = os.environ.get(SYNC_INTENT_ADMISSION_POLICY_ENV)
    if raw_value is None:
        return sync_intent_policy_enabled()
    raw_value = raw_value.strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{SYNC_INTENT_ADMISSION_POLICY_ENV} must be a boolean, got {raw_value!r}")


def sync_intent_ttl_seconds() -> float:
    return _positive_float_env(SYNC_INTENT_TTL_ENV, "600")


def sync_intent_window_groups() -> int | None:
    raw_value = os.environ.get(SYNC_INTENT_WINDOW_GROUPS_ENV)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        groups = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{SYNC_INTENT_WINDOW_GROUPS_ENV} must be a positive integer") from error
    if groups <= 0:
        raise ValueError(f"{SYNC_INTENT_WINDOW_GROUPS_ENV} must be positive, got {groups}")
    return groups


def _positive_float_env(name: str, default: str) -> float:
    raw_value = os.environ.get(name, default)
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number, got {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def sync_intent_quiesce_multiplier() -> float:
    return _positive_float_env(SYNC_INTENT_QUIESCE_MULTIPLIER_ENV, "1.25")


def sync_intent_quiesce_floor_seconds() -> float:
    return _positive_float_env(SYNC_INTENT_QUIESCE_FLOOR_ENV, "2.0")


def sync_intent_abort_retry_interval_seconds() -> float:
    return _positive_float_env(SYNC_INTENT_ABORT_RETRY_INTERVAL_ENV, "0.5")


def sync_intent_abort_timeout_seconds() -> float:
    return _positive_float_env(SYNC_INTENT_ABORT_TIMEOUT_ENV, "15")


def sync_intent_protected_drain_timeout_seconds() -> float:
    return _positive_float_env(SYNC_INTENT_PROTECTED_DRAIN_TIMEOUT_ENV, "600")


@dataclass(frozen=True)
class SyncIntentSnapshot:
    active: bool
    sync_id: int | None = None
    actor_rollout_id: int | None = None
    started_at: float | None = None
    phase: SyncIntentPhase | None = None
    phase_started_at: float | None = None
    estimated_phase_seconds: float | None = None
    expired: bool = False


class SyncIntentController:
    """Lock-protected process-local synchronization intent."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sync_id: int | None = None
        self._actor_rollout_id: int | None = None
        self._started_at: float | None = None
        self._phase: SyncIntentPhase | None = None
        self._phase_started_at: float | None = None
        self._estimated_phase_seconds: float | None = None

    def begin(self, sync_id: int, actor_rollout_id: int) -> SyncIntentSnapshot:
        if sync_id < 0:
            raise ValueError(f"sync_id must be non-negative, got {sync_id}")
        if actor_rollout_id < 0:
            raise ValueError(f"actor_rollout_id must be non-negative, got {actor_rollout_id}")
        with self._lock:
            if self._sync_id is not None:
                if sync_id < self._sync_id:
                    return self._snapshot_locked()
                if sync_id == self._sync_id:
                    if actor_rollout_id != self._actor_rollout_id:
                        raise ValueError(
                            "sync intent identity mismatch: "
                            f"sync_id={sync_id}, existing_actor_rollout_id={self._actor_rollout_id}, "
                            f"requested_actor_rollout_id={actor_rollout_id}"
                        )
                    return self._snapshot_locked()
            now = time.monotonic()
            self._sync_id = sync_id
            self._actor_rollout_id = actor_rollout_id
            self._started_at = now
            self._phase = "data_wait"
            self._phase_started_at = now
            self._estimated_phase_seconds = None
            return self._snapshot_locked()

    def update_phase(
        self,
        sync_id: int,
        phase: SyncIntentPhase,
        estimated_phase_seconds: float | None = None,
    ) -> SyncIntentSnapshot:
        if phase not in _PHASE_ORDER:
            raise ValueError(f"unsupported sync intent phase: {phase!r}")
        if estimated_phase_seconds is not None and estimated_phase_seconds <= 0:
            raise ValueError("estimated_phase_seconds must be positive when provided")
        with self._lock:
            if self._sync_id is None:
                return SyncIntentSnapshot(active=False)
            if sync_id != self._sync_id:
                if sync_id < self._sync_id:
                    return self._snapshot_locked()
                raise ValueError(f"cannot update future sync intent {sync_id}; active sync_id={self._sync_id}")
            assert self._phase is not None
            if _PHASE_ORDER[phase] < _PHASE_ORDER[self._phase]:
                return self._snapshot_locked()
            if phase != self._phase:
                self._phase = phase
                self._phase_started_at = time.monotonic()
            self._estimated_phase_seconds = estimated_phase_seconds
            return self._snapshot_locked()

    def end(self, sync_id: int) -> SyncIntentSnapshot:
        with self._lock:
            if self._sync_id is None:
                return SyncIntentSnapshot(active=False)
            if sync_id < self._sync_id:
                return self._snapshot_locked()
            if sync_id > self._sync_id:
                raise ValueError(f"cannot end future sync intent {sync_id}; active sync_id={self._sync_id}")
            self._clear_locked()
            return SyncIntentSnapshot(active=False)

    def snapshot(self) -> SyncIntentSnapshot:
        with self._lock:
            if self._sync_id is None:
                return SyncIntentSnapshot(active=False)
            assert self._started_at is not None
            if time.monotonic() - self._started_at > sync_intent_ttl_seconds():
                self._clear_locked()
                return SyncIntentSnapshot(active=False, expired=True)
            return self._snapshot_locked()

    def reset(self) -> None:
        with self._lock:
            self._clear_locked()

    def wait_until_inactive(self, sync_id: int, *, poll_interval_seconds: float = 0.01) -> SyncIntentSnapshot:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        while True:
            snapshot = self.snapshot()
            if not snapshot.active or snapshot.sync_id != sync_id:
                return snapshot
            time.sleep(poll_interval_seconds)

    def _snapshot_locked(self) -> SyncIntentSnapshot:
        return SyncIntentSnapshot(
            active=self._sync_id is not None,
            sync_id=self._sync_id,
            actor_rollout_id=self._actor_rollout_id,
            started_at=self._started_at,
            phase=self._phase,
            phase_started_at=self._phase_started_at,
            estimated_phase_seconds=self._estimated_phase_seconds,
        )

    def _clear_locked(self) -> None:
        self._sync_id = None
        self._actor_rollout_id = None
        self._started_at = None
        self._phase = None
        self._phase_started_at = None
        self._estimated_phase_seconds = None


_SYNC_INTENT = SyncIntentController()


def begin_sync_intent(sync_id: int, actor_rollout_id: int) -> SyncIntentSnapshot:
    return _SYNC_INTENT.begin(sync_id, actor_rollout_id)


def end_sync_intent(sync_id: int) -> SyncIntentSnapshot:
    return _SYNC_INTENT.end(sync_id)


def update_sync_intent_phase(
    sync_id: int,
    phase: SyncIntentPhase,
    estimated_phase_seconds: float | None = None,
) -> SyncIntentSnapshot:
    return _SYNC_INTENT.update_phase(sync_id, phase, estimated_phase_seconds)


def get_sync_intent() -> SyncIntentSnapshot:
    return _SYNC_INTENT.snapshot()


def reset_sync_intent() -> None:
    _SYNC_INTENT.reset()


def wait_for_sync_intent_end(sync_id: int, *, poll_interval_seconds: float = 0.01) -> SyncIntentSnapshot:
    return _SYNC_INTENT.wait_until_inactive(sync_id, poll_interval_seconds=poll_interval_seconds)


def plan_intent_guard_fetch(
    *,
    snapshot: SyncIntentSnapshot,
    physical_rollout_id: int,
    old_debt_groups: int,
    completed_debt_groups: int,
    inflight_debt_groups: int,
    observed_group_latency_seconds: float | None,
    default_fetch_groups: int,
    now: float | None = None,
) -> int:
    """Admit debt first and quiesce fresh work near publication."""

    if default_fetch_groups <= 0:
        raise ValueError(f"default_fetch_groups must be positive, got {default_fetch_groups}")
    if min(old_debt_groups, completed_debt_groups, inflight_debt_groups) < 0:
        raise ValueError("debt group counts must be non-negative")
    if observed_group_latency_seconds is not None and observed_group_latency_seconds <= 0:
        raise ValueError("observed_group_latency_seconds must be positive when provided")
    if not snapshot.active:
        return default_fetch_groups
    if snapshot.actor_rollout_id is not None and physical_rollout_id <= snapshot.actor_rollout_id:
        return default_fetch_groups
    missing_debt_groups = max(old_debt_groups - completed_debt_groups - inflight_debt_groups, 0)
    debt_fetch_groups = min(default_fetch_groups, missing_debt_groups)
    if debt_fetch_groups >= default_fetch_groups or not should_admit_fresh(
        snapshot,
        observed_group_latency_seconds=observed_group_latency_seconds,
        now=now,
    ):
        return debt_fetch_groups
    return default_fetch_groups


def plan_dp_aligned_extra_groups(
    *,
    current_groups: int,
    available_extra_groups: int,
    dp_size: int,
) -> int:
    if min(current_groups, available_extra_groups) < 0:
        raise ValueError("group counts must be non-negative")
    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    aligned_total = ((current_groups + available_extra_groups) // dp_size) * dp_size
    return max(aligned_total - current_groups, 0)


def validate_disjoint_rollout_groups(
    accepted_groups: list[list[object]],
    buffered_groups: list[list[object]],
) -> None:
    accepted_ids = {id(sample) for group in accepted_groups for sample in group}
    buffered_ids = {id(sample) for group in buffered_groups for sample in group}
    overlap = accepted_ids & buffered_ids
    if overlap:
        raise RuntimeError(f"Rollout samples have dual accepted/buffer ownership: {len(overlap)} samples")


def should_admit_fresh(
    snapshot: SyncIntentSnapshot,
    *,
    observed_group_latency_seconds: float | None,
    now: float | None = None,
) -> bool:
    if not snapshot.active:
        return True
    if snapshot.phase in (None, "data_wait"):
        return True
    if snapshot.phase in ("quiesce", "weight_sync"):
        return False
    if snapshot.estimated_phase_seconds is None or snapshot.phase_started_at is None:
        return True
    if observed_group_latency_seconds is None:
        return True
    current_time = time.monotonic() if now is None else now
    elapsed = max(current_time - snapshot.phase_started_at, 0.0)
    remaining = max(snapshot.estimated_phase_seconds - elapsed, 0.0)
    quiesce_margin = max(
        sync_intent_quiesce_floor_seconds(),
        observed_group_latency_seconds * sync_intent_quiesce_multiplier(),
    )
    return remaining > quiesce_margin


def plan_adaptive_window_fetch(
    *,
    resident_groups: int,
    baseline_fetch_groups: int,
    remaining_commit_groups: int,
    hedge_groups: int,
    window_initialized: bool,
    completed_buffer_groups: int = 0,
) -> int:
    """Seed a carry-over-aware window, then refill only the commit gap."""

    if min(resident_groups, remaining_commit_groups, hedge_groups, completed_buffer_groups) < 0:
        raise ValueError("resident, remaining commit, hedge, and completed buffer groups must be non-negative")
    if baseline_fetch_groups <= 0:
        raise ValueError("baseline_fetch_groups must be positive")
    if remaining_commit_groups == 0:
        return 0
    if window_initialized:
        desired_window = remaining_commit_groups
    else:
        configured_window = sync_intent_window_groups()
        maximum_window = configured_window if configured_window is not None else baseline_fetch_groups
        fresh_shortfall = max(remaining_commit_groups - completed_buffer_groups, 0)
        bounded_hedge = min(hedge_groups, fresh_shortfall)
        desired_window = min(maximum_window, remaining_commit_groups + bounded_hedge)
    return max(desired_window - resident_groups, 0)


def plan_baseline_window_fetch(
    *,
    resident_groups: int,
    submit_target_groups: int,
    fetch_batch_groups: int,
) -> int:
    """Mirror the ordinary rollout's fixed-batch oversampling admission."""

    if min(resident_groups, submit_target_groups) < 0:
        raise ValueError("resident and submit target groups must be non-negative")
    if fetch_batch_groups <= 0:
        raise ValueError("fetch batch groups must be positive")
    if submit_target_groups == 0 or resident_groups >= submit_target_groups:
        return 0
    return fetch_batch_groups


def plan_carry_aware_oversampling_seed(
    *,
    oversampling_envelope_groups: int,
    adopted_current_groups: int,
    missing_debt_groups: int,
) -> int:
    """Seed missing debt plus the unoccupied part of the current candidate
    envelope."""

    if min(oversampling_envelope_groups, adopted_current_groups, missing_debt_groups) < 0:
        raise ValueError("oversampling envelope, adopted current, and missing debt groups must be non-negative")
    if oversampling_envelope_groups == 0:
        raise ValueError("oversampling_envelope_groups must be positive")
    return missing_debt_groups + max(oversampling_envelope_groups - adopted_current_groups, 0)


def mark_work_origin(
    samples: list[list[object]],
    old_debt_groups: int,
    *,
    fresh_origin: str = "fresh",
) -> None:
    for group_index, group in enumerate(samples):
        if group_index < old_debt_groups:
            origin = "old_debt"
        elif _is_partial_resume_group(group):
            origin = "partial_resume"
        else:
            origin = fresh_origin
        for sample in group:
            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                setattr(sample, "metadata", metadata)
            metadata["work_origin"] = origin


def _is_partial_resume_group(group: list[object]) -> bool:
    for sample in group:
        status = getattr(sample, "status", None)
        status_value = getattr(status, "value", status)
        if status_value == "aborted" or int(getattr(sample, "abort_count", 0) or 0) > 0:
            return True
    return False


def resolve_partition_request_priority(args: object, sample: object) -> int | None:
    if not getattr(args, "sglang_enable_priority_scheduling", False):
        return None
    metadata = getattr(sample, "metadata", None)
    work_origin = metadata.get("work_origin") if isinstance(metadata, dict) else None
    if work_origin is None:
        return None
    if work_origin == "old_debt":
        return OLD_DEBT_REQUEST_PRIORITY
    if work_origin == "partial_resume":
        return CARRYOVER_RESUME_REQUEST_PRIORITY
    return DEFAULT_ROLLOUT_REQUEST_PRIORITY
