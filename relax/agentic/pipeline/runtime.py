# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import base64
import copy
import functools
import json
import os
import signal
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx
import numpy as np
import ray
import torch

from relax.agentic import AGENTIC_CHAT_API_ROUTE_PREFIX, AGENTIC_CHAT_API_SERVICE_NAME
from relax.agentic.pipeline import GroupKey, sample_group_key
from relax.agentic.pipeline.prepare import (
    ExecutionBatchInput,
    PrepareGroupSpec,
    PrepareGroupState,
    PrepareRequestHandle,
)
from relax.agentic.profile import (
    TRACE_KEY,
    mark_agentic_event,
    mark_metadata_agentic_event,
    merge_agentic_trace,
)
from relax.agentic.runner.ipc import LauncherClient, ensure_local_launcher_daemon
from relax.agentic.session.state import (
    FinalizedResultTransport,
    TrainingFieldArtifact,
    check_messages,
    normalize_template_kwargs,
    normalize_tools,
)
from relax.utils.http_utils import post
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)

_AGENTIC_SERVICE_CLIENT: "_ServeHandleChatControlClient | None" = None


@dataclass
class RequestDispatchContext:
    rollout_id: int
    group_key: GroupKey
    expected_count: int
    slot_idx: int
    envelope: Any


class RuntimeSlotState(str, Enum):
    RUNNING_APP = "running_app"
    MATERIALIZING = "materializing"
    MATERIALIZED = "materialized"
    DISCARDED = "discarded"


@dataclass
class RuntimeSlot:
    request_id: str
    session_id: str
    dispatch_context: RequestDispatchContext
    seed_sample: Sample
    managed_session_handle: Any
    managed_session_submitted_at: float
    state: RuntimeSlotState = RuntimeSlotState.RUNNING_APP
    materialization_task: asyncio.Task | None = None
    managed_session_handle_released: bool = False

    @property
    def group_key(self) -> GroupKey:
        return self.dispatch_context.group_key

    @property
    def admission_rollout_id(self) -> int:
        return self.dispatch_context.rollout_id


@dataclass
class RuntimeGroup:
    group_key: GroupKey
    expected_count: int
    admission_rollout_id: int
    request_ids: set[str] = field(default_factory=set)
    materialized_slot_idxs: set[int] = field(default_factory=set)
    materialized_samples_by_slot: dict[int, list[Sample]] = field(default_factory=dict)

    def add_materialized_record(self, record: dict[str, Any], *, store_samples: bool) -> None:
        slot_idx = int(record["slot_idx"])
        self.materialized_slot_idxs.add(slot_idx)
        if store_samples:
            self.materialized_samples_by_slot[slot_idx] = record["samples"]

    def is_complete(self) -> bool:
        return len(self.materialized_slot_idxs) >= self.expected_count

    def slot_count(self) -> int:
        return len(self.materialized_slot_idxs)

    def materialized_group(self) -> list[Sample]:
        completed_group: list[Sample] = []
        for idx in range(self.expected_count):
            samples = self.materialized_samples_by_slot.get(idx)
            if samples is None:
                raise RuntimeError(
                    f"RuntimeGroup cannot build a materialized group before all stored slots are ready: "
                    f"group_key={self.group_key!r}, missing_slot={idx}."
                )
            completed_group.extend(samples)
        return completed_group


@dataclass(frozen=True)
class _MaterializationDrop:
    group_key: GroupKey
    session_id: str
    reason: str
    info: dict[str, Any] = field(default_factory=dict)


RewardValue = float | dict[str, Any] | None
_RESERVED_AGENT_ENV_PREFIXES = ("RELAX_",)


class AgentExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionInput:
    session_id: str
    rollout_mode: str
    group_id: str
    input_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_agent_payload(self) -> dict[str, Any]:
        return copy.deepcopy(self.input_payload)


@dataclass(frozen=True)
class SessionOutput:
    metadata: dict[str, Any] = field(default_factory=dict)
    reward: RewardValue = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SessionOutput":
        if not isinstance(payload, dict):
            raise TypeError(f"SessionOutput payload must be a JSON object, got {type(payload)}")
        unknown_keys = set(payload) - {"metadata", "reward"}
        if unknown_keys:
            raise TypeError("SessionOutput only supports 'metadata' and 'reward'")
        metadata = payload.get("metadata", {})
        reward = payload.get("reward")
        if not isinstance(metadata, dict):
            raise TypeError("SessionOutput 'metadata' must be a JSON object")
        if reward is not None and not isinstance(reward, (int, float, dict)):
            raise TypeError("SessionOutput 'reward' must be null, number, or JSON object")
        return cls(metadata=metadata, reward=reward)


@dataclass(frozen=True)
class ManagedCommandAppSpec:
    command: str = ""
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 1800.0


@dataclass(frozen=True)
class ManagedSessionHandle:
    runner_id: int
    session_handle: str


def _parse_agent_env_items(raw_items: Any) -> dict[str, str]:
    if raw_items is None or raw_items == "":
        return {}
    if not isinstance(raw_items, list) or not all(isinstance(item, str) for item in raw_items):
        raise TypeError("--agent-env must be provided as a list of KEY=VALUE strings.")
    parsed_env: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"--agent-env entry must be KEY=VALUE, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--agent-env entry must include a non-empty key, got {item!r}.")
        if any(key.startswith(prefix) for prefix in _RESERVED_AGENT_ENV_PREFIXES):
            raise ValueError(f"--agent-env does not allow reserved key {key!r}.")
        parsed_env[key] = value
    return parsed_env


def load_agent_app_spec_from_args(args: Any) -> ManagedCommandAppSpec:
    command = args.agent_command
    if not isinstance(command, str) or not command.strip():
        raise RuntimeError("--agent-command is required for the agentic runtime.")
    cwd = args.agent_cwd
    if not isinstance(cwd, str) or not cwd.strip():
        raise RuntimeError("--agent-cwd is required for the agentic runtime.")
    cwd_path = Path(cwd).expanduser()
    if not cwd_path.is_dir():
        raise RuntimeError(f"--agent-cwd must point to an existing directory, got {cwd!r}.")
    agent_env = _parse_agent_env_items(args.agent_env)
    return ManagedCommandAppSpec(
        command=command.strip(),
        cwd=str(cwd_path.resolve()),
        env=agent_env,
        timeout_s=float(args.agent_timeout),
    )


def _metadata_patch(base_metadata: dict[str, Any] | None, updated_metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(base_metadata, dict):
        return dict(updated_metadata)
    patch: dict[str, Any] = {}
    for key, value in updated_metadata.items():
        if key not in base_metadata or base_metadata[key] != value:
            patch[key] = value
    return patch


def _decode_output(encoded: str | None) -> str:
    if not encoded:
        return ""
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8", errors="ignore")


async def execute_managed_session_input(
    *,
    spec: ManagedCommandAppSpec,
    session_input: SessionInput,
    launcher_client: LauncherClient | None = None,
    before_process_launch: Callable[[], Any] | None = None,
    on_process_started: Callable[[], None] | None = None,
    on_process_launched: Callable[[str], None] | None = None,
    on_process_finished: Callable[[str], None] | None = None,
    is_process_timed_out: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    prepare_started_at = time.time()

    tmpdir_handle = tempfile.TemporaryDirectory(prefix="relax-agentic-command-")
    launcher_handle: str | None = None
    process_finished_reported = False

    def _report_process_finished() -> None:
        nonlocal process_finished_reported
        if launcher_handle is None or process_finished_reported:
            return
        process_finished_reported = True
        if on_process_finished is not None:
            on_process_finished(launcher_handle)

    try:
        tmpdir_path = Path(tmpdir_handle.name)
        input_path = tmpdir_path / "session_input.json"
        output_path = tmpdir_path / "session_output.json"
        log_path = tmpdir_path / "command.log"
        env = {str(key): str(value) for key, value in spec.env.items()}
        env.update(
            {
                "RELAX_SESSION_ID": session_input.session_id,
                "RELAX_ROLLOUT_MODE": session_input.rollout_mode,
                "RELAX_GROUP_ID": session_input.group_id,
                "RELAX_SESSION_IO_DIR": str(tmpdir_path),
                "RELAX_OUTPUT_JSON": str(output_path),
            }
        )
        input_path.write_text(
            json.dumps(
                session_input.to_agent_payload(),
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        env["RELAX_INPUT_JSON"] = str(input_path)
        prepare_finished_at = time.time()
        if launcher_client is None:
            launcher_client = LauncherClient(socket_path=ensure_local_launcher_daemon())
        if before_process_launch is not None:
            launch_hook_result = before_process_launch()
            if asyncio.iscoroutine(launch_hook_result):
                await launch_hook_result
        launch_started_at = time.time()
        request_id = session_input.metadata["request_id"]
        launch_payload = await launcher_client.launch(
            command=spec.command,
            cwd=spec.cwd,
            env=env,
        )
        launch_returned_at = time.time()
        process_started_at = float(launch_payload["started_at"])
        spawn_returned_at = float(launch_payload["spawn_returned_at"])
        handle = str(launch_payload["handle"])
        launcher_handle = handle
        if on_process_launched is not None:
            on_process_launched(handle)
        logger.debug(
            "Managed session launched session_id=%s request_id=%s handle=%s pid=%s",
            session_input.session_id,
            request_id,
            handle,
            launch_payload["pid"],
        )
        if on_process_started is not None:
            on_process_started()
        process_wait_finished = False

        try:
            wait_payload = await launcher_client.wait(handle=handle)
            process_wait_finished = True
        except asyncio.CancelledError:
            raise
        finally:
            if process_wait_finished:
                _report_process_finished()
        process_exited_at = float(wait_payload["exited_at"])
        return_code = int(wait_payload["exit_code"])
        logger.debug(
            "Managed session exited session_id=%s request_id=%s handle=%s exit_code=%s",
            session_input.session_id,
            request_id,
            handle,
            return_code,
        )
        stdout_text = _decode_output(wait_payload["stdout_b64"])
        stderr_text = _decode_output(wait_payload["stderr_b64"])
        combined_output = stdout_text + stderr_text
        log_path.write_text(combined_output, encoding="utf-8")

        process_timed_out = bool(is_process_timed_out is not None and is_process_timed_out())
        if process_timed_out:
            raise AgentExecutionError(
                f"Managed command agent timed out after {spec.timeout_s} seconds.\n{combined_output}".rstrip()
            )
        if return_code != 0:
            raise AgentExecutionError(
                f"Managed command agent exited with code {return_code}.\n{combined_output}".rstrip()
            )
        output_read_at = time.time()
        if output_path.exists():
            try:
                session_output = SessionOutput.from_payload(json.loads(output_path.read_text(encoding="utf-8")))
            except Exception as exc:
                raise AgentExecutionError(
                    f"Managed command agent produced invalid output.\n{combined_output}".rstrip()
                ) from exc
        else:
            session_output = SessionOutput()
        payload_ready_at = time.time()
        session_output_metadata = _metadata_patch(
            session_input.input_payload.get("metadata"),
            session_output.metadata,
        )
        payload = {
            "reward": session_output.reward,
            "_session_id": session_input.session_id,
            "_session_output_metadata": session_output_metadata,
            "_agentic_trace_events": {},
        }
        profile = payload["_agentic_trace_events"]
        mark_agentic_event(profile, "managed_prepare_start_at", prepare_started_at)
        mark_agentic_event(profile, "managed_prepare_end_at", prepare_finished_at)
        mark_agentic_event(profile, "managed_launch_start_at", launch_started_at)
        mark_agentic_event(profile, "managed_process_start_at", process_started_at)
        if spawn_returned_at is not None:
            mark_agentic_event(profile, "managed_process_spawn_return_at", spawn_returned_at)
        mark_agentic_event(profile, "managed_launch_return_at", launch_returned_at)
        mark_agentic_event(profile, "managed_process_exit_at", process_exited_at)
        mark_agentic_event(profile, "managed_output_read_at", output_read_at)
        mark_agentic_event(profile, "managed_output_ready_at", payload_ready_at)
        return payload
    finally:
        _report_process_finished()
        tmpdir_handle.cleanup()


def _managed_spec_to_payload(spec: ManagedCommandAppSpec) -> dict[str, Any]:
    return {
        "command": spec.command,
        "cwd": spec.cwd,
        "env": dict(spec.env),
        "timeout_s": spec.timeout_s,
    }


def _managed_spec_from_payload(payload: dict[str, Any]) -> ManagedCommandAppSpec:
    command = payload["command"]
    cwd = payload["cwd"]
    env = payload["env"]
    timeout_s = payload["timeout_s"]
    return ManagedCommandAppSpec(
        command=command.strip(),
        cwd=cwd,
        env={str(key): str(value) for key, value in env.items()},
        timeout_s=float(timeout_s),
    )


def _session_input_to_payload(session_input: SessionInput) -> dict[str, Any]:
    return {
        "session_id": session_input.session_id,
        "group_id": session_input.group_id,
        "rollout_mode": session_input.rollout_mode,
        "input_payload": dict(session_input.input_payload),
        "metadata": dict(session_input.metadata),
    }


def _session_input_from_payload(payload: dict[str, Any]) -> SessionInput:
    session_id = payload["session_id"]
    group_id = payload["group_id"]
    rollout_mode = payload["rollout_mode"]
    input_payload = payload["input_payload"]
    metadata = payload["metadata"]
    return SessionInput(
        session_id=session_id,
        group_id=group_id,
        rollout_mode=rollout_mode,
        input_payload=dict(input_payload),
        metadata=dict(metadata),
    )


@ray.remote(max_restarts=3, max_task_retries=0, num_cpus=0, max_concurrency=256)
class ManagedSessionRunner:
    def __init__(self, *, runner_id: int, launch_capacity: int) -> None:
        self._runner_id = runner_id
        self._capacity = launch_capacity
        self._inflight = 0
        self._launch_permits_in_use = 0
        self._shutdown_requested = False
        self._semaphore = asyncio.BoundedSemaphore(self._capacity)
        self._launcher_client = LauncherClient(socket_path=ensure_local_launcher_daemon())
        self._active_session_ids: set[str] = set()
        self._active_launcher_handles: set[str] = set()
        self._waiting_launch_session_ids: set[str] = set()
        self._recent_completed_session_ids: deque[str] = deque(maxlen=32)
        self._tasks_by_handle: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._session_id_by_handle: dict[str, str] = {}
        self._completed_session_handles: deque[str] = deque()
        self._launcher_handle_by_session_handle: dict[str, str] = {}
        self._timeout_remaining_s_by_handle: dict[str, float] = {}
        self._timeout_active_started_at_by_handle: dict[str, float] = {}
        self._timeout_tasks_by_handle: dict[str, asyncio.Task] = {}
        self._timeout_active_handles: set[str] = set()
        self._timed_out_session_handles: set[str] = set()

    async def submit_sessions(
        self,
        spec_payload: dict[str, Any],
        session_input_payloads: list[dict[str, Any]],
    ) -> list[str]:
        return [
            self._submit_session_task(
                spec_payload=spec_payload,
                session_input_payload=session_input_payload,
            )
            for session_input_payload in session_input_payloads
        ]

    def _submit_session_task(
        self,
        *,
        spec_payload: dict[str, Any],
        session_input_payload: dict[str, Any],
    ) -> str:
        session_handle = uuid4().hex
        session_id = session_input_payload["session_id"]
        task = asyncio.create_task(
            self._run_session(
                session_handle=session_handle,
                spec_payload=spec_payload,
                session_input_payload=session_input_payload,
            )
        )
        task.add_done_callback(lambda _task, handle=session_handle: self._completed_session_handles.append(handle))
        self._tasks_by_handle[session_handle] = task
        self._session_id_by_handle[session_handle] = str(session_id)
        self._timeout_remaining_s_by_handle[session_handle] = float(spec_payload["timeout_s"])
        return session_handle

    def _start_session_timeout_clock(self, session_handle: str) -> None:
        if session_handle not in self._timeout_active_handles:
            return
        if session_handle in self._timeout_active_started_at_by_handle:
            return
        if session_handle not in self._launcher_handle_by_session_handle:
            return
        remaining_s = self._timeout_remaining_s_by_handle.get(session_handle)
        if remaining_s is None:
            return
        self._timeout_active_started_at_by_handle[session_handle] = time.monotonic()
        task = self._timeout_tasks_by_handle.get(session_handle)
        if task is not None and not task.done():
            task.cancel()
        self._timeout_tasks_by_handle[session_handle] = asyncio.create_task(
            self._terminate_session_on_timeout(session_handle=session_handle, delay_s=max(0.0, remaining_s))
        )

    def _pause_session_timeout_clock(self, session_handle: str) -> None:
        self._timeout_active_handles.discard(session_handle)
        active_started_at = self._timeout_active_started_at_by_handle.pop(session_handle, None)
        if active_started_at is not None and session_handle in self._timeout_remaining_s_by_handle:
            elapsed_s = time.monotonic() - active_started_at
            self._timeout_remaining_s_by_handle[session_handle] = max(
                0.0,
                self._timeout_remaining_s_by_handle[session_handle] - elapsed_s,
            )
        task = self._timeout_tasks_by_handle.pop(session_handle, None)
        if task is not None and not task.done():
            task.cancel()

    def _clear_session_timeout_clock(self, session_handle: str) -> None:
        self._pause_session_timeout_clock(session_handle)
        self._timeout_remaining_s_by_handle.pop(session_handle, None)
        self._launcher_handle_by_session_handle.pop(session_handle, None)
        self._timed_out_session_handles.discard(session_handle)

    async def _terminate_session_on_timeout(self, *, session_handle: str, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        if session_handle not in self._timeout_active_started_at_by_handle:
            return
        launcher_handle = self._launcher_handle_by_session_handle.get(session_handle)
        if not launcher_handle:
            return
        session_id = self._session_id_by_handle.get(session_handle, "")
        logger.warning(
            "Managed session active timeout reached; sending SIGTERM runner_id=%s session_id=%s "
            "session_handle=%s launcher_handle=%s timeout_s=%s",
            self._runner_id,
            session_id,
            session_handle,
            launcher_handle,
            self._timeout_remaining_s_by_handle.get(session_handle, 0.0),
        )
        self._timed_out_session_handles.add(session_handle)
        await self._launcher_client.kill(handle=launcher_handle, signal_value=signal.SIGTERM, forget=True)
        task = self._tasks_by_handle.get(session_handle)
        if task is not None and not task.done():
            task.cancel()

    async def set_session_timeouts_active(self, *, session_handles: list[str], active: bool) -> int:
        changed_count = 0
        for session_handle in dict.fromkeys(session_handles):
            if session_handle not in self._tasks_by_handle:
                if active:
                    raise KeyError(f"Unknown managed session handle: {session_handle}")
                continue
            if active:
                if session_handle not in self._timeout_active_handles:
                    changed_count += 1
                self._timeout_active_handles.add(session_handle)
                self._start_session_timeout_clock(session_handle)
            else:
                if session_handle in self._timeout_active_handles:
                    changed_count += 1
                self._pause_session_timeout_clock(session_handle)
        return changed_count

    async def _run_session(
        self,
        *,
        session_handle: str,
        spec_payload: dict[str, Any],
        session_input_payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._inflight += 1
        released = False
        permit_acquired = False
        launch_permit_wait_started_at: float | None = None
        session_id = session_input_payload["session_id"]
        request_id = session_input_payload["metadata"]["request_id"]
        self._active_session_ids.add(session_id)
        logger.debug(
            "ManagedSessionRunner runner_id=%s start session_id=%s request_id=%s inflight=%s",
            self._runner_id,
            session_id,
            request_id,
            self._inflight,
        )

        def _release_permit() -> None:
            nonlocal released
            if released:
                return
            released = True
            if session_id:
                self._waiting_launch_session_ids.discard(session_id)
            if permit_acquired:
                self._semaphore.release()
                self._launch_permits_in_use -= 1
                if self._launch_permits_in_use < 0:
                    raise RuntimeError(
                        "ManagedSessionRunner launch permit counter underflow: "
                        f"runner_id={self._runner_id}, session_id={session_id}, request_id={request_id}."
                    )

        async def _acquire_permit() -> None:
            nonlocal permit_acquired
            nonlocal launch_permit_wait_started_at
            launch_permit_wait_started_at = time.time()
            if session_id:
                self._waiting_launch_session_ids.add(session_id)
            await self._semaphore.acquire()
            permit_acquired = True
            self._launch_permits_in_use += 1
            if self._launch_permits_in_use > self._capacity:
                raise RuntimeError(
                    "ManagedSessionRunner launch permits exceed capacity: "
                    f"runner_id={self._runner_id}, capacity={self._capacity}, "
                    f"permits_in_use={self._launch_permits_in_use}."
                )

        def _record_launcher_handle(handle: str) -> None:
            self._active_launcher_handles.add(handle)
            self._launcher_handle_by_session_handle[session_handle] = handle
            self._start_session_timeout_clock(session_handle)

        try:
            try:
                result_payload = await execute_managed_session_input(
                    spec=_managed_spec_from_payload(spec_payload),
                    session_input=_session_input_from_payload(session_input_payload),
                    launcher_client=self._launcher_client,
                    before_process_launch=_acquire_permit,
                    on_process_started=_release_permit,
                    on_process_launched=_record_launcher_handle,
                    on_process_finished=self._active_launcher_handles.discard,
                    is_process_timed_out=lambda: session_handle in self._timed_out_session_handles,
                )
            except asyncio.CancelledError:
                if session_handle in self._timed_out_session_handles:
                    raise AgentExecutionError(
                        f"Managed command agent timed out after {spec_payload['timeout_s']} seconds."
                    ) from None
                raise
            except AgentExecutionError as exc:
                message = str(exc)
                shutdown_signal_exit = (
                    "Managed command agent exited with code -9" in message
                    or "Managed command agent exited with code -15" in message
                )
                if not self._shutdown_requested or not shutdown_signal_exit:
                    raise
                logger.info(
                    "ManagedSessionRunner runner_id=%s suppressed managed agent exit during shutdown "
                    "session_id=%s request_id=%s error=%s",
                    self._runner_id,
                    session_id,
                    request_id,
                    message.splitlines()[0] if message else type(exc).__name__,
                )
                return {
                    "reward": None,
                    "_session_id": session_id,
                    "_session_output_metadata": {},
                    "_agentic_trace_events": {
                        "managed_terminated_by_shutdown": True,
                        "managed_shutdown_exit_at": time.time(),
                    },
                }
            profile = result_payload["_agentic_trace_events"]
            if not isinstance(profile, dict):
                raise TypeError("Managed result payload '_agentic_trace_events' must be a dict")
            if launch_permit_wait_started_at is not None:
                mark_agentic_event(profile, "managed_launch_queue_enter_at", launch_permit_wait_started_at)
            return result_payload
        finally:
            self._clear_session_timeout_clock(session_handle)
            if not released:
                _release_permit()
            if session_id:
                self._active_session_ids.discard(session_id)
                self._waiting_launch_session_ids.discard(session_id)
                self._recent_completed_session_ids.append(session_id)
            self._inflight -= 1
            if self._inflight < 0:
                raise RuntimeError(
                    "ManagedSessionRunner inflight counter underflow: "
                    f"runner_id={self._runner_id}, session_id={session_id}, request_id={request_id}."
                )
            logger.debug(
                "ManagedSessionRunner runner_id=%s finish session_id=%s request_id=%s inflight=%s",
                self._runner_id,
                session_id,
                request_id,
                self._inflight,
            )

    async def drain_completed_sessions(self, *, session_handles: list[str] | None = None) -> list[str]:
        requested_handles = set(session_handles) if session_handles is not None else None
        ready: list[str] = []
        retained: deque[str] = deque()
        while self._completed_session_handles:
            session_handle = self._completed_session_handles.popleft()
            task = self._tasks_by_handle.get(session_handle)
            if task is None or not task.done():
                continue
            if requested_handles is not None and session_handle not in requested_handles:
                retained.append(session_handle)
                continue
            if task is not None and task.done():
                ready.append(session_handle)
        if retained:
            retained.extend(self._completed_session_handles)
            self._completed_session_handles = retained
        return ready

    async def completed_sessions(self, *, session_handles: list[str]) -> list[str]:
        return [
            session_handle
            for session_handle in session_handles
            if (task := self._tasks_by_handle.get(session_handle)) is not None and task.done()
        ]

    async def collect_session(self, *, session_handle: str) -> dict[str, Any]:
        task = self._tasks_by_handle.get(session_handle)
        if task is None:
            raise KeyError(f"Unknown managed session handle: {session_handle}")
        try:
            return await task
        finally:
            self._tasks_by_handle.pop(session_handle, None)
            self._session_id_by_handle.pop(session_handle, None)

    async def release_sessions(self, *, session_handles: list[str], signal_before_wait: int | None = None) -> int:
        if not session_handles:
            return 0
        unique_handles: list[str] = []
        seen_handles: set[str] = set()
        for session_handle in session_handles:
            if session_handle in seen_handles:
                continue
            if session_handle not in self._tasks_by_handle:
                raise KeyError(f"Unknown managed session handle: {session_handle}")
            seen_handles.add(session_handle)
            unique_handles.append(session_handle)
        tasks = [self._tasks_by_handle[session_handle] for session_handle in unique_handles]
        for session_handle in unique_handles:
            self._clear_session_timeout_clock(session_handle)
        if signal_before_wait is not None:
            for session_handle in unique_handles:
                task = self._tasks_by_handle[session_handle]
                if task.done():
                    continue
                launcher_handle = self._launcher_handle_by_session_handle.get(session_handle)
                if launcher_handle:
                    await self._launcher_client.kill(
                        handle=launcher_handle,
                        signal_value=signal_before_wait,
                        forget=True,
                    )
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for session_handle in unique_handles:
            self._tasks_by_handle.pop(session_handle, None)
            self._session_id_by_handle.pop(session_handle, None)
        if self._completed_session_handles:
            released_handles = set(unique_handles)
            self._completed_session_handles = deque(
                session_handle
                for session_handle in self._completed_session_handles
                if session_handle not in released_handles
            )
        return len(unique_handles)

    async def debug_state(self, *, sample_limit: int = 16) -> dict[str, Any]:
        active_samples = sorted(self._active_session_ids)[:sample_limit]
        waiting_launch_samples = sorted(self._waiting_launch_session_ids)[:sample_limit]
        completed_task_count = sum(1 for task in self._tasks_by_handle.values() if task.done())
        available_launch_slots = self._capacity - self._launch_permits_in_use
        return {
            "runner_id": self._runner_id,
            "launch_capacity": self._capacity,
            "launch_permits_in_use": self._launch_permits_in_use,
            "inflight": self._inflight,
            "tracked_session_tasks": len(self._tasks_by_handle),
            "completed_session_tasks": completed_task_count,
            "completed_session_queue": len(self._completed_session_handles),
            "active_launcher_handle_count": len(self._active_launcher_handles),
            "available_launch_slots": available_launch_slots,
            "active_session_count": len(self._active_session_ids),
            "waiting_launch_count": len(self._waiting_launch_session_ids),
            "timeout_active_session_count": len(self._timeout_active_handles),
            "timeout_tracked_session_count": len(self._timeout_remaining_s_by_handle),
            "timeout_triggered_session_count": len(self._timed_out_session_handles),
            "active_session_samples": active_samples,
            "waiting_launch_samples": waiting_launch_samples,
            "recent_completed_session_samples": list(self._recent_completed_session_ids)[-sample_limit:],
        }

    async def warm_up(self) -> bool:
        await self._launcher_client.ping()
        return True

    async def shutdown(self) -> bool:
        self._shutdown_requested = True
        tasks = list(self._tasks_by_handle.values())
        active_launcher_handles = list(self._active_launcher_handles)
        for handle in active_launcher_handles:
            await self._launcher_client.kill(handle=handle, signal_value=signal.SIGTERM, forget=True)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_launcher_handles.clear()
        self._active_session_ids.clear()
        self._waiting_launch_session_ids.clear()
        self._recent_completed_session_ids.clear()
        self._tasks_by_handle.clear()
        self._session_id_by_handle.clear()
        self._completed_session_handles.clear()
        for task in self._timeout_tasks_by_handle.values():
            if not task.done():
                task.cancel()
        self._launcher_handle_by_session_handle.clear()
        self._timeout_remaining_s_by_handle.clear()
        self._timeout_active_started_at_by_handle.clear()
        self._timeout_tasks_by_handle.clear()
        self._timeout_active_handles.clear()
        self._timed_out_session_handles.clear()
        return True


class ManagedSessionRunnerPool:
    def __init__(self, handles: list[Any], *, capacities: list[int]) -> None:
        if not handles:
            raise ValueError("Managed session session runner pool requires at least one session runner.")
        self._handles = handles
        self._total_capacity = sum(capacities)
        self._active_session_count = 0
        self._reserved_launch_slots = 0
        self._next_index = 0

    def drain_completed_session_handles(
        self, *, session_handles: list[ManagedSessionHandle] | None = None
    ) -> list[ManagedSessionHandle]:
        handles_by_runner_id: dict[int, list[str]] | None = None
        if session_handles is not None:
            if not session_handles:
                return []
            handles_by_runner_id = {}
            for handle in session_handles:
                if handle.runner_id < 0 or handle.runner_id >= len(self._handles):
                    raise RuntimeError(
                        "Managed session handle references an unknown runner: "
                        f"runner_id={handle.runner_id}, runner_count={len(self._handles)}."
                    )
                handles_by_runner_id.setdefault(handle.runner_id, []).append(handle.session_handle)
        refs = []
        ref_to_runner_id: dict[Any, int] = {}
        if handles_by_runner_id is None:
            runner_items = list(enumerate(self._handles))
        else:
            runner_items = [(runner_id, self._handles[runner_id]) for runner_id in sorted(handles_by_runner_id)]
        for runner_id, handle in runner_items:
            requested_handles = None if handles_by_runner_id is None else handles_by_runner_id[runner_id]
            ref = handle.drain_completed_sessions.remote(session_handles=requested_handles)
            refs.append(ref)
            ref_to_runner_id[ref] = runner_id
        completed_handles: list[ManagedSessionHandle] = []
        for ref, session_handles in zip(refs, ray.get(refs), strict=True):
            runner_id = ref_to_runner_id[ref]
            for session_handle in session_handles:
                completed_handles.append(ManagedSessionHandle(runner_id=runner_id, session_handle=str(session_handle)))
        return completed_handles

    def completed_session_handles(self, *, session_handles: list[ManagedSessionHandle]) -> list[ManagedSessionHandle]:
        if not session_handles:
            return []
        handles_by_runner_id: dict[int, list[str]] = {}
        for handle in session_handles:
            if handle.runner_id < 0 or handle.runner_id >= len(self._handles):
                raise RuntimeError(
                    "Managed session handle references an unknown runner: "
                    f"runner_id={handle.runner_id}, runner_count={len(self._handles)}."
                )
            handles_by_runner_id.setdefault(handle.runner_id, []).append(handle.session_handle)
        refs = []
        ref_to_runner_id: dict[Any, int] = {}
        for runner_id in sorted(handles_by_runner_id):
            ref = self._handles[runner_id].completed_sessions.remote(session_handles=handles_by_runner_id[runner_id])
            refs.append(ref)
            ref_to_runner_id[ref] = runner_id
        completed_handles: list[ManagedSessionHandle] = []
        for ref, completed_session_handles in zip(refs, ray.get(refs), strict=True):
            runner_id = ref_to_runner_id[ref]
            for session_handle in completed_session_handles:
                completed_handles.append(ManagedSessionHandle(runner_id=runner_id, session_handle=str(session_handle)))
        return completed_handles

    def available_launch_slots(self) -> int:
        available_launch_slots = self._total_capacity - self._active_session_count - self._reserved_launch_slots
        if available_launch_slots < 0:
            raise RuntimeError(
                "ManagedSessionRunnerPool active task count and reserved launches exceed total launch capacity: "
                f"capacity={self._total_capacity}, active={self._active_session_count}, "
                f"reserved={self._reserved_launch_slots}."
            )
        return available_launch_slots

    def reserve_launch_slots(self, session_count: int) -> None:
        if session_count < 0:
            raise RuntimeError(f"ManagedSessionRunnerPool cannot reserve negative launch slots: {session_count}.")
        if session_count == 0:
            return
        available_launch_slots = self.available_launch_slots()
        if session_count > available_launch_slots:
            raise RuntimeError(
                "ManagedSessionRunnerPool launch reservation exceeds available capacity: "
                f"requested={session_count}, available={available_launch_slots}, "
                f"capacity={self._total_capacity}, active={self._active_session_count}, "
                f"reserved={self._reserved_launch_slots}."
            )
        self._reserved_launch_slots += session_count

    def release_launch_slots(self, session_count: int) -> None:
        if session_count < 0:
            raise RuntimeError(f"ManagedSessionRunnerPool cannot release negative launch slots: {session_count}.")
        if session_count == 0:
            return
        next_reserved_launch_slots = self._reserved_launch_slots - session_count
        if next_reserved_launch_slots < 0:
            raise RuntimeError(
                "ManagedSessionRunnerPool launch reservation underflow: "
                f"reserved={self._reserved_launch_slots}, release={session_count}."
            )
        self._reserved_launch_slots = next_reserved_launch_slots

    def submit_sessions(self, *, spec: ManagedCommandAppSpec, session_inputs: list[SessionInput]) -> list[Any]:
        if not session_inputs:
            return []
        spec_payload = _managed_spec_to_payload(spec)
        grouped_inputs: dict[int, list[tuple[int, SessionInput]]] = {}
        for item_index, session_input in enumerate(session_inputs):
            runner_id = self._next_index % len(self._handles)
            self._next_index += 1
            grouped_inputs.setdefault(runner_id, []).append((item_index, session_input))

        refs = []
        ref_to_runner_id: dict[Any, int] = {}
        for runner_id, items in grouped_inputs.items():
            payloads = [_session_input_to_payload(session_input) for _, session_input in items]
            ref = self._handles[runner_id].submit_sessions.remote(spec_payload, payloads)
            refs.append(ref)
            ref_to_runner_id[ref] = runner_id

        submitted_handles: list[ManagedSessionHandle | None] = [None] * len(session_inputs)
        for ref, session_handles in zip(refs, ray.get(refs), strict=True):
            runner_id = ref_to_runner_id[ref]
            indexed_inputs = grouped_inputs[runner_id]
            if len(session_handles) != len(indexed_inputs):
                raise RuntimeError(
                    "ManagedSessionRunner returned a mismatched number of session handles: "
                    f"runner_id={runner_id}, expected={len(indexed_inputs)}, got={len(session_handles)}."
                )
            for (item_index, _session_input), session_handle in zip(indexed_inputs, session_handles, strict=True):
                managed_handle = ManagedSessionHandle(runner_id=runner_id, session_handle=str(session_handle))
                submitted_handles[item_index] = managed_handle

        missing_handle_count = sum(handle is None for handle in submitted_handles)
        if missing_handle_count:
            raise RuntimeError(f"Managed session batch submission missed {missing_handle_count} handles.")
        self._active_session_count += len(submitted_handles)
        return submitted_handles

    def collect_session_result_ref(self, handle: ManagedSessionHandle) -> Any:
        return self._handles[handle.runner_id].collect_session.remote(session_handle=handle.session_handle)

    def set_session_timeouts_active(self, handles: list[ManagedSessionHandle], *, active: bool) -> list[Any]:
        unique_handles: list[ManagedSessionHandle] = []
        seen_handles: set[tuple[int, str]] = set()
        for handle in handles:
            if handle.runner_id < 0 or handle.runner_id >= len(self._handles):
                raise RuntimeError(
                    "Managed session handle references an unknown runner: "
                    f"runner_id={handle.runner_id}, runner_count={len(self._handles)}."
                )
            handle_key = (handle.runner_id, handle.session_handle)
            if handle_key in seen_handles:
                continue
            seen_handles.add(handle_key)
            unique_handles.append(handle)
        grouped_handles: dict[int, list[str]] = {}
        for handle in unique_handles:
            grouped_handles.setdefault(handle.runner_id, []).append(handle.session_handle)
        return [
            self._handles[runner_id].set_session_timeouts_active.remote(
                session_handles=session_handles,
                active=active,
            )
            for runner_id, session_handles in grouped_handles.items()
        ]

    def release_session_handles(
        self,
        handles: list[ManagedSessionHandle],
        *,
        signal_before_wait: int | None = None,
    ) -> list[Any]:
        unique_handles: list[ManagedSessionHandle] = []
        seen_handles: set[tuple[int, str]] = set()
        for handle in handles:
            if handle.runner_id < 0 or handle.runner_id >= len(self._handles):
                raise RuntimeError(
                    "Managed session handle references an unknown runner: "
                    f"runner_id={handle.runner_id}, runner_count={len(self._handles)}."
                )
            handle_key = (handle.runner_id, handle.session_handle)
            if handle_key in seen_handles:
                continue
            seen_handles.add(handle_key)
            unique_handles.append(handle)
        grouped_handles: dict[int, list[str]] = {}
        for handle in unique_handles:
            grouped_handles.setdefault(handle.runner_id, []).append(handle.session_handle)
        refs = [
            self._handles[runner_id].release_sessions.remote(
                session_handles=session_handles,
                signal_before_wait=signal_before_wait,
            )
            for runner_id, session_handles in grouped_handles.items()
        ]
        return refs

    def mark_session_handles_released(self, released_count: int) -> None:
        self._active_session_count -= released_count
        if self._active_session_count < 0:
            raise RuntimeError(
                "ManagedSessionRunnerPool released more sessions than it submitted: "
                f"released={released_count}, active={self._active_session_count}."
            )

    def warm_up(self) -> list[Any]:
        refs = []
        for handle in self._handles:
            try:
                refs.append(handle.warm_up.remote())
            except Exception:
                continue
        return refs

    def shutdown(self) -> None:
        shutdown_refs = []
        for handle in self._handles:
            try:
                shutdown_refs.append(handle.shutdown.remote())
            except Exception:
                continue
        if shutdown_refs:
            ray.get(shutdown_refs)
        self._handles.clear()
        self._active_session_count = 0
        self._reserved_launch_slots = 0
        self._total_capacity = 0
        self._next_index = 0

    def debug_state(self, *, sample_limit: int = 8) -> list[dict[str, Any]]:
        debug_refs = []
        for handle in self._handles:
            try:
                debug_refs.append(handle.debug_state.remote(sample_limit=sample_limit))
            except Exception:
                continue
        if not debug_refs:
            return []
        try:
            snapshots = ray.get(debug_refs, timeout=10)
        except Exception:
            return []
        return [dict(item) for item in snapshots if isinstance(item, dict)]


def create_managed_session_runner_pool(args: Any, *, total_requests: int) -> ManagedSessionRunnerPool | None:
    resources = get_agentic_runtime_resources(args)

    def _factory(capacities: list[int]) -> ManagedSessionRunnerPool:
        handles = []
        for runner_id, launch_capacity in enumerate(capacities):
            handle = ManagedSessionRunner.options(
                num_cpus=0,
                # Ray default scheduling avoids forcing every node to participate in local agent launch.
                # scheduling_strategy="SPREAD",
            ).remote(runner_id=runner_id, launch_capacity=launch_capacity)
            handles.append(handle)
        return ManagedSessionRunnerPool(handles, capacities=capacities)

    return resources.build_session_runner_pool(total_requests=total_requests, factory=_factory)


@dataclass
class SampleSeedInfo:
    group_index: int | None = None
    index: int | None = None
    label: str | None = None
    train_metadata: dict[str, Any] | None = None


@dataclass
class RequestEnvelope:
    metadata: dict[str, Any] = field(default_factory=dict)
    input_payload: dict[str, Any] = field(default_factory=dict)
    sampling_params: dict[str, Any] | None = None
    session_id: str | None = None
    request_id: str | None = None
    rollout_id: int | None = None
    seed: SampleSeedInfo = field(default_factory=SampleSeedInfo)


def resolve_chat_api_base_url() -> str:
    from relax.utils.utils import get_serve_url

    return f"{get_serve_url(route_prefix=AGENTIC_CHAT_API_ROUTE_PREFIX)}/"


def _sample_metadata(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return metadata


def _normalize_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return check_messages([{"role": "user", "content": prompt}])
    if isinstance(prompt, dict):
        prompt = [prompt]
    return check_messages(prompt or [])


def _transport_image_payload(payload: Any) -> str:
    if isinstance(payload, str) and payload.startswith(("data:image/", "http://", "https://")):
        return payload

    from PIL import Image

    from relax.utils.multimodal.image_utils import load_image

    if isinstance(payload, dict):
        if isinstance(payload.get("bytes"), bytes):
            payload = payload["bytes"]
        elif isinstance(payload.get("path"), str) and payload["path"]:
            payload = payload["path"]
    image = payload if isinstance(payload, Image.Image) else load_image(payload)
    buffer = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _transport_dataset_message_media(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_messages = check_messages(messages)
    rendered_messages: list[dict[str, Any]] = []
    for message in normalized_messages:
        content = message["content"]
        if isinstance(content, str):
            rendered_messages.append(message)
            continue
        rendered_content: list[dict[str, Any]] = []
        for item in content:
            item_type = item.get("type")
            if item_type == "image":
                rendered_content.append(
                    {"type": "image_url", "image_url": {"url": _transport_image_payload(item.get("image"))}}
                )
                continue
            rendered_content.append(copy.deepcopy(item))
        rendered_messages.append({"role": message["role"], "content": rendered_content})
    return rendered_messages


def _sample_messages(sample: Sample) -> list[dict[str, Any]]:
    return _transport_dataset_message_media(_normalize_prompt(sample.prompt))


def _request_envelope_from_sample(
    sample: Sample,
    *,
    session_id: str | None = None,
    request_id: str | None = None,
    rollout_id: int | None = None,
    sampling_params: dict[str, Any] | None = None,
    shared_messages: list[dict[str, Any]] | None = None,
    include_input_payload: bool = True,
) -> RequestEnvelope:
    metadata = _sample_metadata(sample)
    if rollout_id is not None and sample.group_index is None:
        raise ValueError("sample.group_index is required for rollout-managed request envelopes")
    effective_session_id = session_id if isinstance(session_id, str) and session_id else sample.session_id
    messages: list[dict[str, Any]] = []
    input_payload: dict[str, Any] = {}
    if include_input_payload:
        messages = copy.copy(shared_messages) if shared_messages is not None else _sample_messages(sample)
        input_payload["messages"] = messages
        if metadata:
            input_payload["metadata"] = metadata
    effective_sampling_params = sampling_params
    if effective_sampling_params is None:
        sample_sampling_params = getattr(sample, "sampling_params", None)
        if isinstance(sample_sampling_params, dict):
            effective_sampling_params = sample_sampling_params
    return RequestEnvelope(
        metadata=metadata,
        input_payload=input_payload,
        sampling_params=copy.deepcopy(effective_sampling_params) if effective_sampling_params is not None else None,
        session_id=effective_session_id,
        request_id=request_id,
        rollout_id=rollout_id,
        seed=SampleSeedInfo(
            group_index=sample.group_index,
            index=sample.index,
            label=sample.label,
            train_metadata=sample.train_metadata,
        ),
    )


def _request_groups_from_sample_groups(
    groups: list[list[Sample]], *, rollout_id: int | None = None, include_input_payload: bool = True
) -> list[list[RequestEnvelope]]:
    request_groups: list[list[RequestEnvelope]] = []
    for group in groups:
        if not group:
            request_groups.append([])
            continue
        if include_input_payload:
            shared_messages = _sample_messages(group[0])
        else:
            shared_messages = []
        request_groups.append(
            [
                _request_envelope_from_sample(
                    sample,
                    rollout_id=rollout_id,
                    shared_messages=shared_messages,
                    include_input_payload=include_input_payload,
                )
                for sample in group
            ]
        )
    return request_groups


@dataclass(frozen=True)
class _ManagedRequestResult:
    # The managed agent task is complete, but the ChatAPI session is still alive.
    # Finalize/discard belongs to materialization, so this object must never destroy session state by itself.
    result_payload: dict[str, Any]
    session_runner_started_at: float


@dataclass(frozen=True)
class ExecutionDispatch:
    materialized_batches: list[list[dict[str, Any]]] = field(default_factory=list)
    ready_groups: list[list[Sample]] = field(default_factory=list)


@dataclass(frozen=True)
class _MaterializationFailure:
    request_id: str
    session_id: str
    group_key: GroupKey
    error: Exception


@dataclass(frozen=True)
class _PrepareRequestPlan:
    slot_idx: int
    envelope: RequestEnvelope
    session_id: str
    request_sampling_params: dict[str, Any]


class _ServeHandleChatControlClient:
    def __init__(self, *, handle: Any):
        self._handle = handle

    async def aclose(self) -> None:
        return None

    async def finalize_and_discard(
        self,
        *,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        reward: float | dict[str, Any] | None = None,
    ) -> FinalizedResultTransport:
        return await self._handle.finalize_and_discard.remote(
            session_id=session_id,
            metadata=metadata,
            reward=reward,
        )

    async def discard_session(self, *, session_id: str) -> bool:
        return await self._handle.discard_session.remote(session_id=session_id)

    async def register_sessions_batch(self, *, entries: list[dict[str, Any]]) -> int:
        return int(await self._handle.register_sessions_batch.remote(entries=entries))

    async def release_partial_resume_gate(self, *, rollout_id: int) -> int:
        return int(await self._handle.release_partial_resume_gate.remote(rollout_id=rollout_id))

    async def gate_rollout_irs_for_partial_resume(self, *, rollout_id: int) -> int:
        return int(await self._handle.gate_rollout_irs_for_partial_resume.remote(rollout_id=rollout_id))

    async def gate_rollout_irs_for_discard(self, *, rollout_id: int) -> int:
        return int(await self._handle.gate_rollout_irs_for_discard.remote(rollout_id=rollout_id))

    async def gate_all_irs_for_shutdown(self) -> int:
        return int(await self._handle.gate_all_irs_for_shutdown.remote())

    async def active_rollout_request_counts(self, *, rollout_id: int) -> dict[str, int]:
        return dict(await self._handle.active_rollout_request_counts.remote(rollout_id=rollout_id))

    async def abort_rollout_requests(self, *, rollout_id: int) -> dict[str, int]:
        return dict(await self._handle.abort_rollout_requests.remote(rollout_id=rollout_id))

    async def aborted_resume_session_ids(self, *, rollout_id: int) -> list[str]:
        return list(await self._handle.aborted_resume_session_ids.remote(rollout_id=rollout_id))

    async def enter_eval(self) -> int:
        return int(await self._handle.enter_eval.remote())

    async def exit_eval(self) -> int:
        return int(await self._handle.exit_eval.remote())

    async def trim_memory(self) -> dict[str, Any]:
        return dict(await self._handle.trim_memory.remote())

    async def debug_state(self, *, sample_limit: int = 0) -> dict[str, Any]:
        return dict(await self._handle.debug_state.remote(sample_limit=int(sample_limit)))

    async def prepare_group_status(self, *, scope_id: str) -> list[dict[str, Any]]:
        return list(await self._handle.prepare_group_status.remote(scope_id=scope_id))

    async def activate_group_sessions(
        self,
        *,
        scope_id: str,
        groups: list[dict[str, Any]],
        rollout_id: int,
    ) -> dict[str, int]:
        return dict(
            await self._handle.activate_group_sessions.remote(
                scope_id=scope_id,
                groups=groups,
                rollout_id=rollout_id,
            )
        )


def create_agentic_service_client() -> _ServeHandleChatControlClient:
    global _AGENTIC_SERVICE_CLIENT
    if _AGENTIC_SERVICE_CLIENT is not None:
        return _AGENTIC_SERVICE_CLIENT
    from ray import serve

    if not ray.is_initialized():
        raise RuntimeError("Agentic chat control requires a live Ray Serve handle for 'agentic_chat_api'.")
    try:
        handle = serve.get_app_handle(AGENTIC_CHAT_API_SERVICE_NAME)
    except Exception as exc:
        raise RuntimeError("Agentic chat control requires a live Ray Serve handle for 'agentic_chat_api'.") from exc
    _AGENTIC_SERVICE_CLIENT = _ServeHandleChatControlClient(handle=handle)
    return _AGENTIC_SERVICE_CLIENT


async def _resolve_object_async(payload: Any, *, executor: Any = None) -> Any:
    if not ray.is_initialized():
        return payload
    object_ref_type = getattr(ray, "ObjectRef", None)
    if object_ref_type is None or not isinstance(payload, object_ref_type):
        return payload
    loop = asyncio.get_running_loop()
    if executor is not None:
        return await loop.run_in_executor(executor, ray.get, payload)
    if hasattr(payload, "__await__"):
        return await payload
    return await loop.run_in_executor(executor, ray.get, payload)


def _overlay_sample_metadata(
    *,
    base_metadata: dict[str, Any] | None,
    overlay_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    base = base_metadata if isinstance(base_metadata, dict) else {}
    if not isinstance(overlay_metadata, dict) or not overlay_metadata:
        return base

    overlay_trace = overlay_metadata.get(TRACE_KEY)
    if overlay_trace is not None:
        base[TRACE_KEY] = merge_agentic_trace(base.get(TRACE_KEY), overlay_trace)

    for key, value in overlay_metadata.items():
        if key == TRACE_KEY:
            continue
        base[key] = copy.deepcopy(value)
    return base


class RuntimeDomain:
    def __init__(
        self,
        *,
        args,
        scope_id: str,
    ) -> None:
        if not isinstance(scope_id, str) or not scope_id:
            raise RuntimeError("RuntimeDomain requires a non-empty scope_id.")
        self.args = args
        self.scope_id = scope_id
        self.rollout_id: int | None = None
        self.managed_chat_api_base_url: str | None = None
        self.runtime_slots_by_request_id: dict[str, RuntimeSlot] = {}
        self.runtime_groups_by_key: dict[GroupKey, RuntimeGroup] = {}
        self._discarded_group_keys: set[GroupKey] = set()
        self._aborted_resume_session_ids_by_rollout: dict[int, set[str]] = {}
        self._interrupted_close_accounting_last_refresh_at = 0.0
        self._interrupted_close_accounting_refresh_interval_s = 0.5
        self.session_debug_state: dict[str, Any] | None = None
        self._ready_materialized_batches: list[list[dict[str, Any]]] = []
        self._ready_materialized_groups: list[list[Sample]] = []
        self._output_ready_event = asyncio.Event()
        self._ready_materialized_session_count = 0
        self._emitted_materialized_session_count_total = 0
        self._drained_materialized_session_count_total = 0
        self._managed_app_spec = load_agent_app_spec_from_args(args)
        self._runtime_session_counter = 0
        self._runtime_request_counter = 0
        self._notified_session_rollouts: dict[str, int] = {}
        self._rollout_mode = "train"
        self._runtime_resources = get_agentic_runtime_resources(args)
        self._session_runner_pool_lock = threading.Lock()
        self._session_runner_pool_warm_task: asyncio.Task[Any] | None = None
        self._session_runner_pool_total_requests = self._runtime_resources.step_session_capacity()

        self._session_runner_pool: ManagedSessionRunnerPool | None = None
        self._service_client = None

        if ray.is_initialized():
            try:
                create_agentic_service_client()
            except Exception:
                logger.debug("Agentic service handles are not ready for warm-init yet.", exc_info=True)
            self._service_client = _AGENTIC_SERVICE_CLIENT
            self._prewarm_session_runner_pool(total_requests=self._session_runner_pool_total_requests)

    def rebind_step(
        self,
        *,
        args,
        rollout_id: int,
    ) -> None:
        self.args = args
        self.rollout_id = rollout_id
        self._rollout_mode = "train"
        self._managed_app_spec = load_agent_app_spec_from_args(args)
        self._runtime_resources = get_agentic_runtime_resources(args)
        self._session_runner_pool_total_requests = max(
            self._session_runner_pool_total_requests,
            self._runtime_resources.step_session_capacity(),
        )
        self._interrupted_close_accounting_last_refresh_at = 0.0

    def require_rollout_id(self) -> int:
        if self.rollout_id is None:
            raise RuntimeError("RuntimeDomain step is not bound.")
        return self.rollout_id

    async def drop_resident_results(self) -> int:
        dropped_group_keys: set[GroupKey] = set(self._discarded_group_keys)
        dropped_group_keys.update(self.runtime_groups_by_key)
        for records in self._ready_materialized_batches:
            for record in records:
                dropped_group_keys.add(record["group_key"])
        for group in self._ready_materialized_groups:
            dropped_group_keys.add(sample_group_key(group))

        for group_key in list(dropped_group_keys):
            await self._drop_runtime_group_resources(group_key=group_key)
        self.runtime_slots_by_request_id.clear()
        self.runtime_groups_by_key.clear()
        self._discarded_group_keys.clear()
        self._aborted_resume_session_ids_by_rollout.clear()
        self._interrupted_close_accounting_last_refresh_at = 0.0
        self._ready_materialized_batches.clear()
        self._ready_materialized_groups.clear()
        self._ready_materialized_session_count = 0
        self._output_ready_event.clear()
        return len(dropped_group_keys)

    def _register_runtime_slot(
        self,
        *,
        request_id: str,
        session_id: str,
        dispatch_context: RequestDispatchContext,
        seed_sample: Sample,
        managed_session_handle: Any,
        managed_session_submitted_at: float,
    ) -> None:
        if request_id in self.runtime_slots_by_request_id:
            raise RuntimeError(f"RuntimeDomain received duplicate resident slot request_id={request_id!r}.")
        slot = RuntimeSlot(
            request_id=request_id,
            session_id=session_id,
            dispatch_context=dispatch_context,
            seed_sample=seed_sample,
            managed_session_handle=managed_session_handle,
            managed_session_submitted_at=float(managed_session_submitted_at),
        )
        group = self.runtime_groups_by_key.get(dispatch_context.group_key)
        if group is None:
            group = RuntimeGroup(
                group_key=dispatch_context.group_key,
                expected_count=dispatch_context.expected_count,
                admission_rollout_id=dispatch_context.rollout_id,
            )
            self.runtime_groups_by_key[dispatch_context.group_key] = group
        if group.expected_count != dispatch_context.expected_count:
            raise RuntimeError(
                "RuntimeDomain resident group expected_count changed: "
                f"group_key={dispatch_context.group_key!r}, existing={group.expected_count}, "
                f"new={dispatch_context.expected_count}."
            )
        if group.admission_rollout_id != dispatch_context.rollout_id:
            raise RuntimeError(
                "RuntimeDomain resident group admission rollout changed: "
                f"group_key={dispatch_context.group_key!r}, existing={group.admission_rollout_id}, "
                f"new={dispatch_context.rollout_id}."
            )
        group.request_ids.add(request_id)
        self.runtime_slots_by_request_id[request_id] = slot

    def _drop_runtime_slot(self, request_id: str) -> RuntimeSlot | None:
        slot = self.runtime_slots_by_request_id.pop(request_id, None)
        if slot is None:
            return None
        if slot.materialization_task is not None and not slot.materialization_task.done():
            slot.materialization_task.cancel()
        group = self.runtime_groups_by_key.get(slot.group_key)
        if group is not None:
            group.request_ids.discard(request_id)
            if not group.request_ids:
                self.runtime_groups_by_key.pop(slot.group_key, None)
        return slot

    def _drop_runtime_group_slots(self, *, group_key: GroupKey) -> list[RuntimeSlot]:
        group = self.runtime_groups_by_key.pop(group_key, None)
        if group is None:
            return []
        dropped_slots: list[RuntimeSlot] = []
        for request_id in list(group.request_ids):
            slot = self.runtime_slots_by_request_id.pop(request_id, None)
            if slot is None:
                continue
            dropped_slots.append(slot)
        return dropped_slots

    async def _discard_runtime_slots(self, slots: list[RuntimeSlot]) -> None:
        if not slots:
            return
        release_slots = [
            slot
            for slot in slots
            if slot.state != RuntimeSlotState.MATERIALIZING
            if not slot.managed_session_handle_released
            if isinstance(slot.managed_session_handle, ManagedSessionHandle)
        ]
        release_handles = [slot.managed_session_handle for slot in release_slots]
        await self._set_managed_session_timeouts_active(release_handles, active=False)
        finalize_client = self._ensure_service_client()
        discard_tasks = [finalize_client.discard_session(session_id=slot.session_id) for slot in slots]
        results = await asyncio.gather(*discard_tasks, return_exceptions=True)
        failed_results: list[tuple[str, BaseException]] = []
        for slot, result in zip(slots, results, strict=True):
            if isinstance(result, BaseException):
                failed_results.append((slot.session_id, result))
                continue
            if not isinstance(result, bool):
                failed_results.append((slot.session_id, TypeError(f"Unexpected discard result: {result!r}")))
                continue
            self._clear_local_session_state(session_id=slot.session_id)
        await self._release_managed_session_handles(release_handles, signal_before_wait=signal.SIGTERM)
        for slot in release_slots:
            slot.managed_session_handle_released = True
        if failed_results:
            failed_session_ids = ", ".join(session_id for session_id, _ in failed_results)
            raise RuntimeError(f"Failed to discard runtime slots: failed={failed_session_ids}") from (
                failed_results[0][1]
            )

    async def _resolve_ready_runtime_slot_ref(self, slot: RuntimeSlot) -> _ManagedRequestResult:
        result_ready_at = time.time()
        runner_pool = self._session_runner_pool
        if runner_pool is None:
            raise RuntimeError("RuntimeDomain cannot collect a managed session without a runner pool.")
        try:
            result_payload = await _resolve_object_async(
                runner_pool.collect_session_result_ref(slot.managed_session_handle),
                executor=self._runtime_resources.compiler.cpu_executor,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if (
                isinstance(slot.managed_session_handle, ManagedSessionHandle)
                and not slot.managed_session_handle_released
            ):
                runner_pool.mark_session_handles_released(1)
                slot.managed_session_handle_released = True
            raise
        if isinstance(slot.managed_session_handle, ManagedSessionHandle) and not slot.managed_session_handle_released:
            runner_pool.mark_session_handles_released(1)
            slot.managed_session_handle_released = True
        if not isinstance(result_payload, dict):
            raise TypeError(f"Managed result payload must be a dict, got {type(result_payload)}")
        profile = result_payload["_agentic_trace_events"]
        if not isinstance(profile, dict):
            raise TypeError("Managed result payload '_agentic_trace_events' must be a dict")
        mark_agentic_event(profile, "managed_result_ready_at", result_ready_at)
        mark_agentic_event(profile, "managed_result_resolved_at")
        return _ManagedRequestResult(
            result_payload=result_payload,
            session_runner_started_at=slot.managed_session_submitted_at,
        )

    async def _materialize_runtime_slot_task(
        self,
        *,
        request_id: str,
        slot: RuntimeSlot,
    ) -> tuple[str, dict[str, Any] | _MaterializationDrop | _MaterializationFailure]:
        try:
            managed_result = await self._resolve_ready_runtime_slot_ref(slot)
            materialized_record = await self._materialize_managed_result_record(
                dispatch_context=slot.dispatch_context,
                session_id=slot.session_id,
                managed_result=managed_result,
            )
        except Exception as exc:
            materialized_record = _MaterializationFailure(
                request_id=request_id,
                session_id=slot.session_id,
                group_key=slot.group_key,
                error=exc,
            )
        return request_id, materialized_record

    def _start_runtime_slot_materialization(self, *, request_id: str, slot: RuntimeSlot) -> None:
        if slot.state != RuntimeSlotState.RUNNING_APP:
            return
        slot.state = RuntimeSlotState.MATERIALIZING
        slot.materialization_task = asyncio.create_task(
            self._materialize_runtime_slot_task(request_id=request_id, slot=slot)
        )
        slot.materialization_task.add_done_callback(lambda _task: self._output_ready_event.set())

    async def _drop_runtime_group_resources(self, *, group_key: GroupKey) -> None:
        self._discarded_group_keys.add(group_key)
        dropped_slots = self._drop_runtime_group_slots(group_key=group_key)
        self._drop_materialized_group_state(group_key=group_key)
        await self._discard_runtime_slots(dropped_slots)

    async def _drain_finished_runtime_slot_materializations(self) -> str | None:
        done_tasks = [
            slot.materialization_task
            for slot in self.runtime_slots_by_request_id.values()
            if slot.state == RuntimeSlotState.MATERIALIZING
            and slot.materialization_task is not None
            and slot.materialization_task.done()
        ]
        if not done_tasks:
            return None
        materialized_records: list[dict[str, Any]] = []
        materialized_request_ids: list[str] = []
        discarded_group_keys: set[GroupKey] = set()
        unexpected_errors: list[tuple[str, str, BaseException]] = []
        for task in done_tasks:
            request_id, materialized_record = task.result()
            if isinstance(materialized_record, _MaterializationFailure):
                session_local_failure = isinstance(materialized_record.error, AgentExecutionError)
                session_local_failure_type = type(materialized_record.error).__name__
                if isinstance(materialized_record.error, ray.exceptions.RayTaskError):
                    try:
                        cause = materialized_record.error.as_instanceof_cause()
                    except Exception:
                        cause = None
                    session_local_failure = isinstance(cause, AgentExecutionError)
                    if cause is not None:
                        session_local_failure_type = type(cause).__name__
                if session_local_failure:
                    logger.info(
                        "Dropping failed resident agentic session session_id=%s request_id=%s error_type=%s",
                        materialized_record.session_id,
                        materialized_record.request_id,
                        session_local_failure_type,
                    )
                    discarded_group_keys.add(materialized_record.group_key)
                    await self._drop_runtime_group_resources(group_key=materialized_record.group_key)
                    continue
                if materialized_record.group_key not in self._discarded_group_keys:
                    unexpected_errors.append(
                        (materialized_record.request_id, materialized_record.session_id, materialized_record.error)
                    )
                    continue
                discarded_group_keys.add(materialized_record.group_key)
                continue
            slot = self.runtime_slots_by_request_id.get(request_id)
            if slot is None:
                continue
            if isinstance(materialized_record, _MaterializationDrop):
                logger.info(
                    "Dropping finalized resident agentic session session_id=%s request_id=%s reason=%s info=%s",
                    slot.session_id,
                    request_id,
                    materialized_record.reason,
                    materialized_record.info,
                )
                discarded_group_keys.add(slot.group_key)
                await self._drop_runtime_group_resources(group_key=slot.group_key)
                continue
            materialized_records.append(materialized_record)
            materialized_request_ids.append(request_id)
        if materialized_records and discarded_group_keys:
            retained_records: list[dict[str, Any]] = []
            retained_request_ids: list[str] = []
            for request_id, record in zip(materialized_request_ids, materialized_records, strict=True):
                group_key = record["group_key"]
                if group_key in discarded_group_keys or group_key in self._discarded_group_keys:
                    continue
                retained_records.append(record)
                retained_request_ids.append(request_id)
            materialized_records = retained_records
            materialized_request_ids = retained_request_ids
        if materialized_records:
            self._emit_session_materialization_batch(materialized_records)
        for request_id in materialized_request_ids:
            slot = self.runtime_slots_by_request_id.get(request_id)
            if slot is not None:
                slot.state = RuntimeSlotState.MATERIALIZED
            self._drop_runtime_slot(request_id)
        if discarded_group_keys:
            self._output_ready_event.set()
        if unexpected_errors:
            request_id, session_id, exc = unexpected_errors[0]
            raise RuntimeError(
                f"Resident runtime slot materialization failed request_id={request_id} session_id={session_id}."
            ) from exc
        return "slot_completed" if materialized_records or discarded_group_keys else None

    def _clear_local_session_state(self, *, session_id: str) -> None:
        self._notified_session_rollouts.pop(session_id, None)
        for request_id, slot in list(self.runtime_slots_by_request_id.items()):
            if slot.session_id == session_id:
                self._drop_runtime_slot(request_id)

    def drain_discarded_group_keys(self) -> set[GroupKey]:
        discarded_group_keys = set(self._discarded_group_keys)
        self._discarded_group_keys.clear()
        return discarded_group_keys

    def _drop_materialized_group_state(self, *, group_key: GroupKey) -> None:
        self._drop_runtime_group_slots(group_key=group_key)
        removed_ready_records = 0
        retained_ready_batches: list[list[dict[str, Any]]] = []
        for records in self._ready_materialized_batches:
            retained_records = [record for record in records if record["group_key"] != group_key]
            removed_ready_records += len(records) - len(retained_records)
            if retained_records:
                retained_ready_batches.append(retained_records)
        self._ready_materialized_batches = retained_ready_batches
        if removed_ready_records:
            self._ready_materialized_session_count -= removed_ready_records
            if self._ready_materialized_session_count < 0:
                raise RuntimeError("RuntimeDomain ready materialized session count underflow after group discard.")
        self._ready_materialized_groups = [
            group for group in self._ready_materialized_groups if sample_group_key(group) != group_key
        ]

    def debug_snapshot(self) -> dict[str, Any]:
        def _summarize_runtime_groups(groups: dict[GroupKey, RuntimeGroup]) -> list[dict[str, Any]]:
            details: list[dict[str, Any]] = []
            for key, group in list(groups.items())[:8]:
                details.append(
                    {
                        "group_key": key,
                        "expected_count": group.expected_count,
                        "slot_count": group.slot_count(),
                        "running_slots": len(group.request_ids),
                        "admission_rollout_id": group.admission_rollout_id,
                    }
                )
            return details

        runtime_slot_states = {
            state.value: sum(1 for slot in self.runtime_slots_by_request_id.values() if slot.state == state)
            for state in RuntimeSlotState
        }
        return {
            "rollout_id": self.rollout_id,
            "ready_materialized_batches": len(self._ready_materialized_batches),
            "ready_materialized_records": sum(len(batch) for batch in self._ready_materialized_batches),
            "ready_materialized_groups": len(self._ready_materialized_groups),
            "runtime_slots": len(self.runtime_slots_by_request_id),
            "runtime_slot_states": runtime_slot_states,
            "runtime_groups": len(self.runtime_groups_by_key),
            "runtime_materialized_records": sum(group.slot_count() for group in self.runtime_groups_by_key.values()),
            "runtime_group_details": _summarize_runtime_groups(self.runtime_groups_by_key),
            "interrupted_current_groups": self.interrupted_group_count_for_step(
                rollout_id=self.rollout_id,
                previous=False,
            )
            if self.rollout_id is not None
            else 0,
            "interrupted_previous_groups": self.interrupted_group_count_for_step(
                rollout_id=self.rollout_id,
                previous=True,
            )
            if self.rollout_id is not None
            else 0,
            "prepare_gate_blocked_ir_count": self._cached_session_debug_total("prepare_gate_blocked_ir_count"),
            "partial_resume_gate_blocked_ir_count": self._cached_session_debug_total(
                "partial_resume_gate_blocked_ir_count"
            ),
            "emitted_materialized_session_count_total": self._emitted_materialized_session_count_total,
            "drained_materialized_session_count_total": self._drained_materialized_session_count_total,
            "notified_session_rollouts": len(self._notified_session_rollouts),
            "output_ready_event_set": bool(self._output_ready_event.is_set()),
            "session_debug_totals": self._cached_session_debug_totals(),
        }

    def resident_group_keys(self) -> set[GroupKey]:
        ready_batch_group_keys = {
            record["group_key"] for records in self._ready_materialized_batches for record in records
        }
        ready_group_keys = {sample_group_key(group) for group in self._ready_materialized_groups}
        return set(self.runtime_groups_by_key) | ready_batch_group_keys | ready_group_keys

    def accounting_snapshot(self) -> dict[str, int]:
        ready_batch_group_keys = {
            record["group_key"] for records in self._ready_materialized_batches for record in records
        }
        ready_group_keys = {sample_group_key(group) for group in self._ready_materialized_groups}
        resident_group_keys = self.resident_group_keys()
        rollout_id = self.rollout_id
        interrupted_current_groups = 0
        interrupted_previous_groups = 0
        if rollout_id is not None:
            interrupted_current_groups = self.interrupted_group_count_for_step(
                rollout_id=rollout_id,
                previous=False,
            )
            interrupted_previous_groups = self.interrupted_group_count_for_step(
                rollout_id=rollout_id,
                previous=True,
            )
        return {
            "resident_groups": len(resident_group_keys),
            "runtime_groups": len(self.runtime_groups_by_key),
            "ready_materialized_batch_groups": len(ready_batch_group_keys),
            "ready_materialized_groups": len(ready_group_keys),
            "runtime_slots": len(self.runtime_slots_by_request_id),
            "interrupted_current_groups": interrupted_current_groups,
            "interrupted_previous_groups": interrupted_previous_groups,
        }

    def _interrupted_close_accounting_snapshot(self, *, rollout_id: int) -> dict[str, int]:
        return {
            "interrupted_sessions": len(self._aborted_resume_session_ids_by_rollout.get(rollout_id, set())),
            "interrupted_current_groups": self.interrupted_group_count_for_step(
                rollout_id=rollout_id,
                previous=False,
            ),
            "interrupted_previous_groups": self.interrupted_group_count_for_step(
                rollout_id=rollout_id,
                previous=True,
            ),
        }

    async def refresh_interrupted_close_accounting(self) -> dict[str, int]:
        rollout_id = self.require_rollout_id()
        now = time.monotonic()
        if (
            self._interrupted_close_accounting_last_refresh_at > 0.0
            and now - self._interrupted_close_accounting_last_refresh_at
            < self._interrupted_close_accounting_refresh_interval_s
        ):
            return self._interrupted_close_accounting_snapshot(rollout_id=rollout_id)
        client = self._ensure_service_client()
        getter = getattr(client, "aborted_resume_session_ids", None)
        if not callable(getter):
            self._interrupted_close_accounting_last_refresh_at = now
            return self._interrupted_close_accounting_snapshot(rollout_id=rollout_id)
        session_ids = {str(session_id) for session_id in await getter(rollout_id=rollout_id)}
        self._aborted_resume_session_ids_by_rollout = {rollout_id: session_ids}
        self._interrupted_close_accounting_last_refresh_at = time.monotonic()
        return self._interrupted_close_accounting_snapshot(rollout_id=rollout_id)

    def interrupted_group_count_for_step(self, *, rollout_id: int, previous: bool) -> int:
        return len(self._interrupted_group_keys_for_step(rollout_id=rollout_id, previous=previous))

    def _interrupted_group_keys_for_step(self, *, rollout_id: int, previous: bool) -> set[GroupKey]:
        aborted_session_ids = self._aborted_resume_session_ids_by_rollout.get(rollout_id, set())
        if not aborted_session_ids:
            return set()
        accounted_group_keys: set[GroupKey] = set()
        for group_key, group in self.runtime_groups_by_key.items():
            if previous:
                if group.admission_rollout_id >= rollout_id:
                    continue
            elif group.admission_rollout_id != rollout_id:
                continue
            aborted_slot_count = 0
            for request_id in group.request_ids:
                slot = self.runtime_slots_by_request_id.get(request_id)
                if slot is not None and slot.session_id in aborted_session_ids:
                    aborted_slot_count += 1
            if aborted_slot_count == 0:
                continue
            if group.slot_count() + aborted_slot_count >= group.expected_count:
                accounted_group_keys.add(group_key)
        return accounted_group_keys

    def _cached_session_debug_total(self, key: str) -> int | None:
        session_debug_state = self.session_debug_state
        if not isinstance(session_debug_state, dict):
            return None
        if key in session_debug_state:
            return int(session_debug_state.get(key) or 0)
        totals = session_debug_state.get("totals")
        if not isinstance(totals, dict):
            return None
        return int(totals.get(key) or 0)

    def _cached_session_debug_totals(self) -> dict[str, int]:
        session_debug_state = self.session_debug_state
        if not isinstance(session_debug_state, dict):
            return {}
        by_state = session_debug_state.get("by_state")
        if not isinstance(by_state, list):
            return {}
        totals = {
            "active_sessions": int(session_debug_state.get("active_sessions") or 0),
            "queued_irs": 0,
            "protected_sessions": 0,
            "active_irs": 0,
            "total_irs": 0,
            "pending_chat_waiters": 0,
            "prepare_gate_sessions": 0,
            "partial_resume_gate_sessions": 0,
        }
        breakdown = session_debug_state.get("session_breakdown")
        if isinstance(breakdown, dict):
            totals["sessions_with_active_irs"] = int(breakdown.get("with_active_irs") or 0)
            totals["sessions_queued_no_active"] = int(breakdown.get("queued_no_active") or 0)
            totals["sessions_waiters_no_active"] = int(breakdown.get("waiters_no_active") or 0)
            totals["sessions_with_no_irs"] = int(breakdown.get("no_irs") or 0)
        for row in by_state:
            if not isinstance(row, dict):
                continue
            session_count = int(row.get("session_count") or 0)
            totals["queued_irs"] += int(row.get("ir_queue") or 0)
            totals["protected_sessions"] += int(row.get("protected_sessions") or 0)
            totals["active_irs"] += int(row.get("active_irs") or 0)
            totals["total_irs"] += int(row.get("irs_by_id") or 0)
            totals["pending_chat_waiters"] += int(row.get("pending_chat_waiters") or 0)
            gate_reason = row.get("gate_reason")
            if gate_reason == "prepare":
                totals["prepare_gate_sessions"] += session_count
            elif gate_reason == "partial_resume":
                totals["partial_resume_gate_sessions"] += session_count
        return totals

    def ensure_session_runner_pool(self, *, total_requests: int) -> ManagedSessionRunnerPool:
        target_total_requests = max(total_requests, self._session_runner_pool_total_requests)
        runner_pool = self._session_runner_pool
        if runner_pool is not None and self._session_runner_pool_total_requests >= target_total_requests:
            return runner_pool
        with self._session_runner_pool_lock:
            runner_pool = self._session_runner_pool
            if runner_pool is None or self._session_runner_pool_total_requests < target_total_requests:
                runner_pool = create_managed_session_runner_pool(self.args, total_requests=target_total_requests)
                if runner_pool is None:
                    raise RuntimeError("Managed session session runner pool could not be created.")
                self._session_runner_pool = runner_pool
                self._session_runner_pool_total_requests = target_total_requests
        return runner_pool

    def _prewarm_session_runner_pool(self, *, total_requests: int) -> None:
        if not ray.is_initialized():
            return
        runner_pool = self.ensure_session_runner_pool(total_requests=total_requests)
        warm_refs = runner_pool.warm_up()
        if not warm_refs:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._session_runner_pool_warm_task is not None and not self._session_runner_pool_warm_task.done():
            return

        async def _wait_warm_refs() -> None:
            warm_results = await asyncio.gather(*warm_refs, return_exceptions=True)
            for result in warm_results:
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    raise RuntimeError("Managed session runner pool warmup failed") from result

        self._session_runner_pool_warm_task = loop.create_task(_wait_warm_refs())

    def _ensure_service_client(self):
        if self._service_client is None:
            self._service_client = create_agentic_service_client()
        return self._service_client

    async def _refresh_session_debug_state(
        self, *, client: Any | None = None, sample_limit: int = 0
    ) -> dict[str, Any]:
        client = self._ensure_service_client() if client is None else client
        debug_state = getattr(client, "debug_state", None)
        if not callable(debug_state):
            return {}
        try:
            snapshot = dict(await debug_state(sample_limit=int(sample_limit)))
        except Exception:
            return {}
        self.session_debug_state = copy.deepcopy(snapshot)
        return snapshot

    async def prepare_group_status(self) -> list[dict[str, Any]]:
        client = self._ensure_service_client()
        return await client.prepare_group_status(scope_id=self.scope_id)

    def raise_if_prepare_group_completed_before_ready(
        self,
        *,
        group_state: PrepareGroupState,
        total_sessions: int,
        ready_sessions: int,
    ) -> None:
        if not group_state.request_handles:
            return
        runner_pool = self._session_runner_pool
        if runner_pool is None:
            raise RuntimeError("RuntimeDomain cannot validate prepare-owned managed sessions without a runner pool.")
        handle_to_request: dict[ManagedSessionHandle, tuple[str, str]] = {}
        for request_handle in group_state.request_handles:
            managed_handle = request_handle.managed_session_handle
            if not isinstance(managed_handle, ManagedSessionHandle):
                raise RuntimeError(
                    "Prepare-owned request is missing a managed session handle; "
                    f"group_id={group_state.group_id} slot_idx={request_handle.slot_idx}."
                )
            try:
                envelope = group_state.request_group[request_handle.slot_idx]
            except IndexError as exc:
                raise RuntimeError(
                    f"Prepare group {group_state.group_id} has inconsistent slot indexing "
                    f"at slot={request_handle.slot_idx}."
                ) from exc
            handle_to_request[managed_handle] = (envelope.session_id, envelope.request_id)
        completed_handles = runner_pool.completed_session_handles(session_handles=list(handle_to_request))
        if not completed_handles:
            return
        completed_requests = [
            {
                "session_id": handle_to_request[managed_handle][0],
                "request_id": handle_to_request[managed_handle][1],
            }
            for managed_handle in completed_handles
        ]
        raise RuntimeError(
            "Prepare-owned managed agent session completed before producing a chat IR: "
            f"group_id={group_state.group_id}, group_generation={group_state.group_generation}, "
            f"expected_sessions={len(group_state.request_handles)}, total_sessions={total_sessions}, "
            f"ready_sessions={ready_sessions}, completed_requests={completed_requests[:8]}."
        )

    async def activate_group_sessions(
        self,
        *,
        groups: list[dict[str, Any]],
        rollout_id: int,
    ) -> dict[str, int]:
        client = self._ensure_service_client()
        return await client.activate_group_sessions(
            scope_id=self.scope_id,
            groups=groups,
            rollout_id=rollout_id,
        )

    async def _release_managed_session_handles(
        self,
        handles: list[ManagedSessionHandle],
        *,
        signal_before_wait: int | None = None,
    ) -> None:
        if not handles:
            return
        runner_pool = self._session_runner_pool
        if runner_pool is None:
            raise RuntimeError("RuntimeDomain cannot release managed sessions without a runner pool.")
        refs = runner_pool.release_session_handles(handles, signal_before_wait=signal_before_wait)
        results = await asyncio.gather(
            *(_resolve_object_async(ref, executor=self._runtime_resources.compiler.cpu_executor) for ref in refs),
            return_exceptions=True,
        )
        released_count = 0
        failures: list[BaseException] = []
        for result in results:
            if isinstance(result, BaseException):
                failures.append(result)
                continue
            released_count += int(result)
        runner_pool.mark_session_handles_released(released_count)
        if failures:
            raise RuntimeError(f"Failed to release {len(failures)} managed session handle(s).") from failures[0]

    def _running_managed_session_handles(self) -> list[ManagedSessionHandle]:
        return [
            slot.managed_session_handle
            for slot in self.runtime_slots_by_request_id.values()
            if slot.state == RuntimeSlotState.RUNNING_APP
            and isinstance(slot.managed_session_handle, ManagedSessionHandle)
        ]

    async def _set_managed_session_timeouts_active(
        self,
        handles: list[ManagedSessionHandle],
        *,
        active: bool,
    ) -> None:
        if not handles:
            return
        runner_pool = self._session_runner_pool
        if runner_pool is None:
            raise RuntimeError("RuntimeDomain cannot update managed session timeout state without a runner pool.")
        refs = runner_pool.set_session_timeouts_active(handles, active=active)
        results = await asyncio.gather(
            *(_resolve_object_async(ref, executor=self._runtime_resources.compiler.cpu_executor) for ref in refs),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError("Failed to update managed session timeout state.") from failures[0]

    async def discard_prepare_group(
        self,
        *,
        group_state: PrepareGroupState,
        released_session_handles: set[ManagedSessionHandle] | None = None,
    ) -> int:
        # Discarding a prepare group ends its prepare-owned managed session handles through the session API path.
        session_ids = {
            envelope.session_id
            for envelope in group_state.request_group
            if isinstance(envelope.session_id, str) and envelope.session_id
        }
        ordered_session_ids = sorted(session_ids)
        if not ordered_session_ids:
            return 0
        finalize_client = self._ensure_service_client()
        discard_tasks = [finalize_client.discard_session(session_id=session_id) for session_id in ordered_session_ids]
        results = await asyncio.gather(*discard_tasks, return_exceptions=True)
        cleaned_count = 0
        cleaned_session_ids: set[str] = set()
        failed_results: list[tuple[str, BaseException]] = []
        for session_id, result in zip(ordered_session_ids, results):
            if isinstance(result, BaseException):
                failed_results.append((session_id, result))
                continue
            if not isinstance(result, bool):
                failed_results.append((session_id, TypeError(f"Unexpected discard result: {result!r}")))
                continue
            cleaned_count += 1
            cleaned_session_ids.add(session_id)
            self._clear_local_session_state(session_id=session_id)
        release_handles: list[ManagedSessionHandle] = []
        released_session_handles = released_session_handles or set()
        for request_handle in group_state.request_handles:
            try:
                envelope = group_state.request_group[request_handle.slot_idx]
            except IndexError as exc:
                raise RuntimeError(
                    "Prepare-owned request handle has invalid slot index during discard: "
                    f"group_id={group_state.group_id} slot_idx={request_handle.slot_idx}."
                ) from exc
            if envelope.session_id not in cleaned_session_ids:
                continue
            managed_session_handle = request_handle.managed_session_handle
            if not isinstance(managed_session_handle, ManagedSessionHandle):
                raise RuntimeError(
                    "Prepare-owned request is missing a managed session handle during discard: "
                    f"group_id={group_state.group_id} slot_idx={request_handle.slot_idx}."
                )
            if managed_session_handle in released_session_handles:
                continue
            release_handles.append(managed_session_handle)
        await self._release_managed_session_handles(release_handles, signal_before_wait=signal.SIGTERM)
        if failed_results:
            failed_session_ids = ", ".join(session_id for session_id, _ in failed_results)
            raise RuntimeError(f"Failed to discard prepare group sessions: failed={failed_session_ids}") from (
                failed_results[0][1]
            )
        return cleaned_count

    async def _finalize_managed_result_payload(
        self,
        *,
        result_payload: dict[str, Any],
        session_runner_started_at: float,
    ) -> FinalizedResultTransport:
        if not isinstance(result_payload, dict):
            raise TypeError(f"Managed result payload must be a dict, got {type(result_payload)}")
        framework_events = result_payload["_agentic_trace_events"]
        if not isinstance(framework_events, dict):
            raise TypeError("Managed result payload '_agentic_trace_events' must be a dict")
        finalize_metadata = result_payload["_session_output_metadata"]
        if not isinstance(finalize_metadata, dict):
            raise TypeError("Managed result payload '_session_output_metadata' must be a dict")
        mark_agentic_event(framework_events, "managed_session_runner_start_at", session_runner_started_at)
        mark_agentic_event(framework_events, "managed_session_runner_end_at")
        finalize_client = self._ensure_service_client()
        mark_agentic_event(framework_events, "managed_finalize_request_start_at")
        transport = await finalize_client.finalize_and_discard(
            session_id=result_payload["_session_id"],
            metadata=copy.deepcopy(finalize_metadata),
            reward=result_payload["reward"],
        )
        mark_agentic_event(framework_events, "managed_finalize_request_end_at")
        transport_metadata = copy.deepcopy(transport.metadata) if isinstance(transport.metadata, dict) else {}
        if framework_events:
            transport_metadata[TRACE_KEY] = merge_agentic_trace(
                transport_metadata.get(TRACE_KEY),
                {"events": framework_events},
            )
        transport.metadata = transport_metadata
        return transport

    async def release_partial_resume_gate(self) -> int:
        """Release partial-resume-gated sessions for the bound rollout step."""
        client = self._ensure_service_client()
        released = int(await client.release_partial_resume_gate(rollout_id=self.require_rollout_id()))
        await self._set_managed_session_timeouts_active(self._running_managed_session_handles(), active=True)
        return released

    async def enter_eval(self) -> int:
        client = self._ensure_service_client()
        entered = int(await client.enter_eval())
        self._rollout_mode = "eval"
        return entered

    async def exit_eval(self) -> int:
        client = self._ensure_service_client()
        exited = int(await client.exit_eval())
        self._rollout_mode = "train"
        return exited

    def _managed_app_spec_for_launch(self) -> ManagedCommandAppSpec:
        base_url = self.managed_chat_api_base_url
        if not isinstance(base_url, str) or not base_url:
            base_url = resolve_chat_api_base_url()
            self.managed_chat_api_base_url = base_url
        return replace(
            self._managed_app_spec,
            env={
                **self._managed_app_spec.env,
                "RELAX_BASE_URL": base_url,
            },
        )

    def _build_managed_session_input(self, *, envelope: RequestEnvelope) -> SessionInput:
        if not isinstance(envelope.session_id, str) or not envelope.session_id:
            raise ValueError("Managed agentic request requires envelope.session_id")
        if envelope.seed.group_index is None:
            raise ValueError("Managed agentic request requires envelope.seed.group_index")
        return SessionInput(
            session_id=envelope.session_id,
            group_id=str(envelope.seed.group_index),
            rollout_mode=self._rollout_mode,
            input_payload=copy.deepcopy(envelope.input_payload),
            metadata={
                "request_id": envelope.request_id,
            },
        )

    def _session_id_for(self, *, envelope: RequestEnvelope, seed_sample: Sample, slot_idx: int) -> str:
        session_id = envelope.session_id
        if isinstance(session_id, str) and session_id:
            return session_id
        generated = f"agentic_session_{self._runtime_session_counter}_{seed_sample.group_index}_{slot_idx}"
        self._runtime_session_counter += 1
        return generated

    def _request_id_for(self, *, envelope: RequestEnvelope, session_id: str) -> str:
        if isinstance(envelope.request_id, str) and envelope.request_id:
            return envelope.request_id
        request_id = f"req_runtime_{session_id}_{self._runtime_request_counter}"
        self._runtime_request_counter += 1
        return request_id

    def _session_sampling_params_for(self, *, envelope: RequestEnvelope, slot_idx: int) -> dict[str, Any]:
        if isinstance(envelope.sampling_params, dict):
            return copy.deepcopy(envelope.sampling_params)
        sampling_params: dict[str, Any] = {
            "temperature": self.args.rollout_temperature,
            "top_p": self.args.rollout_top_p,
            "top_k": self.args.rollout_top_k,
            "max_new_tokens": self.args.rollout_max_response_len,
            "stop": self.args.rollout_stop,
            "stop_token_ids": self.args.rollout_stop_token_ids,
            "skip_special_tokens": self.args.rollout_skip_special_tokens,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }
        if self.args.sglang_enable_deterministic_inference:
            sampling_params["sampling_seed"] = self.args.rollout_seed + slot_idx
        return {key: value for key, value in sampling_params.items() if value is not None}

    async def _register_managed_sessions_batch(self, entries: list[dict[str, Any]]) -> None:
        pending_entries: list[dict[str, Any]] = []
        for entry in entries:
            session_id = entry["session_id"]
            rollout_id = entry["rollout_id"]
            if self._notified_session_rollouts.get(session_id) == rollout_id:
                continue
            pending_entries.append(entry)
        if not pending_entries:
            return
        client = self._ensure_service_client()
        registered = await client.register_sessions_batch(entries=pending_entries)
        if registered != len(pending_entries):
            raise RuntimeError(
                f"Managed session registration mismatch: expected {len(pending_entries)}, got {registered}"
            )
        for entry in pending_entries:
            self._notified_session_rollouts[entry["session_id"]] = entry["rollout_id"]
        await self._refresh_session_debug_state(client=client)

    def has_pending_runtime_work(self) -> bool:
        return bool(self._ready_materialized_batches or self.runtime_slots_by_request_id)

    def has_inflight_work(self) -> bool:
        return bool(self.runtime_slots_by_request_id)

    def has_ready_output(self) -> bool:
        return bool(self._ready_materialized_batches)

    def drain_ready_execution(self) -> ExecutionDispatch:
        materialized_session_count = self._ready_materialized_session_count
        self._ready_materialized_session_count = 0
        self._drained_materialized_session_count_total += materialized_session_count
        return self.drain_ready_dispatch()

    def drain_ready_dispatch(self) -> ExecutionDispatch:
        materialized_batches = list(self._ready_materialized_batches)
        self._ready_materialized_batches.clear()
        ready_groups = list(self._ready_materialized_groups)
        self._ready_materialized_groups.clear()
        return ExecutionDispatch(
            materialized_batches=materialized_batches,
            ready_groups=ready_groups,
        )

    async def gate_rollout_irs_for_partial_resume(self) -> int:
        """Gate not-yet-backend-started IRs for partial resume; resident
        managed tasks keep running."""
        await self._set_managed_session_timeouts_active(self._running_managed_session_handles(), active=False)
        client = self._ensure_service_client()
        locked = await client.gate_rollout_irs_for_partial_resume(rollout_id=self.require_rollout_id())
        await self._refresh_session_debug_state(client=client)
        return locked

    async def gate_rollout_irs_for_discard(self) -> int:
        """Gate current-rollout IRs before non-partial resident tail
        discard."""
        await self._set_managed_session_timeouts_active(self._running_managed_session_handles(), active=False)
        client = self._ensure_service_client()
        locked = await client.gate_rollout_irs_for_discard(rollout_id=self.require_rollout_id())
        await self._refresh_session_debug_state(client=client)
        return locked

    async def gate_all_irs_for_shutdown(self) -> int:
        """Gate all train IRs before terminal shutdown or error cleanup."""
        await self._set_managed_session_timeouts_active(self._running_managed_session_handles(), active=False)
        client = self._ensure_service_client()
        locked = await client.gate_all_irs_for_shutdown()
        await self._refresh_session_debug_state(client=client)
        return locked

    async def active_rollout_request_counts(self) -> dict[str, int]:
        client = self._ensure_service_client()
        counts = dict(await client.active_rollout_request_counts(rollout_id=self.require_rollout_id()))
        await self._refresh_session_debug_state(client=client)
        return counts

    async def abort_rollout_requests(self) -> dict[str, int]:
        client = self._ensure_service_client()
        result = dict(await client.abort_rollout_requests(rollout_id=self.require_rollout_id()))
        await self._refresh_session_debug_state(client=client)
        return result

    async def trim_agentic_session_shards(self, *, reason: str = "manual") -> dict[str, Any]:
        client = self._ensure_service_client()
        result = dict(await client.trim_memory())
        logger.info(
            "Agentic session shard trim completed reason=%s ok=%s trimmed=%s active_sessions=%s active_requests=%s",
            reason,
            result.get("ok"),
            result.get("trimmed_count"),
            result.get("active_sessions"),
            result.get("active_requests"),
        )
        await self._refresh_session_debug_state(client=client)
        return result

    async def _activate_leased_group_sessions(self, batch_input: ExecutionBatchInput) -> dict[str, int]:
        leased_groups = list(batch_input.leased_groups)
        if not leased_groups:
            return {"activated_sessions": 0, "started_sessions": 0}
        client = self._ensure_service_client()
        activation = await client.activate_group_sessions(
            scope_id=self.scope_id,
            groups=[
                {
                    "group_id": str(group_id),
                    "group_generation": group_generation,
                }
                for group_id, group_generation in leased_groups
            ],
            rollout_id=batch_input.rollout_id,
        )
        activated_sessions = activation.get("activated_sessions") or 0
        started_sessions = activation.get("started_sessions") or 0
        await self._refresh_session_debug_state(client=client)
        return {
            "activated_sessions": activated_sessions,
            "started_sessions": started_sessions,
        }

    @staticmethod
    def _leased_group_states_from_input(batch_input: ExecutionBatchInput) -> list[PrepareGroupState]:
        return list(batch_input.leased_group_states)

    @classmethod
    def _leased_request_views(
        cls, batch_input: ExecutionBatchInput
    ) -> list[tuple[PrepareGroupState, PrepareRequestHandle, RequestEnvelope, Sample]]:
        views: list[tuple[PrepareGroupState, PrepareRequestHandle, RequestEnvelope, Sample]] = []
        for group_state in cls._leased_group_states_from_input(batch_input):
            for handle in group_state.request_handles:
                slot_idx = handle.slot_idx
                try:
                    envelope = group_state.request_group[slot_idx]
                    seed_sample = group_state.sample_group[slot_idx]
                except IndexError as exc:
                    raise RuntimeError(
                        f"Prepare group {group_state.group_id} has inconsistent slot indexing at slot={slot_idx}."
                    ) from exc
                views.append((group_state, handle, envelope, seed_sample))
        return views

    @classmethod
    def _refresh_batch_debug_from_input(cls, batch_input: ExecutionBatchInput, batch_debug: dict[str, Any]) -> None:
        leased_request_views = cls._leased_request_views(batch_input)
        batch_debug["leased_group_ids"] = list(batch_input.leased_group_ids)
        batch_debug["leased_group_count"] = len(batch_input.leased_group_ids)
        batch_debug["request_handle_count"] = len(leased_request_views)
        batch_debug["request_id_samples"] = [envelope.request_id for _, _, envelope, _ in leased_request_views[:8]]
        batch_debug["session_request_samples"] = [
            (
                envelope.session_id,
                envelope.request_id,
            )
            for _, _, envelope, _ in leased_request_views[:8]
        ]

    async def start_batch(
        self,
        *,
        batch_input: ExecutionBatchInput,
        retire_leased_groups_cb: Callable[[list[str]], None] | None = None,
    ) -> int:
        leased_group_count = len(batch_input.leased_group_states)
        if leased_group_count <= 0:
            raise RuntimeError("RuntimeDomain cannot start an empty execution batch.")
        if leased_group_count != len(batch_input.leased_group_ids):
            raise RuntimeError(
                "RuntimeDomain execution batch has inconsistent leased group counts: "
                f"states={leased_group_count}, ids={len(batch_input.leased_group_ids)}."
            )
        leased_request_count = len(self._leased_request_views(batch_input))
        if leased_request_count <= 0:
            raise RuntimeError(
                f"RuntimeDomain execution batch has no leased requests for {leased_group_count} leased groups."
            )
        activation_counts = await self._activate_leased_group_sessions(batch_input)
        activated_sessions = activation_counts.get("activated_sessions") or 0
        started_sessions = activation_counts.get("started_sessions") or 0
        if activated_sessions != leased_request_count or started_sessions != leased_request_count:
            raise RuntimeError(
                "RuntimeDomain activation did not take ownership of every leased request: "
                f"leased_requests={leased_request_count}, activated_sessions={activated_sessions}, "
                f"started_sessions={started_sessions}."
            )
        managed_session_handles = self._admit_runtime_slots(batch_input)
        await self._set_managed_session_timeouts_active(managed_session_handles, active=True)
        if retire_leased_groups_cb is not None and batch_input.leased_group_ids:
            retire_leased_groups_cb(list(batch_input.leased_group_ids))
        logger.info(
            "Agentic step admission rollout=%s leased_groups=%s activated_sessions=%s started_sessions=%s resident_slots=%s",
            batch_input.rollout_id,
            leased_group_count,
            activated_sessions,
            started_sessions,
            len(self.runtime_slots_by_request_id),
        )
        return leased_group_count

    def _admit_runtime_slots(self, batch_input: ExecutionBatchInput) -> list[ManagedSessionHandle]:
        managed_session_handles: list[ManagedSessionHandle] = []
        for group_state, request_handle, envelope, seed_sample in self._leased_request_views(batch_input):
            managed_session_handle = request_handle.managed_session_handle
            if managed_session_handle is None:
                raise RuntimeError(
                    "Leased prepare request is missing managed_session_handle; runtime must not relaunch agent process."
                )
            if not isinstance(managed_session_handle, ManagedSessionHandle):
                raise RuntimeError("Leased prepare request has an invalid managed_session_handle.")
            session_id = envelope.session_id
            request_id = envelope.request_id
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError("Leased prepare request is missing session_id.")
            if not isinstance(request_id, str) or not request_id:
                raise RuntimeError("Leased prepare request is missing request_id.")
            self._register_runtime_slot(
                request_id=request_id,
                session_id=session_id,
                dispatch_context=self._dispatch_context_for(
                    rollout_id=batch_input.rollout_id,
                    slot_idx=request_handle.slot_idx,
                    sample_group=group_state.sample_group,
                    envelope=envelope,
                ),
                seed_sample=seed_sample,
                managed_session_handle=managed_session_handle,
                managed_session_submitted_at=float(request_handle.managed_session_submitted_at),
            )
            managed_session_handles.append(managed_session_handle)
        return managed_session_handles

    @staticmethod
    def _prepare_request_registration_entries(
        request_plans: list[Any],
        *,
        scope_id: str,
        group_id: str | None = None,
        group_generation: int = 0,
        gate_reason: str | None = None,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for request_plan in request_plans:
            group_index = request_plan.envelope.seed.group_index
            if group_index is None:
                raise RuntimeError(
                    "rollout-managed session registration requires envelope.seed.group_index to be populated"
                )
            entries.append(
                {
                    "session_id": request_plan.envelope.session_id,
                    "scope_id": scope_id,
                    "rollout_id": request_plan.envelope.rollout_id
                    if request_plan.envelope.rollout_id is not None
                    else -1,
                    "sampling_params": copy.deepcopy(request_plan.request_sampling_params),
                    "session_seed": {
                        "group_index": group_index,
                        "index": request_plan.envelope.seed.index,
                        "label": request_plan.envelope.seed.label,
                        "train_metadata": copy.deepcopy(request_plan.envelope.seed.train_metadata),
                        "metadata": copy.deepcopy(request_plan.envelope.metadata)
                        if isinstance(request_plan.envelope.metadata, dict)
                        else {},
                    },
                    "group_id": group_id,
                    "group_generation": group_generation,
                    "gate_reason": gate_reason,
                }
            )
        return entries

    async def start_prepare_group_sessions(
        self,
        *,
        prepare_groups: list[PrepareGroupSpec],
        runner_pool,
    ) -> list[PrepareGroupState]:
        if not prepare_groups:
            return []
        loop = asyncio.get_running_loop()
        sample_groups = [prepare_group.sample_group for prepare_group in prepare_groups]
        request_groups = await loop.run_in_executor(
            self._runtime_resources.compiler.cpu_executor,
            functools.partial(
                _request_groups_from_sample_groups,
                sample_groups,
                include_input_payload=self.args.rollout_global_dataset,
            ),
        )
        group_records: list[dict[str, Any]] = []
        all_request_plans: list[_PrepareRequestPlan] = []
        registration_entries: list[dict[str, Any]] = []
        for prepare_group, sample_group, request_group in zip(
            prepare_groups, sample_groups, request_groups, strict=True
        ):
            group_id = prepare_group.group_id
            group_generation = prepare_group.group_generation
            request_plans: list[_PrepareRequestPlan] = []
            for slot_idx, (envelope, seed_sample) in enumerate(zip(request_group, sample_group, strict=True)):
                session_id = self._session_id_for(
                    envelope=envelope,
                    seed_sample=seed_sample,
                    slot_idx=slot_idx,
                )
                envelope.session_id = session_id
                seed_sample.session_id = session_id
                request_sampling_params = (
                    copy.deepcopy(envelope.sampling_params)
                    if isinstance(envelope.sampling_params, dict)
                    else self._session_sampling_params_for(
                        envelope=envelope,
                        slot_idx=slot_idx,
                    )
                )
                request_plan = _PrepareRequestPlan(
                    slot_idx=slot_idx,
                    envelope=envelope,
                    session_id=session_id,
                    request_sampling_params=request_sampling_params,
                )
                request_plans.append(request_plan)
                all_request_plans.append(request_plan)
            registration_entries.extend(
                self._prepare_request_registration_entries(
                    request_plans,
                    scope_id=self.scope_id,
                    group_id=group_id,
                    group_generation=group_generation,
                    gate_reason="prepare",
                )
            )
            group_records.append(
                {
                    "group_id": group_id,
                    "group_generation": group_generation,
                    "sample_group": sample_group,
                    "request_group": request_group,
                    "request_plans": request_plans,
                }
            )
        managed_session_handles: list[ManagedSessionHandle] = []
        try:
            await self._register_managed_sessions_batch(entries=registration_entries)
            session_inputs: list[SessionInput] = []
            submission_records: list[tuple[int, _PrepareRequestPlan]] = []
            for group_index, group_record in enumerate(group_records):
                for request_plan in group_record["request_plans"]:
                    if not isinstance(request_plan.envelope.request_id, str) or not request_plan.envelope.request_id:
                        request_plan.envelope.request_id = self._request_id_for(
                            envelope=request_plan.envelope,
                            session_id=request_plan.session_id,
                        )
                    session_inputs.append(self._build_managed_session_input(envelope=request_plan.envelope))
                    submission_records.append((group_index, request_plan))
            session_submitted_at = time.time()
            managed_session_handles = runner_pool.submit_sessions(
                spec=self._managed_app_spec_for_launch(),
                session_inputs=session_inputs,
            )
            if len(managed_session_handles) != len(submission_records):
                raise RuntimeError(
                    "Managed session batch submission returned a mismatched number of handles: "
                    f"expected={len(submission_records)}, got={len(managed_session_handles)}."
                )
            request_handles_by_group: list[list[PrepareRequestHandle]] = [[] for _ in group_records]
            for (group_index, request_plan), managed_session_handle in zip(
                submission_records, managed_session_handles, strict=True
            ):
                if not isinstance(request_plan.envelope.request_id, str) or not request_plan.envelope.request_id:
                    raise RuntimeError("Submitted prepare request is missing request_id.")
                request_handles_by_group[group_index].append(
                    PrepareRequestHandle(
                        slot_idx=request_plan.slot_idx,
                        managed_session_handle=managed_session_handle,
                        managed_session_submitted_at=session_submitted_at,
                    )
                )
            return [
                PrepareGroupState(
                    group_id=str(group_record["group_id"]),
                    group_generation=int(group_record["group_generation"]),
                    sample_group=group_record["sample_group"],
                    request_group=group_record["request_group"],
                    request_handles=request_handles,
                    status="warming",
                )
                for group_record, request_handles in zip(group_records, request_handles_by_group, strict=True)
            ]
        except BaseException:
            try:
                finalize_client = self._ensure_service_client()
            except Exception:
                finalize_client = None
            if finalize_client is not None:
                discard_tasks = [
                    finalize_client.discard_session(session_id=request_plan.session_id)
                    for request_plan in all_request_plans
                    if isinstance(request_plan.session_id, str) and request_plan.session_id
                ]
                if discard_tasks:
                    await asyncio.gather(*discard_tasks, return_exceptions=True)
            for request_plan in all_request_plans:
                session_id = request_plan.session_id
                if isinstance(session_id, str) and session_id:
                    self._clear_local_session_state(session_id=session_id)
            await self._release_managed_session_handles(managed_session_handles)
            raise

    @staticmethod
    def _dispatch_context_for(
        *,
        rollout_id: int,
        slot_idx: int,
        sample_group: list[Sample],
        envelope: RequestEnvelope,
    ) -> RequestDispatchContext:
        return RequestDispatchContext(
            rollout_id=rollout_id,
            group_key=sample_group_key(sample_group),
            expected_count=len(sample_group),
            slot_idx=slot_idx,
            envelope=envelope,
        )

    async def _discard_registered_sessions(self) -> int:
        session_ids: set[str] = set(self._notified_session_rollouts)
        for slot in self.runtime_slots_by_request_id.values():
            if isinstance(slot.session_id, str) and slot.session_id:
                session_ids.add(slot.session_id)
        if not session_ids:
            return 0

        try:
            finalize_client = self._ensure_service_client()
        except BaseException as exc:  # noqa: BLE001
            logger.warning(
                "Failed to create finalize client during runtime shutdown session cleanup: error=%s",
                exc,
            )
            raise RuntimeError("Failed to create finalize client during runtime shutdown session cleanup") from exc
        ordered_session_ids = sorted(session_ids)
        results = await asyncio.gather(
            *(finalize_client.discard_session(session_id=session_id) for session_id in ordered_session_ids),
            return_exceptions=True,
        )
        cleaned_count = 0
        cleaned_session_ids: set[str] = set()
        failed_results: list[tuple[str, BaseException]] = []
        for session_id, result in zip(ordered_session_ids, results):
            if isinstance(result, BaseException):
                failed_results.append((session_id, result))
                continue
            if not isinstance(result, bool):
                failed_results.append((session_id, TypeError(f"Unexpected discard result: {result!r}")))
                continue
            cleaned_count += 1
            cleaned_session_ids.add(session_id)
            # Idempotent "already absent" cleanup is still considered terminal for runtime-local cleanup state.
            self._clear_local_session_state(session_id=session_id)
        if cleaned_count:
            logger.debug(
                "Discarded or confirmed missing registered agentic sessions count=%s session_ids=%s",
                cleaned_count,
                sorted(cleaned_session_ids)[:16],
            )
        if failed_results:
            for session_id, result in failed_results:
                logger.warning(
                    "Failed to discard registered agentic session during runtime shutdown: session_id=%s error=%s",
                    session_id,
                    result,
                )
            failed_session_ids = ", ".join(session_id for session_id, _ in failed_results)
            raise RuntimeError(
                f"Failed to discard registered agentic sessions during runtime shutdown: failed={failed_session_ids}"
            ) from failed_results[0][1]
        return cleaned_count

    async def shutdown(self) -> None:
        await self.drop_resident_results()
        discarded_sessions = await self._discard_registered_sessions()
        if discarded_sessions:
            logger.info(
                "Discarded registered agentic sessions during runtime shutdown count=%s",
                discarded_sessions,
            )
        if self._session_runner_pool_warm_task is not None:
            warm_task = self._session_runner_pool_warm_task
            warm_task.cancel()
            self._session_runner_pool_warm_task = None
            try:
                await warm_task
            except asyncio.CancelledError:
                pass
        if self._service_client is not None:
            await self._service_client.aclose()
            self._service_client = None
        if self._session_runner_pool is not None:
            self._session_runner_pool.shutdown()
            self._session_runner_pool = None
        self._emitted_materialized_session_count_total = 0
        self._drained_materialized_session_count_total = 0
        self.session_debug_state = None
        self.managed_chat_api_base_url = None
        self._notified_session_rollouts.clear()

    async def wait_for_next_runtime_slot(self, *, timeout_s: float | None = None) -> str | None:
        if self._ready_materialized_batches:
            self._output_ready_event.clear()
            return "output_ready"
        materialization_progress = await self._drain_finished_runtime_slot_materializations()
        if materialization_progress is not None:
            return materialization_progress
        running_slots = [
            slot for slot in self.runtime_slots_by_request_id.values() if slot.state == RuntimeSlotState.RUNNING_APP
        ]
        if not running_slots:
            return None
        runner_pool = self._session_runner_pool
        if runner_pool is None:
            raise RuntimeError("RuntimeDomain has running managed sessions without a runner pool.")
        handle_to_request_id = {slot.managed_session_handle: slot.request_id for slot in running_slots}
        poll_sleep_s = 0.05 if timeout_s is None else max(0.0, float(timeout_s))
        owned_handles = list(handle_to_request_id)
        ready_handles = runner_pool.drain_completed_session_handles(session_handles=owned_handles)
        if not ready_handles and poll_sleep_s > 0.0:
            await asyncio.sleep(poll_sleep_s)
            ready_handles = runner_pool.drain_completed_session_handles(session_handles=owned_handles)
        if not ready_handles:
            return None
        for ready_handle in ready_handles:
            request_id = handle_to_request_id[ready_handle]
            slot = self.runtime_slots_by_request_id.get(request_id)
            if slot is None:
                continue
            self._start_runtime_slot_materialization(request_id=request_id, slot=slot)
        materialization_progress = await self._drain_finished_runtime_slot_materializations()
        if materialization_progress is not None:
            return materialization_progress
        return "slot_materializing"

    async def _materialize_managed_result_record(
        self,
        *,
        dispatch_context: RequestDispatchContext,
        session_id: str,
        managed_result: _ManagedRequestResult,
    ) -> dict[str, Any] | _MaterializationDrop:
        # This is the only runtime path that turns an ended managed agent task into a finalized sample.
        result_transport = await self._finalize_managed_result_payload(
            result_payload=managed_result.result_payload,
            session_runner_started_at=managed_result.session_runner_started_at,
        )
        if result_transport.status in {"discarded", "non_finalizable"}:
            return _MaterializationDrop(
                group_key=dispatch_context.group_key,
                session_id=session_id,
                reason=result_transport.status,
                info=copy.deepcopy(result_transport.metadata) if isinstance(result_transport.metadata, dict) else {},
            )
        return await self._materialize_transport_record(
            dispatch_context=dispatch_context,
            session_id=session_id,
            result_transport=result_transport,
        )

    async def _materialize_transport_record(
        self,
        *,
        dispatch_context: RequestDispatchContext,
        session_id: str,
        result_transport: FinalizedResultTransport,
    ) -> dict[str, Any]:
        exported_samples = await self._export_samples_from_transport(
            session_id=session_id,
            envelope=dispatch_context.envelope,
            result_transport=result_transport,
        )
        self._notified_session_rollouts.pop(session_id, None)
        return {
            "rollout_id": dispatch_context.rollout_id,
            "group_key": dispatch_context.group_key,
            "expected_count": dispatch_context.expected_count,
            "slot_idx": dispatch_context.slot_idx,
            "session_id": session_id,
            "samples": exported_samples,
        }

    def _emit_session_materialization_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        ready_at = time.time()
        self._ready_materialized_session_count += len(records)
        self._emitted_materialized_session_count_total += len(records)
        for record in records:
            for sample in record["samples"]:
                mark_metadata_agentic_event(sample.metadata, "materialize_ready_at", ready_at)
        for record in records:
            group = self.runtime_groups_by_key.get(record["group_key"])
            if group is None:
                raise RuntimeError(f"RuntimeDomain cannot materialize unknown resident group: {record['group_key']!r}")
            group.add_materialized_record(record, store_samples=bool(self.args.group_rm))
            if group.is_complete() and self.args.group_rm:
                completed_group = group.materialized_group()
                self._ready_materialized_groups.append(completed_group)
        self._ready_materialized_batches.append(list(records))
        self._output_ready_event.set()

    async def _export_samples_from_transport(
        self,
        *,
        session_id: str,
        envelope: RequestEnvelope,
        result_transport: FinalizedResultTransport,
    ) -> list[Sample]:
        if result_transport.status in {"discarded", "non_finalizable"}:
            raise RuntimeError(f"Cannot export samples from {result_transport.status!r} transport.")
        artifact_ref = result_transport.artifact_ref
        if artifact_ref is None:
            raise RuntimeError("Agentic export requires artifact_ref in FinalizedResultTransport")
        artifact = await _resolve_object_async(artifact_ref, executor=self._runtime_resources.compiler.cpu_executor)
        if isinstance(artifact, TrainingFieldArtifact):
            sample = artifact.to_sample()
        elif isinstance(artifact, dict):
            sample = TrainingFieldArtifact(sample_payload=artifact).to_sample()
        else:
            raise RuntimeError(
                "Agentic export requires artifact_ref to resolve to TrainingFieldArtifact-compatible payload."
            )
        sample.group_index = envelope.seed.group_index
        sample.index = envelope.seed.index
        sample.label = envelope.seed.label
        sample.train_metadata = envelope.seed.train_metadata
        sample.metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.session_id = session_id
        _overlay_sample_metadata(base_metadata=sample.metadata, overlay_metadata=envelope.metadata)
        _overlay_sample_metadata(base_metadata=sample.metadata, overlay_metadata=result_transport.metadata)
        mark_metadata_agentic_event(sample.metadata, "materialize_harvest_at")
        return [sample]


# Runtime backend resources


_DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]
_COMPILER_RESOURCE_CACHE_LOCK = threading.Lock()
_COMPILER_RESOURCE_CACHE: dict[tuple[str, int], "AgenticCompilerResources"] = {}
_MAX_COMPILER_RESOURCE_CACHE_ENTRIES = 4


def _default_cpu_worker_count() -> int:
    cpu_count = os.cpu_count() or 8
    return max(16, min(64, cpu_count * 4))


@dataclass(frozen=True)
class AgenticCompilerResources:
    tokenizer: Any
    processor: Any
    cpu_executor: Executor
    processor_pool: Any | None = None

    def shutdown(self) -> None:
        try:
            self.cpu_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.cpu_executor.shutdown(wait=False)
        except Exception as exc:
            raise RuntimeError("Failed to shutdown compiler cpu executor") from exc

        if self.processor_pool is not None:
            try:
                self.processor_pool.shutdown(wait=False)
            except Exception as exc:
                raise RuntimeError("Failed to shutdown processor pool") from exc


@dataclass(frozen=True)
class RuntimeDomainResources:
    compiler: AgenticCompilerResources
    target_session_count: int

    def step_session_capacity(self) -> int:
        return self.target_session_count

    def materialization_cap(self, *, total_requests: int) -> int:
        return min(total_requests, self.compiler.cpu_executor._max_workers)

    def session_runner_pool_size(self, *, total_requests: int) -> int:
        launch_limit = max(self.target_session_count, total_requests)
        if launch_limit <= 0 or total_requests <= 0:
            return 0
        desired_by_load = ceil(total_requests / 16)
        node_count = 1
        try:
            import ray

            if ray.is_initialized():
                node_count = sum(1 for node in ray.nodes() if node.get("Alive"))
        except Exception as exc:
            raise RuntimeError("Failed to inspect Ray nodes for launch session runner pool sizing") from exc
        return min(total_requests, launch_limit, max(node_count, desired_by_load))

    def session_runner_capacities(self, *, total_requests: int) -> list[int]:
        launch_limit = max(self.target_session_count, total_requests)
        runner_count = self.session_runner_pool_size(total_requests=total_requests)
        if launch_limit <= 0 or runner_count <= 0:
            return []
        base, remainder = divmod(launch_limit, runner_count)
        capacities = [base + (1 if idx < remainder else 0) for idx in range(runner_count)]
        return [capacity for capacity in capacities if capacity > 0]

    def build_session_runner_pool(self, *, total_requests: int, factory: Callable[[list[int]], Any]) -> Any | None:
        capacities = self.session_runner_capacities(total_requests=total_requests)
        if not capacities:
            return None
        return factory(capacities)


def _bootstrap_processor_pool(args: Any):
    pool_size = args.mm_processor_pool_size
    if pool_size <= 0:
        return None
    from relax.utils.data.processor_pool import ProcessorPool

    return ProcessorPool(
        model_path=args.hf_checkpoint,
        pool_size=pool_size,
        trust_remote_code=True,
    )


def _pop_oldest_cache_entry(cache: dict[Any, Any]) -> tuple[Any, Any] | None:
    if not cache:
        return None
    oldest_key = next(iter(cache))
    return oldest_key, cache.pop(oldest_key)


def clear_agentic_runtime_caches() -> None:
    global _AGENTIC_SERVICE_CLIENT
    _AGENTIC_SERVICE_CLIENT = None

    with _COMPILER_RESOURCE_CACHE_LOCK:
        compiler_resources = list(_COMPILER_RESOURCE_CACHE.values())
        _COMPILER_RESOURCE_CACHE.clear()
    for resources in compiler_resources:
        resources.shutdown()
    logger.info("Agentic runtime caches cleared.")


def agentic_prepare_pool_target_from_args(args: Any) -> int:
    oversample_group_count = args.over_sampling_batch_size
    configured_pool_size = args.agentic_prepare_pool_size
    default_pool_target = oversample_group_count
    prepare_pool_target_groups = (
        configured_pool_size if configured_pool_size not in {None, ""} else default_pool_target
    )
    if prepare_pool_target_groups < oversample_group_count:
        raise RuntimeError(
            f"agentic_prepare_pool_size ({prepare_pool_target_groups}) must be >= "
            f"over_sampling_batch_size ({oversample_group_count}); pool must hold at least one sampling batch"
        )
    return prepare_pool_target_groups


def agentic_target_session_count_from_args(args: Any) -> int:
    oversample_group_count = args.over_sampling_batch_size
    prepare_pool_target_groups = agentic_prepare_pool_target_from_args(args)
    tail_carry_group_count = args.rollout_batch_size if getattr(args, "fully_async", False) else 0
    sessions_per_group = args.n_samples_per_prompt
    return (tail_carry_group_count + oversample_group_count + prepare_pool_target_groups) * sessions_per_group


def get_agentic_runtime_resources(args: Any) -> RuntimeDomainResources:
    hf_checkpoint = args.hf_checkpoint
    pool_size = args.mm_processor_pool_size
    cache_key = (hf_checkpoint, pool_size)
    with _COMPILER_RESOURCE_CACHE_LOCK:
        compiler_resources = _COMPILER_RESOURCE_CACHE.get(cache_key)
        if compiler_resources is None:
            evicted = None
            if len(_COMPILER_RESOURCE_CACHE) >= _MAX_COMPILER_RESOURCE_CACHE_ENTRIES:
                evicted = _pop_oldest_cache_entry(_COMPILER_RESOURCE_CACHE)
            from relax.utils.data.processing_utils import load_processor, load_tokenizer

            compiler_resources = AgenticCompilerResources(
                tokenizer=load_tokenizer(hf_checkpoint, trust_remote_code=True),
                processor=load_processor(hf_checkpoint, trust_remote_code=True),
                cpu_executor=ThreadPoolExecutor(max_workers=_default_cpu_worker_count()),
                processor_pool=_bootstrap_processor_pool(args),
            )
            _COMPILER_RESOURCE_CACHE[cache_key] = compiler_resources
            logger.info(
                "Created agentic compiler resources "
                f"(hf_checkpoint={hf_checkpoint}, mm_processor_pool_size={pool_size})"
            )
        else:
            _COMPILER_RESOURCE_CACHE.pop(cache_key)
            _COMPILER_RESOURCE_CACHE[cache_key] = compiler_resources
            evicted = None
    if evicted is not None:
        evicted[1].shutdown()
    return RuntimeDomainResources(
        compiler=compiler_resources,
        target_session_count=agentic_target_session_count_from_args(args),
    )


def _processing_utils():
    from relax.utils.data import processing_utils

    return processing_utils


def _normalize_multimodal_inputs_sync(multimodal_inputs: dict[str, Any] | None) -> dict[str, Any] | None:
    if not multimodal_inputs:
        return None

    normalized: dict[str, Any] = {}
    images = list(multimodal_inputs.get("images") or [])
    if images:
        from PIL import Image

        from relax.utils.multimodal.image_utils import load_image

        normalized_images = []
        for image in images:
            if image is None:
                continue
            if isinstance(image, Image.Image):
                normalized_images.append(image)
                continue
            if isinstance(image, dict):
                image_bytes = image.get("bytes")
                image_path = image.get("path")
                if isinstance(image_bytes, (bytes, bytearray)):
                    normalized_images.append(load_image(bytes(image_bytes)))
                    continue
                if isinstance(image_path, str) and image_path:
                    normalized_images.append(load_image(image_path))
                    continue
            normalized_images.append(load_image(image))
        if normalized_images:
            normalized["images"] = normalized_images

    videos = list(multimodal_inputs.get("videos") or [])
    if videos:
        normalized["videos"] = videos

    audio_items = list(multimodal_inputs.get("audio") or [])
    if audio_items:
        normalized["audio"] = audio_items

    return normalized or None


@dataclass
class EncodedMessages:
    train_prompt_ids: list[int]
    backend_prompt_ids: list[int]
    backend_image_data: list[str] = field(default_factory=list)
    backend_audio_data: list[str] = field(default_factory=list)
    backend_video_data: list[str] = field(default_factory=list)
    multimodal_train_inputs: dict[str, Any] | None = None
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class BackendGenerateResult:
    new_tokens: list[int]
    new_log_probs: list[float]
    finish_type: str
    meta_info: dict[str, Any]
    elapsed: float


class BackendContextLengthExceededError(RuntimeError):
    pass


class SGLangMessageCompiler:
    def __init__(
        self,
        *,
        tokenizer: Any,
        processor: Any,
        processor_pool: Any | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        use_audio_in_video: bool = False,
        cpu_executor: Any | None = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.processor_pool = processor_pool
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.use_audio_in_video = use_audio_in_video
        self.cpu_executor = cpu_executor

    @staticmethod
    def _merge_timing(*timings: dict[str, float]) -> dict[str, float]:
        merged: dict[str, float] = {}
        for timing in timings:
            if not timing:
                continue
            merged.update({key: float(value) for key, value in timing.items()})
        return merged

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> str:
        chat_template_kwargs = (
            self.apply_chat_template_kwargs
            if chat_template_kwargs is None
            else normalize_template_kwargs(chat_template_kwargs)
        )
        return self.tokenizer.apply_chat_template(
            check_messages(messages),
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    def _build_prompt_sync(
        self,
        normalized_messages: list[dict[str, Any]],
        normalized_tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> tuple[str, list[int], dict[str, float]]:
        template_started_at = time.monotonic()
        prompt_text = self._apply_chat_template(
            normalized_messages,
            tools=normalized_tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        template_elapsed_s = time.monotonic() - template_started_at

        tokenize_started_at = time.monotonic()
        backend_prompt_ids = list(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        tokenize_elapsed_s = time.monotonic() - tokenize_started_at
        return (
            prompt_text,
            backend_prompt_ids,
            {
                "apply_chat_template_elapsed_s": template_elapsed_s,
                "tokenizer_encode_elapsed_s": tokenize_elapsed_s,
            },
        )

    def _build_observation_prompt_sync(
        self,
        normalized_messages: list[dict[str, Any]],
        normalized_tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> tuple[str, list[int], int, dict[str, float]]:
        chat_template_kwargs = (
            self.apply_chat_template_kwargs
            if chat_template_kwargs is None
            else normalize_template_kwargs(chat_template_kwargs)
        )
        template_started_at = time.monotonic()
        dummy_prompt = self.tokenizer.apply_chat_template(
            check_messages(_DUMMY_MESSAGES),
            tools=normalized_tools,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs,
        ).rstrip("\n")
        formatted_prompt = self.tokenizer.apply_chat_template(
            check_messages(_DUMMY_MESSAGES + normalized_messages),
            tools=normalized_tools,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        template_elapsed_s = time.monotonic() - template_started_at

        tokenize_started_at = time.monotonic()
        trim_length = len(self.tokenizer.encode(dummy_prompt, add_special_tokens=False))
        backend_prompt_ids = list(self.tokenizer.encode(formatted_prompt, add_special_tokens=False))
        tokenize_elapsed_s = time.monotonic() - tokenize_started_at
        if trim_length:
            backend_prompt_ids = backend_prompt_ids[trim_length:]
        return (
            formatted_prompt,
            backend_prompt_ids,
            trim_length,
            {
                "apply_chat_template_elapsed_s": template_elapsed_s,
                "tokenizer_encode_elapsed_s": tokenize_elapsed_s,
            },
        )

    async def _compute_multimodal_inputs_async(
        self,
        normalized_messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        from relax.utils.data.processing_utils import process_vision_info

        started_at = time.monotonic()
        loop = asyncio.get_running_loop()
        multimodal_inputs = await loop.run_in_executor(
            self.cpu_executor,
            process_vision_info,
            normalized_messages,
            self.processor,
            self.use_audio_in_video,
        )
        return multimodal_inputs, {"process_vision_info_elapsed_s": time.monotonic() - started_at}

    async def _run_processor_async(
        self,
        prompt_text: str,
        multimodal_inputs: dict[str, Any],
        *,
        trim_length: int = 0,
    ) -> tuple[list[int], dict[str, Any] | None, dict[str, float]]:
        loop = asyncio.get_running_loop()
        started_at = time.monotonic()
        if self.processor_pool is not None:
            from relax.utils.data.processor_pool import prepare_mm_inputs_for_ipc, process_sample_in_worker

            mm_inputs_ipc = prepare_mm_inputs_for_ipc(multimodal_inputs)
            processor_kwargs = {
                "use_audio_in_video": self.use_audio_in_video,
                "return_mm_token_type_ids": False,
            }
            train_prompt_ids, multimodal_train_inputs = await loop.run_in_executor(
                self.processor_pool.executor,
                process_sample_in_worker,
                prompt_text,
                mm_inputs_ipc,
                processor_kwargs,
            )
            train_prompt_ids = list(train_prompt_ids)
        else:

            def _run_processor_sync():
                from relax.utils.data.processing_utils import (
                    adapt_processor_kwargs,
                    expand_kimi_k25_placeholders,
                    remap_mm_train_inputs,
                )

                processor_kwargs = adapt_processor_kwargs(
                    self.processor,
                    multimodal_inputs,
                    {
                        "use_audio_in_video": self.use_audio_in_video,
                        "return_mm_token_type_ids": False,
                    },
                )
                processor_output = self.processor(
                    text=prompt_text,
                    **processor_kwargs,
                )
                prompt_ids = list(processor_output["input_ids"][0])
                train_inputs = {
                    key: (torch.from_numpy(value) if isinstance(value, np.ndarray) else value)
                    for key, value in processor_output.items()
                    if key not in {"input_ids", "attention_mask"}
                } or None
                train_inputs = remap_mm_train_inputs(self.processor, train_inputs)
                prompt_ids = expand_kimi_k25_placeholders(self.processor, prompt_ids, train_inputs)
                return prompt_ids, train_inputs

            train_prompt_ids, multimodal_train_inputs = await loop.run_in_executor(
                self.cpu_executor,
                _run_processor_sync,
            )

        if trim_length:
            train_prompt_ids = train_prompt_ids[trim_length:]
        return (
            train_prompt_ids,
            multimodal_train_inputs,
            {"processor_elapsed_s": time.monotonic() - started_at},
        )

    async def _encode_media_async(
        self,
        multimodal_inputs: dict[str, Any] | None,
    ) -> tuple[list[str], list[str], list[str], dict[str, float]]:
        if not multimodal_inputs:
            return [], [], [], {}
        started_at = time.monotonic()
        images = list(multimodal_inputs.get("images") or [])
        videos = list(multimodal_inputs.get("videos") or [])
        audio_items = list(multimodal_inputs.get("audio") or [])

        processing_utils = _processing_utils()
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(self.cpu_executor, processing_utils.encode_image_for_rollout_engine, image)
            for image in images
        ]
        tasks.extend(
            loop.run_in_executor(
                self.cpu_executor,
                processing_utils.encode_video_tensor_for_rollout_engine,
                video,
            )
            for video in videos
        )
        for audio in audio_items:
            if isinstance(audio, tuple) and len(audio) == 2:
                waveform, sample_rate = audio
                tasks.append(
                    loop.run_in_executor(
                        self.cpu_executor,
                        processing_utils.encode_audio_for_rollout_engine,
                        waveform,
                        sample_rate,
                    )
                )

        if not tasks:
            return [], [], [], {}

        results = await asyncio.gather(*tasks)
        offset = 0
        image_data = list(results[offset : offset + len(images)])
        offset += len(images)
        video_data = list(results[offset : offset + len(videos)])
        offset += len(videos)
        audio_data = list(results[offset:])
        return image_data, audio_data, video_data, {"media_encode_elapsed_s": time.monotonic() - started_at}

    async def encode_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None = None,
        multimodal_inputs: dict[str, Any] | None = None,
    ) -> EncodedMessages:
        total_started_at = time.monotonic()
        normalized_messages = check_messages(messages)
        normalized_tools = normalize_tools(tools)
        loop = asyncio.get_running_loop()

        prompt_future = loop.run_in_executor(
            self.cpu_executor,
            self._build_prompt_sync,
            normalized_messages,
            normalized_tools,
            chat_template_kwargs,
        )
        if self.processor and multimodal_inputs is None:
            multimodal_future = asyncio.create_task(self._compute_multimodal_inputs_async(normalized_messages))
        else:
            multimodal_future = None
        normalized_override_future = None
        if multimodal_inputs is not None:
            normalized_override_future = loop.run_in_executor(
                self.cpu_executor,
                _normalize_multimodal_inputs_sync,
                multimodal_inputs,
            )
        prompt_text, backend_prompt_ids, prompt_timing = await prompt_future
        multimodal_inputs = None
        multimodal_timing: dict[str, float] = {}
        if multimodal_future is not None:
            multimodal_inputs, multimodal_timing = await multimodal_future
        elif normalized_override_future is not None:
            multimodal_inputs = await normalized_override_future

        train_prompt_ids = list(backend_prompt_ids)
        multimodal_train_inputs = None
        processor_timing: dict[str, float] = {}
        media_timing: dict[str, float] = {}
        image_data: list[str] = []
        audio_data: list[str] = []
        video_data: list[str] = []

        processor_task = None
        media_task = None
        if self.processor:
            processor_task = asyncio.create_task(self._run_processor_async(prompt_text, multimodal_inputs or {}))
        if multimodal_inputs:
            media_task = asyncio.create_task(self._encode_media_async(multimodal_inputs))

        if processor_task is not None:
            train_prompt_ids, multimodal_train_inputs, processor_timing = await processor_task
        if media_task is not None:
            image_data, audio_data, video_data, media_timing = await media_task

        timing = self._merge_timing(prompt_timing, multimodal_timing, processor_timing, media_timing)
        timing["total_elapsed_s"] = time.monotonic() - total_started_at
        return EncodedMessages(
            train_prompt_ids=train_prompt_ids,
            backend_prompt_ids=backend_prompt_ids,
            backend_image_data=image_data,
            backend_audio_data=audio_data,
            backend_video_data=video_data,
            multimodal_train_inputs=multimodal_train_inputs,
            timing=timing,
        )

    async def encode_observation_delta(
        self,
        messages_delta: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None = None,
        multimodal_inputs: dict[str, Any] | None = None,
    ) -> EncodedMessages:
        total_started_at = time.monotonic()
        normalized_messages = check_messages(messages_delta)
        normalized_tools = normalize_tools(tools)
        loop = asyncio.get_running_loop()

        prompt_future = loop.run_in_executor(
            self.cpu_executor,
            self._build_observation_prompt_sync,
            normalized_messages,
            normalized_tools,
            chat_template_kwargs,
        )
        if self.processor and multimodal_inputs is None:
            multimodal_future = asyncio.create_task(self._compute_multimodal_inputs_async(normalized_messages))
        else:
            multimodal_future = None
        normalized_override_future = None
        if multimodal_inputs is not None:
            normalized_override_future = loop.run_in_executor(
                self.cpu_executor,
                _normalize_multimodal_inputs_sync,
                multimodal_inputs,
            )

        formatted_prompt, backend_prompt_ids, trim_length, prompt_timing = await prompt_future
        multimodal_inputs = None
        multimodal_timing: dict[str, float] = {}
        if multimodal_future is not None:
            multimodal_inputs, multimodal_timing = await multimodal_future
        elif normalized_override_future is not None:
            multimodal_inputs = await normalized_override_future

        train_prompt_ids = list(backend_prompt_ids)
        multimodal_train_inputs = None
        processor_timing: dict[str, float] = {}
        media_timing: dict[str, float] = {}
        image_data: list[str] = []
        audio_data: list[str] = []
        video_data: list[str] = []

        processor_task = None
        media_task = None
        if self.processor:
            processor_task = asyncio.create_task(
                self._run_processor_async(formatted_prompt, multimodal_inputs or {}, trim_length=trim_length)
            )
        if multimodal_inputs:
            media_task = asyncio.create_task(self._encode_media_async(multimodal_inputs))

        if processor_task is not None:
            train_prompt_ids, multimodal_train_inputs, processor_timing = await processor_task
        if media_task is not None:
            image_data, audio_data, video_data, media_timing = await media_task

        timing = self._merge_timing(prompt_timing, multimodal_timing, processor_timing, media_timing)
        timing["total_elapsed_s"] = time.monotonic() - total_started_at
        return EncodedMessages(
            train_prompt_ids=train_prompt_ids,
            backend_prompt_ids=backend_prompt_ids,
            backend_image_data=image_data,
            backend_audio_data=audio_data,
            backend_video_data=video_data,
            multimodal_train_inputs=multimodal_train_inputs,
            timing=timing,
        )


def _extract_output_tokens_and_log_probs(
    meta_info: dict[str, Any],
    *,
    output_ids: list[int],
    tokenizer: Any,
    processor: Any | None = None,
) -> tuple[list[int], list[float]]:
    output_token_logprobs = meta_info.get("output_token_logprobs")
    if output_token_logprobs:
        tokens = [item[1] for item in output_token_logprobs]
        log_probs = [item[0] for item in output_token_logprobs]
        return _sanitize_multimodal_output_tokens(tokens, tokenizer=tokenizer, processor=processor), log_probs
    return _sanitize_multimodal_output_tokens(output_ids, tokenizer=tokenizer, processor=processor), []


def _sanitize_multimodal_output_tokens(
    tokens: list[int],
    *,
    tokenizer: Any | None,
    processor: Any | None = None,
) -> list[int]:
    if tokenizer is None or not tokens:
        return tokens
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return tokens
    sanitized = list(tokens)
    for token_name, warning_label in (
        ("image_token_id", "Image"),
        ("audio_token_id", "Audio"),
        ("video_token_id", "Video"),
    ):
        special_token_id = getattr(tokenizer, token_name, None)
        if special_token_id is None:
            continue
        while special_token_id in sanitized:
            index = sanitized.index(special_token_id)
            sanitized[index] = int(pad_token_id)
            logger.warning(
                "%s token found in output tokens, replaced with pad_token_id. Consider updating the model's stop "
                "condition to stop at %s if you want to avoid this.",
                warning_label,
                token_name,
            )
    if processor is not None:
        from relax.utils.data.processing_utils import sanitize_kimi_k25_response_tokens

        kimi_sanitized = sanitize_kimi_k25_response_tokens(processor, sanitized)
        if kimi_sanitized is not sanitized:
            replaced = sum(1 for old, new in zip(sanitized, kimi_sanitized, strict=True) if old != new)
            if replaced:
                logger.warning(
                    "K2.x: replaced %s stray <|media_pad|> token(s) in rollout response with pad_token_id.",
                    replaced,
                )
            sanitized = kimi_sanitized
    return sanitized


def _is_context_length_error(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code != 400:
        return False
    response_text = exc.response.text or ""
    if "maximum context length" in response_text or "Requested token count exceeds" in response_text:
        return True
    try:
        payload = json.loads(response_text)
    except Exception:
        return False
    error_message = (payload.get("error") or {}).get("message", "")
    return "maximum context length" in error_message or "Requested token count exceeds" in error_message


class SGLangBackendAdapter:
    def __init__(self, args: Any, *, compiler_resources: AgenticCompilerResources | None = None):
        router_port = args.sglang_router_port
        self._resolved_router_ip = args.sglang_router_ip
        self._resolved_router_port = None if router_port is None else int(router_port)
        self._apply_chat_template_kwargs = args.apply_chat_template_kwargs
        self._use_rollout_routing_replay = args.use_rollout_routing_replay
        self._router_policy = args.sglang_router_policy
        resources = compiler_resources or get_agentic_runtime_resources(args).compiler
        self.tokenizer = resources.tokenizer
        self.compiler = SGLangMessageCompiler(
            tokenizer=resources.tokenizer,
            processor=resources.processor,
            processor_pool=resources.processor_pool,
            apply_chat_template_kwargs=self._apply_chat_template_kwargs,
            use_audio_in_video=args.use_audio_in_video,
            cpu_executor=resources.cpu_executor,
        )

    async def generate(
        self,
        *,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
        request_id: str | None = None,
        image_data: list[str] | None = None,
        audio_data: list[str] | None = None,
        video_data: list[str] | None = None,
        return_logprob: bool = True,
    ) -> BackendGenerateResult:
        router_ip, router_port = self._resolved_router_ip, self._resolved_router_port
        if not router_ip or not router_port:
            raise RuntimeError("SGLang router address is required for agentic chat generation.")
        payload = {
            "input_ids": list(input_ids),
            "sampling_params": dict(sampling_params),
            "return_logprob": bool(return_logprob),
        }
        if request_id:
            payload["request_id"] = request_id
        if self._use_rollout_routing_replay:
            payload["return_routed_experts"] = True
        if image_data:
            payload["image_data"] = list(image_data)
        if audio_data:
            payload["audio_data"] = list(audio_data)
        if video_data:
            payload["video_data"] = list(video_data)
        headers = None
        if self._router_policy == "consistent_hashing" and session_id:
            headers = {"X-SMG-Routing-Key": session_id}
        url = f"http://{router_ip}:{router_port}/generate"
        started = time.time()
        try:
            output = await post(url, payload, headers=headers)
        except httpx.HTTPStatusError as exc:
            if _is_context_length_error(exc):
                raise BackendContextLengthExceededError(exc.response.text) from exc
            raise
        elapsed = time.time() - started
        meta_info = dict(output.get("meta_info", {}))
        new_tokens, new_log_probs = _extract_output_tokens_and_log_probs(
            meta_info,
            output_ids=output["output_ids"],
            tokenizer=self.tokenizer,
            processor=self.compiler.processor,
        )
        finish_type = str(meta_info.get("finish_reason", {}).get("type", "stop"))
        return BackendGenerateResult(
            new_tokens=new_tokens,
            new_log_probs=new_log_probs,
            finish_type=finish_type,
            meta_info=meta_info,
            elapsed=elapsed,
        )
