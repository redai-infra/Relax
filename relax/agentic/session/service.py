# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Group-affine SessionShard lifecycle and OpenAI chat routing."""

from __future__ import annotations

import asyncio
import copy
import ctypes
import hashlib
import json
import threading
import time
from argparse import Namespace
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, cast

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from ray import serve
from starlette.requests import ClientDisconnect

from relax.agentic import AGENTIC_CHAT_API_ROUTE_PREFIX, AGENTIC_CHAT_API_SERVICE_NAME
from relax.agentic.pipeline import (
    RuntimeGroupError,
    SessionExportTransport,
    SessionGroupProgress,
    SessionShardProgress,
    SessionSpec,
    TrainingFieldArtifact,
)
from relax.agentic.pipeline.runtime import (
    BackendContextLengthExceededError,
    SGLangBackendAdapter,
    finish_before_cancellation,
)
from relax.agentic.profile import (
    agentic_trace_events,
    mark_agentic_event,
    mark_agentic_event_once,
    mark_sample_agentic_event,
)
from relax.agentic.runner import (
    AgentExecutionError,
    ManagedAgentLauncher,
    ManagedAgentProcess,
    SessionInput,
    load_agent_app_spec_from_args,
)
from relax.agentic.session.admission import compute_reservation_tokens
from relax.agentic.session.admission_coordinator import AdmissionCoordinator, RayAdmissionClient
from relax.agentic.session.state import (
    InflightRequest,
    MsgNode,
    RequestKind,
    SessionForest,
    _merge_export_metadata,
    _messages_tools_template_state_hash,
    _multimodal_inputs_from_messages,
    check_messages,
    normalize_template_kwargs,
    normalize_tools,
)
from relax.utils.logging_utils import get_logger
from relax.utils.types import get_spec_token_counts


# Stable actor-name prefix embedded in opaque Session route tokens. Renaming it
# invalidates every live token and therefore requires a fleet-wide restart.
AGENTIC_SESSION_SHARD_NAME_PREFIX = "agentic_session_shard"
# Fixed fleet topology: one Serve replica per Session Shard.
# Changing it alters Ray placement and Serve capacity and requires an
# explicit deployment-level validation.
_DEFAULT_SESSION_SHARD_COUNT = 16
# Per-replica transport admission ceiling for agentic chat requests. Keep this
# wider than expected concurrent and pre-lease IR fanout so Ray Serve admission
# does not participate in Session first-request barriers. Runtime residency and
# SGLang permits remain the resource limits.
_AGENTIC_CHAT_MAX_ONGOING_REQUESTS = 9182
# Bound glibc arena growth and make free chunks eligible for malloc_trim().
# These settings mitigate long-lived Shard Private Dirty; they cannot fully
# recover fragmentation that only process exit releases.
_AGENTIC_SHARD_ALLOCATOR_ENV = {
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_TRIM_THRESHOLD_": "0",
}
# Maximum concurrent top-level Ray methods per Shard. Excess calls queue in
# Ray's actor mailbox; this is not the SGLang generation capacity. Too small a
# value can starve long-lived chat/progress RPCs, while an unnecessarily large
# value permits more live Python coroutines and memory pressure.
_AGENTIC_SHARD_MAX_CONCURRENCY = 4096
# Ray concurrency-group width for permit-acquire RPCs. Excess acquires queue;
# the actual fleet generation capacity comes from sglang_server_concurrency.
_SGLANG_PERMIT_CONCURRENCY = 1024
# Separate control lane for permit release and lifecycle RPCs. Keeping it
# independent prevents queued acquires from blocking cleanup; undersizing it
# delays capacity return and can stall the fleet.
_SGLANG_PERMIT_CONTROL_CONCURRENCY = 128
_SESSION_CLOSE_TIMEOUT_S = 2.0
_ADMISSION_SHUTDOWN_TIMEOUT_S = 5.0

# Identity sentinels encode the complete result-cell state machine without a
# parallel status field. They must remain unique objects and must never cross a
# Ray serialization boundary.
_EMPTY = object()
_DROPPED = object()
_DROPPED_TAKEN = object()
_TAKEN = object()

# Shared FastAPI schema mounted by the Ray Serve deployment. Creating multiple
# app instances would duplicate route registration and drift from the fixed
# OpenAI ingress contract.
app = FastAPI()
logger = get_logger(__name__)
_DEPLOYED_ADMISSION_COORDINATOR: Optional[Any] = None


def agentic_session_shard_name(index: int) -> str:
    """Return the stable actor name for one SessionShard."""

    return f"{AGENTIC_SESSION_SHARD_NAME_PREFIX}_{index}"


def resolve_chat_api_base_url() -> str:
    from relax.utils.utils import get_serve_url

    return f"{get_serve_url(route_prefix=AGENTIC_CHAT_API_ROUTE_PREFIX)}/"


def _resolve_fastapi_request_endpoint(func: Callable[..., Any]) -> Callable[..., Any]:
    """Let FastAPI inspect the class-bound Request endpoint correctly."""

    func.__annotations__["request"] = Request
    func.__annotations__["return"] = JSONResponse
    return func


async def _wait_process(process: ManagedAgentProcess) -> Optional[BaseException]:
    """Return process failure as data so watcher cancellation stays
    unambiguous."""

    try:
        await process.wait()
    except asyncio.CancelledError as error:
        return error
    except Exception as error:
        return error
    return None


def _extend_token_ids(target: set[int], value: Any) -> None:
    if isinstance(value, int):
        target.add(value)
    elif isinstance(value, (list, tuple, set)):
        target.update(int(item) for item in value)


def _last_token_is_stop_token(
    *,
    token_ids: list[int],
    tokenizer: Any,
    sampling_params: dict[str, Any],
) -> bool:
    if not token_ids:
        return False
    stop_token_ids: set[int] = set()
    _extend_token_ids(stop_token_ids, getattr(tokenizer, "eos_token_id", None))
    _extend_token_ids(stop_token_ids, getattr(tokenizer, "eos_token_ids", None))
    _extend_token_ids(stop_token_ids, getattr(tokenizer, "additional_stop_token_ids", None))
    _extend_token_ids(stop_token_ids, sampling_params.get("stop_token_ids"))
    return token_ids[-1] in stop_token_ids


def _backend_request_id(ir: InflightRequest) -> str:
    return f"{ir.request_id}:{ir.abort_count}"


def _openai_token_logprobs_payload(
    *,
    tokenizer: Any,
    token_ids: list[int],
    token_logprobs: list[float],
) -> dict[str, Any]:
    content = []
    for position, token_id in enumerate(token_ids):
        logprob = token_logprobs[position]
        token = str(tokenizer.decode([token_id], skip_special_tokens=False))
        content.append(
            {
                "token": token,
                "logprob": float(logprob),
                "bytes": list(token.encode("utf-8")),
                "top_logprobs": [],
            }
        )
    return {"content": content, "refusal": None}


def _decode_response_payload(
    *,
    args: Namespace,
    tokenizer: Any,
    token_ids: list[int],
    tools: list[dict[str, Any]],
    parent_state_hash: str,
) -> tuple[list[dict[str, Any]], bool]:
    if not token_ids:
        raise RuntimeError("Generation backend returned a terminal response without output tokens")
    text = str(tokenizer.decode(token_ids, skip_special_tokens=False))
    if not text:
        raise RuntimeError("Tokenizer decoded non-empty output tokens to empty text")
    reasoning_text: Optional[str] = None
    reasoning_parser_name = args.agentic_reasoning_parser
    if reasoning_parser_name:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        parsed_reasoning, parsed_text = ReasoningParser(
            model_type=str(reasoning_parser_name),
            stream_reasoning=False,
        ).parse_non_stream(text)
        reasoning_text = parsed_reasoning if isinstance(parsed_reasoning, str) and parsed_reasoning else None
        text = parsed_text if isinstance(parsed_text, str) else ""

    tool_calls: list[dict[str, Any]] = []
    tool_call_parser_name = args.agentic_tool_call_parser
    if tool_call_parser_name and tools:
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(
            [Tool.model_validate(tool) for tool in tools],
            str(tool_call_parser_name),
        )
        parsed_text, call_items = parser.parse_non_stream(text)
        text = parsed_text if isinstance(parsed_text, str) else ""
        for call_index, call_item in enumerate(call_items):
            call_payload = json.dumps(
                {
                    "call_index": call_index,
                    "parent_state_hash": parent_state_hash,
                    "token_ids": token_ids,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            tool_calls.append(
                {
                    "id": f"call_{hashlib.sha256(call_payload.encode()).hexdigest()[:24]}",
                    "type": "function",
                    "function": {
                        "name": str(call_item.name or ""),
                        "arguments": call_item.parameters,
                    },
                }
            )

    if not text and reasoning_text is None and not tool_calls:
        raise RuntimeError("Response parsers produced an empty assistant message")

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning_text:
        message["reasoning_content"] = reasoning_text
    if tool_calls:
        message["tool_calls"] = tool_calls
    return check_messages([message]), bool(tool_calls)


def _decode_routed_experts(
    *,
    args: Namespace,
    meta_info: dict[str, Any],
    token_count: int,
) -> Any:
    encoded = meta_info.get("routed_experts")
    if not encoded or token_count <= 1:
        return None
    import numpy as np
    import pybase64

    return np.frombuffer(
        pybase64.b64decode(str(encoded).encode("ascii")),
        dtype=np.int32,
    ).reshape(
        token_count - 1,
        args.num_layers,
        args.moe_router_topk,
    )


class AgenticChatRequestError(HTTPException):
    """OpenAI-compatible request failure returned through the chat gateway."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request_error",
        param: Optional[str] = None,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.message = message
        self.code = code
        self.param = param
        self.error_type = error_type


def _session_discarded_error(session_id: str) -> AgenticChatRequestError:
    return AgenticChatRequestError(
        f"Unknown or discarded agentic session {session_id!r}.",
        code="session_discarded",
        param="session_id",
        status_code=404,
        error_type="not_found_error",
    )


def _openai_error_result(
    message: str,
    *,
    code: str = "invalid_request_error",
    param: Optional[str] = None,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
) -> dict[str, Any]:
    return {
        "_http_status": status_code,
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        },
    }


def _openai_error_from_exception(error: AgenticChatRequestError) -> dict[str, Any]:
    return _openai_error_result(
        error.message,
        code=error.code,
        param=error.param,
        status_code=error.status_code,
        error_type=error.error_type,
    )


def _openai_context_length_error_result(
    *,
    max_context_len: Optional[int],
    prompt_tokens: Optional[int],
    requested_completion_tokens: Optional[int] = None,
) -> dict[str, Any]:
    if max_context_len is None:
        message = "This model's maximum context length was exceeded. Please reduce the length of the messages."
    elif prompt_tokens is None:
        message = (
            f"This model's maximum context length is {max_context_len} tokens. "
            "Please reduce the length of the messages."
        )
    elif requested_completion_tokens is None:
        message = (
            f"This model's maximum context length is {max_context_len} tokens. "
            f"However, your messages resulted in {prompt_tokens} tokens. "
            "Please reduce the length of the messages."
        )
    else:
        total_tokens = prompt_tokens + requested_completion_tokens
        message = (
            f"This model's maximum context length is {max_context_len} tokens. "
            f"However, your messages resulted in {prompt_tokens} tokens and requested "
            f"{requested_completion_tokens} completion tokens ({total_tokens} tokens total). "
            "Please reduce the length of the messages or max_completion_tokens."
        )
    return _openai_error_result(
        message,
        code="context_length_exceeded",
        param="messages",
    )


def _openai_context_length_error(*, max_context_len: int, prompt_tokens: int) -> AgenticChatRequestError:
    error = cast(
        dict[str, Any],
        _openai_context_length_error_result(
            max_context_len=max_context_len,
            prompt_tokens=prompt_tokens,
        )["error"],
    )
    return AgenticChatRequestError(
        error["message"],
        code=error["code"],
        param=error["param"],
    )


def _openai_error_response(result: dict[str, Any]) -> JSONResponse:
    status_code = int(result.get("_http_status") or 400)
    payload = {key: value for key, value in result.items() if key != "_http_status"}
    headers = {}
    error = result.get("error")
    if isinstance(error, dict) and (
        error.get("type") == "internal_error" or error.get("code") in {"internal_error", "session_discarded"}
    ):
        headers["x-should-retry"] = "false"
    return JSONResponse(payload, status_code=status_code, headers=headers)


def _session_id_from_request(request: Request) -> str:
    header = request.headers.get("Authorization")
    if header is None:
        raise AgenticChatRequestError(
            "Missing Authorization header",
            code="authentication_error",
            status_code=401,
            error_type="authentication_error",
        )
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AgenticChatRequestError(
            "Authorization header must be 'Bearer <token>'",
            code="authentication_error",
            status_code=401,
            error_type="authentication_error",
        )
    return token


def _normalized_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    def fail(message: str, *, param: Optional[str] = None) -> None:
        raise AgenticChatRequestError(message, param=param)

    if "messages" not in payload:
        fail("messages is required", param="messages")
    messages = payload["messages"]
    if not isinstance(messages, list):
        fail("messages must be a list", param="messages")

    tools = payload.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            fail("tools must be a list", param="tools")
        if any(not isinstance(tool, dict) for tool in tools):
            fail("tools entries must be JSON objects", param="tools")

    chat_template_kwargs = payload.get("chat_template_kwargs")
    if chat_template_kwargs is not None:
        if not isinstance(chat_template_kwargs, dict):
            fail("chat_template_kwargs must be a JSON object", param="chat_template_kwargs")
        reserved = sorted({"add_generation_prompt", "tokenize", "tools"}.intersection(chat_template_kwargs))
        if reserved:
            fail(
                f"chat_template_kwargs cannot set reserved keys: {', '.join(reserved)}",
                param="chat_template_kwargs",
            )

    if "stream" in payload and payload["stream"] not in {None, False}:
        fail("stream is not supported", param="stream")
    if "n" in payload and payload["n"] != 1:
        fail("n must be 1", param="n")
    requested_logprobs = payload.get("logprobs", False)
    if requested_logprobs is None:
        logprobs = False
    elif isinstance(requested_logprobs, bool):
        logprobs = requested_logprobs
    else:
        fail("logprobs must be a boolean", param="logprobs")
    if "top_logprobs" in payload and payload["top_logprobs"] is not None:
        fail("top_logprobs is not supported", param="top_logprobs")
    if "functions" in payload and payload["functions"] not in (None, []):
        fail("functions are not supported", param="functions")
    if "function_call" in payload and payload["function_call"] not in (None, "none"):
        fail("function_call is not supported", param="function_call")

    for field_name in ("max_completion_tokens", "max_tokens"):
        value = payload.get(field_name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            fail(f"{field_name} must be a positive integer", param=field_name)
    max_completion_tokens = payload.get("max_completion_tokens") or payload.get("max_tokens")

    stop = payload.get("stop")
    if (
        "stop" in payload
        and stop is not None
        and not (isinstance(stop, str) or (isinstance(stop, list) and all(isinstance(item, str) for item in stop)))
    ):
        fail("stop must be a string or list of strings", param="stop")

    try:
        messages = check_messages(messages)
    except (TypeError, ValueError) as error:
        fail(str(error), param="messages")

    return {
        "messages": messages,
        "tools": normalize_tools(tools),
        "chat_template_kwargs": normalize_template_kwargs(chat_template_kwargs),
        "model": payload.get("model"),
        "logprobs": logprobs,
        "max_completion_tokens": max_completion_tokens,
        "stop": stop,
        "seed": payload.get("seed"),
    }


class _NonFinalizableExportError(AgentExecutionError):
    """Controlled Session exit without an exportable committed response."""


class SessionPhase(Enum):
    ACTIVE = auto()
    FINALIZING = auto()


class GroupOwnership(Enum):
    PREPARE = auto()
    RUNTIME = auto()


_SessionResultPayload = SessionExportTransport | RuntimeGroupError | None


@dataclass(eq=False)
class _SessionResultCell:
    """Single-publish, single-take terminal cell for one Session."""

    payload: object = _EMPTY

    @property
    def terminal(self) -> bool:
        return self.payload is not _EMPTY

    @property
    def available(self) -> bool:
        return self.payload is not _EMPTY and self.payload is not _DROPPED_TAKEN and self.payload is not _TAKEN

    @property
    def dropped(self) -> bool:
        return self.payload is _DROPPED or self.payload is _DROPPED_TAKEN

    def publish(self, result: _SessionResultPayload) -> None:
        if self.terminal:
            raise RuntimeGroupError("session result already published")
        self.payload = _DROPPED if result is None else result

    def take(self) -> _SessionResultPayload:
        if not self.available:
            raise RuntimeGroupError("session result is not available")
        if self.payload is _DROPPED:
            self.payload = _DROPPED_TAKEN
            return None
        result = cast(_SessionResultPayload, self.payload)
        self.payload = _TAKEN
        return result


@dataclass
class SessionResources:
    """Agent process capabilities owned by one Session cleanup task."""

    process: ManagedAgentProcess
    process_wait: asyncio.Task[Optional[BaseException]]
    watcher_task: asyncio.Task[None]


@dataclass(eq=False)
class ResidentGroup:
    """Shard-local owner of ordered Sessions and their result cells."""

    rollout_mode: str
    group_id: str
    result_cells: Dict[str, _SessionResultCell]
    sessions: list["_SessionRecord"]
    ownership: GroupOwnership = GroupOwnership.PREPARE
    change_revision: int = 0
    drop_task: Optional[asyncio.Task[None]] = None

    @property
    def first_request_barrier_crossed(self) -> bool:
        return (
            self.ownership is GroupOwnership.PREPARE
            and self.first_error() is None
            and not self.dropped
            and all(session.phase is SessionPhase.ACTIVE and session.live_irs for session in self.sessions)
        )

    @property
    def interrupted(self) -> bool:
        if self.ownership is GroupOwnership.PREPARE or self.first_error() is not None or self.dropped:
            return False
        unfinished = sum(not cell.terminal for cell in self.result_cells.values())
        return unfinished > 0 and unfinished == sum(session.interrupted for session in self.sessions)

    @property
    def protected(self) -> bool:
        return any(session.protected_until_finalize for session in self.sessions)

    @property
    def terminal(self) -> bool:
        return all(cell.terminal for cell in self.result_cells.values())

    def first_error(self) -> Optional[RuntimeGroupError]:
        for cell in self.result_cells.values():
            if isinstance(cell.payload, RuntimeGroupError):
                return cell.payload
        return None

    @property
    def dropped(self) -> bool:
        return any(cell.dropped for cell in self.result_cells.values())


@dataclass(eq=False)
class _SessionRecord:
    """Shard-local owner of one Session's Forest, IR, process, and result."""

    group: ResidentGroup
    session_id: str
    session_sampling_params: Dict[str, Any]
    result_cell: _SessionResultCell
    rollout_id: int = 0
    next_ir_sequence: int = 0
    forest: Optional[SessionForest] = None
    phase: SessionPhase = SessionPhase.ACTIVE
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    live_irs: set[InflightRequest] = field(default_factory=set)
    queued_irs: deque[InflightRequest] = field(default_factory=deque)
    resp_state_hash_by_request_id: Dict[str, str] = field(default_factory=dict)
    resources: Optional[SessionResources] = None
    finish_task: Optional["asyncio.Task[Optional[BaseException]]"] = None
    protection_pending_until_resume: bool = False
    protected_until_finalize: bool = False

    @property
    def interrupted(self) -> bool:
        """Report a wholly parked Session carrying a real backend prefix."""

        return (
            self.phase is SessionPhase.ACTIVE
            and not self.protected_until_finalize
            and bool(self.live_irs)
            and all(ir.runner_task is None and ir in self.queued_irs for ir in self.live_irs)
            and any(ir.pending_status == "aborted" and ir.abort_count > 0 for ir in self.live_irs)
        )


@ray.remote(
    max_concurrency=_AGENTIC_SHARD_MAX_CONCURRENCY,
    concurrency_groups={
        "sglang_request_permit": _SGLANG_PERMIT_CONCURRENCY,
        "sglang_request_control": _SGLANG_PERMIT_CONTROL_CONCURRENCY,
    },
)
class AgenticSessionShard:
    """Group-affine owner of every live Session, IR, process, and barrier."""

    def __init__(
        self,
        args: Namespace,
        sglang_request_capacity: Optional[int],
        sglang_request_limiter: Optional[Any],
        admission_coordinator: Optional[Any],
    ) -> None:
        self.args = args
        self._groups: Dict[str, ResidentGroup] = {}
        # OpenAI ingress index. ResidentGroup.sessions owns each Session's
        # lifecycle; this table resolves a route token to that direct ref.
        self._session_records: Dict[str, _SessionRecord] = {}
        self._train_generation_open = False
        self._state_revision = 0
        self._state_changed = asyncio.Event()
        self._sglang_request_semaphore = (
            threading.BoundedSemaphore(sglang_request_capacity) if sglang_request_capacity is not None else None
        )
        self._sglang_request_limiter = sglang_request_limiter
        self._admission_client = (
            RayAdmissionClient(admission_coordinator) if admission_coordinator is not None else None
        )
        self._permit_cleanup_tasks: set["asyncio.Task[None]"] = set()
        self._lifecycle_close_count = 0
        self._lifecycle_close_failure_count = 0
        self._generation_backend = SGLangBackendAdapter(args)
        self._agent_launcher = ManagedAgentLauncher(
            load_agent_app_spec_from_args(args),
            resolve_chat_api_base_url(),
        )

    @ray.method(concurrency_group="sglang_request_permit")
    async def acquire_sglang_request_permit(self) -> None:
        """Queue one backend attempt on the fleet-global limit."""

        semaphore = cast(threading.BoundedSemaphore, self._sglang_request_semaphore)
        while not semaphore.acquire(blocking=False):
            await asyncio.sleep(0.01)

    @ray.method(concurrency_group="sglang_request_control")
    async def release_sglang_request_permit(self) -> None:
        """Return one backend-attempt permit to the fleet."""

        semaphore = cast(threading.BoundedSemaphore, self._sglang_request_semaphore)
        semaphore.release()

    def _notify_state_change(self, group: Optional[ResidentGroup] = None) -> None:
        """Advance the Shard event cursor after owned refs change."""

        self._state_revision += 1
        if group is not None:
            group.change_revision = self._state_revision
        self._state_changed.set()

    def _current_progress(self, known_revision: int) -> SessionShardProgress:
        """Project progress from Groups changed after one cursor."""

        groups = tuple(group for group in self._groups.values() if group.change_revision > known_revision)
        return SessionShardProgress(
            revision=self._state_revision,
            groups=tuple(
                SessionGroupProgress(
                    group_id=group.group_id,
                    ready_for_lease=group.first_request_barrier_crossed,
                    takeable_session_ids=tuple(
                        session_id for session_id, cell in group.result_cells.items() if cell.available
                    ),
                    interrupted=group.interrupted,
                    protected=group.protected,
                )
                for group in groups
            ),
        )

    async def wait_for_state_change(self, known_revision: int) -> SessionShardProgress:
        """Wait for one Shard change and return a ref-derived view.

        One Runtime observer holds this RPC for each Shard. The number of live
        Groups, Sessions, and IRs therefore does not create more waiting actor
        methods.
        """

        while self._state_revision == known_revision:
            self._state_changed.clear()
            if self._state_revision != known_revision:
                break
            await self._state_changed.wait()
        return self._current_progress(known_revision)

    async def wake_state_change_waiters(self) -> None:
        """Release Runtime observers that are ending their borrowed
        lifecycle."""

        self._notify_state_change()

    async def start_group(
        self,
        rollout_mode: str,
        group_id: str,
        session_specs: Tuple[SessionSpec, ...],
    ) -> None:
        """Register one group and start every Session agent."""

        if group_id in self._groups:
            raise RuntimeGroupError(f"group token is already resident: {group_id}")

        result_cells = {session_spec.session_id: _SessionResultCell() for session_spec in session_specs}
        group = ResidentGroup(
            rollout_mode=rollout_mode,
            group_id=group_id,
            result_cells=result_cells,
            sessions=[],
        )
        for sample_position, session_spec in enumerate(session_specs):
            session_id = session_spec.session_id
            group.sessions.append(
                _SessionRecord(
                    group=group,
                    session_id=session_id,
                    session_sampling_params=self._session_sampling_params(session_spec, sample_position),
                    result_cell=result_cells[session_id],
                    forest=SessionForest.create_empty(
                        session_id=session_id,
                        group_index=session_spec.group_index,
                        index=session_spec.index,
                        label=session_spec.label,
                        train_metadata=session_spec.train_metadata,
                        metadata=session_spec.metadata,
                    ),
                )
            )
        self._groups[group_id] = group
        self._session_records.update((session.session_id, session) for session in group.sessions)

        try:
            for session, session_spec in zip(group.sessions, session_specs, strict=True):
                self._start_session(
                    session,
                    SessionInput(
                        session_id=session.session_id,
                        group_id=group_id,
                        rollout_mode=rollout_mode,
                        input_payload=session_spec.input_payload,
                    ),
                )
        except BaseException:
            await self.drop_group(group_id)
            raise

    def _session_sampling_params(
        self,
        session_spec: SessionSpec,
        sample_position: int,
    ) -> Dict[str, Any]:
        """Apply sampling precedence before the first OpenAI request."""

        sampling_params = {
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
            sampling_params["sampling_seed"] = self.args.rollout_seed + sample_position
        if session_spec.sampling_params is not None:
            sampling_params.update(session_spec.sampling_params)
        return {key: value for key, value in sampling_params.items() if value is not None}

    @staticmethod
    def _accumulate_request_meta(request, *, meta_info: dict[str, Any]) -> None:
        weight_version = meta_info.get("weight_version")
        if weight_version is not None:
            request.pending_weight_version_delta.append(str(weight_version))
        spec_accept_token_num, spec_draft_token_num = get_spec_token_counts(meta_info)
        request.pending_spec_delta["spec_accept_token_num"] += spec_accept_token_num
        request.pending_spec_delta["spec_draft_token_num"] += spec_draft_token_num
        request.pending_spec_delta["spec_verify_ct"] += int(meta_info.get("spec_verify_ct", 0) or 0)
        request.pending_spec_delta["completion_token_num"] += int(meta_info.get("completion_tokens", 0) or 0)
        request.pending_prefix_cache_delta["cached_tokens"] += int(meta_info.get("cached_tokens", 0) or 0)
        request.pending_prefix_cache_delta["total_prompt_tokens"] += int(meta_info.get("prompt_tokens", 0) or 0)

    async def lease_group(self, group_id: str, rollout_id: int) -> None:
        """Move one prepared group ref into Runtime and release its IR gate."""

        group = self._groups[group_id]
        self._raise_group_error(group)
        if group.dropped:
            raise RuntimeGroupError(f"cannot lease group that failed during Prepare: {group.group_id}")
        if not group.first_request_barrier_crossed:
            raise RuntimeGroupError(f"cannot lease group before every first IR exists: {group.group_id}")
        for session in group.sessions:
            async with session.lock:
                session.rollout_id = rollout_id
        group.ownership = GroupOwnership.RUNTIME
        await self._set_group_timeout_active(group, self._group_generation_open(group))
        await self._open_group(group)

    def _group_generation_open(self, group: ResidentGroup) -> bool:
        """Derive gate state from the Runtime-owned rollout mode."""

        return group.rollout_mode != "train" or self._train_generation_open

    def _max_context_len(self, group: ResidentGroup) -> Optional[int]:
        return self.args.eval_max_context_len if group.rollout_mode == "eval" else self.args.rollout_max_context_len

    async def _open_group(self, group: ResidentGroup) -> None:
        """Release a prepared group when the resident rollout step is open."""

        self._raise_group_error(group)
        if group.ownership is not GroupOwnership.RUNTIME or group.dropped or group.terminal:
            return
        if self._group_generation_open(group):
            for session in tuple(group.sessions):
                if session.result_cell.terminal:
                    continue
                async with session.lock:
                    if not self._group_generation_open(group) or session.phase is not SessionPhase.ACTIVE:
                        continue
                    self._dispatch_queued_irs_locked(session)
        # Scheduling or gating changes the Group's fully-async close credit.
        self._notify_state_change(group)

    async def pause_generation(self) -> int:
        """Gate train generation and park requests actually aborted."""

        self._train_generation_open = False
        try:
            await self._interrupt_running_requests()
        finally:
            await self._set_runtime_timeouts_active(False)
        self._notify_state_change()
        return self._state_revision

    async def _interrupt_running_requests(self) -> None:
        """Ask selected backend attempts to return their generated prefix."""

        runner_tasks: list[asyncio.Task[None]] = []
        backend_request_ids: list[str] = []
        try:
            for group in tuple(self._groups.values()):
                self._raise_group_error(group)
                if (
                    group.rollout_mode != "train"
                    or group.ownership is GroupOwnership.PREPARE
                    or group.dropped
                    or group.terminal
                ):
                    continue
                for session in tuple(group.sessions):
                    if session.result_cell.terminal:
                        continue
                    async with session.lock:
                        if session.phase is not SessionPhase.ACTIVE:
                            continue
                        if session.protected_until_finalize:
                            continue
                        requeued_irs: list[InflightRequest] = []
                        for ir in session.live_irs:
                            runner = ir.runner_task
                            if runner is None:
                                continue
                            if not ir.backend_started:
                                ir.runner_task = None
                                requeued_irs.append(ir)
                                runner.cancel()
                                runner_tasks.append(runner)
                                continue
                            backend_request_ids.append(_backend_request_id(ir))
                            runner_tasks.append(runner)
                        session.queued_irs.extendleft(reversed(requeued_irs))
                self._notify_state_change(group)
        finally:
            abort_outcomes = await asyncio.gather(
                *(self._generation_backend.abort_request(request_id) for request_id in backend_request_ids),
                return_exceptions=True,
            )
            await asyncio.gather(*runner_tasks, return_exceptions=True)
            for outcome in abort_outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome

    async def resume_generation(self, rollout_id: int) -> int:
        """Open the step and release prepared or interrupted Runtime groups."""

        # Keep the gate closed until every resident Session carries this step's
        # rollout ID. A concurrent OpenAI request therefore cannot start with
        # the preceding step's ID.
        for group in tuple(self._groups.values()):
            if group.rollout_mode == "train" and group.ownership is GroupOwnership.RUNTIME:
                for session in tuple(group.sessions):
                    async with session.lock:
                        if session.phase is SessionPhase.ACTIVE:
                            session.rollout_id = rollout_id
                            if session.protection_pending_until_resume:
                                session.protection_pending_until_resume = False
                                session.protected_until_finalize = True
                                for ir in session.live_irs:
                                    ir.kind = RequestKind.PROTECTED

        runtime_groups = tuple(
            group
            for group in self._groups.values()
            if group.rollout_mode == "train" and group.ownership is GroupOwnership.RUNTIME
        )
        await asyncio.gather(*(self._set_group_timeout_active(group, True) for group in runtime_groups))
        self._train_generation_open = True
        for group in runtime_groups:
            await self._open_group(group)
        self._notify_state_change()
        return self._state_revision

    async def _set_runtime_timeouts_active(self, active: bool) -> None:
        """Set timeout activity for every Runtime-owned group."""

        await asyncio.gather(
            *(
                self._set_group_timeout_active(group, active)
                for group in self._groups.values()
                if group.rollout_mode == "train" and group.ownership is GroupOwnership.RUNTIME
            )
        )

    @staticmethod
    async def _set_group_timeout_active(group: ResidentGroup, active: bool) -> None:
        """Set timeout activity through direct process capabilities."""

        await asyncio.gather(
            *(
                session.resources.process.set_timeout_active(active)
                for session in group.sessions
                if session.phase is SessionPhase.ACTIVE
                and session.resources is not None
                and not session.result_cell.terminal
                and (active or not session.protected_until_finalize)
            )
        )

    async def take_session_results(
        self,
        group_id: str,
        session_ids: Tuple[str, ...],
    ) -> Tuple[_SessionResultPayload, ...]:
        """Move announced Session payloads out in one group RPC.

        The event stream carries lightweight Session tokens. Batching their
        direct returns avoids both per-Session Ray calls and nested ``ray.put``
        ObjectRefs while each actor-local cell releases its heavy payload.
        """

        group = self._groups[group_id]
        return tuple(group.result_cells[session_id].take() for session_id in session_ids)

    async def release_group(self, group_id: str) -> None:
        """Release a remotely addressed group after all Session results are
        available."""

        group = self._groups[group_id]
        self._raise_group_error(group)
        if not group.terminal:
            raise RuntimeGroupError(f"cannot release unfinished group: {group_id}")
        self._groups.pop(group_id)
        self._notify_state_change()

    async def drop_group(self, group_id: str) -> None:
        """Discard one group through an idempotent cleanup path."""

        group = self._groups.get(group_id)
        if group is None:
            # Driver cancellation may retry after the remote release committed.
            return
        if group.drop_task is None:
            group.drop_task = asyncio.create_task(
                self._drop_group(group),
                name=f"drop-group:{group.group_id}",
            )
        await finish_before_cancellation(
            group.drop_task,
            f"drop-group:{group.group_id}",
        )

    async def chat(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any],
        max_completion_tokens: Optional[int],
        stop: Optional[list[str] | str],
        seed: Optional[int],
        logprobs: bool = False,
    ) -> dict[str, Any]:
        """Resolve one HTTP route token to the actor-local Session ref."""

        try:
            session = self._session_records.get(session_id)
            if session is None:
                raise _session_discarded_error(session_id)
            return await self._chat(
                session=session,
                messages=messages,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
                max_completion_tokens=max_completion_tokens,
                stop=stop,
                seed=seed,
                logprobs=logprobs,
            )
        except AgenticChatRequestError as error:
            return _openai_error_from_exception(error)

    async def mark_chat_service_response_ready(
        self,
        *,
        session_id: str,
        request_id: str,
        remote_return_at: float,
        response_ready_at: float,
        http_return_at: float,
    ) -> bool:
        """Complete the HTTP-return trace on its response node."""

        session = self._session_records.get(session_id)
        if session is None:
            return False
        async with session.lock:
            forest = session.forest
            state_hash = session.resp_state_hash_by_request_id.get(request_id)
            if forest is None or state_hash is None:
                return False
            response_node = forest.nodes_by_hash.get(state_hash)
            if response_node is None:
                return False
            events = agentic_trace_events(response_node.export_metadata_patch)
            mark_agentic_event_once(events, "chat_service_remote_return_at", remote_return_at)
            mark_agentic_event_once(events, "chat_service_response_ready_at", response_ready_at)
            mark_agentic_event_once(events, "chat_service_http_return_at", http_return_at)
            return True

    async def shutdown(self) -> None:
        """Shut down the Shard through normal group cleanup."""

        await finish_before_cancellation(self._shutdown(), "session-shard-shutdown")

    @ray.method(concurrency_group="sglang_request_control")
    async def trim_memory(self) -> Dict[str, Any]:
        """Run allocator maintenance outside Session control flow."""

        active_sessions = sum(len(group.sessions) for group in self._groups.values())
        active_requests = sum(len(session.live_irs) for group in self._groups.values() for session in group.sessions)
        try:
            trimmed = int(ctypes.CDLL("libc.so.6").malloc_trim(0))
        except Exception as error:
            return {
                "ok": False,
                "error": str(error)[:500],
                "active_sessions": active_sessions,
                "active_requests": active_requests,
            }
        return {
            "ok": True,
            "trimmed": trimmed,
            "active_sessions": active_sessions,
            "active_requests": active_requests,
        }

    @ray.method(concurrency_group="sglang_request_control")
    async def health(self) -> dict[str, Any]:
        """Project liveness from the Shard's current Session refs."""

        sessions = tuple(self._session_records.values())
        return {
            "ok": True,
            "active_sessions": len(sessions),
            "active_requests": sum(len(session.live_irs) for session in sessions),
            "forest_nodes": sum(
                len(session.forest.nodes_by_hash) for session in sessions if session.forest is not None
            ),
            "ir_queue": {"queued": sum(len(session.queued_irs) for session in sessions)},
        }

    @ray.method(concurrency_group="sglang_request_control")
    async def debug_state(self, *, sample_limit: int = 8) -> dict[str, Any]:
        """Project actor-local refs into a debug dictionary."""

        groups = tuple(self._group_debug_state(group) for group in self._groups.values())
        session_ids = tuple(self._session_records)
        return {
            "train_generation_open": self._train_generation_open,
            "groups": groups,
            "group_ids": tuple(group["group_id"] for group in groups),
            "group_count": len(groups),
            "session_ids": session_ids[:sample_limit],
            "session_count": len(session_ids),
            "retained_request_count": sum(group["retained_request_count"] for group in groups),
            "backend_request_count": sum(group["backend_request_count"] for group in groups),
            "live_process_count": sum(group["live_process_count"] for group in groups),
            "interrupted_group_ids": tuple(group["group_id"] for group in groups if group["interrupted"]),
            "permit_cleanup_task_count": len(self._permit_cleanup_tasks),
        }

    @ray.method(concurrency_group="sglang_request_control")
    async def agentic_kv_metrics(self, reset: bool = False) -> dict[str, float]:
        if not self.args.agentic_session_lifecycle:
            return {}
        metrics = {
            "session/lifecycle_enabled": 1.0,
            "session/close": float(self._lifecycle_close_count),
            "session/close_failure": float(self._lifecycle_close_failure_count),
        }
        if reset:
            self._lifecycle_close_count = 0
            self._lifecycle_close_failure_count = 0
        return metrics

    def _group_debug_state(self, group: ResidentGroup) -> dict[str, Any]:
        """Project one group from owned refs."""

        sessions = tuple(group.sessions)
        return {
            "rollout_mode": group.rollout_mode,
            "group_id": group.group_id,
            "generation_open": self._group_generation_open(group),
            "ownership": group.ownership,
            "live_session_count": len(sessions),
            "published_session_result_count": sum(cell.terminal for cell in group.result_cells.values()),
            "retained_request_count": sum(len(session.live_irs) for session in sessions),
            "backend_request_count": sum(ir.backend_started for session in sessions for ir in session.live_irs),
            "live_process_count": sum(session.resources is not None for session in sessions),
            "interrupted_session_count": sum(session.interrupted for session in sessions),
            "ready_for_lease": group.ownership is GroupOwnership.PREPARE and group.first_request_barrier_crossed,
            "interrupted": group.interrupted,
            "terminal": group.terminal,
        }

    async def _shutdown(self) -> None:
        """Drop resident groups, then release the Shard-owned backend."""

        self._train_generation_open = False
        first_error: Optional[BaseException] = None
        for group in tuple(self._groups.values()):
            try:
                await self.drop_group(group.group_id)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        permit_cleanup_outcomes = await asyncio.gather(
            *tuple(self._permit_cleanup_tasks),
            return_exceptions=True,
        )
        self._permit_cleanup_tasks.clear()
        for outcome in permit_cleanup_outcomes:
            if first_error is None and isinstance(outcome, BaseException):
                first_error = outcome
        try:
            await self._generation_backend.shutdown()
        except BaseException as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error

    async def _chat(
        self,
        *,
        session: _SessionRecord,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any],
        max_completion_tokens: Optional[int],
        stop: Optional[list[str] | str],
        seed: Optional[int],
        logprobs: bool,
    ) -> dict[str, Any]:
        """Commit one full OpenAI request to the Forest before scheduling
        it."""

        chat_request_arrive_at = time.time()
        wall_started_at = time.monotonic()
        group = session.group
        chat_template_kwargs = normalize_template_kwargs(
            {
                **self._generation_backend.compiler.apply_chat_template_kwargs,
                **chat_template_kwargs,
            }
        )
        async with session.lock:
            chat_lock_acquired_at = time.time()
            if session.phase is not SessionPhase.ACTIVE:
                raise _session_discarded_error(session.session_id)
            forest = cast(SessionForest, session.forest)
            sampling_params = copy.deepcopy(session.session_sampling_params)
            if max_completion_tokens is not None:
                sampling_params["max_new_tokens"] = int(max_completion_tokens)
            if stop is not None:
                sampling_params["stop"] = stop
            if seed is not None:
                sampling_params["sampling_seed"] = int(seed)
            parent, observation = self._match_parent_state(
                forest=forest,
                messages=messages,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            )
            generation_parent, compiler_timing = await self._append_observation(
                session=session,
                forest=forest,
                parent=parent,
                messages_delta=observation,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            )
            kind = forest.resolve_request_kind(
                abort_count=generation_parent.abort_count,
                protected_abort_count_threshold=(
                    self.args.partial_rollout_max_aborted_count if self.args.partial_rollout else None
                ),
            )
            if session.protected_until_finalize:
                kind = RequestKind.PROTECTED
            prefix = forest.build_execution_prefix(generation_parent.state_hash)
            max_context_len = self._max_context_len(group)
            if max_context_len is not None:
                prompt_tokens = len(prefix.train_token_prefix)
                context_budget = int(max_context_len) - prompt_tokens
                if context_budget <= 0:
                    raise _openai_context_length_error(
                        max_context_len=int(max_context_len),
                        prompt_tokens=prompt_tokens,
                    )
                if int(sampling_params["max_new_tokens"]) > context_budget:
                    sampling_params["max_new_tokens"] = context_budget
            request_id = f"req_{session.session_id}_{session.next_ir_sequence}"
            session.next_ir_sequence += 1
            waiter = asyncio.get_running_loop().create_future()
            ir = InflightRequest(
                request_id=request_id,
                parent_state_hash=generation_parent.state_hash,
                rollout_id=session.rollout_id,
                kind=kind,
                abort_count=generation_parent.abort_count,
                waiter=waiter,
                wall_started_at=wall_started_at,
                sampling_params=sampling_params,
                logprobs=logprobs,
                history_train_token_prefix=prefix.train_token_prefix,
                history_rollout_token_prefix=prefix.rollout_token_prefix,
                history_backend_image_data=prefix.backend_image_data,
                history_backend_audio_data=prefix.backend_audio_data,
                history_backend_video_data=prefix.backend_video_data,
            )
            profile = agentic_trace_events(ir.pending_export_metadata_patch)
            profile.update(copy.deepcopy(compiler_timing))
            mark_agentic_event(profile, "chat_request_arrive_at", chat_request_arrive_at)
            mark_agentic_event(profile, "chat_lock_acquired_at", chat_lock_acquired_at)
            mark_agentic_event(profile, "ir_created_at")
            session.live_irs.add(ir)
            session.queued_irs.append(ir)
            self._dispatch_queued_irs_locked(session)
        self._notify_state_change(group)
        try:
            return await asyncio.shield(ir.waiter)
        except AgenticChatRequestError as error:
            return _openai_error_from_exception(error)
        except Exception:
            return _openai_error_result(
                "Internal error while handling agentic chat request.",
                code="internal_error",
                status_code=500,
                error_type="internal_error",
            )

    @staticmethod
    def _match_parent_state(
        *,
        forest: SessionForest,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any],
    ) -> tuple[MsgNode, list[dict[str, Any]]]:
        for prefix_length in range(len(messages), 0, -1):
            prefix_hash = _messages_tools_template_state_hash(
                messages[:prefix_length],
                tools,
                chat_template_kwargs,
            )
            parent = forest.nodes_by_hash.get(prefix_hash)
            if parent is not None:
                return parent, messages[prefix_length:]
        root_state_hash = forest.root_state_hash
        assert root_state_hash is not None
        return forest.nodes_by_hash[root_state_hash], messages

    async def _append_observation(
        self,
        *,
        session: _SessionRecord,
        forest: SessionForest,
        parent: MsgNode,
        messages_delta: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any],
    ) -> tuple[MsgNode, dict[str, float]]:
        if not messages_delta:
            return parent, {}
        if parent.state_hash == forest.root_state_hash:
            encoded = await self._generation_backend.compiler.encode_messages(
                messages_delta,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
                multimodal_inputs=_multimodal_inputs_from_messages(messages_delta),
            )
            abort_count = 0
            node_tools = tools
            node_chat_template_kwargs = chat_template_kwargs
        else:
            subtree_root = forest.subtree_root_node(parent.state_hash)
            assert subtree_root is not None
            if tools != subtree_root.tools:
                raise RuntimeGroupError(f"tools changed inside session subtree: {session.session_id}")
            if chat_template_kwargs != subtree_root.chat_template_kwargs:
                raise RuntimeGroupError(f"chat template changed inside session subtree: {session.session_id}")
            encoded = await self._generation_backend.compiler.encode_observation_delta(
                messages_delta,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
                multimodal_inputs=_multimodal_inputs_from_messages(messages_delta),
            )
            abort_count = parent.abort_count
            node_tools = None
            node_chat_template_kwargs = None
        self._raise_if_observation_exceeds_context(
            session=session,
            forest=forest,
            parent=parent,
            observation_train_tokens=encoded.train_prompt_ids,
        )
        node = forest.append_obs(
            parent_state_hash=parent.state_hash,
            rollout_id=session.rollout_id,
            abort_count=abort_count,
            messages_delta=messages_delta,
            train_token_delta=encoded.train_prompt_ids,
            rollout_token_delta=encoded.backend_prompt_ids,
            multimodal_train_inputs_delta=encoded.multimodal_train_inputs,
            backend_image_data_delta=encoded.backend_image_data,
            backend_audio_data_delta=encoded.backend_audio_data,
            backend_video_data_delta=encoded.backend_video_data,
            tools=node_tools,
            chat_template_kwargs=node_chat_template_kwargs,
        )
        return node, dict(encoded.timing)

    def _raise_if_observation_exceeds_context(
        self,
        *,
        session: _SessionRecord,
        forest: SessionForest,
        parent: MsgNode,
        observation_train_tokens: list[int],
    ) -> None:
        max_context_len = self._max_context_len(session.group)
        if max_context_len is None:
            return
        prompt_tokens = forest.train_token_count(parent.state_hash) + len(observation_train_tokens)
        if prompt_tokens >= int(max_context_len):
            raise _openai_context_length_error(
                max_context_len=int(max_context_len),
                prompt_tokens=prompt_tokens,
            )

    def _start_session(self, session: _SessionRecord, session_input: SessionInput) -> None:
        """Attach a record-owned process to one Session."""

        process = self._agent_launcher.start_agent(session_input)
        process_wait = asyncio.create_task(
            _wait_process(process),
            name=f"agent-process-wait:{session.session_id}",
        )
        watcher_task = asyncio.create_task(
            self._watch_agent_process(session, process_wait),
            name=f"agent-process-watch:{session.session_id}",
        )
        session.resources = SessionResources(
            process=process,
            process_wait=process_wait,
            watcher_task=watcher_task,
        )

    @staticmethod
    def _raise_group_error(group: ResidentGroup) -> None:
        """Surface Session export errors at each group control edge."""

        error = group.first_error()
        if error is not None:
            raise error

    def _dispatch_queued_irs_locked(self, session: _SessionRecord) -> None:
        """Dispatch every queued IR allowed by the Session gate.

        Drain the Session's queued IRs by traversing direct refs. A runner task
        records dispatch; backend execution begins later at the generation
        boundary.
        """

        if session.phase is not SessionPhase.ACTIVE:
            return
        group = session.group
        if group.ownership is not GroupOwnership.RUNTIME:
            return
        if not self._group_generation_open(group) and not session.protected_until_finalize:
            return
        while session.queued_irs:
            ir = session.queued_irs.popleft()
            remaining_tokens = int(ir.sampling_params["max_new_tokens"]) - len(ir.pending_token_delta)
            if remaining_tokens == 0:
                # Reactivate every aborted IR, then reject a zero remaining
                # budget before another backend call or Forest commit.
                session.live_irs.remove(ir)
                ir.waiter.set_result(
                    _openai_context_length_error_result(
                        max_context_len=self._max_context_len(group),
                        prompt_tokens=len(ir.history_train_token_prefix) + len(ir.pending_token_delta),
                    )
                )
                continue
            ir.rollout_id = session.rollout_id
            ir.backend_started = False
            profile = agentic_trace_events(ir.pending_export_metadata_patch)
            mark_agentic_event(profile, "ir_activated_at")
            mark_agentic_event(profile, "generation_queue_enter_at")
            ir.runner_task = asyncio.create_task(
                self._run_ir(session, ir),
                name=f"chat:{session.session_id}:{ir.request_id}",
            )

    def _fail_ir_locked(
        self,
        session: _SessionRecord,
        ir: InflightRequest,
        error: Exception,
    ) -> None:
        """Release one failed IR, wake its caller, and continue sibling IRs."""

        session.live_irs.remove(ir)
        ir.runner_task = None
        ir.waiter.set_exception(error)
        self._dispatch_queued_irs_locked(session)

    async def _acquire_sglang_request_permit(self) -> None:
        """Acquire the fleet-global permit before backend entry."""

        try:
            if self._sglang_request_semaphore is not None:
                await self.acquire_sglang_request_permit()
                return

            acquire_ref = self._sglang_request_limiter.acquire_sglang_request_permit.remote()
            try:
                await asyncio.shield(acquire_ref)
            except asyncio.CancelledError:
                cleanup_task = asyncio.create_task(
                    self._release_sglang_request_permit_after_cancel(acquire_ref),
                    name="sglang-permit-cancel",
                )
                self._permit_cleanup_tasks.add(cleanup_task)
                cleanup_task.add_done_callback(self._discard_successful_permit_cleanup)
                raise
        except Exception as error:
            raise RuntimeGroupError(f"SGLang permit acquire failed: {type(error).__name__}: {error}") from error

    async def _release_sglang_request_permit(self) -> None:
        """Return one acquired permit before forwarding cancellation."""

        try:
            release = (
                self.release_sglang_request_permit()
                if self._sglang_request_semaphore is not None
                else self._sglang_request_limiter.release_sglang_request_permit.remote()
            )
            await finish_before_cancellation(release, "sglang-permit-release")
        except Exception as error:
            raise RuntimeGroupError(f"SGLang permit release failed: {type(error).__name__}: {error}") from error

    async def _release_sglang_request_permit_after_cancel(self, acquire_ref: Any) -> None:
        """Return a remote permit acquired after runner cancellation."""

        try:
            await acquire_ref
        except Exception as error:
            raise RuntimeGroupError(f"SGLang permit acquire failed: {type(error).__name__}: {error}") from error
        await self._release_sglang_request_permit()

    def _discard_successful_permit_cleanup(self, task: "asyncio.Task[None]") -> None:
        """Release successful compensation refs; retain failures."""

        if not task.cancelled() and task.exception() is None:
            self._permit_cleanup_tasks.discard(task)

    @asynccontextmanager
    async def _admission_lease(self, session: _SessionRecord, ir: InflightRequest):
        """Hold one execution-token lease for the current backend attempt."""

        lease_id = None
        admission_client = self._admission_client
        scope_allowed = self.args.agentic_admission_scope == "all" or session.group.rollout_mode == "train"
        if admission_client is not None and scope_allowed:
            remaining_tokens = int(ir.sampling_params["max_new_tokens"]) - len(ir.pending_token_delta)
            prompt_tokens = len(ir.history_train_token_prefix) + len(ir.pending_token_delta)
            expected_decode_cap = self.args.agentic_admission_expected_decode_cap
            if expected_decode_cap is not None:
                remaining_tokens = min(remaining_tokens, int(expected_decode_cap))
            ticket_id = _backend_request_id(ir)
            decision = await admission_client.acquire(
                {
                    "ticket_id": ticket_id,
                    "reservation_tokens": compute_reservation_tokens(
                        prompt_tokens=prompt_tokens,
                        remaining_completion_tokens=remaining_tokens,
                    ),
                    "protected": ir.kind is RequestKind.PROTECTED or session.protected_until_finalize,
                }
            )
            lease_id = decision.get("lease_id")
        try:
            yield
        finally:
            if lease_id is not None and admission_client is not None:
                await finish_before_cancellation(
                    admission_client.release(lease_id),
                    "agentic-admission-lease-release",
                )

    @asynccontextmanager
    async def _sglang_request_permit(self):
        """Hold one fleet request permit around backend execution."""

        await self._acquire_sglang_request_permit()
        try:
            yield
        finally:
            await self._release_sglang_request_permit()

    async def _run_ir(
        self,
        session: _SessionRecord,
        ir: InflightRequest,
    ) -> None:
        """Run one IR while keeping all partial deltas on that ref."""

        group = session.group
        runner_task = asyncio.current_task()

        def is_current() -> bool:
            return session.phase is SessionPhase.ACTIVE and ir in session.live_irs and ir.runner_task is runner_task

        async with session.lock:
            if not is_current():
                return
            if not self._group_generation_open(group) and not session.protected_until_finalize:
                ir.runner_task = None
                session.queued_irs.appendleft(ir)
                self._notify_state_change(group)
                return

        try:
            async with self._admission_lease(session, ir):
                async with self._sglang_request_permit():
                    async with session.lock:
                        if not is_current():
                            return
                        if not self._group_generation_open(group) and not session.protected_until_finalize:
                            ir.runner_task = None
                            session.queued_irs.appendleft(ir)
                            self._notify_state_change(group)
                            return
                        backend_request_id = _backend_request_id(ir)
                        ir.backend_started = True
                        remaining_tokens = int(ir.sampling_params["max_new_tokens"]) - len(ir.pending_token_delta)
                        mark_agentic_event(
                            agentic_trace_events(ir.pending_export_metadata_patch),
                            "generation_start_at",
                        )

                    try:
                        result = await self._generation_backend.generate(
                            input_ids=ir.history_rollout_token_prefix + ir.pending_token_delta,
                            sampling_params={**ir.sampling_params, "max_new_tokens": remaining_tokens},
                            session_id=session.session_id,
                            request_id=backend_request_id,
                            image_data=ir.history_backend_image_data,
                            audio_data=ir.history_backend_audio_data,
                            video_data=ir.history_backend_video_data,
                            return_logprob=group.rollout_mode == "train" or ir.logprobs,
                        )
                    finally:
                        mark_agentic_event(
                            agentic_trace_events(ir.pending_export_metadata_patch),
                            "generation_end_at",
                        )
        except asyncio.CancelledError:
            async with session.lock:
                expected_runtime_cancel = session.phase is not SessionPhase.ACTIVE or ir.runner_task is not runner_task
                if not expected_runtime_cancel:
                    ir.runner_task = None
                    ir.backend_started = False
            if not expected_runtime_cancel:
                await self._handle_infra_failure(
                    session,
                    RuntimeGroupError(f"chat runner cancelled itself for {session.session_id}/{ir.request_id}"),
                )
            return
        except BackendContextLengthExceededError:
            async with session.lock:
                if is_current():
                    ir.runner_task = None
                    ir.backend_started = False
                    session.live_irs.remove(ir)
                    ir.waiter.set_result(
                        _openai_context_length_error_result(
                            max_context_len=self._max_context_len(group),
                            prompt_tokens=len(ir.history_train_token_prefix) + len(ir.pending_token_delta),
                            requested_completion_tokens=remaining_tokens,
                        )
                    )
                    self._dispatch_queued_irs_locked(session)
            self._notify_state_change(group)
            return
        except RuntimeGroupError as error:
            async with session.lock:
                if ir in session.live_irs and ir.runner_task is runner_task:
                    ir.runner_task = None
                    ir.backend_started = False
            await self._handle_infra_failure(session, error)
            return
        except Exception as error:
            async with session.lock:
                if is_current():
                    ir.backend_started = False
                    self._fail_ir_locked(session, ir, error)
            self._notify_state_change(group)
            return

        async with session.lock:
            if not is_current():
                return
            ir.runner_task = None
            ir.backend_started = False
            try:
                self._apply_generate_result(ir, result)
                finish_type = result.finish_type
                if _last_token_is_stop_token(
                    token_ids=result.new_tokens,
                    tokenizer=self._generation_backend.tokenizer,
                    sampling_params=ir.sampling_params,
                ):
                    finish_type = "stop"
            except Exception as error:
                self._fail_ir_locked(session, ir, error)
                self._notify_state_change(group)
                return

            if finish_type == "abort":
                ir.abort_count += 1
                protected_abort_count_threshold = (
                    self.args.partial_rollout_max_aborted_count if self.args.partial_rollout else None
                )
                ir.kind = cast(SessionForest, session.forest).resolve_request_kind(
                    abort_count=ir.abort_count,
                    resumed=True,
                )
                ir.pending_status = "aborted"
                if protected_abort_count_threshold is not None and ir.abort_count >= protected_abort_count_threshold:
                    # Activate protection at the next rollout boundary, after this interrupted step has closed.
                    session.protection_pending_until_resume = True
                session.queued_irs.append(ir)
                self._dispatch_queued_irs_locked(session)
                self._notify_state_change(group)
                return

            ir.pending_status = "truncated" if finish_type == "length" else "completed"
            try:
                payload = self._terminal_response_locked(
                    session=session,
                    ir=ir,
                    finish_type=finish_type,
                )
            except Exception as error:
                self._fail_ir_locked(session, ir, error)
                self._notify_state_change(group)
                return
            session.live_irs.remove(ir)
            ir.waiter.set_result(payload)
            self._dispatch_queued_irs_locked(session)
        self._notify_state_change(group)

    def _apply_generate_result(
        self,
        ir: InflightRequest,
        result: Any,
    ) -> None:
        """Accumulate one backend attempt on the stable IR ref."""

        ir.pending_token_delta.extend(result.new_tokens)
        ir.pending_logprob_delta.extend(result.new_log_probs)
        ir.pending_generation_elapsed_s += result.elapsed
        ir.latest_backend_meta = dict(result.meta_info)
        self._accumulate_request_meta(ir, meta_info=result.meta_info)
        ir.pending_routed_experts = _decode_routed_experts(
            args=self.args,
            meta_info=result.meta_info,
            token_count=len(ir.history_train_token_prefix) + len(ir.pending_token_delta),
        )

    def _terminal_response_locked(
        self,
        *,
        session: _SessionRecord,
        ir: InflightRequest,
        finish_type: str,
    ) -> dict[str, Any]:
        """Commit one terminal IR to its Forest and build its OpenAI
        response."""

        mark_agentic_event(agentic_trace_events(ir.pending_export_metadata_patch), "chat_end_at")
        ir.pending_export_metadata_patch.update(
            {
                "request_id": ir.request_id,
                "request_kind": ir.kind.value,
                "base_state_hash": ir.parent_state_hash,
            }
        )
        forest = cast(SessionForest, session.forest)
        response_messages, has_tool_calls = _decode_response_payload(
            args=self.args,
            tokenizer=self._generation_backend.tokenizer,
            token_ids=ir.pending_token_delta,
            tools=forest.subtree_tools(ir.parent_state_hash),
            parent_state_hash=ir.parent_state_hash,
        )
        response_node = forest.append_resp(
            parent_state_hash=ir.parent_state_hash,
            rollout_id=ir.rollout_id,
            abort_count=ir.abort_count,
            messages_delta=response_messages,
            token_delta=ir.pending_token_delta,
            logprob_delta=ir.pending_logprob_delta,
            weight_version_delta=ir.pending_weight_version_delta,
            spec_delta=ir.pending_spec_delta,
            prefix_cache_delta=ir.pending_prefix_cache_delta,
            wall_elapsed_s=time.monotonic() - ir.wall_started_at,
            generation_elapsed_s=ir.pending_generation_elapsed_s,
            status=ir.pending_status,
            rollout_routed_experts=ir.pending_routed_experts,
            export_metadata_patch=ir.pending_export_metadata_patch,
        )
        session.resp_state_hash_by_request_id[ir.request_id] = response_node.state_hash
        prompt_tokens = int(ir.latest_backend_meta.get("prompt_tokens", 0))
        completion_tokens = len(ir.pending_token_delta)
        payload = {
            "request_id": ir.request_id,
            "message": response_messages[0],
            "logprobs": (
                _openai_token_logprobs_payload(
                    tokenizer=self._generation_backend.tokenizer,
                    token_ids=ir.pending_token_delta,
                    token_logprobs=ir.pending_logprob_delta,
                )
                if ir.logprobs
                else None
            ),
            "finish_reason": "length" if finish_type == "length" else ("tool_calls" if has_tool_calls else "stop"),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        response_node.wall_elapsed_s = time.monotonic() - ir.wall_started_at
        return payload

    async def _watch_agent_process(
        self,
        session: _SessionRecord,
        process_wait: "asyncio.Task[Optional[BaseException]]",
    ) -> None:
        """Watch one process and finalize its Session result."""

        process_error = await asyncio.shield(process_wait)

        group = session.group
        if isinstance(process_error, AgentExecutionError):
            logger.error(
                "Dropping failed resident agentic session session_id=%s group_id=%s error_type=%s\n%s",
                session.session_id,
                group.group_id,
                type(process_error).__name__,
                process_error,
            )
            await self._handle_controlled_agent_failure(session)
            return
        if (
            group.rollout_mode == "eval"
            and group.ownership is GroupOwnership.PREPARE
            and not group.first_request_barrier_crossed
        ):
            if process_error is None:
                logger.error(
                    "Dropping agentic eval session that exited before its first request "
                    "session_id=%s group_id=%s error_type=silent-success",
                    session.session_id,
                    group.group_id,
                )
                await self._handle_controlled_agent_failure(session)
            else:
                if not isinstance(process_error, Exception):
                    process_error = RuntimeGroupError(
                        f"agent session task cancelled unexpectedly for {session.session_id}"
                    )
                await self._handle_infra_failure(session, process_error)
            return

        if process_error is None:
            await finish_before_cancellation(
                self._finalize_terminal_session(session),
                f"session-finalize:{session.session_id}",
            )
            return
        if not isinstance(process_error, Exception):
            process_error = RuntimeGroupError(f"agent session task cancelled unexpectedly for {session.session_id}")
        await self._handle_infra_failure(session, process_error)

    async def _handle_controlled_agent_failure(
        self,
        session: _SessionRecord,
    ) -> None:
        """Map a controlled agent failure to whole-group drop."""

        await finish_before_cancellation(
            self._finish_session(session, None),
            f"dropped-session-cleanup:{session.session_id}",
        )

    async def _handle_infra_failure(
        self,
        session: _SessionRecord,
        error: Exception,
    ) -> None:
        """Map process or transport failure to a Runtime group error."""

        wrapped = (
            error
            if isinstance(error, RuntimeGroupError)
            else RuntimeGroupError(f"agent process failed for {session.session_id}: {type(error).__name__}: {error}")
        )
        await finish_before_cancellation(
            self._finish_session(session, wrapped),
            f"failed-session-cleanup:{session.session_id}",
        )

    async def _finalize_terminal_session(self, session: _SessionRecord) -> None:
        """Build the compact export, release Session resources, then
        publish."""

        finalize_arrive_at = time.time()
        outcome: _SessionResultPayload
        async with session.lock:
            finalize_lock_acquired_at = time.time()
            if session.phase is not SessionPhase.ACTIVE:
                return
            session.phase = SessionPhase.FINALIZING
            try:
                output = cast(SessionResources, session.resources).process.output
                finalize_events = {
                    "finalize_arrive_at": finalize_arrive_at,
                    "finalize_lock_acquired_at": finalize_lock_acquired_at,
                    "finalize_start_at": time.time(),
                }
                outcome = self._build_session_export_transport(
                    session=session,
                    reward=output.reward,
                    metadata=output.metadata,
                    output_records=output.records,
                    finalize_events=finalize_events,
                )
            except AgentExecutionError as error:
                logger.error(
                    "Dropping non-finalizable agentic session session_id=%s group_id=%s error_type=%s\n%s",
                    session.session_id,
                    session.group.group_id,
                    type(error).__name__,
                    error,
                )
                outcome = None
            except Exception as error:
                outcome = RuntimeGroupError(
                    f"agent export failed for {session.session_id}: {type(error).__name__}: {error}"
                )
        await self._finish_session(session, outcome)

    def _build_session_export_transport(
        self,
        *,
        session: _SessionRecord,
        reward: Any,
        metadata: dict[str, Any],
        output_records: tuple[dict[str, Any], ...],
        finalize_events: dict[str, float],
    ) -> SessionExportTransport:
        forest = cast(SessionForest, session.forest)
        export_payloads: list[dict[str, Any]] = []
        if output_records:
            for output_record in output_records:
                unit_metadata = copy.deepcopy(metadata)
                _merge_export_metadata(unit_metadata, output_record["metadata"])
                state_hash = _messages_tools_template_state_hash(
                    output_record["messages"],
                    output_record.get("tools", []),
                    normalize_template_kwargs(
                        {
                            **self._generation_backend.compiler.apply_chat_template_kwargs,
                            **output_record.get("chat_template_kwargs", {}),
                        }
                    ),
                )
                if state_hash not in forest.nodes_by_hash:
                    raise _NonFinalizableExportError("explicit export does not match a committed Forest state")
                sample = self._build_sample(
                    forest=forest,
                    state_hash=state_hash,
                    reward=output_record.get("reward"),
                    metadata=unit_metadata,
                    mutate_node=False,
                    finalize_events=finalize_events,
                )
                export_payloads.append(
                    self._export_payload(
                        name=output_record["name"],
                        sample=sample,
                    )
                )
        else:
            state_hash = self._implicit_export_state_hash(forest)
            sample = self._build_sample(
                forest=forest,
                state_hash=state_hash,
                reward=reward,
                metadata=metadata,
                mutate_node=True,
                finalize_events=finalize_events,
            )
            export_payloads.append(self._export_payload(name=None, sample=sample))
        return SessionExportTransport(exports=tuple(export_payloads))

    def _build_sample(
        self,
        *,
        forest: SessionForest,
        state_hash: str,
        reward: Any,
        metadata: dict[str, Any],
        mutate_node: bool,
        finalize_events: dict[str, float],
    ) -> Any:
        lineage = forest.lineage(state_hash)
        if not any(node.kind == "resp" for node in lineage):
            raise _NonFinalizableExportError("export leaf has no committed response")
        node = lineage[-1]
        if mutate_node:
            _merge_export_metadata(node.export_metadata_patch, metadata)
            agentic_trace_events(node.export_metadata_patch).update(finalize_events)
        sample = forest.build_sample(leaf_state_hash=state_hash, tokenizer=self._generation_backend.tokenizer)
        if reward is not None:
            sample.reward = copy.deepcopy(reward)
        if not mutate_node:
            _merge_export_metadata(sample.metadata, metadata)
            agentic_trace_events(sample.metadata).update(finalize_events)
        mark_sample_agentic_event(sample, "finalize_end_at")
        return sample

    @staticmethod
    def _implicit_export_state_hash(forest: SessionForest) -> str:
        exportable = [
            leaf_hash
            for leaf_hash in forest.export_leaf_hashes()
            if any(node.kind == "resp" for node in forest.lineage(leaf_hash))
        ]
        if len(exportable) != 1:
            raise _NonFinalizableExportError(f"implicit export requires one committed leaf, found {len(exportable)}")
        return exportable[0]

    @staticmethod
    def _export_payload(*, name: Optional[str], sample: Any) -> dict[str, Any]:
        mark_sample_agentic_event(sample, "sample_export_ready_at")
        return {
            "name": name,
            "sample_payload": TrainingFieldArtifact.from_sample(sample).sample_payload,
        }

    async def _finish_session(
        self,
        session: _SessionRecord,
        outcome: _SessionResultPayload,
    ) -> Optional[BaseException]:
        """Give cleanup, publication, and Group removal one task owner."""

        async with session.lock:
            finish_task = session.finish_task
            if finish_task is None:
                session.phase = SessionPhase.FINALIZING
                finish_task = asyncio.create_task(
                    self._finish_session_once(session, outcome),
                    name=f"session-finish:{session.session_id}",
                )
                session.finish_task = finish_task
        return await asyncio.shield(finish_task)

    async def _finish_session_once(
        self,
        session: _SessionRecord,
        outcome: _SessionResultPayload,
    ) -> Optional[BaseException]:
        """Release Session refs, publish once, then retire the lightweight
        record."""

        group = session.group
        pre_backend_tasks: list[asyncio.Task[None]] = []
        backend_tasks: list[asyncio.Task[None]] = []
        backend_request_ids: list[str] = []
        process_wait: Optional["asyncio.Task[Optional[BaseException]]"] = None
        resources: Optional[SessionResources] = None
        async with session.lock:
            irs = tuple(session.live_irs)
            for ir in irs:
                runner_task = ir.runner_task
                if runner_task is not None:
                    if ir.backend_started:
                        backend_request_ids.append(_backend_request_id(ir))
                        backend_tasks.append(runner_task)
                    else:
                        pre_backend_tasks.append(runner_task)
            resources = session.resources
            if resources is not None:
                process_wait = resources.process_wait

        for runner_task in pre_backend_tasks:
            runner_task.cancel()
        abort_outcomes = await asyncio.gather(
            *(self._generation_backend.abort_request(request_id) for request_id in backend_request_ids),
            return_exceptions=True,
        )
        await asyncio.gather(*pre_backend_tasks, *backend_tasks, return_exceptions=True)
        lifecycle_closed = True
        if self.args.agentic_session_lifecycle:
            lifecycle_closed = await self._generation_backend.close_session(
                session.session_id,
                timeout_s=_SESSION_CLOSE_TIMEOUT_S,
            )
        async with session.lock:
            for ir in irs:
                ir.runner_task = None
                ir.backend_started = False
                ir.waiter.set_result(_openai_error_from_exception(_session_discarded_error(session.session_id)))
            session.live_irs.clear()
            session.queued_irs.clear()
        if resources is not None:
            await asyncio.gather(resources.process.terminate_and_join(), return_exceptions=True)
        if process_wait is not None:
            process_wait.cancel()
            await asyncio.gather(process_wait, return_exceptions=True)
        async with session.lock:
            session.forest = None
            session.resp_state_hash_by_request_id.clear()
            session.resources = None
        del self._session_records[session.session_id]

        cleanup_error = next(
            (abort_outcome for abort_outcome in abort_outcomes if isinstance(abort_outcome, BaseException)),
            None,
        )
        if self.args.agentic_session_lifecycle:
            self._lifecycle_close_count += 1
            self._lifecycle_close_failure_count += int(not lifecycle_closed)
        if cleanup_error is not None:
            if isinstance(outcome, RuntimeGroupError):
                outcome = RuntimeGroupError(
                    f"{outcome}; cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
            else:
                outcome = RuntimeGroupError(
                    f"agent cleanup failed for {session.session_id}: {type(cleanup_error).__name__}: {cleanup_error}"
                )

        session.result_cell.publish(outcome)
        group.sessions.remove(session)
        self._notify_state_change(group)
        return cleanup_error

    async def _drop_group(self, group: ResidentGroup) -> None:
        """Discard a group and release every Session owner."""

        group_id = group.group_id
        if self._groups.get(group_id) is not group:
            return
        sessions = tuple(group.sessions)
        resources = tuple(session.resources for session in sessions if session.resources is not None)
        watcher_tasks = tuple(resource.watcher_task for resource in resources)
        for watcher in watcher_tasks:
            watcher.cancel()

        cleanup_outcomes = await asyncio.gather(
            *(self._finish_session(session, None) for session in sessions),
            return_exceptions=True,
        )
        await asyncio.gather(*watcher_tasks, return_exceptions=True)
        self._groups.pop(group_id, None)
        self._notify_state_change()
        for outcome in cleanup_outcomes:
            if isinstance(outcome, BaseException):
                raise outcome


def create_agentic_session_shards(
    config: Namespace,
    admission_coordinator: Optional[Any],
) -> Tuple[Tuple[str, Any], ...]:
    """Create the named Shard fleet for one Serve deployment.

    Every Shard is the resource-visible actor that directly owns its short-
    lived Session records and agent processes.
    """

    sglang_request_capacity = (
        config.sglang_server_concurrency * config.rollout_num_gpus // config.rollout_num_gpus_per_engine
    )
    shard_entries: list[tuple[str, Any]] = []
    try:
        for placement in range(_DEFAULT_SESSION_SHARD_COUNT):
            actor_name = agentic_session_shard_name(placement)
            shard = AgenticSessionShard.options(
                name=actor_name,
                # The head advertises zero CPU; this positive reservation keeps process owners on workers.
                num_cpus=0.25,
                max_restarts=0,
                scheduling_strategy="SPREAD",
                runtime_env={"env_vars": dict(_AGENTIC_SHARD_ALLOCATOR_ENV)},
            ).remote(
                config,
                sglang_request_capacity if placement == 0 else None,
                shard_entries[0][1] if placement > 0 else None,
                admission_coordinator,
            )
            shard_entries.append((actor_name, shard))
    except BaseException:
        _shutdown_agentic_session_shards(tuple(handle for _name, handle in shard_entries))
        raise
    return tuple(shard_entries)


@serve.deployment
@serve.ingress(app)
class AgenticChatAPIService:
    """OpenAI chat ingress and deployment-owned SessionShard directory."""

    def __init__(
        self,
        args: Namespace,
        session_shards: Tuple[Tuple[str, Any], ...],
        admission_coordinator: Optional[Any],
    ) -> None:
        self.args = args
        self._session_shards = dict(session_shards)
        self._admission_coordinator = admission_coordinator

    def _shard_handle(self, session_id: str) -> Any:
        actor_name = session_id.rsplit(".session-", maxsplit=1)[0]
        return self._session_shards[actor_name]

    @app.get("/health")
    @app.get("/healthz")
    async def healthz(self) -> dict[str, Any]:
        snapshots = await asyncio.gather(*(shard.health.remote() for shard in self._session_shards.values()))
        return {"ok": True, "shards": snapshots}

    @app.get("/debug_state")
    async def debug_state(self, sample_limit: int = 8) -> dict[str, Any]:
        snapshots = await asyncio.gather(
            *(shard.debug_state.remote(sample_limit=sample_limit) for shard in self._session_shards.values())
        )
        return {
            "totals": {
                "group_count": sum(snapshot["group_count"] for snapshot in snapshots),
                "session_count": sum(snapshot["session_count"] for snapshot in snapshots),
                "retained_request_count": sum(snapshot["retained_request_count"] for snapshot in snapshots),
                "backend_request_count": sum(snapshot["backend_request_count"] for snapshot in snapshots),
                "live_process_count": sum(snapshot["live_process_count"] for snapshot in snapshots),
            },
            "shards": snapshots,
        }

    @app.get("/models")
    @app.get("/v1/models")
    async def models(self) -> JSONResponse:
        checkpoint = self.args.hf_checkpoint
        model_id = Path(checkpoint).name if isinstance(checkpoint, str) and checkpoint else None
        data = []
        if model_id is not None:
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "relax",
                }
            )
        return JSONResponse({"object": "list", "data": data})

    @app.post("/")
    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    @_resolve_fastapi_request_endpoint
    async def chat_completions(self, request: Request) -> JSONResponse:
        """Normalize and forward one OpenAI-compatible chat request."""

        try:
            payload = await request.json()
        except ClientDisconnect:
            return JSONResponse(
                {
                    "error": {
                        "message": "client disconnected before chat request body was read",
                        "type": "client_disconnect",
                        "code": "client_disconnect",
                    }
                },
                status_code=499,
            )
        except ValueError as error:
            return _openai_error_response(_openai_error_result(f"Invalid request body: {error}", param="body"))
        try:
            if not isinstance(payload, dict):
                raise AgenticChatRequestError("request body must be a JSON object", param="body")
            normalized = _normalized_chat_request(payload)
            session_id = _session_id_from_request(request)
        except AgenticChatRequestError as error:
            return _openai_error_response(_openai_error_from_exception(error))

        try:
            shard = self._shard_handle(session_id)
        except KeyError:
            return _openai_error_response(_openai_error_from_exception(_session_discarded_error(session_id)))

        try:
            response = await shard.chat.remote(
                session_id=session_id,
                messages=normalized["messages"],
                tools=normalized["tools"],
                chat_template_kwargs=normalized["chat_template_kwargs"],
                max_completion_tokens=normalized["max_completion_tokens"],
                stop=normalized["stop"],
                seed=normalized["seed"],
                logprobs=normalized["logprobs"],
            )
        except (ray.exceptions.RayTaskError, ray.exceptions.TaskCancelledError) as error:
            if isinstance(error, ray.exceptions.RayTaskError) and not isinstance(
                error.as_instanceof_cause(), ray.exceptions.TaskCancelledError
            ):
                raise
            return JSONResponse(
                {
                    "error": {
                        "message": "client disconnected before chat completion was produced",
                        "type": "client_disconnect",
                        "code": "client_disconnect",
                    }
                },
                status_code=499,
            )
        if isinstance(response.get("error"), dict):
            return _openai_error_response(response)
        remote_return_at = time.time()
        response_ready_at = time.time()
        http_return_at = time.time()
        try:
            await shard.mark_chat_service_response_ready.remote(
                session_id=session_id,
                request_id=response["request_id"],
                remote_return_at=remote_return_at,
                response_ready_at=response_ready_at,
                http_return_at=http_return_at,
            )
        except Exception as error:
            logger.warning(
                "Failed to record chat service response trace for session=%s request=%s: %s",
                session_id,
                response["request_id"],
                error,
            )
        response_payload = {
            "id": f"chatcmpl_{session_id}_{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": response["message"],
                    "logprobs": response["logprobs"] if normalized["logprobs"] else None,
                    "finish_reason": response["finish_reason"],
                }
            ],
            "usage": response["usage"],
        }
        if isinstance(normalized["model"], str):
            response_payload["model"] = normalized["model"]
        return JSONResponse(response_payload)

    async def runtime_resources(self) -> tuple[Tuple[Tuple[str, Any], ...], Optional[Any]]:
        """Lend Shards and their shared admission Coordinator to Runtime."""

        return tuple(self._session_shards.items()), self._admission_coordinator


def deploy_agentic_chat_api_services(
    *,
    config: Namespace,
    runtime_env: dict[str, Any] | None,
) -> None:
    """Create Shards and bind their handles into Ray Serve.

    Request and Serve capacities are derived from the Agentic configuration.
    """

    from relax.distributed.ray.rollout import _resolve_sglang_config, _start_router

    resolved_config = _resolve_sglang_config(config)
    router_ip, router_port = _start_router(
        config,
        has_pd_disaggregation=resolved_config.has_pd_disaggregation,
        force_new=False,
    )
    config.sglang_router_ip = router_ip
    config.sglang_router_port = router_port

    global _DEPLOYED_ADMISSION_COORDINATOR

    shard_entries: Tuple[Tuple[str, Any], ...] = ()
    admission_coordinator = None
    try:
        if config.agentic_program_admission:
            admission_coordinator = AdmissionCoordinator.options(num_cpus=0, max_restarts=0).remote(config)
            ray.get(admission_coordinator.start.remote())
        shard_entries = create_agentic_session_shards(
            config=config,
            admission_coordinator=admission_coordinator,
        )
        deployment = AgenticChatAPIService.options(
            num_replicas=_DEFAULT_SESSION_SHARD_COUNT,
            max_ongoing_requests=_AGENTIC_CHAT_MAX_ONGOING_REQUESTS,
            ray_actor_options={"runtime_env": runtime_env},
        )
        serve.run(
            deployment.bind(config, shard_entries, admission_coordinator),
            name=AGENTIC_CHAT_API_SERVICE_NAME,
            route_prefix=AGENTIC_CHAT_API_ROUTE_PREFIX,
        )
        _DEPLOYED_ADMISSION_COORDINATOR = admission_coordinator
    except BaseException:
        _shutdown_agentic_session_shards(tuple(handle for _name, handle in shard_entries))
        _shutdown_admission_coordinator(admission_coordinator)
        raise


def shutdown_agentic_chat_api_services() -> None:
    """Delete Serve ingress and join Shards after Runtime borrowers stop."""

    global _DEPLOYED_ADMISSION_COORDINATOR

    admission_coordinator = _DEPLOYED_ADMISSION_COORDINATOR
    _DEPLOYED_ADMISSION_COORDINATOR = None
    try:
        serve.delete(AGENTIC_CHAT_API_SERVICE_NAME)
    except Exception as error:
        logger.warning("Failed to delete Agentic Serve ingress during shutdown: %s", error)
    finally:
        shards = []
        for index in range(_DEFAULT_SESSION_SHARD_COUNT):
            try:
                shards.append(ray.get_actor(agentic_session_shard_name(index)))
            except Exception:
                continue
        _shutdown_agentic_session_shards(tuple(shards))
        _shutdown_admission_coordinator(admission_coordinator)


def _shutdown_admission_coordinator(coordinator: Optional[Any]) -> None:
    if coordinator is None:
        return
    try:
        ray.get(coordinator.shutdown.remote(), timeout=_ADMISSION_SHUTDOWN_TIMEOUT_S)
    except Exception as error:
        logger.warning("Failed to join Agentic admission coordinator: %s", error)
    finally:
        try:
            ray.kill(coordinator, no_restart=True)
        except Exception as error:
            logger.warning("Failed to kill Agentic admission coordinator: %s", error)


def _shutdown_agentic_session_shards(shards: Tuple[Any, ...]) -> None:
    """Best-effort join and kill of the Shards that actually exist."""

    if not shards:
        return
    try:
        ray.get([shard.shutdown.remote() for shard in shards])
    except Exception as error:
        logger.warning("Failed to join one or more Agentic SessionShards: %s", error)
    finally:
        for shard in shards:
            try:
                ray.kill(shard, no_restart=True)
            except Exception as error:
                logger.warning("Failed to kill Agentic SessionShard: %s", error)
