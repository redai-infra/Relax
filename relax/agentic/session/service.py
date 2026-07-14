# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy
import ctypes
import hashlib
import json
import threading
import time
import zlib
from argparse import Namespace
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from ray import serve
from starlette.requests import ClientDisconnect

from relax.agentic import AGENTIC_CHAT_API_ROUTE_PREFIX, AGENTIC_CHAT_API_SERVICE_NAME
from relax.agentic.pipeline.runtime import (
    BackendContextLengthExceededError,
    SGLangBackendAdapter,
    agentic_target_session_count_from_args,
    get_agentic_runtime_resources,
)
from relax.agentic.profile import (
    agentic_trace_events,
    mark_agentic_event,
    mark_agentic_event_once,
    mark_metadata_agentic_event,
)
from relax.agentic.session.state import (
    FinalizedResultTransport,
    InflightRequest,
    MsgNode,
    RequestKind,
    SessionForest,
    TrainingFieldArtifact,
    _messages_tools_template_state_hash,
    _multimodal_inputs_from_messages,
    check_messages,
    normalize_template_kwargs,
    normalize_tools,
)
from relax.utils.http_utils import get, init_http_client, post
from relax.utils.logging_utils import get_logger


app = FastAPI(title="Relax Agentic Chat API")
logger = get_logger(__name__)

AGENTIC_SESSION_SHARD_NAME_PREFIX = "agentic_session_shard"
_DEFAULT_SESSION_SHARD_COUNT = 16
_STALE_SESSION_SHARD_CLEANUP_LIMIT = 64
_AGENTIC_SHARD_ALLOCATOR_ENV = {
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_TRIM_THRESHOLD_": "0",
}


def agentic_session_shard_name(index: int) -> str:
    return f"{AGENTIC_SESSION_SHARD_NAME_PREFIX}_{int(index)}"


def deploy_agentic_chat_api_services(*, config, runtime_env) -> None:
    from relax.distributed.ray.rollout import _resolve_sglang_config, _start_router

    resolved_config = _resolve_sglang_config(config)
    has_pd = resolved_config.has_pd_disaggregation
    router_ip, router_port = _start_router(config, has_pd_disaggregation=has_pd, force_new=False)
    config.sglang_router_ip = router_ip
    config.sglang_router_port = router_port
    session_shards = create_agentic_session_shards(config=config)
    chat_deployment = AgenticChatAPIService.options(
        ray_actor_options={"runtime_env": runtime_env},
        num_replicas=_DEFAULT_SESSION_SHARD_COUNT,
        max_ongoing_requests=agentic_target_session_count_from_args(config),
    ).bind(config, session_shards)
    serve.run(chat_deployment, name=AGENTIC_CHAT_API_SERVICE_NAME, route_prefix=AGENTIC_CHAT_API_ROUTE_PREFIX)
    logger.info("AgenticChatAPIService deployed at %s", AGENTIC_CHAT_API_ROUTE_PREFIX)


def shutdown_agentic_chat_api_services() -> None:
    try:
        serve.delete(AGENTIC_CHAT_API_SERVICE_NAME)
    except Exception:
        pass
    for idx in range(_STALE_SESSION_SHARD_CLEANUP_LIMIT):
        try:
            ray.kill(ray.get_actor(agentic_session_shard_name(idx)), no_restart=True)
        except Exception:
            pass
    logger.info("Agentic chat API services shut down.")


class SessionGateReason(str, Enum):
    PREPARE = "prepare"
    PARTIAL_RESUME = "partial_resume"
    TERMINAL_SHUTDOWN = "terminal_shutdown"


class IRBlockedReason(str, Enum):
    PREPARE_GATE = "prepare_gate"
    PARTIAL_RESUME_GATE = "partial_resume_gate"
    TERMINAL_SHUTDOWN_GATE = "terminal_shutdown_gate"


_WAITING_REASON_PREPARE_GATE = IRBlockedReason.PREPARE_GATE.value
_WAITING_REASON_PARTIAL_RESUME_GATE = IRBlockedReason.PARTIAL_RESUME_GATE.value
_WAITING_REASON_TERMINAL_SHUTDOWN_GATE = IRBlockedReason.TERMINAL_SHUTDOWN_GATE.value
_GATE_REASON_PREPARE = SessionGateReason.PREPARE.value
_GATE_REASON_PARTIAL_RESUME = SessionGateReason.PARTIAL_RESUME.value
_GATE_REASON_TERMINAL_SHUTDOWN = SessionGateReason.TERMINAL_SHUTDOWN.value
_GATE_WAITING_REASONS = {
    _GATE_REASON_PREPARE: _WAITING_REASON_PREPARE_GATE,
    _GATE_REASON_PARTIAL_RESUME: _WAITING_REASON_PARTIAL_RESUME_GATE,
    _GATE_REASON_TERMINAL_SHUTDOWN: _WAITING_REASON_TERMINAL_SHUTDOWN_GATE,
}


@dataclass
class _SessionRecord:
    forest: SessionForest | None = None
    rollout_id: int = -1
    scope_id: str | None = None
    group_id: str | None = None
    group_generation: int = 0
    next_ir_sequence: int = 0
    session_sampling_params: dict[str, Any] = field(default_factory=dict)
    session_seed: dict[str, Any] = field(default_factory=dict)
    resp_state_hash_by_request_id: dict[str, str] = field(default_factory=dict)
    irs_by_id: dict[str, InflightRequest] = field(default_factory=dict)
    ir_queue: deque[str] = field(default_factory=deque)
    active_ir_runner_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    pending_chat_waiters: dict[str, asyncio.Future[Any]] = field(default_factory=dict)
    gate_reason: str | None = None
    protected_until_finalize: bool = False


@dataclass(frozen=True)
class IRReleaseDecision:
    allow: bool
    blocked_reason: str | None = None


class AgenticChatRequestError(HTTPException):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request_error",
        param: str | None = None,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.message = message
        self.code = code
        self.param = param
        self.status_code = int(status_code)
        self.error_type = error_type


def _session_discarded_error(session_id: str) -> AgenticChatRequestError:
    return AgenticChatRequestError(
        f"Unknown or discarded agentic session {session_id!r}.",
        code="session_discarded",
        param="session_id",
        status_code=404,
        error_type="not_found_error",
    )


def _normalize_session_gate_reason(gate_reason: SessionGateReason | str | None) -> str | None:
    if gate_reason is None:
        return None
    if isinstance(gate_reason, SessionGateReason):
        return gate_reason.value
    if isinstance(gate_reason, str) and not gate_reason:
        return None
    try:
        return SessionGateReason(str(gate_reason)).value
    except ValueError as exc:
        raise ValueError(f"Unknown session gate_reason: {gate_reason!r}") from exc


def _blocked_reason_for_gate(gate_reason: SessionGateReason | str | None) -> str | None:
    normalized_gate_reason = _normalize_session_gate_reason(gate_reason)
    if normalized_gate_reason is None:
        return None
    return _GATE_WAITING_REASONS[normalized_gate_reason]


def _decide_ir_release(record: _SessionRecord) -> IRReleaseDecision:
    blocked_reason = _blocked_reason_for_gate(record.gate_reason)
    if blocked_reason == _WAITING_REASON_PARTIAL_RESUME_GATE and record.protected_until_finalize:
        return IRReleaseDecision(allow=True)
    if blocked_reason is not None:
        return IRReleaseDecision(allow=False, blocked_reason=blocked_reason)
    return IRReleaseDecision(allow=True)


def _count_gate_blocked_irs(record: _SessionRecord) -> tuple[int, int]:
    blocked_reason = _decide_ir_release(record=record).blocked_reason
    waiter_count = len(record.pending_chat_waiters)
    if blocked_reason == _WAITING_REASON_PREPARE_GATE:
        return waiter_count, 0
    if blocked_reason == _WAITING_REASON_PARTIAL_RESUME_GATE:
        return 0, waiter_count
    return 0, 0


def _openai_error_result(
    message: str,
    *,
    code: str = "invalid_request_error",
    param: str | None = None,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
) -> dict[str, Any]:
    return {
        "_http_status": int(status_code),
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        },
    }


def _openai_error_from_exc(exc: AgenticChatRequestError) -> dict[str, Any]:
    return _openai_error_result(
        exc.message,
        code=exc.code,
        param=exc.param,
        status_code=exc.status_code,
        error_type=exc.error_type,
    )


def _openai_context_length_error_result(
    *,
    max_context_len: int | None,
    prompt_tokens: int | None,
    requested_completion_tokens: int | None = None,
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


async def _sglang_worker_urls(args: Namespace) -> list[str]:
    router_ip = args.sglang_router_ip
    router_port = args.sglang_router_port
    if not router_ip or not router_port:
        return []
    base_url = f"http://{router_ip}:{router_port}"
    worker_query_error: Exception | None = None
    try:
        response = await get(f"{base_url}/workers")
        workers = response.get("workers", [])
        if isinstance(workers, list):
            urls = [
                worker.get("url")
                for worker in workers
                if isinstance(worker, dict) and worker.get("url") and bool(worker.get("is_healthy", False))
            ]
            if urls:
                return urls
    except Exception as exc:
        worker_query_error = exc
    try:
        response = await get(f"{base_url}/list_workers")
        urls = response.get("urls", [])
        if isinstance(urls, list):
            return [url for url in urls if isinstance(url, str) and url]
    except Exception as exc:
        if worker_query_error is not None:
            raise RuntimeError("Failed to query worker urls from both /workers and /list_workers") from exc
        raise RuntimeError("Failed to query worker urls from /list_workers") from exc
    return []


def _shard_index_for_session(session_id: str, shard_count: int) -> int:
    return zlib.crc32(session_id.encode("utf-8")) % shard_count


def _session_id_from_request(*, request: Request) -> str:
    header = request.headers.get("Authorization")
    if not header:
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
    def fail(
        message: str,
        *,
        code: str = "invalid_request_error",
        param: str | None = None,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
    ) -> None:
        raise AgenticChatRequestError(
            message,
            code=code,
            param=param,
            status_code=status_code,
            error_type=error_type,
        )

    if "messages" not in payload:
        fail("messages is required", param="messages")
    messages = payload["messages"]
    if not isinstance(messages, list):
        fail("messages must be a list", param="messages")

    tools = payload.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            fail("tools must be a list", param="tools")
        if any(not isinstance(item, dict) for item in tools):
            fail("tools entries must be JSON objects", param="tools")
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
        fail("chat_template_kwargs must be a JSON object", param="chat_template_kwargs")
    if chat_template_kwargs is not None:
        reserved_chat_template_kwargs = {"add_generation_prompt", "tokenize", "tools"}
        reserved = sorted(reserved_chat_template_kwargs.intersection(chat_template_kwargs))
        if reserved:
            fail(
                f"chat_template_kwargs cannot set reserved keys: {', '.join(reserved)}",
                param="chat_template_kwargs",
            )

    # TODO: support some of these following parameters
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
    max_completion_tokens = payload.get("max_completion_tokens")
    if max_completion_tokens is not None and not (
        isinstance(max_completion_tokens, int)
        and not isinstance(max_completion_tokens, bool)
        and max_completion_tokens > 0
    ):
        fail("max_completion_tokens must be a positive integer", param="max_completion_tokens")

    stop = payload.get("stop")
    if (
        "stop" in payload
        and stop is not None
        and not (isinstance(stop, str) or (isinstance(stop, list) and all(isinstance(item, str) for item in stop)))
    ):
        fail("stop must be a string or list of strings", param="stop")

    try:
        messages = check_messages(messages)
    except (TypeError, ValueError) as exc:
        fail(str(exc), param="messages")

    return {
        "messages": messages,
        "tools": normalize_tools(tools),
        "chat_template_kwargs": normalize_template_kwargs(chat_template_kwargs),
        "model": payload.get("model"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "logprobs": logprobs,
        "max_completion_tokens": max_completion_tokens,
        "stop": stop,
        "seed": payload.get("seed"),
    }


def _openai_finish_reason(response_finish_reason: str) -> str:
    if response_finish_reason == "stop":
        return "stop"
    if response_finish_reason in {"length", "context_length"}:
        return "length"
    raise RuntimeError(
        f"Internal finish_reason {response_finish_reason!r} cannot be exposed via OpenAI-compatible API"
    )


def _transport_status_from_finish_type(finish_type: str) -> str:
    if finish_type == "length":
        return "truncated"
    if finish_type == "abort":
        return "aborted"
    return "completed"


def _extend_token_id_set(token_ids: set[int], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, int):
        token_ids.add(int(value))
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _extend_token_id_set(token_ids, item)


def _last_token_is_stop_token(*, token_ids: list[int], tokenizer: Any, sampling_params: dict[str, Any]) -> bool:
    if not token_ids:
        return False
    stop_token_ids: set[int] = set()
    _extend_token_id_set(stop_token_ids, getattr(tokenizer, "eos_token_id", None))
    _extend_token_id_set(stop_token_ids, getattr(tokenizer, "eos_token_ids", None))
    _extend_token_id_set(stop_token_ids, getattr(tokenizer, "additional_stop_token_ids", None))
    _extend_token_id_set(stop_token_ids, sampling_params.get("stop_token_ids"))
    return int(token_ids[-1]) in stop_token_ids


def _stable_tool_call_id(*, parent_state_hash: str, token_ids: list[int], call_index: int) -> str:
    payload = json.dumps(
        {
            "call_index": int(call_index),
            "parent_state_hash": parent_state_hash,
            "token_ids": list(token_ids),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"call_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _sglang_parser_tools(tools: list[dict[str, Any]]) -> list[Any]:
    try:
        from sglang.srt.entrypoints.openai.protocol import Tool
    except Exception as exc:
        raise AgenticChatRequestError(
            f"Failed to import SGLang Tool protocol model: {exc}",
            code="agentic_tool_call_parser_error",
            status_code=500,
            error_type="internal_error",
        ) from exc

    try:
        return [Tool.model_validate(tool) for tool in tools]
    except Exception as exc:
        raise AgenticChatRequestError(
            f"Invalid tools for agentic tool-call parser: {exc}",
            code="invalid_tools",
            param="tools",
        ) from exc


def _postprocess_assistant_message(
    *,
    args: Namespace,
    text: str,
    tools: list[dict[str, Any]],
    parent_state_hash: str,
    token_ids: list[int],
) -> tuple[dict[str, Any], bool]:
    reasoning_text = None
    normal_text = text
    reasoning_parser_name = getattr(args, "agentic_reasoning_parser", None)
    if reasoning_parser_name:
        try:
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            reasoning_parser = ReasoningParser(model_type=str(reasoning_parser_name), stream_reasoning=False)
            parsed_reasoning, parsed_text = reasoning_parser.parse_non_stream(normal_text)
        except Exception as exc:
            raise AgenticChatRequestError(
                f"Failed to parse reasoning content with parser {reasoning_parser_name!r}: {exc}",
                code="agentic_reasoning_parser_error",
                status_code=500,
                error_type="internal_error",
            ) from exc
        reasoning_text = parsed_reasoning if isinstance(parsed_reasoning, str) and parsed_reasoning else None
        normal_text = parsed_text if isinstance(parsed_text, str) else ""

    tool_calls: list[dict[str, Any]] = []
    tool_call_parser_name = getattr(args, "agentic_tool_call_parser", None)
    if tool_call_parser_name and tools:
        try:
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            parser_tools = _sglang_parser_tools(tools)
            tool_call_parser = FunctionCallParser(parser_tools, str(tool_call_parser_name))
            parsed_text, call_items = tool_call_parser.parse_non_stream(normal_text)
        except AgenticChatRequestError:
            raise
        except Exception as exc:
            raise AgenticChatRequestError(
                f"Failed to parse tool calls with parser {tool_call_parser_name!r}: {exc}",
                code="agentic_tool_call_parser_error",
                status_code=500,
                error_type="internal_error",
            ) from exc
        normal_text = parsed_text if isinstance(parsed_text, str) else ""
        for call_index, call_item in enumerate(call_items):
            tool_calls.append(
                {
                    "id": _stable_tool_call_id(
                        parent_state_hash=parent_state_hash,
                        token_ids=token_ids,
                        call_index=call_index,
                    ),
                    "type": "function",
                    "function": {
                        "name": str(call_item.name or ""),
                        "arguments": call_item.parameters,
                    },
                }
            )

    message: dict[str, Any] = {"role": "assistant", "content": normal_text}
    if not normal_text and (reasoning_text or tool_calls):
        message["content"] = None
    if reasoning_text:
        message["reasoning_content"] = reasoning_text
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, bool(tool_calls)


def _decode_response_payload(
    *,
    args: Namespace,
    tokenizer: Any,
    token_ids: list[int],
    tools: list[dict[str, Any]],
    parent_state_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    if not token_ids:
        response_message = {"role": "assistant", "content": ""}
        return [], response_message, False
    text = str(tokenizer.decode(token_ids, skip_special_tokens=False))
    if not text:
        response_message = {"role": "assistant", "content": ""}
        return [], response_message, False
    response_message, has_tool_calls = _postprocess_assistant_message(
        args=args,
        text=text,
        tools=tools,
        parent_state_hash=parent_state_hash,
        token_ids=token_ids,
    )
    return [response_message], response_message, has_tool_calls


def _openai_token_logprobs_payload(
    *,
    tokenizer: Any,
    token_ids: list[int],
    token_logprobs: list[float],
) -> dict[str, Any]:
    content = []
    for idx, token_id in enumerate(token_ids):
        logprob = token_logprobs[idx] if idx < len(token_logprobs) else -9999.0
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


def _decode_routed_experts(*, args: Namespace, meta_info: dict[str, Any], token_count: int) -> Any:
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


@ray.remote(
    max_concurrency=1024,
    concurrency_groups={
        "sglang_request_permit": 1024,
        "sglang_request_control": 128,
    },
)
class AgenticSessionShard:
    def __init__(
        self,
        config_payload: dict[str, Any] | Namespace | None = None,
        *,
        sglang_request_capacity: int | None = None,
        sglang_request_limiter: Any | None = None,
    ) -> None:
        if isinstance(config_payload, Namespace):
            self.args = config_payload
        else:
            self.args = Namespace(**dict(config_payload or {}))
        init_http_client(self.args)
        resources = get_agentic_runtime_resources(self.args)
        self.backend = SGLangBackendAdapter(self.args, compiler_resources=resources.compiler)
        self._session_records: dict[str, _SessionRecord] = {}
        self._session_locks: dict[str, Any] = {}
        self._evaluating = 0
        self._terminal_ir_gate_closed = False
        self._sglang_request_semaphore = (
            threading.BoundedSemaphore(sglang_request_capacity) if sglang_request_capacity is not None else None
        )
        self._sglang_request_limiter = sglang_request_limiter

    def _ensure_session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    def _get_session_lock(self, session_id: str) -> asyncio.Lock | None:
        return self._session_locks.get(session_id)

    @ray.method(concurrency_group="sglang_request_permit")
    async def acquire_sglang_request_permit(self) -> None:
        if self._sglang_request_semaphore is None:
            raise RuntimeError("SGLang request permit owner has no semaphore.")
        while True:
            if self._sglang_request_semaphore.acquire(blocking=False):
                return
            await asyncio.sleep(0.01)

    @ray.method(concurrency_group="sglang_request_control")
    async def release_sglang_request_permit(self) -> None:
        if self._sglang_request_semaphore is None:
            raise RuntimeError("SGLang request permit owner has no semaphore.")
        self._sglang_request_semaphore.release()

    @staticmethod
    def _is_non_train_session_record(record: _SessionRecord) -> bool:
        return record.scope_id != "train"

    @ray.method(concurrency_group="sglang_request_control")
    async def health(self) -> dict[str, Any]:
        queued_pending = 0
        for record in self._session_records.values():
            queued_pending += len(record.ir_queue)
        return {
            "ok": True,
            "active_sessions": len(self._session_records),
            "active_locks": len(self._session_locks),
            "active_requests": sum(len(record.irs_by_id) for record in self._session_records.values()),
            "forest_nodes": sum(
                len(record.forest.nodes_by_hash)
                for record in self._session_records.values()
                if record.forest is not None
            ),
            "ir_queue": {
                "queued": queued_pending,
            },
        }

    @ray.method(concurrency_group="sglang_request_control")
    async def trim_memory(self) -> dict[str, Any]:
        try:
            libc = ctypes.CDLL("libc.so.6")
            trimmed = int(libc.malloc_trim(0))
        except Exception as exc:
            logger.warning("AgenticSessionShard malloc_trim failed: %s", exc)
            return {
                "ok": False,
                "error": str(exc)[:500],
                "active_sessions": len(self._session_records),
                "active_requests": sum(len(record.irs_by_id) for record in self._session_records.values()),
            }
        return {
            "ok": True,
            "trimmed": trimmed,
            "active_sessions": len(self._session_records),
            "active_requests": sum(len(record.irs_by_id) for record in self._session_records.values()),
        }

    @ray.method(concurrency_group="sglang_request_control")
    async def debug_state(self, *, sample_limit: int = 8) -> dict[str, Any]:
        state_counts: Counter[str | None] = Counter()
        state_irs: Counter[str | None] = Counter()
        state_active: Counter[str | None] = Counter()
        state_queued: Counter[str | None] = Counter()
        state_protected_sessions: Counter[str | None] = Counter()
        state_waiters: Counter[str | None] = Counter()
        state_samples: dict[str | None, list[dict[str, Any]]] = {}
        rollout_id_counts: Counter[int] = Counter()
        per_rollout_irs: Counter[int] = Counter()
        per_rollout_active: Counter[int] = Counter()
        per_rollout_queued: Counter[int] = Counter()
        per_rollout_waiters: Counter[int] = Counter()
        sessions_with_active = 0
        sessions_with_queued_no_active = 0
        sessions_with_waiters_no_active = 0
        sessions_with_no_irs = 0
        prepare_gate_blocked_ir_count = 0
        partial_resume_gate_blocked_ir_count = 0
        for session_id, record in self._session_records.items():
            ir_count = len(record.irs_by_id)
            active_count = len(record.active_ir_runner_tasks)
            queued_count = len(record.ir_queue)
            protected_session_count = 1 if record.protected_until_finalize else 0
            waiter_count = len(record.pending_chat_waiters)
            key = record.gate_reason
            state_counts[key] += 1
            state_irs[key] += ir_count
            state_active[key] += active_count
            state_queued[key] += queued_count
            state_protected_sessions[key] += protected_session_count
            state_waiters[key] += waiter_count
            rollout_id_counts[record.rollout_id] += 1
            per_rollout_irs[record.rollout_id] += ir_count
            per_rollout_active[record.rollout_id] += active_count
            per_rollout_queued[record.rollout_id] += queued_count
            per_rollout_waiters[record.rollout_id] += waiter_count
            prepare_blocked, partial_resume_blocked = _count_gate_blocked_irs(record)
            prepare_gate_blocked_ir_count += prepare_blocked
            partial_resume_gate_blocked_ir_count += partial_resume_blocked
            if active_count > 0:
                sessions_with_active += 1
            elif queued_count > 0:
                sessions_with_queued_no_active += 1
                if waiter_count > 0:
                    sessions_with_waiters_no_active += 1
            elif waiter_count > 0:
                sessions_with_waiters_no_active += 1
            if ir_count == 0 and waiter_count == 0:
                sessions_with_no_irs += 1
            samples_for_state = state_samples.setdefault(key, [])
            if len(samples_for_state) < sample_limit:
                samples_for_state.append(
                    {
                        "session_id": session_id,
                        "rollout_id": record.rollout_id,
                        "group_id": record.group_id,
                        "group_generation": record.group_generation,
                        "ir_count": ir_count,
                        "active_ir_count": active_count,
                        "ir_queue": queued_count,
                        "protected_until_finalize": record.protected_until_finalize,
                        "pending_waiters": waiter_count,
                    }
                )

        def _state_rows() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for key, count in state_counts.most_common():
                rows.append(
                    {
                        "gate_reason": key,
                        "session_count": count,
                        "irs_by_id": state_irs[key],
                        "active_irs": state_active[key],
                        "ir_queue": state_queued[key],
                        "protected_sessions": state_protected_sessions[key],
                        "pending_chat_waiters": state_waiters[key],
                        "samples": list(state_samples.get(key, [])),
                    }
                )
            return rows

        return {
            "evaluating": self._evaluating,
            "active_sessions": len(self._session_records),
            "active_locks": len(self._session_locks),
            "session_breakdown": {
                "with_active_irs": sessions_with_active,
                "queued_no_active": sessions_with_queued_no_active,
                "waiters_no_active": sessions_with_waiters_no_active,
                "no_irs": sessions_with_no_irs,
            },
            "prepare_gate_blocked_ir_count": prepare_gate_blocked_ir_count,
            "partial_resume_gate_blocked_ir_count": partial_resume_gate_blocked_ir_count,
            "per_rollout": [
                {
                    "rollout_id": rollout_id,
                    "session_count": rollout_id_counts[rollout_id],
                    "irs_by_id": per_rollout_irs[rollout_id],
                    "active_irs": per_rollout_active[rollout_id],
                    "ir_queue": per_rollout_queued[rollout_id],
                    "pending_chat_waiters": per_rollout_waiters[rollout_id],
                }
                for rollout_id in sorted(rollout_id_counts)
            ],
            "by_state": _state_rows(),
        }

    async def aborted_resume_session_ids(self, *, rollout_id: int) -> list[str]:
        session_ids: list[str] = []
        for session_id in list(self._session_records):
            lock = self._get_session_lock(session_id)
            if lock is None:
                continue
            async with lock:
                current = self._session_records.get(session_id)
                if current is None:
                    continue
                if self._is_non_train_session_record(current):
                    continue
                if current.gate_reason != _GATE_REASON_PARTIAL_RESUME:
                    continue
                if current.protected_until_finalize:
                    continue
                for ir in current.irs_by_id.values():
                    if ir.rollout_id != rollout_id:
                        continue
                    if ir.pending_status == "aborted" and ir.abort_count > 0:
                        session_ids.append(session_id)
                        break
        return session_ids

    async def enter_eval(self) -> int:
        self._evaluating += 1
        return self._evaluating

    async def exit_eval(self) -> int:
        self._evaluating -= 1
        if self._evaluating < 0:
            raise RuntimeError("AgenticSessionShard eval counter underflow.")
        return self._evaluating

    def _session_stats(self, record: _SessionRecord) -> dict[str, int]:
        forest = record.forest
        return {
            "request_count": len(record.irs_by_id),
            "node_count": len(forest.nodes_by_hash) if forest is not None else 0,
        }

    def _template_kwargs(self) -> dict[str, Any]:
        compiler = getattr(self.backend, "compiler", None)
        return normalize_template_kwargs(getattr(compiler, "apply_chat_template_kwargs", None))

    def _default_session_seed(self) -> dict[str, Any]:
        return {
            "group_index": None,
            "index": None,
            "label": None,
            "train_metadata": None,
            "metadata": {
                "chat_template_kwargs": self._template_kwargs(),
            },
        }

    def _normalized_session_seed(
        self,
        *,
        seed: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(seed, dict):
            return self._default_session_seed()
        normalized = copy.deepcopy(seed)
        metadata = normalized.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = copy.deepcopy(metadata)
        metadata["chat_template_kwargs"] = self._template_kwargs()
        normalized["metadata"] = metadata
        return normalized

    async def _encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
        multimodal_inputs: dict[str, Any] | None = None,
    ):
        return await self.backend.compiler.encode_messages(
            messages,
            tools=tools or [],
            chat_template_kwargs=chat_template_kwargs,
            multimodal_inputs=multimodal_inputs,
        )

    async def _register_session(
        self,
        *,
        session_id: str,
        scope_id: str,
        rollout_id: int,
        group_id: str | None = None,
        group_generation: int = 0,
        gate_reason: str | None = None,
        sampling_params: dict[str, Any] | None = None,
        session_seed: dict[str, Any] | None = None,
    ) -> bool:
        if not isinstance(scope_id, str) or not scope_id:
            raise RuntimeError("Agentic session registration requires a non-empty scope_id.")
        async with self._ensure_session_lock(session_id):
            record = self._session_records.get(session_id)
            if record is None:
                record = _SessionRecord(
                    forest=None,
                    rollout_id=rollout_id,
                    scope_id=scope_id,
                    session_sampling_params=copy.deepcopy(sampling_params)
                    if sampling_params is not None
                    else self._default_sampling_params(sample_index=None),
                    session_seed=copy.deepcopy(session_seed) if isinstance(session_seed, dict) else {},
                    group_id=str(group_id) if isinstance(group_id, str) and group_id else None,
                    group_generation=group_generation,
                )
                self._set_session_gate_locked(record=record, gate_reason=gate_reason)
                self._session_records[session_id] = record
            else:
                if record.scope_id != scope_id:
                    raise RuntimeError(
                        f"Agentic session {session_id!r} is already registered in scope "
                        f"{record.scope_id!r}, got {scope_id!r}."
                    )
                record.rollout_id = rollout_id
                if sampling_params is not None:
                    record.session_sampling_params = copy.deepcopy(sampling_params)
                if isinstance(session_seed, dict) and session_seed:
                    record.session_seed = copy.deepcopy(session_seed)
                if isinstance(group_id, str) and group_id:
                    record.group_id = str(group_id)
                record.group_generation = group_generation
                if isinstance(gate_reason, str):
                    self._set_session_gate_locked(record=record, gate_reason=gate_reason)
        return True

    async def register_sessions_batch(self, *, entries: list[dict[str, Any]]) -> int:
        normalized_entries = []
        for entry in entries:
            session_id = entry["session_id"]
            scope_id = entry["scope_id"]
            if not isinstance(scope_id, str) or not scope_id:
                raise RuntimeError("Agentic session batch registration requires a non-empty scope_id.")
            normalized_entries.append(
                dict(
                    session_id=session_id,
                    scope_id=scope_id,
                    rollout_id=entry["rollout_id"],
                    group_id=entry["group_id"],
                    group_generation=entry["group_generation"],
                    gate_reason=entry["gate_reason"],
                    sampling_params=copy.deepcopy(entry["sampling_params"]),
                    session_seed=copy.deepcopy(entry["session_seed"]),
                )
            )
        if not normalized_entries:
            return 0
        await asyncio.gather(
            *(
                self._register_session(
                    session_id=entry["session_id"],
                    scope_id=entry["scope_id"],
                    rollout_id=entry["rollout_id"],
                    group_id=entry["group_id"],
                    group_generation=entry["group_generation"],
                    gate_reason=entry["gate_reason"],
                    sampling_params=entry["sampling_params"],
                    session_seed=entry["session_seed"],
                )
                for entry in normalized_entries
            )
        )
        return len(normalized_entries)

    async def mark_chat_service_response_ready(
        self,
        *,
        session_id: str,
        request_id: str,
        remote_return_at: float | None,
        response_ready_at: float | None,
        http_return_at: float | None,
    ) -> bool:
        lock = self._get_session_lock(session_id)
        if lock is None:
            return False
        async with lock:
            record = self._session_records.get(session_id)
            if record is None:
                return False
            leaf_state_hash = record.resp_state_hash_by_request_id.get(request_id)
            if leaf_state_hash is None:
                return False
            leaf = record.forest.nodes_by_hash.get(leaf_state_hash) if record.forest is not None else None
            if leaf is None or leaf.kind != "resp":
                return False
            profile = agentic_trace_events(leaf.export_metadata_patch)
            if isinstance(remote_return_at, (int, float)):
                mark_agentic_event_once(profile, "chat_service_remote_return_at", float(remote_return_at))
            if isinstance(response_ready_at, (int, float)):
                mark_agentic_event_once(profile, "chat_service_response_ready_at", float(response_ready_at))
            if isinstance(http_return_at, (int, float)):
                mark_agentic_event_once(profile, "chat_service_http_return_at", float(http_return_at))
            return True

    async def _ensure_record(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        rollout_id: int,
        sampling_params: dict[str, Any] | None = None,
    ) -> tuple[_SessionRecord, dict[str, float] | None]:
        record = self._session_records.get(session_id)
        if record is not None:
            record.rollout_id = rollout_id
            if sampling_params is not None:
                record.session_sampling_params = copy.deepcopy(sampling_params)
            if self._terminal_ir_gate_closed and not self._is_non_train_session_record(record):
                self._set_session_gate_locked(record=record, gate_reason=_GATE_REASON_TERMINAL_SHUTDOWN)
            if record.forest is not None:
                return record, None
        if record is None:
            raise _session_discarded_error(session_id)
        seed = self._normalized_session_seed(
            seed=record.session_seed,
        )
        record.session_seed = seed
        if sampling_params is None and not record.session_sampling_params:
            record.session_sampling_params = self._default_sampling_params(sample_index=seed.get("index"))
        record.forest = SessionForest.create_empty(
            session_id=session_id,
            group_index=seed.get("group_index"),
            index=seed.get("index"),
            label=seed.get("label"),
            train_metadata=copy.deepcopy(seed.get("train_metadata")),
            metadata=copy.deepcopy(seed.get("metadata")),
        )
        return record, None

    def _match_parent_state_hash(
        self,
        *,
        forest: SessionForest,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        for prefix_len in range(len(messages), 0, -1):
            prefix_hash = _messages_tools_template_state_hash(
                messages[:prefix_len],
                tools,
                chat_template_kwargs,
            )
            if prefix_hash in forest.nodes_by_hash:
                return prefix_hash, messages[prefix_len:]
        return forest.root_state_hash, messages

    def _raise_if_observation_exceeds_context(
        self,
        *,
        forest: SessionForest,
        parent_state_hash: str,
        observation_rollout_tokens: list[int],
    ) -> None:
        rollout_max_context_len = self.args.rollout_max_context_len
        if rollout_max_context_len is None:
            return
        prompt_tokens = forest.rollout_token_count(parent_state_hash) + len(observation_rollout_tokens)
        if prompt_tokens < int(rollout_max_context_len):
            return
        error = _openai_context_length_error_result(
            max_context_len=rollout_max_context_len,
            prompt_tokens=prompt_tokens,
        )["error"]
        raise AgenticChatRequestError(
            error["message"],
            code=error["code"],
            param=error["param"],
        )

    async def _append_subtree_root_observation(
        self,
        *,
        forest: SessionForest,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any],
        rollout_id: int,
    ) -> tuple[str, dict[str, float] | None]:
        root_state_hash = forest.root_state_hash
        encoded = await self._encode_messages(
            messages=messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            multimodal_inputs=_multimodal_inputs_from_messages(messages),
        )
        self._raise_if_observation_exceeds_context(
            forest=forest,
            parent_state_hash=root_state_hash,
            observation_rollout_tokens=list(encoded.backend_prompt_ids),
        )
        subtree_root = forest.append_obs(
            parent_state_hash=root_state_hash,
            rollout_id=rollout_id,
            abort_count=0,
            messages_delta=messages,
            train_token_delta=list(encoded.train_prompt_ids),
            rollout_token_delta=list(encoded.backend_prompt_ids),
            multimodal_train_inputs_delta=copy.deepcopy(encoded.multimodal_train_inputs),
            backend_image_data_delta=list(encoded.backend_image_data),
            backend_audio_data_delta=list(encoded.backend_audio_data),
            backend_video_data_delta=list(encoded.backend_video_data),
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        return subtree_root.state_hash, dict(encoded.timing)

    async def _append_observation_if_needed(
        self,
        *,
        forest: SessionForest,
        parent_state_hash: str,
        obs_delta: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any],
        rollout_id: int,
    ) -> tuple[str, dict[str, float] | None]:
        if not obs_delta:
            return parent_state_hash, None
        if parent_state_hash == forest.root_state_hash:
            return await self._append_subtree_root_observation(
                forest=forest,
                messages=obs_delta,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
                rollout_id=rollout_id,
            )
        encoded_obs = await self.backend.compiler.encode_observation_delta(
            obs_delta,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            multimodal_inputs=_multimodal_inputs_from_messages(obs_delta),
        )
        self._raise_if_observation_exceeds_context(
            forest=forest,
            parent_state_hash=parent_state_hash,
            observation_rollout_tokens=list(encoded_obs.backend_prompt_ids),
        )
        parent_subtree_root = forest.subtree_root_node(parent_state_hash)
        parent_tools = parent_subtree_root.tools if parent_subtree_root is not None else []
        if tools != parent_tools:
            raise RuntimeError(f"Session {forest.session_id!r} tools diverged inside an existing subtree")
        parent_chat_template_kwargs = (
            parent_subtree_root.chat_template_kwargs if parent_subtree_root is not None else {}
        )
        if chat_template_kwargs != parent_chat_template_kwargs:
            raise RuntimeError(
                f"Session {forest.session_id!r} chat_template_kwargs diverged inside an existing subtree"
            )
        parent_abort_count = forest.nodes_by_hash[parent_state_hash].abort_count
        obs_node = forest.append_obs(
            parent_state_hash=parent_state_hash,
            rollout_id=rollout_id,
            abort_count=parent_abort_count,
            messages_delta=obs_delta,
            train_token_delta=list(encoded_obs.train_prompt_ids),
            rollout_token_delta=list(encoded_obs.backend_prompt_ids),
            multimodal_train_inputs_delta=copy.deepcopy(encoded_obs.multimodal_train_inputs),
            backend_image_data_delta=list(encoded_obs.backend_image_data),
            backend_audio_data_delta=list(encoded_obs.backend_audio_data),
            backend_video_data_delta=list(encoded_obs.backend_video_data),
            tools=None,
            chat_template_kwargs=None,
        )
        return obs_node.state_hash, dict(encoded_obs.timing)

    def _default_sampling_params(self, *, sample_index: int | None) -> dict[str, Any]:
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
            sampling_params["sampling_seed"] = self.args.rollout_seed + int(sample_index or 0)
        return {key: value for key, value in sampling_params.items() if value is not None}

    def _budget_sampling_params(
        self,
        *,
        forest: SessionForest,
        generation_parent_hash: str,
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        budgeted = dict(sampling_params)
        current_context_tokens = forest.rollout_token_count(generation_parent_hash)
        rollout_max_context_len = self.args.rollout_max_context_len
        if rollout_max_context_len is not None:
            context_budget = max(0, int(rollout_max_context_len) - current_context_tokens)
            if "max_new_tokens" in budgeted:
                budgeted["max_new_tokens"] = min(int(budgeted["max_new_tokens"]), context_budget)
            else:
                budgeted["max_new_tokens"] = context_budget
        return budgeted

    @staticmethod
    def _accumulate_request_meta(request, *, meta_info: dict[str, Any]) -> None:
        weight_version = meta_info.get("weight_version")
        if weight_version is not None:
            request.pending_weight_version_delta.append(str(weight_version))
        request.pending_spec_delta["spec_accept_token_num"] += int(meta_info.get("spec_accept_token_num", 0) or 0)
        request.pending_spec_delta["spec_draft_token_num"] += int(meta_info.get("spec_draft_token_num", 0) or 0)
        request.pending_spec_delta["spec_verify_ct"] += int(meta_info.get("spec_verify_ct", 0) or 0)
        request.pending_spec_delta["completion_token_num"] += int(meta_info.get("completion_tokens", 0) or 0)
        request.pending_prefix_cache_delta["cached_tokens"] += int(meta_info.get("cached_tokens", 0) or 0)
        request.pending_prefix_cache_delta["total_prompt_tokens"] += int(meta_info.get("prompt_tokens", 0) or 0)

    @staticmethod
    def _remove_ir_from_queue(record: _SessionRecord, ir_id: str) -> None:
        while ir_id in record.ir_queue:
            record.ir_queue.remove(ir_id)

    def _enqueue_ir_locked(self, record: _SessionRecord, ir: InflightRequest) -> None:
        self._remove_ir_from_queue(record, ir.request_id)
        if record.protected_until_finalize:
            ir.kind = RequestKind.PROTECTED
        ir.backend_started = False
        record.ir_queue.append(ir.request_id)

    def _pop_next_ir_locked(self, record: _SessionRecord) -> InflightRequest | None:
        while record.ir_queue:
            ir_id = record.ir_queue.popleft()
            ir = record.irs_by_id.get(ir_id)
            if ir is not None:
                return ir
        return None

    def _complete_waiter_locked(
        self,
        record: _SessionRecord,
        ir_id: str,
        *,
        result: dict[str, Any] | None = None,
        exc: Exception | None = None,
    ) -> None:
        waiter = record.pending_chat_waiters.pop(ir_id, None)
        if waiter is None or waiter.done():
            return
        if exc is not None:
            waiter.set_exception(exc)
            return
        waiter.set_result(result)

    def _release_ir_locked(self, record: _SessionRecord, ir_id: str) -> None:
        self._remove_ir_from_queue(record, ir_id)
        record.irs_by_id.pop(ir_id, None)
        record.active_ir_runner_tasks.pop(ir_id, None)

    @staticmethod
    def _clear_removed_record_state(record: _SessionRecord) -> None:
        record.forest = None
        record.group_id = None
        record.group_generation = 0
        record.session_sampling_params.clear()
        record.session_seed.clear()
        record.resp_state_hash_by_request_id.clear()
        record.irs_by_id.clear()
        record.ir_queue.clear()
        record.active_ir_runner_tasks.clear()
        record.pending_chat_waiters.clear()
        record.gate_reason = None
        record.protected_until_finalize = False

    def _remove_ir_from_runnable_state_locked(self, record: _SessionRecord, ir_id: str) -> None:
        # A gated IR must stop being runnable locally. Backend-started IRs are handled by abort_all.
        self._remove_ir_from_queue(record, ir_id)
        record.active_ir_runner_tasks.pop(ir_id, None)

    @staticmethod
    def _mark_record_protected_locked(record: _SessionRecord) -> None:
        record.protected_until_finalize = True
        for ir in record.irs_by_id.values():
            ir.kind = RequestKind.PROTECTED
        AgenticSessionShard._assert_protected_kind_invariant(record)

    @staticmethod
    def _assert_protected_kind_invariant(record: _SessionRecord) -> None:
        for ir in record.irs_by_id.values():
            if record.protected_until_finalize and ir.kind != RequestKind.PROTECTED:
                raise RuntimeError("Protected session contains a non-protected IR.")
            if not record.protected_until_finalize and ir.kind == RequestKind.PROTECTED:
                raise RuntimeError("Protected IR exists outside a protected session.")

    def _protected_abort_count_threshold(self) -> int | None:
        if not getattr(self.args, "partial_rollout", False):
            return None
        return self.args.partial_rollout_max_aborted_count

    def _set_session_gate_locked(self, *, record: _SessionRecord, gate_reason: SessionGateReason | str | None) -> bool:
        normalized_gate_reason = _normalize_session_gate_reason(gate_reason)
        changed = record.gate_reason != normalized_gate_reason
        record.gate_reason = normalized_gate_reason
        return changed

    def _build_ir_locked(
        self,
        *,
        record: _SessionRecord,
        session_id: str,
        parent_state_hash: str,
        sampling_params: dict[str, Any],
        logprobs: bool,
        chat_request_arrive_at: float,
        chat_lock_acquired_at: float,
        bootstrap_compiler_timing: dict[str, float] | None,
        observation_compiler_timing: dict[str, float] | None,
    ) -> InflightRequest:
        parent_abort_count = record.forest.nodes_by_hash[parent_state_hash].abort_count
        request_kind = record.forest.resolve_request_kind(
            abort_count=parent_abort_count,
            resumed=False,
            protected_abort_count_threshold=self._protected_abort_count_threshold(),
        )
        if request_kind == RequestKind.PROTECTED:
            self._mark_record_protected_locked(record)
        prefix = record.forest.build_execution_prefix(parent_state_hash)
        request_id = f"req_{session_id}_{record.next_ir_sequence}"
        record.next_ir_sequence += 1
        ir = InflightRequest(
            request_id=request_id,
            parent_state_hash=parent_state_hash,
            rollout_id=record.rollout_id,
            kind=RequestKind.PROTECTED if record.protected_until_finalize else request_kind,
            abort_count=parent_abort_count,
            sampling_params=copy.deepcopy(sampling_params),
            logprobs=bool(logprobs),
            history_train_token_prefix=prefix.train_token_prefix,
            history_rollout_token_prefix=prefix.rollout_token_prefix,
            history_backend_image_data=prefix.backend_image_data,
            history_backend_audio_data=prefix.backend_audio_data,
            history_backend_video_data=prefix.backend_video_data,
        )
        profile = agentic_trace_events(ir.pending_export_metadata_patch)
        ir_created_at = time.time()
        mark_agentic_event(profile, "ir_created_at", ir_created_at)
        mark_agentic_event(profile, "chat_request_arrive_at", chat_request_arrive_at)
        mark_agentic_event(profile, "chat_lock_acquired_at", chat_lock_acquired_at)
        if bootstrap_compiler_timing:
            profile.update(copy.deepcopy(bootstrap_compiler_timing))
        if observation_compiler_timing:
            profile.update(copy.deepcopy(observation_compiler_timing))
        record.irs_by_id[ir.request_id] = ir
        self._assert_protected_kind_invariant(record)
        return ir

    def _requeue_aborted_ir_locked(
        self,
        *,
        record: _SessionRecord,
        ir_id: str,
        ir: InflightRequest,
    ) -> None:
        ir.abort_count += 1
        ir.kind = record.forest.resolve_request_kind(
            abort_count=ir.abort_count,
            resumed=True,
            protected_abort_count_threshold=self._protected_abort_count_threshold(),
        )
        if ir.kind == RequestKind.PROTECTED:
            self._mark_record_protected_locked(record)
        if record.protected_until_finalize:
            ir.kind = RequestKind.PROTECTED
        self._assert_protected_kind_invariant(record)
        ir.pending_status = "aborted"
        gate_reason = (
            _GATE_REASON_TERMINAL_SHUTDOWN
            if record.gate_reason == _GATE_REASON_TERMINAL_SHUTDOWN
            else _GATE_REASON_PARTIAL_RESUME
        )
        self._gate_active_ir_release_locked(
            record=record,
            ir_id=ir_id,
            ir=ir,
            gate_reason=gate_reason,
        )

    def _gate_active_ir_release_locked(
        self,
        *,
        record: _SessionRecord,
        ir_id: str,
        ir: InflightRequest,
        gate_reason: SessionGateReason | str | None = None,
    ) -> None:
        """Move a not-yet-backend-started active IR behind the session gate."""
        if gate_reason is None:
            gate_reason = (
                _GATE_REASON_TERMINAL_SHUTDOWN
                if record.gate_reason == _GATE_REASON_TERMINAL_SHUTDOWN
                else _GATE_REASON_PARTIAL_RESUME
            )
        self._set_session_gate_locked(record=record, gate_reason=gate_reason)
        self._remove_ir_from_runnable_state_locked(record, ir_id)
        self._enqueue_ir_locked(record, ir)

    def _should_requeue_active_ir_locked(self, *, record: _SessionRecord) -> bool:
        decision = _decide_ir_release(record=record)
        if decision.allow:
            return False
        if record.protected_until_finalize and decision.blocked_reason != _WAITING_REASON_TERMINAL_SHUTDOWN_GATE:
            return False
        return True

    @staticmethod
    def _is_interrupt_backpressure_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code != 503:
            return False
        text = ""
        try:
            text = str(response.text or "")
        except Exception:
            text = ""
        return "no_available_workers" in text

    def _maybe_start_next_ir_locked(self, *, session_id: str, record: _SessionRecord) -> bool:
        if not _decide_ir_release(record=record).allow:
            return False
        started = False
        while True:
            ir = self._pop_next_ir_locked(record)
            if ir is None:
                break
            if ir.sampling_params.get("max_new_tokens", 1) == 0:
                self._complete_waiter_locked(
                    record,
                    ir.request_id,
                    result=_openai_context_length_error_result(
                        max_context_len=self.args.rollout_max_context_len,
                        prompt_tokens=len(ir.history_rollout_token_prefix) + len(ir.pending_rollout_token_delta),
                    ),
                )
                self._release_ir_locked(record, ir.request_id)
                started = True
                continue
            ir.rollout_id = record.rollout_id
            ir.runner_epoch += 1
            profile = agentic_trace_events(ir.pending_export_metadata_patch)
            mark_agentic_event(profile, "ir_activated_at")
            # This is a shard-local IR runner coroutine. It owns one backend generation request while the
            # resident managed agent task remains blocked on the session chat API.
            runner_epoch = ir.runner_epoch
            record.active_ir_runner_tasks[ir.request_id] = asyncio.create_task(
                self._run_ir(session_id=session_id, ir_id=ir.request_id, runner_epoch=runner_epoch)
            )
            started = True
        return started

    def _activate_record_locked(self, *, session_id: str, record: _SessionRecord, rollout_id: int) -> bool:
        record.rollout_id = rollout_id
        gate_reason = (
            _GATE_REASON_TERMINAL_SHUTDOWN
            if self._terminal_ir_gate_closed and not self._is_non_train_session_record(record)
            else None
        )
        self._set_session_gate_locked(record=record, gate_reason=gate_reason)
        return self._maybe_start_next_ir_locked(session_id=session_id, record=record)

    async def prepare_group_status(self, *, scope_id: str) -> list[dict[str, Any]]:
        if not isinstance(scope_id, str) or not scope_id:
            raise RuntimeError("prepare_group_status requires a non-empty scope_id.")
        counts: dict[tuple[str, int], dict[str, Any]] = {}
        for record in self._session_records.values():
            if record.scope_id != scope_id:
                continue
            if record.gate_reason != _GATE_REASON_PREPARE or not record.group_id:
                continue
            key = (str(record.group_id), record.group_generation)
            entry = counts.setdefault(
                key,
                {
                    "group_id": record.group_id,
                    "group_generation": record.group_generation,
                    "total_sessions": 0,
                    "ready_sessions": 0,
                },
            )
            entry["total_sessions"] += 1
            if record.irs_by_id:
                entry["ready_sessions"] += 1
        return list(counts.values())

    async def activate_group_sessions(
        self,
        *,
        scope_id: str,
        groups: list[dict[str, Any]],
        rollout_id: int,
    ) -> dict[str, int]:
        if not isinstance(scope_id, str) or not scope_id:
            raise RuntimeError("activate_group_sessions requires a non-empty scope_id.")
        group_keys = {(str(item["group_id"]), int(item["group_generation"])) for item in groups}
        if not group_keys:
            return {"activated_sessions": 0, "started_sessions": 0}
        activated_sessions = 0
        started_irs = 0
        for session_id, record in list(self._session_records.items()):
            if record.scope_id != scope_id:
                continue
            if (record.group_id, record.group_generation) not in group_keys:
                continue
            lock = self._get_session_lock(session_id)
            if lock is None:
                continue
            async with lock:
                current = self._session_records.get(session_id)
                if current is None:
                    continue
                if current.scope_id != scope_id:
                    continue
                if (current.group_id, current.group_generation) not in group_keys:
                    continue
                activated_sessions += 1
                if self._activate_record_locked(session_id=session_id, record=current, rollout_id=rollout_id):
                    started_irs += 1
        return {"activated_sessions": activated_sessions, "started_sessions": started_irs}

    def _apply_generate_result(self, ir: InflightRequest, *, result: Any) -> None:
        ir.pending_train_token_delta.extend(result.new_tokens)
        ir.pending_rollout_token_delta.extend(result.new_tokens)
        ir.pending_loss_mask_delta.extend([1] * len(result.new_tokens))
        ir.pending_logprob_delta.extend(result.new_log_probs)
        ir.pending_wall_elapsed_s += float(result.elapsed)
        ir.pending_generation_elapsed_s += float(result.elapsed)
        ir.pending_status = _transport_status_from_finish_type(result.finish_type)
        ir.latest_backend_meta = copy.deepcopy(result.meta_info)
        self._accumulate_request_meta(ir, meta_info=result.meta_info)
        ir.pending_routed_experts = _decode_routed_experts(
            args=self.args,
            meta_info=result.meta_info,
            token_count=len(ir.history_train_token_prefix) + len(ir.pending_train_token_delta),
        )

    def _terminal_response_locked(
        self, *, record: _SessionRecord, ir: InflightRequest, finish_type: str
    ) -> dict[str, Any]:
        ir.pending_export_metadata_patch["request_id"] = ir.request_id
        ir.pending_export_metadata_patch["request_kind"] = ir.kind.value
        ir.pending_export_metadata_patch["base_state_hash"] = ir.parent_state_hash
        response_messages_delta, response_message, has_tool_calls = _decode_response_payload(
            args=self.args,
            tokenizer=self.backend.tokenizer,
            token_ids=ir.pending_train_token_delta,
            tools=record.forest.subtree_tools(ir.parent_state_hash),
            parent_state_hash=ir.parent_state_hash,
        )
        resp_node = record.forest.append_resp(
            parent_state_hash=ir.parent_state_hash,
            rollout_id=ir.rollout_id,
            abort_count=ir.abort_count,
            messages_delta=response_messages_delta,
            train_token_delta=ir.pending_train_token_delta,
            rollout_token_delta=ir.pending_rollout_token_delta,
            loss_mask_delta=ir.pending_loss_mask_delta,
            logprob_delta=ir.pending_logprob_delta,
            weight_version_delta=ir.pending_weight_version_delta,
            spec_delta=ir.pending_spec_delta,
            prefix_cache_delta=ir.pending_prefix_cache_delta,
            wall_elapsed_s=ir.pending_wall_elapsed_s,
            generation_elapsed_s=ir.pending_generation_elapsed_s,
            status=ir.pending_status or "completed",
            rollout_routed_experts=ir.pending_routed_experts,
            export_metadata_patch=ir.pending_export_metadata_patch,
        )
        record.resp_state_hash_by_request_id[ir.request_id] = resp_node.state_hash
        resp_profile = agentic_trace_events(resp_node.export_metadata_patch)
        prompt_tokens = int(ir.latest_backend_meta.get("prompt_tokens", 0) or 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(ir.pending_train_token_delta),
            "total_tokens": prompt_tokens + len(ir.pending_train_token_delta),
        }
        finish_reason = _openai_finish_reason("length" if ir.pending_status == "truncated" else finish_type)
        if has_tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        chat_ended_at = time.time()
        mark_agentic_event(resp_profile, "chat_end_at", chat_ended_at)
        return {
            "request_id": ir.request_id,
            "message": response_message,
            "logprobs": (
                _openai_token_logprobs_payload(
                    tokenizer=self.backend.tokenizer,
                    token_ids=ir.pending_train_token_delta,
                    token_logprobs=ir.pending_logprob_delta,
                )
                if ir.logprobs
                else None
            ),
            "finish_reason": finish_reason,
            "usage": usage,
        }

    async def _release_sglang_request_permit_after_cancel(self, *, limiter: Any, acquire_ref: Any) -> None:
        try:
            await acquire_ref
        except Exception:
            logger.exception("Cancelled SGLang permit acquire failed before taking a remote permit.")
            return
        try:
            await limiter.release_sglang_request_permit.remote()
        except Exception:
            logger.exception("Failed to release remote SGLang permit acquired after cancellation.")

    async def _run_ir(self, *, session_id: str, ir_id: str, runner_epoch: int) -> None:
        lock = self._get_session_lock(session_id)
        if lock is None:
            return
        async with lock:
            record = self._session_records.get(session_id)
            if record is None:
                return
            ir = record.irs_by_id.get(ir_id)
            if ir is None or ir_id not in record.active_ir_runner_tasks or ir.runner_epoch != runner_epoch:
                return
            if self._should_requeue_active_ir_locked(record=record):
                self._gate_active_ir_release_locked(record=record, ir_id=ir_id, ir=ir)
                self._maybe_start_next_ir_locked(session_id=session_id, record=record)
                return
            profile = agentic_trace_events(ir.pending_export_metadata_patch)
            mark_agentic_event(profile, "generation_queue_enter_at")
        permit_acquired = False
        generation_profile: dict[str, Any] | None = None
        try:
            if self._sglang_request_semaphore is not None:
                while True:
                    if self._sglang_request_semaphore.acquire(blocking=False):
                        permit_acquired = True
                        break
                    await asyncio.sleep(0.01)
            elif self._sglang_request_limiter is not None:
                acquire_ref = self._sglang_request_limiter.acquire_sglang_request_permit.remote()
                try:
                    await asyncio.shield(acquire_ref)
                except asyncio.CancelledError:
                    asyncio.create_task(
                        self._release_sglang_request_permit_after_cancel(
                            limiter=self._sglang_request_limiter,
                            acquire_ref=acquire_ref,
                        )
                    )
                    raise
                permit_acquired = True
            lock = self._get_session_lock(session_id)
            if lock is None:
                return
            async with lock:
                record = self._session_records.get(session_id)
                if record is None:
                    return
                ir = record.irs_by_id.get(ir_id)
                if ir is None or ir_id not in record.active_ir_runner_tasks or ir.runner_epoch != runner_epoch:
                    return
                if self._should_requeue_active_ir_locked(record=record):
                    self._gate_active_ir_release_locked(record=record, ir_id=ir_id, ir=ir)
                    self._maybe_start_next_ir_locked(session_id=session_id, record=record)
                    return
                ir.backend_started = True
                generation_profile = agentic_trace_events(ir.pending_export_metadata_patch)
            mark_agentic_event(generation_profile, "generation_start_at")
            result = await self.backend.generate(
                input_ids=list(ir.history_rollout_token_prefix) + list(ir.pending_rollout_token_delta),
                sampling_params=ir.sampling_params,
                session_id=session_id,
                request_id=ir.request_id,
                image_data=ir.history_backend_image_data,
                audio_data=ir.history_backend_audio_data,
                video_data=ir.history_backend_video_data,
                return_logprob=record.scope_id == "train" or ir.logprobs,
            )
        except BackendContextLengthExceededError:
            lock = self._get_session_lock(session_id)
            if lock is None:
                return
            async with lock:
                record = self._session_records.get(session_id)
                if record is None:
                    return
                self._complete_waiter_locked(
                    record,
                    ir_id,
                    result=_openai_context_length_error_result(
                        max_context_len=self.args.rollout_max_context_len,
                        prompt_tokens=len(ir.history_rollout_token_prefix) + len(ir.pending_rollout_token_delta),
                        requested_completion_tokens=ir.sampling_params.get("max_new_tokens"),
                    ),
                )
                self._release_ir_locked(record, ir_id)
            return
        except Exception as exc:
            if generation_profile is not None:
                mark_agentic_event(generation_profile, "generation_end_at")
            lock = self._get_session_lock(session_id)
            if lock is None:
                return
            async with lock:
                record = self._session_records.get(session_id)
                if record is None:
                    return
                ir = record.irs_by_id.get(ir_id)
                if (
                    ir is not None
                    and ir_id in record.active_ir_runner_tasks
                    and self._should_requeue_active_ir_locked(record=record)
                    and self._is_interrupt_backpressure_error(exc)
                ):
                    self._requeue_aborted_ir_locked(record=record, ir_id=ir_id, ir=ir)
                    self._maybe_start_next_ir_locked(session_id=session_id, record=record)
                    return
                self._complete_waiter_locked(record, ir_id, exc=exc)
                self._release_ir_locked(record, ir_id)
            return
        finally:
            if permit_acquired:
                if self._sglang_request_semaphore is not None:
                    self._sglang_request_semaphore.release()
                else:
                    await self._sglang_request_limiter.release_sglang_request_permit.remote()

        lock = self._get_session_lock(session_id)
        if lock is None:
            return
        async with lock:
            record = self._session_records.get(session_id)
            if record is None:
                return
            ir = record.irs_by_id.get(ir_id)
            if ir is None or ir_id not in record.active_ir_runner_tasks:
                return
            profile = agentic_trace_events(ir.pending_export_metadata_patch)
            mark_agentic_event(profile, "generation_end_at")
            self._apply_generate_result(ir, result=result)
            finish_type = result.finish_type
            # SGLang may process an abort after the decode step that produced EOS, returning
            # finish_reason=abort while output_ids already end with a stop token.
            if _last_token_is_stop_token(
                token_ids=result.new_tokens,
                tokenizer=self.backend.tokenizer,
                sampling_params=ir.sampling_params,
            ):
                finish_type = "stop"
                ir.pending_status = _transport_status_from_finish_type(finish_type)
            if finish_type == "abort":
                # SGLang aborts only the active backend generation request. The managed agent task remains
                # resident, and its blocked chat call resumes after this IR is requeued.
                if "max_new_tokens" in ir.sampling_params:
                    remaining = ir.sampling_params["max_new_tokens"] - len(result.new_tokens)
                    ir.sampling_params["max_new_tokens"] = remaining
                self._requeue_aborted_ir_locked(record=record, ir_id=ir_id, ir=ir)
                self._maybe_start_next_ir_locked(session_id=session_id, record=record)
                return
            try:
                payload = self._terminal_response_locked(record=record, ir=ir, finish_type=finish_type)
            except AgenticChatRequestError as exc:
                self._complete_waiter_locked(
                    record,
                    ir_id,
                    result=_openai_error_result(
                        exc.message,
                        code=exc.code,
                        param=exc.param,
                        status_code=exc.status_code,
                        error_type=exc.error_type,
                    ),
                )
                self._release_ir_locked(record, ir_id)
                self._maybe_start_next_ir_locked(session_id=session_id, record=record)
                return
            except Exception:
                logger.exception(
                    "Failed to build terminal agentic chat response for session_id=%s request_id=%s",
                    session_id,
                    ir_id,
                )
                self._complete_waiter_locked(
                    record,
                    ir_id,
                    result=_openai_error_result(
                        "Internal error while building agentic chat response.",
                        code="internal_error",
                        status_code=500,
                        error_type="internal_error",
                    ),
                )
                self._release_ir_locked(record, ir_id)
                self._maybe_start_next_ir_locked(session_id=session_id, record=record)
                return
            self._complete_waiter_locked(record, ir_id, result=payload)
            self._release_ir_locked(record, ir_id)
            self._maybe_start_next_ir_locked(session_id=session_id, record=record)

    async def release_partial_resume_gate(self, *, rollout_id: int) -> int:
        """Release sessions blocked by the partial-resume gate for the next
        rollout step.

        Prepare-gated sessions are released only by activate_group_sessions().
        """
        opened = 0
        for session_id in list(self._session_records):
            lock = self._get_session_lock(session_id)
            if lock is None:
                continue
            async with lock:
                current = self._session_records.get(session_id)
                if current is None:
                    continue
                if self._is_non_train_session_record(current):
                    continue
                current.rollout_id = rollout_id
                if current.gate_reason == _GATE_REASON_PARTIAL_RESUME:
                    if self._activate_record_locked(
                        session_id=session_id,
                        record=current,
                        rollout_id=current.rollout_id,
                    ):
                        opened += 1
                    continue
                if self._maybe_start_next_ir_locked(session_id=session_id, record=current):
                    opened += 1
        return opened

    async def _gate_records_for_ir_release(
        self,
        *,
        gate_reason: str,
        rollout_id: int | None,
        include_all_rollouts: bool,
        lock_protected: bool,
        skip_prepare_gate: bool,
    ) -> int:
        """Apply a session gate and park active IRs that have not reached
        SGLang."""
        locked = 0
        for session_id in list(self._session_records):
            lock = self._get_session_lock(session_id)
            if lock is None:
                continue
            async with lock:
                current = self._session_records.get(session_id)
                if current is None:
                    continue
                if self._is_non_train_session_record(current):
                    continue
                if skip_prepare_gate and current.gate_reason == _GATE_REASON_PREPARE:
                    continue
                record_matches_rollout = include_all_rollouts or current.rollout_id == rollout_id
                active_irs = []
                for ir_id in tuple(current.active_ir_runner_tasks):
                    ir = current.irs_by_id.get(ir_id)
                    if ir is None:
                        continue
                    if not include_all_rollouts and ir.rollout_id != rollout_id:
                        continue
                    active_irs.append((ir_id, ir))
                if not record_matches_rollout and not active_irs:
                    continue
                if not lock_protected and current.protected_until_finalize:
                    continue
                if self._set_session_gate_locked(record=current, gate_reason=gate_reason):
                    locked += 1
                for ir_id, ir in active_irs:
                    if ir.backend_started:
                        continue
                    self._gate_active_ir_release_locked(
                        record=current,
                        ir_id=ir_id,
                        ir=ir,
                        gate_reason=gate_reason,
                    )
                    locked += 1
        return locked

    async def gate_rollout_irs_for_partial_resume(self, *, rollout_id: int) -> int:
        """Gate current-rollout IRs so unfinished partial work can resume in a
        later step."""
        return await self._gate_records_for_ir_release(
            gate_reason=_GATE_REASON_PARTIAL_RESUME,
            rollout_id=rollout_id,
            include_all_rollouts=False,
            lock_protected=False,
            skip_prepare_gate=True,
        )

    async def gate_rollout_irs_for_discard(self, *, rollout_id: int) -> int:
        """Gate current-rollout IRs for non-partial tail discard."""
        return await self._gate_records_for_ir_release(
            gate_reason=_GATE_REASON_TERMINAL_SHUTDOWN,
            rollout_id=rollout_id,
            include_all_rollouts=False,
            lock_protected=True,
            skip_prepare_gate=True,
        )

    async def gate_all_irs_for_shutdown(self) -> int:
        """Gate every train IR before terminal shutdown or cleanup."""
        self._terminal_ir_gate_closed = True
        return await self._gate_records_for_ir_release(
            gate_reason=_GATE_REASON_TERMINAL_SHUTDOWN,
            rollout_id=None,
            include_all_rollouts=True,
            lock_protected=True,
            skip_prepare_gate=False,
        )

    async def active_rollout_request_counts(self, *, rollout_id: int) -> dict[str, int]:
        protected_active = 0
        abortable_active = 0
        for session_id in list(self._session_records):
            lock = self._get_session_lock(session_id)
            if lock is None:
                continue
            async with lock:
                current = self._session_records.get(session_id)
                if current is None:
                    continue
                if self._is_non_train_session_record(current):
                    continue
                if current.gate_reason == _GATE_REASON_PREPARE:
                    continue
                record_matches_rollout = current.rollout_id == rollout_id
                active_irs = []
                for ir_id in current.active_ir_runner_tasks:
                    ir = current.irs_by_id.get(ir_id)
                    if ir is None:
                        continue
                    if ir.rollout_id == rollout_id:
                        record_matches_rollout = True
                        active_irs.append(ir)
                    else:
                        continue
                if not record_matches_rollout:
                    continue
                if current.protected_until_finalize and current.gate_reason != _GATE_REASON_TERMINAL_SHUTDOWN:
                    protected_active += 1
                    continue
                if any(ir.backend_started for ir in active_irs):
                    abortable_active += 1
        return {
            "protected_active": protected_active,
            "abortable_active": abortable_active,
            "evaluating": self._evaluating,
        }

    def _finalize_sample_from_leaf(
        self,
        *,
        record: _SessionRecord,
        leaf_node: MsgNode,
        reward: float | dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ):
        metadata_patch = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
        finalize_started_at = time.time()
        leaf_profile = agentic_trace_events(leaf_node.export_metadata_patch)
        mark_agentic_event(leaf_profile, "finalize_start_at", finalize_started_at)
        if reward is not None:
            leaf_node.reward = copy.deepcopy(reward)
        if metadata_patch:
            leaf_node.export_metadata_patch.update(metadata_patch)
        sample = record.forest.build_sample(
            leaf_state_hash=leaf_node.state_hash,
            tokenizer=self.backend.tokenizer,
            # mask_offpolicy_in_partial_rollout=bool(
            #     self.args.partial_rollout and self.args.mask_offpolicy_in_partial_rollout
            # ),
        )
        finalize_ended_at = time.time()
        mark_metadata_agentic_event(sample.metadata, "finalize_end_at", finalize_ended_at)
        return sample

    def _exportable_leaf_node(self, record: _SessionRecord) -> MsgNode | None:
        if record.forest is None:
            return None
        leaf_hashes = record.forest.export_leaf_hashes()
        if len(leaf_hashes) != 1:
            return None
        lineage = record.forest.lineage(leaf_hashes[0])
        if not any(node.kind == "resp" for node in lineage):
            return None
        return lineage[-1]

    @staticmethod
    def _build_transport_from_sample(sample) -> FinalizedResultTransport:
        artifact = TrainingFieldArtifact.from_sample(sample)
        if ray.is_initialized():
            artifact_ref = ray.put(artifact)
        else:
            artifact_ref = artifact
        return FinalizedResultTransport(
            status=sample.status.value,
            artifact_ref=artifact_ref,
        )

    async def chat(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
        temperature: float | None,
        top_p: float | None,
        max_completion_tokens: int | None,
        stop: list[str] | str | None,
        seed: int | None,
        logprobs: bool = False,
    ) -> dict[str, Any]:
        chat_request_arrive_at = time.time()
        try:
            lock = self._get_session_lock(session_id)
            if lock is None:
                raise _session_discarded_error(session_id)
            async with lock:
                chat_lock_acquired_at = time.time()
                existing_record = self._session_records.get(session_id)
                tools = tools or []
                chat_template_kwargs = {**self._template_kwargs(), **(chat_template_kwargs or {})}
                existing_rollout_id = existing_record.rollout_id if existing_record is not None else 0
                record, bootstrap_compiler_timing = await self._ensure_record(
                    session_id=session_id,
                    messages=messages,
                    rollout_id=existing_rollout_id,
                )
                sampling_params: dict[str, Any] = copy.deepcopy(record.session_sampling_params)
                if temperature is not None:
                    sampling_params["temperature"] = temperature
                if top_p is not None:
                    sampling_params["top_p"] = top_p
                if max_completion_tokens is not None:
                    sampling_params["max_new_tokens"] = int(max_completion_tokens)
                if stop is not None:
                    sampling_params["stop"] = stop
                if seed is not None:
                    sampling_params["sampling_seed"] = int(seed)
                parent_state_hash, obs_delta = self._match_parent_state_hash(
                    forest=record.forest,
                    messages=messages,
                    tools=tools,
                    chat_template_kwargs=chat_template_kwargs,
                )
                generation_parent_hash, observation_compiler_timing = await self._append_observation_if_needed(
                    forest=record.forest,
                    parent_state_hash=parent_state_hash,
                    obs_delta=obs_delta,
                    tools=tools,
                    chat_template_kwargs=chat_template_kwargs,
                    rollout_id=record.rollout_id,
                )
                sampling_params = self._budget_sampling_params(
                    forest=record.forest,
                    generation_parent_hash=generation_parent_hash,
                    sampling_params=sampling_params,
                )
                ir = self._build_ir_locked(
                    record=record,
                    session_id=session_id,
                    parent_state_hash=generation_parent_hash,
                    sampling_params=sampling_params,
                    logprobs=logprobs,
                    chat_request_arrive_at=chat_request_arrive_at,
                    chat_lock_acquired_at=chat_lock_acquired_at,
                    bootstrap_compiler_timing=bootstrap_compiler_timing,
                    observation_compiler_timing=observation_compiler_timing,
                )
                waiter = asyncio.get_running_loop().create_future()
                record.pending_chat_waiters[ir.request_id] = waiter
                if ir.sampling_params.get("max_new_tokens", 1) == 0:
                    self._complete_waiter_locked(
                        record,
                        ir.request_id,
                        result=_openai_context_length_error_result(
                            max_context_len=self.args.rollout_max_context_len,
                            prompt_tokens=len(ir.history_rollout_token_prefix) + len(ir.pending_rollout_token_delta),
                        ),
                    )
                    self._release_ir_locked(record, ir.request_id)
                else:
                    self._enqueue_ir_locked(record, ir)
                    self._maybe_start_next_ir_locked(session_id=session_id, record=record)
        except AgenticChatRequestError as exc:
            return _openai_error_from_exc(exc)
        del (
            messages,
            tools,
            chat_template_kwargs,
            existing_record,
            record,
            sampling_params,
            parent_state_hash,
            obs_delta,
            generation_parent_hash,
            ir,
            bootstrap_compiler_timing,
            observation_compiler_timing,
        )
        try:
            return await waiter
        except AgenticChatRequestError as exc:
            return _openai_error_from_exc(exc)
        except Exception:
            logger.exception("Agentic chat waiter failed for session_id=%s", session_id)
            return _openai_error_result(
                "Internal error while handling agentic chat request.",
                code="internal_error",
                status_code=500,
                error_type="internal_error",
            )

    async def _abort_backend_request_ids(self, request_ids: list[str]) -> None:
        if not request_ids:
            return
        urls = await _sglang_worker_urls(self.args)
        if not urls:
            raise RuntimeError(
                "Cannot abort discarded agentic session requests because no SGLang worker urls are available."
            )
        results = await asyncio.gather(
            *(post(f"{url}/abort_request", {"rid": request_id}) for request_id in request_ids for url in urls),
            return_exceptions=True,
        )
        failed = sum(1 for result in results if isinstance(result, BaseException))
        if failed:
            raise RuntimeError(f"Failed to abort {failed} discarded agentic session request(s).")

    async def _finish_discarded_session(
        self,
        *,
        session_id: str,
        lock: asyncio.Lock,
        removed: _SessionRecord | None,
        active_tasks: list[asyncio.Task[Any]],
        backend_request_ids: list[str],
        waiters: list[asyncio.Future[Any]],
        stats: dict[str, int] | None,
    ) -> bool:
        def _log_discarded_runner_result(task: asyncio.Task[Any]) -> None:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(
                    "Discarded agentic session session_id=%s active IR runner finished with error: %s",
                    session_id,
                    exc,
                )

        for active_runner_task in active_tasks:
            if not active_runner_task.done():
                active_runner_task.cancel()
        await self._abort_backend_request_ids(backend_request_ids)
        for active_runner_task in active_tasks:
            active_runner_task.add_done_callback(_log_discarded_runner_result)
        for waiter in waiters:
            if waiter.done():
                continue
            waiter.set_result(
                _openai_error_result(
                    f"Agentic session {session_id!r} was discarded.",
                    code="session_discarded",
                    param="session_id",
                    status_code=404,
                    error_type="not_found_error",
                )
            )
        async with lock:
            current_lock = self._session_locks.get(session_id)
            if current_lock is lock and session_id not in self._session_records:
                self._session_locks.pop(session_id, None)
        if removed is not None:
            logger.debug(
                "Discarded agentic session session_id=%s rollout_id=%s node_count=%s request_count=%s active_sessions=%s active_locks=%s",
                session_id,
                removed.rollout_id,
                stats["node_count"] if stats is not None else 0,
                stats["request_count"] if stats is not None else 0,
                len(self._session_records),
                len(self._session_locks),
            )
        return removed is not None

    def _discard_session_locked(
        self, *, session_id: str
    ) -> tuple[
        _SessionRecord | None,
        dict[str, int] | None,
        list[asyncio.Task[Any]],
        list[asyncio.Future[Any]],
        list[str],
    ]:
        removed = self._session_records.pop(session_id, None)
        stats = self._session_stats(removed) if removed is not None else None
        active_tasks: list[asyncio.Task[Any]] = []
        waiters: list[asyncio.Future[Any]] = []
        backend_request_ids: list[str] = []
        if removed is not None:
            active_tasks = list(removed.active_ir_runner_tasks.values())
            waiters = list(removed.pending_chat_waiters.values())
            backend_request_ids = [ir.request_id for ir in removed.irs_by_id.values() if ir.backend_started]
            self._clear_removed_record_state(removed)
        return removed, stats, active_tasks, waiters, backend_request_ids

    async def finalize_and_discard(
        self,
        *,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        reward: float | dict[str, Any] | None = None,
    ) -> FinalizedResultTransport:
        finalize_arrive_at = time.time()
        lock = self._get_session_lock(session_id)
        if lock is None:
            return FinalizedResultTransport(
                status="discarded",
                metadata={"discard_reason": "already_discarded"},
            )
        active_tasks: list[asyncio.Task[Any]] = []
        waiters: list[asyncio.Future[Any]] = []
        removed = None
        stats = None
        transport = None
        async with lock:
            finalize_lock_acquired_at = time.time()
            record = self._session_records.get(session_id)
            if record is None:
                return FinalizedResultTransport(
                    status="discarded",
                    metadata={"discard_reason": "already_discarded"},
                )
            leaf_node = self._exportable_leaf_node(record)
            if leaf_node is None:
                transport = FinalizedResultTransport(
                    status="non_finalizable",
                    metadata={"discard_reason": "no_committed_response"},
                )
            else:
                profile = agentic_trace_events(leaf_node.export_metadata_patch)
                mark_agentic_event(profile, "finalize_arrive_at", finalize_arrive_at)
                mark_agentic_event(profile, "finalize_lock_acquired_at", finalize_lock_acquired_at)
                sample = self._finalize_sample_from_leaf(
                    record=record,
                    leaf_node=leaf_node,
                    reward=reward,
                    metadata=metadata,
                )
                transport = self._build_transport_from_sample(sample)
            removed, stats, active_tasks, waiters, backend_request_ids = self._discard_session_locked(
                session_id=session_id
            )
        await self._finish_discarded_session(
            session_id=session_id,
            lock=lock,
            removed=removed,
            active_tasks=active_tasks,
            backend_request_ids=backend_request_ids,
            waiters=waiters,
            stats=stats,
        )
        return transport

    async def discard_session(self, *, session_id: str) -> bool:
        lock = self._get_session_lock(session_id)
        if lock is None:
            return False
        active_tasks: list[asyncio.Task[Any]] = []
        waiters: list[asyncio.Future[Any]] = []
        removed = None
        stats = None
        async with lock:
            removed, stats, active_tasks, waiters, backend_request_ids = self._discard_session_locked(
                session_id=session_id
            )
        return await self._finish_discarded_session(
            session_id=session_id,
            lock=lock,
            removed=removed,
            active_tasks=active_tasks,
            backend_request_ids=backend_request_ids,
            waiters=waiters,
            stats=stats,
        )


def create_agentic_session_shards(config: Namespace):
    shard_count = _DEFAULT_SESSION_SHARD_COUNT
    request_capacity = config.sglang_server_concurrency * config.rollout_num_gpus // config.rollout_num_gpus_per_engine
    for idx in range(max(shard_count, _STALE_SESSION_SHARD_CLEANUP_LIMIT)):
        try:
            ray.kill(ray.get_actor(agentic_session_shard_name(idx)), no_restart=True)
        except ValueError:
            pass
    shard_handles = []
    for idx in range(shard_count):
        shard_name = agentic_session_shard_name(idx)
        shard_handles.append(
            AgenticSessionShard.options(
                num_cpus=0.25,
                # Spread the shards across nodes. With the default (PACK) scheduling and a
                # tiny num_cpus, Ray packs all shards onto a single node;
                scheduling_strategy="SPREAD",
                name=shard_name,
                runtime_env={"env_vars": dict(_AGENTIC_SHARD_ALLOCATOR_ENV)},
            ).remote(
                config,
                sglang_request_capacity=request_capacity if idx == 0 else None,
                sglang_request_limiter=shard_handles[0] if idx > 0 else None,
            )
        )
    return shard_handles


@serve.deployment
@serve.ingress(app)
class AgenticChatAPIService:
    def __init__(self, config: Namespace, session_shards) -> None:
        self.args = config
        self._shard_handles = list(session_shards)
        if not self._shard_handles:
            raise RuntimeError("Agentic chat API requires at least one session shard")
        init_http_client(self.args)

    def _shard_handle(self, session_id: str):
        shard_idx = _shard_index_for_session(session_id, len(self._shard_handles))
        return self._shard_handles[shard_idx]

    @app.get("/healthz")
    async def healthz(self):
        snapshots = await asyncio.gather(*(handle.health.remote() for handle in self._shard_handles))
        return {"ok": True, "shards": snapshots}

    @app.get("/health")
    async def health(self):
        return await self.healthz()

    @app.get("/debug_state")
    async def debug_state(self, sample_limit: int = 8):
        snapshots = await asyncio.gather(
            *(handle.debug_state.remote(sample_limit=int(sample_limit)) for handle in self._shard_handles)
        )
        totals: dict[str, int] = {
            "active_sessions": 0,
            "active_locks": 0,
            "with_active_irs": 0,
            "queued_no_active": 0,
            "waiters_no_active": 0,
            "no_irs": 0,
            "irs_by_id": 0,
            "active_irs": 0,
            "ir_queue": 0,
            "protected_sessions": 0,
            "pending_chat_waiters": 0,
            "prepare_gate_blocked_ir_count": 0,
            "partial_resume_gate_blocked_ir_count": 0,
        }
        for snapshot in snapshots:
            totals["active_sessions"] += int(snapshot.get("active_sessions") or 0)
            totals["active_locks"] += int(snapshot.get("active_locks") or 0)
            totals["prepare_gate_blocked_ir_count"] += int(snapshot.get("prepare_gate_blocked_ir_count") or 0)
            totals["partial_resume_gate_blocked_ir_count"] += int(
                snapshot.get("partial_resume_gate_blocked_ir_count") or 0
            )
            breakdown = snapshot.get("session_breakdown") or {}
            for key in ("with_active_irs", "queued_no_active", "waiters_no_active", "no_irs"):
                totals[key] += int(breakdown.get(key) or 0)
            for row in snapshot.get("by_state") or []:
                for key in (
                    "irs_by_id",
                    "active_irs",
                    "ir_queue",
                    "protected_sessions",
                    "pending_chat_waiters",
                ):
                    totals[key] += int(row.get(key) or 0)
        return {"totals": totals, "shards": snapshots}

    async def aborted_resume_session_ids(self, *, rollout_id: int) -> list[str]:
        batches = await asyncio.gather(
            *(handle.aborted_resume_session_ids.remote(rollout_id=rollout_id) for handle in self._shard_handles)
        )
        session_ids: list[str] = []
        for batch in batches:
            session_ids.extend(str(session_id) for session_id in batch)
        return session_ids

    async def register_sessions_batch(self, *, entries: list[dict[str, Any]]) -> int:
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for entry in entries:
            session_id = entry["session_id"]
            handle = self._shard_handle(session_id)
            grouped.setdefault(handle, []).append(entry)
        if not grouped:
            return 0
        counts = await asyncio.gather(
            *(handle.register_sessions_batch.remote(entries=batch) for handle, batch in grouped.items())
        )
        return sum(int(count or 0) for count in counts)

    async def prepare_group_status(self, *, scope_id: str) -> list[dict[str, Any]]:
        snapshots = await asyncio.gather(
            *(handle.prepare_group_status.remote(scope_id=scope_id) for handle in self._shard_handles)
        )
        aggregated: dict[tuple[str, int], dict[str, Any]] = {}
        for batch in snapshots:
            for item in batch:
                group_id = item["group_id"]
                generation = int(item["group_generation"])
                key = (group_id, generation)
                entry = aggregated.setdefault(
                    key,
                    {
                        "group_id": group_id,
                        "group_generation": generation,
                        "total_sessions": 0,
                        "ready_sessions": 0,
                    },
                )
                entry["total_sessions"] += int(item.get("total_sessions") or 0)
                entry["ready_sessions"] += int(item.get("ready_sessions") or 0)
        return list(aggregated.values())

    async def activate_group_sessions(
        self,
        *,
        scope_id: str,
        groups: list[dict[str, Any]],
        rollout_id: int,
    ) -> dict[str, int]:
        counts = await asyncio.gather(
            *(
                handle.activate_group_sessions.remote(
                    scope_id=scope_id,
                    groups=groups,
                    rollout_id=rollout_id,
                )
                for handle in self._shard_handles
            )
        )
        activated_sessions = 0
        started_sessions = 0
        for item in counts:
            if isinstance(item, dict):
                activated_sessions += int(item.get("activated_sessions") or 0)
                started_sessions += int(item.get("started_sessions") or 0)
        return {
            "activated_sessions": activated_sessions,
            "started_sessions": started_sessions,
        }

    async def finalize_and_discard(
        self,
        *,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        reward: float | dict[str, Any] | None = None,
    ) -> FinalizedResultTransport:
        return await self._shard_handle(session_id).finalize_and_discard.remote(
            session_id=session_id,
            metadata=metadata,
            reward=reward,
        )

    async def discard_session(self, *, session_id: str) -> bool:
        return await self._shard_handle(session_id).discard_session.remote(session_id=session_id)

    async def release_partial_resume_gate(self, *, rollout_id: int) -> int:
        counts = await asyncio.gather(
            *(handle.release_partial_resume_gate.remote(rollout_id=rollout_id) for handle in self._shard_handles)
        )
        return sum(int(count or 0) for count in counts)

    async def gate_rollout_irs_for_partial_resume(self, *, rollout_id: int) -> int:
        counts = await asyncio.gather(
            *(
                handle.gate_rollout_irs_for_partial_resume.remote(rollout_id=rollout_id)
                for handle in self._shard_handles
            )
        )
        return sum(int(count or 0) for count in counts)

    async def gate_rollout_irs_for_discard(self, *, rollout_id: int) -> int:
        counts = await asyncio.gather(
            *(handle.gate_rollout_irs_for_discard.remote(rollout_id=rollout_id) for handle in self._shard_handles)
        )
        return sum(int(count or 0) for count in counts)

    async def gate_all_irs_for_shutdown(self) -> int:
        counts = await asyncio.gather(*(handle.gate_all_irs_for_shutdown.remote() for handle in self._shard_handles))
        return sum(int(count or 0) for count in counts)

    async def active_rollout_request_counts(self, *, rollout_id: int) -> dict[str, int]:
        snapshots = await asyncio.gather(
            *(handle.active_rollout_request_counts.remote(rollout_id=rollout_id) for handle in self._shard_handles)
        )
        protected_active = 0
        abortable_active = 0
        evaluating = 0
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            protected_active += int(snapshot.get("protected_active", 0) or 0)
            abortable_active += int(snapshot.get("abortable_active", 0) or 0)
            evaluating += int(snapshot.get("evaluating", 0) or 0)
        return {
            "protected_active": protected_active,
            "abortable_active": abortable_active,
            "evaluating": evaluating,
        }

    async def abort_rollout_requests(self, *, rollout_id: int) -> dict[str, int]:
        counts = await self.active_rollout_request_counts(rollout_id=rollout_id)
        if counts["evaluating"] > 0 or counts["protected_active"] > 0:
            return {**counts, "abort_requested_workers": 0, "abort_failed_workers": 0}
        urls = await _sglang_worker_urls(self.args)
        if not urls:
            raise RuntimeError("Cannot abort active rollout requests because no SGLang worker urls are available.")
        results = await asyncio.gather(
            *(post(f"{url}/abort_request", {"abort_all": True}) for url in urls),
            return_exceptions=True,
        )
        failed = 0
        for url, result in zip(urls, results):
            if isinstance(result, BaseException):
                failed += 1
                logger.warning("Failed to abort SGLang worker at %s: %s", url, result)
        if failed:
            raise RuntimeError(f"Failed to abort {failed}/{len(urls)} SGLang workers.")
        return {
            **counts,
            "abort_requested_workers": len(urls) - failed,
            "abort_failed_workers": failed,
        }

    async def enter_eval(self) -> int:
        counts = await asyncio.gather(*(handle.enter_eval.remote() for handle in self._shard_handles))
        return sum(int(count or 0) for count in counts)

    async def exit_eval(self) -> int:
        counts = await asyncio.gather(*(handle.exit_eval.remote() for handle in self._shard_handles))
        return sum(int(count or 0) for count in counts)

    async def trim_memory(self) -> dict[str, Any]:
        results = await asyncio.gather(
            *(handle.trim_memory.remote() for handle in self._shard_handles),
            return_exceptions=True,
        )
        snapshots = []
        for idx, result in enumerate(results):
            snapshot = {"slot_index": idx}
            if isinstance(result, BaseException):
                snapshot.update({"ok": False, "error": str(result)[:500]})
            elif isinstance(result, dict):
                snapshot.update(result)
            else:
                snapshot.update({"ok": False, "error": f"Unexpected trim result: {result!r}"[:500]})
            snapshots.append(snapshot)
        return {
            "ok": all(bool(snapshot.get("ok")) for snapshot in snapshots),
            "shard_count": len(snapshots),
            "trimmed_count": sum(1 for snapshot in snapshots if int(snapshot.get("trimmed") or 0)),
            "active_sessions": sum(int(snapshot.get("active_sessions") or 0) for snapshot in snapshots),
            "active_requests": sum(int(snapshot.get("active_requests") or 0) for snapshot in snapshots),
            "shards": snapshots,
        }

    async def _chat_completions_impl(self, request: Request) -> JSONResponse:
        request_arrive_at = time.time()
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
        except ValueError as exc:
            return _openai_error_response(
                _openai_error_result(
                    f"Invalid request body: {exc}",
                    param="body",
                    status_code=400,
                    error_type="invalid_request_error",
                )
            )
        request_json_done_at = time.time()
        try:
            if not isinstance(payload, dict):
                raise AgenticChatRequestError("request body must be a JSON object", param="body")
            request_payload = _normalized_chat_request(payload)
            session_id = _session_id_from_request(request=request)
        except AgenticChatRequestError as exc:
            return _openai_error_response(_openai_error_from_exc(exc))
        request_ready_at = time.time()
        shard_dispatch_at = time.time()
        try:
            response = await self._shard_handle(session_id).chat.remote(
                session_id=session_id,
                messages=request_payload["messages"],
                tools=request_payload["tools"],
                chat_template_kwargs=request_payload["chat_template_kwargs"],
                temperature=request_payload["temperature"],
                top_p=request_payload["top_p"],
                max_completion_tokens=request_payload["max_completion_tokens"],
                stop=request_payload["stop"],
                seed=request_payload["seed"],
                logprobs=request_payload["logprobs"],
            )
        except (ray.exceptions.RayTaskError, ray.exceptions.TaskCancelledError) as exc:
            if isinstance(exc, ray.exceptions.RayTaskError) and not isinstance(
                exc.as_instanceof_cause(), ray.exceptions.TaskCancelledError
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
        if isinstance(response, dict) and isinstance(response.get("error"), dict):
            return _openai_error_response(response)
        service_remote_return_at = time.time()
        service_profile = {
            "chat_http_request_arrive_at": request_arrive_at,
            "chat_http_request_json_done_at": request_json_done_at,
            "chat_http_request_ready_at": request_ready_at,
            "chat_shard_rpc_dispatch_at": shard_dispatch_at,
            "chat_service_remote_return_at": service_remote_return_at,
        }
        response_ready_at = time.time()
        mark_agentic_event(service_profile, "chat_service_response_ready_at", response_ready_at)
        http_return_at = time.time()
        mark_agentic_event(service_profile, "chat_service_http_return_at", http_return_at)
        try:
            await self._shard_handle(session_id).mark_chat_service_response_ready.remote(
                session_id=session_id,
                request_id=response["request_id"],
                remote_return_at=service_remote_return_at,
                response_ready_at=response_ready_at,
                http_return_at=http_return_at,
            )
        except (ray.exceptions.RayTaskError, ray.exceptions.TaskCancelledError) as exc:
            if isinstance(exc, ray.exceptions.RayTaskError) and not isinstance(
                exc.as_instanceof_cause(), ray.exceptions.TaskCancelledError
            ):
                logger.warning(
                    "Failed to mark chat service response ready for session=%s request=%s: %s",
                    session_id,
                    response["request_id"],
                    exc,
                )
        except Exception as exc:
            logger.warning(
                "Failed to mark chat service response ready for session=%s request=%s: %s",
                session_id,
                response["request_id"],
                exc,
            )
        choice = {
            "index": 0,
            "message": response["message"],
            "logprobs": response["logprobs"] if request_payload["logprobs"] else None,
            "finish_reason": response["finish_reason"],
        }
        response_payload = {
            "id": f"chatcmpl_{session_id}_{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [choice],
            "usage": response["usage"],
        }
        if isinstance(request_payload["model"], str):
            response_payload["model"] = request_payload["model"]
        return JSONResponse(response_payload)

    @app.post("/")
    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def bare_chat_completions(self, request: Request) -> JSONResponse:
        return await self._chat_completions_impl(request)

    @app.get("/models")
    @app.get("/v1/models")
    async def models(self) -> JSONResponse:
        checkpoint = self.args.hf_checkpoint
        model_id = Path(checkpoint).name if isinstance(checkpoint, str) and checkpoint else None
        data = []
        if isinstance(model_id, str) and model_id:
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "relax",
                }
            )
        return JSONResponse(
            {
                "object": "list",
                "data": data,
            }
        )
