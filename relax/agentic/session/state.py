# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import torch

from relax.agentic.profile import TRACE_KEY, merge_agentic_trace
from relax.utils.types import Sample


MsgKind = Literal["obs", "resp"]
_ALLOWED_MESSAGE_ROLES = {"user", "assistant", "tool", "system"}


class RequestKind(str, Enum):
    FRESH = "fresh"
    RESUMED = "resumed"
    PROTECTED = "protected"


_EMPTY_SPEC_DELTA = {
    "spec_accept_token_num": 0,
    "spec_draft_token_num": 0,
    "spec_verify_ct": 0,
    "completion_token_num": 0,
}
_EMPTY_PREFIX_CACHE_DELTA = {
    "cached_tokens": 0,
    "total_prompt_tokens": 0,
}
_TRAINING_ARTIFACT_ARRAY_FIELDS: list[tuple[str, Any]] = [
    ("tokens", np.int32),
    ("rollout_tokens", np.int32),
    ("loss_mask", np.uint8),
    ("rollout_log_probs", np.float64),
    ("teacher_log_probs", np.float64),
    ("teacher_topk_token_ids", np.int32),
    ("rollout_routed_experts", np.int32),
]


def check_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages)}")

    ensured: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{index}] must be a dict, got {type(message)}")
        if "role" not in message:
            raise ValueError(f"messages[{index}] must include role")
        role = message["role"]
        if not isinstance(role, str):
            raise TypeError(f"messages[{index}].role must be a string, got {type(role)}")
        if role not in _ALLOWED_MESSAGE_ROLES:
            allowed_roles = ", ".join(sorted(_ALLOWED_MESSAGE_ROLES))
            raise ValueError(f"messages[{index}].role must be one of: {allowed_roles}")
        if "content" not in message:
            raise ValueError(f"messages[{index}] must include content")
        content = message["content"]
        if (
            content is None
            or (isinstance(content, str) and not content.strip())
            or (isinstance(content, list) and not content)
        ):
            raise ValueError(f"messages[{index}].content must not be empty")
        if isinstance(content, str):
            ensured.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise TypeError(f"messages[{index}].content must be a list or string, got {type(content)}")
        rendered_content = []
        for item_index, item in enumerate(content):
            if not isinstance(item, dict):
                raise TypeError(f"messages[{index}].content[{item_index}] must be a dict, got {type(item)}")
            if item.get("type") == "text" and isinstance(item.get("text"), str) and not item["text"].strip():
                raise ValueError(f"messages[{index}].content[{item_index}].text must not be empty")
            rendered_content.append(copy.deepcopy(item))
        ensured.append({"role": role, "content": rendered_content})
    return ensured


def iter_message_content_parts(message: dict[str, Any]):
    content = message.get("content")
    if isinstance(content, str):
        yield {"type": "text", "text": content}
        return
    for item in content or []:
        yield item


def normalize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if tools is None:
        return normalized
    if not isinstance(tools, list):
        raise TypeError(f"tools must be a list, got {type(tools)}")
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise TypeError(f"tools[{index}] must be a dict, got {type(tool)}")
        normalized.append(tool)
    return normalized


def normalize_template_kwargs(template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    def _normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        if isinstance(value, tuple):
            return [_normalize(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return copy.deepcopy(value)

    return _normalize(template_kwargs or {})


def messages_tools_template_state_hash(
    messages: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
    template_kwargs: dict[str, Any] | None,
) -> str:
    payload = json.dumps(
        {
            "messages": check_messages(messages),
            "tools": normalize_tools(tools),
            "template_kwargs": normalize_template_kwargs(template_kwargs),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class FinalizedResultTransport:
    """Session finalization result.

    Sample statuses (completed, truncated, aborted, failed) carry training data
    in artifact_ref. discarded means the session was already cleaned by
    rollout-side discard. non_finalizable means the session exited without an
    exportable terminal response. Both are runtime-local drops.
    """

    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_ref: Any = None


@dataclass
class TrainingFieldArtifact:
    sample_payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sample(cls, sample: Sample) -> "TrainingFieldArtifact":
        return cls(sample_payload=_compact_sample_payload(sample.to_dict()))

    def to_sample(self) -> Sample:
        return Sample.from_dict(_expand_sample_payload(self.sample_payload))


def _compact_sample_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compacted = copy.deepcopy(payload)

    tokens = compacted.get("tokens")
    rollout_tokens = compacted.get("rollout_tokens")
    if isinstance(tokens, list) and isinstance(rollout_tokens, list) and rollout_tokens == tokens:
        compacted["rollout_tokens"] = None
        compacted["_rollout_tokens_shared"] = True

    for field_name, dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
        value = compacted.get(field_name)
        if not isinstance(value, list) or not value:
            continue
        try:
            compacted[field_name] = np.asarray(value, dtype=dtype)
        except Exception:
            compacted[field_name] = copy.deepcopy(value)
    return compacted


def _expand_sample_payload(payload: dict[str, Any]) -> dict[str, Any]:
    expanded = copy.deepcopy(payload)

    def _tolist(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    tokens = _tolist(expanded.get("tokens"))
    if isinstance(tokens, list):
        expanded["tokens"] = tokens

    rollout_tokens = _tolist(expanded.get("rollout_tokens"))
    if expanded.pop("_rollout_tokens_shared", False):
        expanded["rollout_tokens"] = list(tokens or [])
    elif isinstance(rollout_tokens, list):
        expanded["rollout_tokens"] = rollout_tokens

    for field_name, _dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
        if field_name in {"tokens", "rollout_tokens"}:
            continue
        value = _tolist(expanded.get(field_name))
        if isinstance(value, list):
            expanded[field_name] = value
    return expanded


def _copy_dict_of_lists(data: dict[str, Any] | None) -> dict[str, Any] | None:
    return copy.deepcopy(data) if data is not None else None


def _merge_multimodal_train_inputs(base: dict[str, Any] | None, delta: dict[str, Any] | None) -> dict[str, Any] | None:
    if base is None and delta is None:
        return None
    merged = copy.deepcopy(base or {})
    for key, value in (delta or {}).items():
        if value is None:
            continue
        if key in merged and isinstance(merged[key], torch.Tensor) and isinstance(value, torch.Tensor):
            merged[key] = torch.cat([merged[key], value], dim=0)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _extend_media_data(base: list[str], delta: list[str] | None) -> list[str]:
    if delta:
        base.extend(list(delta))
    return base


def _merge_export_metadata(static_metadata: dict[str, Any], export_patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(static_metadata)
    patch = copy.deepcopy(export_patch)
    patch_trace = patch.pop(TRACE_KEY, None)
    if patch_trace is not None:
        merged[TRACE_KEY] = merge_agentic_trace(merged.get(TRACE_KEY), patch_trace)
    merged.update(patch)
    return merged


def _sum_counter_dict(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    summed = dict(left)
    for key, value in right.items():
        summed[key] = int(summed.get(key, 0)) + int(value or 0)
    return summed


def _decode_tokens(
    *,
    tokenizer: Any,
    token_ids: list[int],
) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _multimodal_inputs_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    multimodal_inputs = {
        "images": [],
        # "videos": [],
        # "audio": [],
    }
    for message in check_messages(messages):
        for item in iter_message_content_parts(message):
            item_type = item.get("type")
            if item_type == "image_url" and item.get("image_url") is not None:
                multimodal_inputs["images"].append(copy.deepcopy(item["image_url"]["url"]))
            # elif item_type == "video" and item.get("video") is not None:
            #     multimodal_inputs["videos"].append(copy.deepcopy(item.get("video")))
            # elif item_type == "audio" and item.get("audio") is not None:
            #     multimodal_inputs["audio"].append(copy.deepcopy(item.get("audio")))
    compact = {key: value for key, value in multimodal_inputs.items() if value}
    return compact or None


@dataclass
class MsgNode:
    kind: MsgKind
    state_hash: str
    parent_state_hash: str | None
    rollout_id: int
    abort_count: int
    messages_delta: list[dict[str, Any]] = field(default_factory=list)
    train_token_delta: list[int] = field(default_factory=list)
    rollout_token_delta: list[int] = field(default_factory=list)
    loss_mask_delta: list[int] = field(default_factory=list)
    logprob_delta: list[float] = field(default_factory=list)
    multimodal_train_inputs_delta: dict[str, Any] | None = None
    backend_image_data_delta: list[str] = field(default_factory=list)
    backend_audio_data_delta: list[str] = field(default_factory=list)
    backend_video_data_delta: list[str] = field(default_factory=list)
    weight_version_delta: list[str] = field(default_factory=list)
    spec_delta: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_SPEC_DELTA))
    prefix_cache_delta: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_PREFIX_CACHE_DELTA))
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    wall_elapsed_s: float = 0.0
    generation_elapsed_s: float = 0.0
    status: str | None = None
    terminal_count: int = 0
    reward: float | dict[str, Any] | None = None
    remove_sample: bool = False
    teacher_log_probs: list[float] | None = None
    rollout_routed_experts: Any = None
    export_metadata_patch: dict[str, Any] = field(default_factory=dict)


@dataclass
class InflightRequest:
    request_id: str
    parent_state_hash: str
    rollout_id: int
    kind: RequestKind
    abort_count: int
    sampling_params: dict[str, Any] = field(default_factory=dict)
    logprobs: bool = False
    history_train_token_prefix: list[int] = field(default_factory=list)
    history_rollout_token_prefix: list[int] = field(default_factory=list)
    history_backend_image_data: list[str] = field(default_factory=list)
    history_backend_audio_data: list[str] = field(default_factory=list)
    history_backend_video_data: list[str] = field(default_factory=list)
    pending_train_token_delta: list[int] = field(default_factory=list)
    pending_rollout_token_delta: list[int] = field(default_factory=list)
    pending_loss_mask_delta: list[int] = field(default_factory=list)
    pending_logprob_delta: list[float] = field(default_factory=list)
    pending_weight_version_delta: list[str] = field(default_factory=list)
    pending_spec_delta: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_SPEC_DELTA))
    pending_prefix_cache_delta: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_PREFIX_CACHE_DELTA))
    pending_wall_elapsed_s: float = 0.0
    pending_generation_elapsed_s: float = 0.0
    pending_status: str | None = None
    pending_routed_experts: Any = None
    pending_export_metadata_patch: dict[str, Any] = field(default_factory=dict)
    latest_backend_meta: dict[str, Any] = field(default_factory=dict)
    backend_started: bool = False
    runner_epoch: int = 0


@dataclass(frozen=True)
class ExecutionPrefix:
    train_token_prefix: list[int] = field(default_factory=list)
    rollout_token_prefix: list[int] = field(default_factory=list)
    backend_image_data: list[str] = field(default_factory=list)
    backend_audio_data: list[str] = field(default_factory=list)
    backend_video_data: list[str] = field(default_factory=list)


@dataclass
class SessionForest:
    session_id: str
    group_index: int | None
    index: int | None
    label: str | None
    train_metadata: dict[str, Any] | None
    static_metadata: dict[str, Any] = field(default_factory=dict)
    root_state_hash: str | None = None
    root_state_hashes: set[str] = field(default_factory=set)
    nodes_by_hash: dict[str, MsgNode] = field(default_factory=dict)
    children_by_hash: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def create_empty(
        cls,
        *,
        session_id: str,
        group_index: int | None = None,
        index: int | None = None,
        label: str | None = None,
        train_metadata: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SessionForest":
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required for SessionForest.create_empty")
        static_metadata = copy.deepcopy(metadata or {})
        forest = cls(
            session_id=session_id,
            group_index=group_index,
            index=index,
            label=label,
            train_metadata=copy.deepcopy(train_metadata),
            static_metadata=static_metadata,
        )
        forest.append_obs(
            parent_state_hash=None,
            rollout_id=0,
            abort_count=0,
            messages_delta=[],
            train_token_delta=[],
            rollout_token_delta=[],
            multimodal_train_inputs_delta=None,
            backend_image_data_delta=[],
            backend_audio_data_delta=[],
            backend_video_data_delta=[],
            tools=None,
            chat_template_kwargs=normalize_template_kwargs(static_metadata.get("chat_template_kwargs")),
        )
        return forest

    @staticmethod
    def resolve_request_kind(
        *,
        abort_count: int = 0,
        resumed: bool = False,
        protected_abort_count_threshold: int | None = None,
    ) -> RequestKind:
        if protected_abort_count_threshold is not None and abort_count >= protected_abort_count_threshold:
            return RequestKind.PROTECTED
        if resumed or abort_count > 0:
            return RequestKind.RESUMED
        return RequestKind.FRESH

    def _node(self, state_hash: str) -> MsgNode:
        return self.nodes_by_hash[state_hash]

    def lineage(self, state_hash: str) -> list[MsgNode]:
        lineage: list[MsgNode] = []
        current_hash: str | None = state_hash
        while current_hash is not None:
            node = self._node(current_hash)
            lineage.append(node)
            current_hash = node.parent_state_hash
        lineage.reverse()
        return lineage

    def full_messages(self, state_hash: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for node in self.lineage(state_hash):
            if node.messages_delta:
                messages.extend(copy.deepcopy(node.messages_delta))
        return messages

    def rollout_token_count(self, state_hash: str | None) -> int:
        if state_hash is None:
            return 0
        total = 0
        for node in self.lineage(state_hash):
            total += len(node.rollout_token_delta)
        return total

    def build_execution_prefix(self, state_hash: str | None) -> ExecutionPrefix:
        if state_hash is None:
            return ExecutionPrefix()
        train_token_prefix: list[int] = []
        rollout_token_prefix: list[int] = []
        backend_image_data: list[str] = []
        backend_audio_data: list[str] = []
        backend_video_data: list[str] = []
        for node in self.lineage(state_hash):
            train_token_prefix.extend(node.train_token_delta)
            rollout_token_prefix.extend(node.rollout_token_delta)
            _extend_media_data(backend_image_data, node.backend_image_data_delta)
            _extend_media_data(backend_audio_data, node.backend_audio_data_delta)
            _extend_media_data(backend_video_data, node.backend_video_data_delta)
        return ExecutionPrefix(
            train_token_prefix=train_token_prefix,
            rollout_token_prefix=rollout_token_prefix,
            backend_image_data=backend_image_data,
            backend_audio_data=backend_audio_data,
            backend_video_data=backend_video_data,
        )

    def current_tools(self, state_hash: str | None) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        if state_hash is None:
            return tools
        for node in self.lineage(state_hash):
            if node.tools is not None:
                tools = normalize_tools(node.tools)
        return tools

    def current_chat_template_kwargs(self, state_hash: str | None) -> dict[str, Any]:
        chat_template_kwargs = normalize_template_kwargs(self.static_metadata.get("chat_template_kwargs"))
        if state_hash is None:
            return chat_template_kwargs
        for node in self.lineage(state_hash):
            if node.chat_template_kwargs is not None:
                chat_template_kwargs = normalize_template_kwargs(node.chat_template_kwargs)
        return chat_template_kwargs

    def export_terminal_hashes(self) -> list[str]:
        terminal_hashes: list[str] = []
        for state_hash, node in self.nodes_by_hash.items():
            if state_hash in self.root_state_hashes:
                continue
            if node.terminal_count <= 0:
                continue
            terminal_hashes.extend([state_hash] * node.terminal_count)
        return terminal_hashes

    def subtree_root_hash(self, state_hash: str) -> str | None:
        lineage = self.lineage(state_hash)
        if not lineage:
            return None
        if len(lineage) == 1:
            return None
        return lineage[1].state_hash

    def subtree_tools(self, state_hash: str) -> list[dict[str, Any]]:
        subtree_root_hash = self.subtree_root_hash(state_hash)
        if subtree_root_hash is None:
            return []
        subtree_root = self.nodes_by_hash[subtree_root_hash]
        return normalize_tools(subtree_root.tools)

    def subtree_chat_template_kwargs(self, state_hash: str) -> dict[str, Any]:
        subtree_root_hash = self.subtree_root_hash(state_hash)
        if subtree_root_hash is None:
            return {}
        subtree_root = self.nodes_by_hash[subtree_root_hash]
        return normalize_template_kwargs(subtree_root.chat_template_kwargs)

    def _mark_parent_non_terminal(self, parent_state_hash: str | None) -> None:
        if parent_state_hash is None:
            return
        parent_node = self.nodes_by_hash[parent_state_hash]
        if parent_node.terminal_count > 0:
            parent_node.terminal_count -= 1

    def _register_node(self, node: MsgNode) -> MsgNode:
        existing = self.nodes_by_hash.get(node.state_hash)
        if existing is not None:
            if existing.kind != node.kind:
                raise ValueError(f"State hash collision with mismatched kinds: {existing.kind} vs {node.kind}")
            if existing.state_hash not in self.root_state_hashes:
                existing.terminal_count += 1
            return existing

        node.terminal_count = 0 if node.parent_state_hash is None else 1
        self.nodes_by_hash[node.state_hash] = node
        if node.parent_state_hash is None:
            if self.root_state_hash is None:
                self.root_state_hash = node.state_hash
            self.root_state_hashes.add(node.state_hash)
        else:
            children = self.children_by_hash.setdefault(node.parent_state_hash, [])
            parent_was_leaf = not children
            if node.state_hash not in children:
                children.append(node.state_hash)
                if parent_was_leaf:
                    self._mark_parent_non_terminal(node.parent_state_hash)
        return node

    def _next_state_hash(
        self,
        *,
        parent_state_hash: str | None,
        messages_delta: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> str:
        parent_messages = self.full_messages(parent_state_hash) if parent_state_hash is not None else []
        full_messages = parent_messages + check_messages(messages_delta)
        current_tools = normalize_tools(tools) if tools is not None else self.current_tools(parent_state_hash)
        current_chat_template_kwargs = (
            normalize_template_kwargs(chat_template_kwargs)
            if chat_template_kwargs is not None
            else self.current_chat_template_kwargs(parent_state_hash)
        )
        return messages_tools_template_state_hash(full_messages, current_tools, current_chat_template_kwargs)

    def append_obs(
        self,
        *,
        parent_state_hash: str | None,
        rollout_id: int,
        abort_count: int,
        messages_delta: list[dict[str, Any]],
        train_token_delta: list[int],
        rollout_token_delta: list[int],
        multimodal_train_inputs_delta: dict[str, Any] | None = None,
        backend_image_data_delta: list[str] | None = None,
        backend_audio_data_delta: list[str] | None = None,
        backend_video_data_delta: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        wall_elapsed_s: float = 0.0,
        generation_elapsed_s: float = 0.0,
    ) -> MsgNode:
        state_hash = self._next_state_hash(
            parent_state_hash=parent_state_hash,
            messages_delta=messages_delta,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        return self._register_node(
            MsgNode(
                kind="obs",
                state_hash=state_hash,
                parent_state_hash=parent_state_hash,
                rollout_id=rollout_id,
                abort_count=abort_count,
                messages_delta=check_messages(messages_delta),
                train_token_delta=list(train_token_delta),
                rollout_token_delta=list(rollout_token_delta),
                loss_mask_delta=[],
                logprob_delta=[],
                multimodal_train_inputs_delta=_copy_dict_of_lists(multimodal_train_inputs_delta),
                backend_image_data_delta=list(backend_image_data_delta or []),
                backend_audio_data_delta=list(backend_audio_data_delta or []),
                backend_video_data_delta=list(backend_video_data_delta or []),
                tools=normalize_tools(tools) if tools is not None else None,
                chat_template_kwargs=(
                    normalize_template_kwargs(chat_template_kwargs) if chat_template_kwargs is not None else None
                ),
                wall_elapsed_s=float(wall_elapsed_s),
                generation_elapsed_s=float(generation_elapsed_s),
            )
        )

    def append_resp(
        self,
        *,
        parent_state_hash: str,
        rollout_id: int,
        abort_count: int,
        messages_delta: list[dict[str, Any]],
        train_token_delta: list[int],
        rollout_token_delta: list[int],
        loss_mask_delta: list[int] | None = None,
        logprob_delta: list[float],
        weight_version_delta: list[str] | None = None,
        spec_delta: dict[str, int] | None = None,
        prefix_cache_delta: dict[str, int] | None = None,
        wall_elapsed_s: float = 0.0,
        generation_elapsed_s: float = 0.0,
        status: str = "completed",
        reward: float | dict[str, Any] | None = None,
        remove_sample: bool = False,
        teacher_log_probs: list[float] | None = None,
        rollout_routed_experts: Any = None,
        export_metadata_patch: dict[str, Any] | None = None,
    ) -> MsgNode:
        state_hash = self._next_state_hash(
            parent_state_hash=parent_state_hash,
            messages_delta=messages_delta,
            tools=None,
            chat_template_kwargs=None,
        )
        return self._register_node(
            MsgNode(
                kind="resp",
                state_hash=state_hash,
                parent_state_hash=parent_state_hash,
                rollout_id=rollout_id,
                abort_count=abort_count,
                messages_delta=check_messages(messages_delta),
                train_token_delta=list(train_token_delta),
                rollout_token_delta=list(rollout_token_delta),
                loss_mask_delta=list(loss_mask_delta)
                if loss_mask_delta is not None
                else ([1] * len(train_token_delta)),
                logprob_delta=list(logprob_delta),
                weight_version_delta=[str(item) for item in (weight_version_delta or [])],
                spec_delta=_sum_counter_dict(dict(_EMPTY_SPEC_DELTA), spec_delta or {}),
                prefix_cache_delta=_sum_counter_dict(dict(_EMPTY_PREFIX_CACHE_DELTA), prefix_cache_delta or {}),
                wall_elapsed_s=float(wall_elapsed_s),
                generation_elapsed_s=float(generation_elapsed_s),
                status=str(status),
                reward=copy.deepcopy(reward),
                remove_sample=remove_sample,
                teacher_log_probs=copy.deepcopy(teacher_log_probs),
                rollout_routed_experts=copy.deepcopy(rollout_routed_experts),
                export_metadata_patch=copy.deepcopy(export_metadata_patch or {}),
            )
        )

    def build_agentic_trace_turns(
        self,
        *,
        leaf_state_hash: str,
    ) -> list[dict[str, Any]]:
        lineage = self.lineage(leaf_state_hash)
        if not lineage:
            raise ValueError(f"Unknown state hash: {leaf_state_hash}")
        turns: list[dict[str, Any]] = []
        for node in lineage:
            if node.kind != "resp":
                continue
            patch = copy.deepcopy(node.export_metadata_patch)
            patch_trace = patch.get(TRACE_KEY)
            events = merge_agentic_trace(None, patch_trace).get("events") if patch_trace is not None else None
            turns.append(
                {
                    "turn_idx": len(turns),
                    "resp_state_hash": node.state_hash,
                    "parent_state_hash": node.parent_state_hash,
                    "rollout_id": node.rollout_id,
                    "request_id": patch.get("request_id"),
                    "request_kind": patch.get("request_kind"),
                    "base_state_hash": patch.get("base_state_hash"),
                    "abort_count": node.abort_count,
                    "status": str(node.status),
                    "response_length": len(node.train_token_delta),
                    "wall_elapsed_s": float(node.wall_elapsed_s),
                    "generation_elapsed_s": float(node.generation_elapsed_s),
                    "events": copy.deepcopy(events) if isinstance(events, dict) else {},
                }
            )
        return turns

    def build_sample(
        self,
        *,
        leaf_state_hash: str,
        tokenizer: Any,
        # mask_offpolicy_in_partial_rollout: bool = False,
    ) -> Sample:
        lineage = self.lineage(leaf_state_hash)
        if not lineage:
            raise ValueError(f"Unknown state hash: {leaf_state_hash}")
        leaf = lineage[-1]
        response_nodes = [node for node in lineage if node.kind == "resp"]
        if not response_nodes:
            raise ValueError("A sample export leaf must have at least one resp node in its lineage.")

        tokens: list[int] = []
        rollout_tokens: list[int] = []
        continuation_train_tokens: list[int] = []
        loss_mask: list[int] = []
        rollout_log_probs: list[float] = []
        multimodal_train_inputs: dict[str, Any] | None = None
        weight_versions: list[str] = []
        spec_info = dict(_EMPTY_SPEC_DELTA)
        prefix_cache_info = dict(_EMPTY_PREFIX_CACHE_DELTA)
        wall_elapsed_s = 0.0
        generation_elapsed_s = 0.0

        for idx, node in enumerate(lineage):
            tokens.extend(node.train_token_delta)
            rollout_tokens.extend(node.rollout_token_delta)
            wall_elapsed_s += float(node.wall_elapsed_s)
            generation_elapsed_s += float(node.generation_elapsed_s)
            multimodal_train_inputs = _merge_multimodal_train_inputs(
                multimodal_train_inputs, node.multimodal_train_inputs_delta
            )
            if node.kind == "resp":
                continuation_train_tokens.extend(node.train_token_delta)
                node_loss_mask = list(node.loss_mask_delta)
                # if mask_offpolicy_in_partial_rollout and node.rollout_id < leaf.rollout_id:
                #     node_loss_mask = [0] * len(node_loss_mask)
                loss_mask.extend(node_loss_mask)
                rollout_log_probs.extend(node.logprob_delta)
                weight_versions.extend(node.weight_version_delta)
                spec_info = _sum_counter_dict(spec_info, node.spec_delta)
                prefix_cache_info = _sum_counter_dict(prefix_cache_info, node.prefix_cache_delta)
                continue
            if idx == 0:
                continue
            if not any(item.kind == "resp" for item in lineage[:idx]):
                continue
            continuation_train_tokens.extend(node.train_token_delta)
            loss_mask.extend(node.loss_mask_delta or ([0] * len(node.train_token_delta)))
            rollout_log_probs.extend(node.logprob_delta or ([0.0] * len(node.train_token_delta)))

        prompt_train_token_count = len(tokens) - len(continuation_train_tokens)
        if prompt_train_token_count < 0:
            raise RuntimeError(
                "SessionForest export has more continuation train tokens than total train tokens: "
                f"total={len(tokens)}, continuation={len(continuation_train_tokens)}."
            )
        effective_prompt = _decode_tokens(tokenizer=tokenizer, token_ids=tokens[:prompt_train_token_count])
        if not effective_prompt:
            raise RuntimeError(
                f"SessionForest {self.session_id!r} export produced no decoded prompt; "
                "every exported leaf must have at least one observation node in its lineage."
            )
        merged_metadata = _merge_export_metadata(self.static_metadata, leaf.export_metadata_patch)
        if "start_rollout_id" not in merged_metadata:
            merged_metadata["start_rollout_id"] = response_nodes[0].rollout_id
        merged_metadata["tools"] = self.subtree_tools(leaf_state_hash)
        merged_metadata["chat_template_kwargs"] = self.subtree_chat_template_kwargs(leaf_state_hash)
        trace = merge_agentic_trace(merged_metadata.get(TRACE_KEY), None)
        turns = self.build_agentic_trace_turns(leaf_state_hash=leaf_state_hash)
        trace.update(
            {
                "session_id": self.session_id,
                "leaf_state_hash": leaf_state_hash,
                "terminal_status": str(leaf.status or ""),
                "turn_count": len(turns),
                "turns": turns,
            }
        )
        merged_metadata[TRACE_KEY] = trace
        if leaf.kind == "obs":
            status = Sample.Status.TRUNCATED.value
        else:
            status = leaf.status
        if status is None:
            response_statuses = [node.status for node in response_nodes if node.status is not None]
            status = response_statuses[-1] if response_statuses else Sample.Status.COMPLETED.value
        sample = Sample(
            group_index=self.group_index,
            index=self.index,
            prompt=effective_prompt,
            tokens=tokens,
            rollout_tokens=rollout_tokens,
            multimodal_inputs=_multimodal_inputs_from_messages(self.full_messages(leaf_state_hash)),
            multimodal_train_inputs=multimodal_train_inputs,
            response=_decode_tokens(tokenizer=tokenizer, token_ids=continuation_train_tokens),
            response_length=len(continuation_train_tokens),
            label=self.label,
            reward=copy.deepcopy(leaf.reward),
            loss_mask=loss_mask,
            weight_versions=weight_versions,
            rollout_log_probs=rollout_log_probs,
            rollout_routed_experts=copy.deepcopy(leaf.rollout_routed_experts),
            remove_sample=leaf.remove_sample,
            abort_count=leaf.abort_count,
            teacher_log_probs=copy.deepcopy(leaf.teacher_log_probs),
            status=Sample.Status(str(status)),
            metadata=merged_metadata,
            train_metadata=copy.deepcopy(self.train_metadata),
            session_id=self.session_id,
            non_generation_time=wall_elapsed_s - generation_elapsed_s,
        )
        sample.spec_info = Sample.SpecInfo.from_dict(spec_info)
        sample.prefix_cache_info = Sample.PrefixCacheInfo.from_dict(prefix_cache_info)
        return sample
