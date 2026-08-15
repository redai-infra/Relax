# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Managed GraphGPO ALFWorld agent with explicit per-turn exports."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from examples.graphgpo.action_parser import parse_action
from examples.graphgpo.alfworld_env import AlfWorldSnapshot, AlfWorldTextEnv
from examples.graphgpo.graph_credit import SUCCESS
from examples.graphgpo.prompt import (
    HistoryTurn,
    build_prompt,
    format_reference_observation,
)
from examples.graphgpo.state import state_key_v1


TASK_MARKER = "Your task is to: "
DEFAULT_MAX_STEPS = 50
SUCCESS_REWARD = 10.0
INVALID_PENALTY = 0.1


class ChatCompletionClient(Protocol):
    """Minimal injectable interface used by :func:`run_episode`."""

    async def complete(self, *, messages: list[dict[str, Any]]) -> str:
        """Return one assistant message for a fresh per-turn conversation."""


class OpenAICompatibleChatClient:
    """Lazy OpenAI SDK adapter for Relax's chat-completions service."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "model",
        timeout_s: float = 1200.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s must be a positive number")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(float(top_p))
            or not 0 < top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1]")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = float(timeout_s)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._max_tokens = max_tokens
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("the OpenAI SDK is required by the managed GraphGPO agent") from exc
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_s, connect=30.0),
                max_retries=0,
            )
        return self._client

    async def complete(self, *, messages: list[dict[str, Any]]) -> str:
        response = await self._ensure_client().chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
        )
        if not response.choices:
            raise RuntimeError("chat completion returned no choices")
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("chat completion returned non-text content")
        return content

    async def close(self) -> None:
        if self._client is None:
            return
        client = self._client
        self._client = None
        close_method = getattr(client, "close", None)
        if callable(close_method):
            await close_method()


def stable_group_seed(group_id: str) -> int:
    """Derive the same ALFWorld seed for every slot in one rollout group."""

    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a non-empty string")
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


def stable_row_id(
    *,
    task_id: object,
    rollout_group_id: str,
    trajectory_id: str,
    turn_index: int,
) -> str:
    """Return a deterministic identity for one explicit GraphGPO turn."""

    if isinstance(task_id, bool) or not isinstance(task_id, (str, int)):
        raise ValueError("task_id must be a JSON string or integer")
    if isinstance(task_id, str) and not task_id:
        raise ValueError("task_id must not be empty")
    if not isinstance(rollout_group_id, str) or not rollout_group_id:
        raise ValueError("rollout_group_id must be a non-empty string")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be a non-empty string")
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError("turn_index must be a non-negative integer")
    encoded = json.dumps(
        ["graphgpo-row-v1", task_id, rollout_group_id, trajectory_id, turn_index],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"graphgpo-row-v1:{hashlib.sha256(encoded).hexdigest()}"


def extract_task_description(initial_observation: str) -> str:
    """Extract the frozen ALFWorld task suffix or fail explicitly."""

    if not isinstance(initial_observation, str):
        raise TypeError("initial_observation must be a string")
    marker_index = initial_observation.find(TASK_MARKER)
    if marker_index < 0:
        raise ValueError("Task description not found in text observation.")
    task_description = initial_observation[marker_index + len(TASK_MARKER) :].strip()
    if not task_description:
        raise ValueError("Task description is empty in text observation.")
    return task_description


def _required_string(
    name: str,
    *,
    explicit: str | None,
    environment_key: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    value = explicit
    if value is None:
        source = os.environ if environ is None else environ
        value = source.get(environment_key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required; pass it explicitly or set {environment_key}")
    return value


def _task_id_from_snapshot(
    snapshot: AlfWorldSnapshot,
    *,
    fallback_task_id: object | None,
    data_root: str | None,
) -> object:
    if fallback_task_id is not None:
        if isinstance(fallback_task_id, bool) or not isinstance(fallback_task_id, (str, int)):
            raise ValueError("fallback task_id must be a JSON string or integer")
        if isinstance(fallback_task_id, str) and not fallback_task_id:
            raise ValueError("fallback task_id must not be empty")

    if snapshot.gamefile:
        normalized_gamefile = snapshot.gamefile.replace("\\", "/")
        gamefile_path = PurePosixPath(normalized_gamefile)
        if ".." in gamefile_path.parts:
            raise ValueError("ALFWorld gamefile must not contain parent traversal")
        if gamefile_path.is_absolute():
            if not data_root:
                raise ValueError("ALFWORLD_DATA is required to normalize an absolute gamefile")
            normalized_root = data_root.replace("\\", "/")
            data_root_path = PurePosixPath(normalized_root)
            if not data_root_path.is_absolute():
                raise ValueError("ALFWORLD_DATA must be absolute when ALFWorld returns an absolute gamefile")
            try:
                gamefile_path = gamefile_path.relative_to(data_root_path)
            except ValueError as exc:
                raise ValueError("ALFWorld gamefile is outside ALFWORLD_DATA") from exc
        resolved_task_id = gamefile_path.as_posix()
        if fallback_task_id is not None:
            declared_task_id = (
                fallback_task_id.replace("\\", "/") if isinstance(fallback_task_id, str) else fallback_task_id
            )
            if declared_task_id != resolved_task_id:
                raise ValueError("ALFWorld reset gamefile does not match the declared task_id")
        return resolved_task_id
    if fallback_task_id is None:
        raise ValueError(
            "task_id is unavailable: reset info has no extra.gamefile and session metadata has no explicit task_id"
        )
    return fallback_task_id


def _require_max_steps(max_steps: int) -> int:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    return max_steps


def _sampling_float(
    metadata: Mapping[str, Any],
    source: Mapping[str, str],
    *,
    metadata_key: str,
    environment_key: str,
    default: float,
    minimum_exclusive: float | None = None,
    maximum: float | None = None,
) -> float:
    raw: object = metadata.get(
        metadata_key,
        source.get(environment_key, default),
    )
    if isinstance(raw, str):
        try:
            raw = float(raw)
        except ValueError as exc:
            raise ValueError(f"{environment_key} must be a real number") from exc
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError(f"{metadata_key} must be a finite real number")
    value = float(raw)
    if minimum_exclusive is not None and value <= minimum_exclusive:
        raise ValueError(f"{metadata_key} must be greater than {minimum_exclusive}")
    if minimum_exclusive is None and value < 0:
        raise ValueError(f"{metadata_key} must be non-negative")
    if maximum is not None and value > maximum:
        raise ValueError(f"{metadata_key} must be at most {maximum}")
    return value


def _sampling_max_tokens(
    metadata: Mapping[str, Any],
    source: Mapping[str, str],
) -> int:
    raw: object = metadata.get(
        "max_tokens",
        source.get("GRAPHGPO_MAX_TOKENS", 512),
    )
    if isinstance(raw, str):
        try:
            raw = int(raw)
        except ValueError as exc:
            raise ValueError("GRAPHGPO_MAX_TOKENS must be an integer") from exc
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return raw


def _finalize_records(
    records: list[dict[str, Any]],
    *,
    success: bool,
    invalid_count: int,
) -> list[dict[str, Any]]:
    if not records:
        raise RuntimeError("an ALFWorld episode must export at least one turn")
    episode_return = SUCCESS_REWARD * float(success) - INVALID_PENALTY * invalid_count
    for record in records:
        record["metadata"]["success"] = success
        record["metadata"]["episode_return"] = episode_return
        record["reward"] = episode_return
    if success:
        last_metadata = records[-1]["metadata"]
        last_metadata["next_state_key"] = SUCCESS
        last_metadata["terminal"] = True
        last_metadata["truncated"] = False
    return records


async def run_episode(
    *,
    chat_client: ChatCompletionClient,
    env: AlfWorldTextEnv,
    trajectory_id: str | None = None,
    rollout_group_id: str | None = None,
    fallback_task_id: object | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run one trajectory and return JSONL-ready explicit turn records."""

    records: list[dict[str, Any]] = []
    history: list[HistoryTurn] = []
    invalid_count = 0
    final_success = False
    source = os.environ if environ is None else environ

    try:
        max_steps = _require_max_steps(max_steps)
        resolved_trajectory_id = _required_string(
            "trajectory_id",
            explicit=trajectory_id,
            environment_key="RELAX_SESSION_ID",
            environ=environ,
        )
        snapshot = env.reset()
        task_description = extract_task_description(snapshot.raw_observation)
        task_id = _task_id_from_snapshot(
            snapshot,
            fallback_task_id=fallback_task_id,
            data_root=source.get("ALFWORLD_DATA"),
        )
        if rollout_group_id is not None and (not isinstance(rollout_group_id, str) or not rollout_group_id):
            raise ValueError("rollout_group_id must be a non-empty string")

        for turn_index in range(max_steps):
            state_key = state_key_v1(
                snapshot.raw_observation,
                snapshot.tracker,
                snapshot.admissible_commands,
            )
            prompt = build_prompt(
                task_description=task_description,
                raw_observation=format_reference_observation(
                    snapshot.raw_observation,
                    snapshot.tracker,
                ),
                admissible_commands=snapshot.admissible_commands,
                history=history,
                step_index=turn_index,
            )
            request_messages = [{"role": "user", "content": prompt}]
            response_text = await chat_client.complete(messages=request_messages)
            if not isinstance(response_text, str):
                raise TypeError("chat_client.complete() must return a string")
            parsed_action = parse_action(response_text)
            if not parsed_action.is_valid:
                invalid_count += 1

            next_snapshot = env.step(parsed_action.action)
            final_success = bool(next_snapshot.won)
            reached_horizon = turn_index + 1 == max_steps
            terminal = bool(final_success or (next_snapshot.done and not reached_horizon))
            truncated = bool(reached_horizon and not final_success)
            next_state_key = (
                SUCCESS
                if final_success
                else state_key_v1(
                    next_snapshot.raw_observation,
                    next_snapshot.tracker,
                    next_snapshot.admissible_commands,
                )
            )
            turn_id = f"turn_{turn_index:03d}"
            identity_metadata: dict[str, Any] = {}
            if rollout_group_id is not None:
                identity_metadata = {
                    "rollout_group_id": rollout_group_id,
                    "turn_id": turn_id,
                    "row_id": stable_row_id(
                        task_id=task_id,
                        rollout_group_id=rollout_group_id,
                        trajectory_id=resolved_trajectory_id,
                        turn_index=turn_index,
                    ),
                }
            records.append(
                {
                    "name": turn_id,
                    "messages": [
                        dict(request_messages[0]),
                        {"role": "assistant", "content": response_text},
                    ],
                    "metadata": {
                        "task_id": task_id,
                        "trajectory_id": resolved_trajectory_id,
                        "turn_index": turn_index,
                        "state_key": state_key,
                        "action": parsed_action.action,
                        "next_state_key": next_state_key,
                        "is_action_valid": parsed_action.is_valid,
                        "success": False,
                        "terminal": terminal,
                        "truncated": truncated,
                        "episode_return": 0.0,
                        **identity_metadata,
                    },
                    "reward": 0.0,
                }
            )
            history.append(
                HistoryTurn(
                    action=parsed_action.action,
                    observation=format_reference_observation(
                        next_snapshot.raw_observation,
                        next_snapshot.tracker,
                    ),
                )
            )
            snapshot = next_snapshot
            if terminal or truncated:
                break
    finally:
        env.close()

    return _finalize_records(
        records,
        success=final_success,
        invalid_count=invalid_count,
    )


def read_session_input(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("session input must be a JSON object")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("session input metadata must be a JSON object")
    return payload


def write_session_output(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Write explicit records as JSONL, never as a JSON array."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    lines: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("every output record must be a mapping")
        lines.append(
            json.dumps(
                dict(record),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    if not lines:
        raise ValueError("at least one explicit record is required")
    try:
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    except FileNotFoundError:
        # The managed runtime can remove its temporary directory after timeout.
        pass


def _session_metadata(payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, MutableMapping):
        raise TypeError("session input metadata must be a mutable JSON object")
    return metadata


async def run_managed_session(
    *,
    session_input: Mapping[str, Any],
    chat_client: ChatCompletionClient | None = None,
    env: AlfWorldTextEnv | None = None,
    group_id: str | None = None,
    trajectory_id: str | None = None,
    task_id: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve managed-process inputs and run one ALFWorld episode."""

    metadata = _session_metadata(session_input)
    resolved_group_id = _required_string(
        "group_id",
        explicit=group_id,
        environment_key="RELAX_GROUP_ID",
        environ=environ,
    )
    resolved_trajectory_id = _required_string(
        "trajectory_id",
        explicit=trajectory_id,
        environment_key="RELAX_SESSION_ID",
        environ=environ,
    )
    source = os.environ if environ is None else environ

    owns_client = chat_client is None
    if chat_client is None:
        chat_client = OpenAICompatibleChatClient(
            api_key=source.get("OPENAI_API_KEY") or source.get("RELAX_SESSION_ID", ""),
            base_url=source.get("OPENAI_BASE_URL") or source.get("RELAX_BASE_URL", ""),
            model=source.get("OPENAI_MODEL", "model"),
            temperature=_sampling_float(
                metadata,
                source,
                metadata_key="temperature",
                environment_key="GRAPHGPO_TEMPERATURE",
                default=1.0,
            ),
            top_p=_sampling_float(
                metadata,
                source,
                metadata_key="top_p",
                environment_key="GRAPHGPO_TOP_P",
                default=1.0,
                minimum_exclusive=0.0,
                maximum=1.0,
            ),
            max_tokens=_sampling_max_tokens(metadata, source),
        )

    fallback_task_id = task_id if task_id is not None else metadata.get("task_id")
    if env is None:
        config_path = metadata.get("alfworld_config_path") or source.get("ALFWORLD_CONFIG_PATH")
        if not isinstance(config_path, str) or not config_path:
            raise ValueError("ALFWorld config is required in metadata.alfworld_config_path or ALFWORLD_CONFIG_PATH")
        train_eval = metadata.get("alfworld_train_eval", "train")
        if not isinstance(train_eval, str) or not train_eval:
            raise ValueError("metadata.alfworld_train_eval must be a string")
        env = AlfWorldTextEnv(
            config_path=config_path,
            seed=stable_group_seed(resolved_group_id),
            train_eval=train_eval,
            game_file=(fallback_task_id if isinstance(fallback_task_id, str) else None),
        )

    max_steps = metadata.get("max_steps", DEFAULT_MAX_STEPS)
    try:
        return await run_episode(
            chat_client=chat_client,
            env=env,
            trajectory_id=resolved_trajectory_id,
            rollout_group_id=resolved_group_id,
            fallback_task_id=fallback_task_id,
            max_steps=max_steps,
            environ=environ,
        )
    finally:
        if owns_client and isinstance(chat_client, OpenAICompatibleChatClient):
            await chat_client.close()


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphGPO ALFWorld managed agent session.")
    parser.add_argument("--input-json")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    source = os.environ if environ is None else environ
    args.input_json = args.input_json or source.get("RELAX_INPUT_JSON")
    args.output_json = args.output_json or source.get("RELAX_OUTPUT_JSON")
    if not args.input_json:
        parser.error("--input-json or the RELAX_INPUT_JSON environment variable is required")
    if not args.output_json:
        parser.error("--output-json or the RELAX_OUTPUT_JSON environment variable is required")
    return args


def main() -> None:
    args = parse_args()
    session_input = read_session_input(args.input_json)
    records = asyncio.run(run_managed_session(session_input=session_input))
    write_session_output(args.output_json, records)


if __name__ == "__main__":
    main()
