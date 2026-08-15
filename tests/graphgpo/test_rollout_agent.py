# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import examples.graphgpo.rollout_agent as rollout_agent
from examples.graphgpo.alfworld_env import AlfWorldSnapshot
from examples.graphgpo.graph_credit import SUCCESS
from examples.graphgpo.rollout_agent import (
    extract_task_description,
    parse_args,
    run_episode,
    run_managed_session,
    stable_group_seed,
    stable_row_id,
    write_session_output,
)
from examples.graphgpo.state import TrackerState


def _snapshot(
    observation: str,
    commands: tuple[str, ...],
    *,
    done: bool = False,
    won: bool = False,
    gamefile: str | None = "games/task-alpha/game.tw-pddl",
) -> AlfWorldSnapshot:
    return AlfWorldSnapshot(
        raw_observation=observation,
        admissible_commands=commands,
        won=won,
        done=done,
        gamefile=gamefile,
        tracker=TrackerState.from_mapping(
            {
                "location": "middle of a room",
                "holding": "nothing",
                "history_items": {},
                "item_location": {},
            }
        ),
    )


class FakeEpisodeEnv:
    def __init__(
        self,
        initial: AlfWorldSnapshot,
        transitions: list[AlfWorldSnapshot],
    ) -> None:
        self.initial = initial
        self.transitions = list(transitions)
        self.actions: list[str] = []
        self.closed = False

    def reset(self) -> AlfWorldSnapshot:
        return self.initial

    def step(self, action: str) -> AlfWorldSnapshot:
        self.actions.append(action)
        return self.transitions.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeChatClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(self, *, messages: list[dict[str, Any]]) -> str:
        self.calls.append(messages)
        return self.responses.pop(0)


def _valid_response(action: str) -> str:
    return f"<think>reason</think><action>{action}</action>"


def test_stable_group_seed_is_group_shared_and_deterministic() -> None:
    assert stable_group_seed("group-a") == stable_group_seed("group-a")
    assert stable_group_seed("group-a") != stable_group_seed("group-b")
    assert 0 <= stable_group_seed("group-a") < 2**31


def test_stable_row_id_binds_task_group_trajectory_and_turn() -> None:
    row_id = stable_row_id(
        task_id="task-a",
        rollout_group_id="group-a",
        trajectory_id="trajectory-a",
        turn_index=0,
    )
    assert row_id == stable_row_id(
        task_id="task-a",
        rollout_group_id="group-a",
        trajectory_id="trajectory-a",
        turn_index=0,
    )
    assert row_id != stable_row_id(
        task_id="task-a",
        rollout_group_id="group-a",
        trajectory_id="trajectory-a",
        turn_index=1,
    )


def test_extract_task_description_is_strict() -> None:
    assert extract_task_description("Intro. Your task is to: cool the mug") == "cool the mug"
    with pytest.raises(ValueError, match="not found"):
        extract_task_description("No task marker")
    with pytest.raises(ValueError, match="empty"):
        extract_task_description("Your task is to: ")


def test_parse_args_uses_managed_runtime_environment_paths() -> None:
    args = parse_args(
        [],
        environ={
            "RELAX_INPUT_JSON": "/tmp/session_input.json",
            "RELAX_OUTPUT_JSON": "/tmp/session_output.json",
        },
    )
    assert args.input_json == "/tmp/session_input.json"
    assert args.output_json == "/tmp/session_output.json"

    explicit = parse_args(
        [
            "--input-json",
            "explicit-input.json",
            "--output-json",
            "explicit-output.json",
        ],
        environ={
            "RELAX_INPUT_JSON": "ignored-input.json",
            "RELAX_OUTPUT_JSON": "ignored-output.json",
        },
    )
    assert explicit.input_json == "explicit-input.json"
    assert explicit.output_json == "explicit-output.json"


@pytest.mark.asyncio
async def test_run_episode_exports_fresh_per_turn_messages_and_last_two_history() -> None:
    env = FakeEpisodeEnv(
        _snapshot(
            "Intro. Your task is to: inspect the room",
            ("action zero", "help"),
        ),
        [
            _snapshot("observation one", ("action one",)),
            _snapshot("observation two", ("action two",)),
            _snapshot("observation three", ("action three",)),
            _snapshot("observation four", ("look",), done=True),
        ],
    )
    client = FakeChatClient(
        [
            _valid_response("action zero"),
            _valid_response("action one"),
            _valid_response("action two"),
            _valid_response("action three"),
        ]
    )

    records = await run_episode(
        chat_client=client,
        env=env,  # type: ignore[arg-type]
        trajectory_id="session-1",
    )

    assert len(records) == 4
    assert all(len(call) == 1 for call in client.calls)
    assert all(call[0]["role"] == "user" for call in client.calls)
    assert all(len(record["messages"]) == 2 for record in records)
    assert all([message["role"] for message in record["messages"]] == ["user", "assistant"] for record in records)
    fourth_prompt = client.calls[3][0]["content"]
    assert "observation one" not in fourth_prompt
    assert "observation two" in fourth_prompt
    assert "observation three" in fourth_prompt
    assert "'help'" not in client.calls[0][0]["content"]
    assert env.closed is True


@pytest.mark.asyncio
async def test_parser_valid_action_outside_commands_stays_valid_and_is_executed() -> None:
    env = FakeEpisodeEnv(
        _snapshot(
            "Intro. Your task is to: look around",
            ("look",),
        ),
        [_snapshot("Nothing happens.", ("look",), done=True)],
    )
    client = FakeChatClient([_valid_response("dance")])

    records = await run_episode(
        chat_client=client,
        env=env,  # type: ignore[arg-type]
        trajectory_id="session-outside-commands",
    )

    assert env.actions == ["dance"]
    assert records[0]["metadata"]["is_action_valid"] is True
    assert records[0]["metadata"]["episode_return"] == 0.0


@pytest.mark.asyncio
async def test_run_episode_success_backfills_return_and_success_sink() -> None:
    env = FakeEpisodeEnv(
        _snapshot(
            "Intro. Your task is to: take the apple",
            ("look",),
        ),
        [
            _snapshot("Nothing happens.", ("take apple",)),
            _snapshot("You win.", ("look",), won=True, done=True),
        ],
    )
    client = FakeChatClient(
        [
            "malformed response",
            _valid_response("take apple"),
        ]
    )

    records = await run_episode(
        chat_client=client,
        env=env,  # type: ignore[arg-type]
        trajectory_id="session-success",
    )

    assert [record["name"] for record in records] == ["turn_000", "turn_001"]
    assert env.actions[0] == "malformed response"[-30:]
    assert all(record["metadata"]["success"] is True for record in records)
    assert all(record["metadata"]["episode_return"] == 9.9 for record in records)
    assert all(record["reward"] == 9.9 for record in records)
    assert records[0]["metadata"]["is_action_valid"] is False
    assert records[0]["metadata"]["terminal"] is False
    assert records[-1]["metadata"]["next_state_key"] == SUCCESS
    assert records[-1]["metadata"]["terminal"] is True
    assert records[-1]["metadata"]["truncated"] is False
    assert records[-1]["metadata"]["task_id"] == ("games/task-alpha/game.tw-pddl")


@pytest.mark.asyncio
async def test_run_episode_marks_unsuccessful_done_and_max_steps() -> None:
    done_env = FakeEpisodeEnv(
        _snapshot("Your task is to: wait", ("look",)),
        [_snapshot("Episode ended.", ("look",), done=True)],
    )
    done_records = await run_episode(
        chat_client=FakeChatClient([_valid_response("look")]),
        env=done_env,  # type: ignore[arg-type]
        trajectory_id="done",
    )
    assert done_records[-1]["metadata"]["terminal"] is True
    assert done_records[-1]["metadata"]["truncated"] is False
    assert done_records[-1]["metadata"]["success"] is False

    truncated_env = FakeEpisodeEnv(
        _snapshot("Your task is to: wait", ("look",)),
        [
            _snapshot("Still waiting one.", ("look",)),
            _snapshot("Still waiting two.", ("look",)),
        ],
    )
    truncated_records = await run_episode(
        chat_client=FakeChatClient([_valid_response("look"), _valid_response("look")]),
        env=truncated_env,  # type: ignore[arg-type]
        trajectory_id="truncated",
        max_steps=2,
    )
    assert len(truncated_records) == 2
    assert truncated_records[-1]["metadata"]["terminal"] is False
    assert truncated_records[-1]["metadata"]["truncated"] is True

    horizon_done_env = FakeEpisodeEnv(
        _snapshot("Your task is to: wait", ("look",)),
        [
            _snapshot("Still waiting one.", ("look",)),
            _snapshot("Time limit reached.", ("look",), done=True),
        ],
    )
    horizon_done_records = await run_episode(
        chat_client=FakeChatClient([_valid_response("look"), _valid_response("look")]),
        env=horizon_done_env,  # type: ignore[arg-type]
        trajectory_id="horizon-done",
        max_steps=2,
    )
    assert horizon_done_records[-1]["metadata"]["terminal"] is False
    assert horizon_done_records[-1]["metadata"]["truncated"] is True


@pytest.mark.asyncio
async def test_run_episode_uses_metadata_task_id_only_without_gamefile() -> None:
    env = FakeEpisodeEnv(
        _snapshot(
            "Your task is to: wait",
            ("look",),
            gamefile=None,
        ),
        [_snapshot("Done.", ("look",), done=True, gamefile=None)],
    )
    records = await run_episode(
        chat_client=FakeChatClient([_valid_response("look")]),
        env=env,  # type: ignore[arg-type]
        trajectory_id="session-fallback-task",
        fallback_task_id="public-task-id",
    )
    assert records[0]["metadata"]["task_id"] == "public-task-id"


@pytest.mark.asyncio
async def test_run_episode_uses_public_relative_task_id_for_absolute_gamefile() -> None:
    gamefile = "/srv/task37/alfworld/json_2.1.1/train/pick_and_place_simple-Apple-None-Fridge-1/trial-1/game.tw-pddl"
    env = FakeEpisodeEnv(
        _snapshot("Your task is to: wait", ("look",), gamefile=gamefile),
        [_snapshot("Done.", ("look",), done=True, gamefile=gamefile)],
    )
    records = await run_episode(
        chat_client=FakeChatClient([_valid_response("look")]),
        env=env,  # type: ignore[arg-type]
        trajectory_id="session-relative-task",
        environ={"ALFWORLD_DATA": "/srv/task37/alfworld"},
    )
    assert records[0]["metadata"]["task_id"] == (
        "json_2.1.1/train/pick_and_place_simple-Apple-None-Fridge-1/trial-1/game.tw-pddl"
    )


@pytest.mark.asyncio
async def test_run_episode_rejects_gamefile_outside_data_root() -> None:
    env = FakeEpisodeEnv(
        _snapshot(
            "Your task is to: wait",
            ("look",),
            gamefile="/other/data/game.tw-pddl",
        ),
        [],
    )
    with pytest.raises(ValueError, match="outside ALFWORLD_DATA"):
        await run_episode(
            chat_client=FakeChatClient([]),
            env=env,  # type: ignore[arg-type]
            trajectory_id="session-outside-root",
            environ={"ALFWORLD_DATA": "/srv/task37/alfworld"},
        )
    assert env.closed is True


@pytest.mark.asyncio
async def test_run_episode_rejects_declared_task_mismatch() -> None:
    env = FakeEpisodeEnv(
        _snapshot(
            "Your task is to: wait",
            ("look",),
            gamefile="/srv/task37/alfworld/json_2.1.1/train/a/game.tw-pddl",
        ),
        [],
    )
    with pytest.raises(ValueError, match="declared task_id"):
        await run_episode(
            chat_client=FakeChatClient([]),
            env=env,  # type: ignore[arg-type]
            trajectory_id="session-task-mismatch",
            fallback_task_id="json_2.1.1/train/b/game.tw-pddl",
            environ={"ALFWORLD_DATA": "/srv/task37/alfworld"},
        )
    assert env.closed is True


@pytest.mark.asyncio
async def test_run_episode_closes_env_on_errors_and_missing_ids() -> None:
    class RaisingClient:
        async def complete(self, *, messages: list[dict[str, Any]]) -> str:
            raise RuntimeError("chat failed")

    env = FakeEpisodeEnv(
        _snapshot("Your task is to: wait", ("look",)),
        [],
    )
    with pytest.raises(RuntimeError, match="chat failed"):
        await run_episode(
            chat_client=RaisingClient(),
            env=env,  # type: ignore[arg-type]
            trajectory_id="session-error",
        )
    assert env.closed is True

    missing_id_env = FakeEpisodeEnv(
        _snapshot("Your task is to: wait", ("look",)),
        [],
    )
    with pytest.raises(ValueError, match="RELAX_SESSION_ID"):
        await run_episode(
            chat_client=FakeChatClient([]),
            env=missing_id_env,  # type: ignore[arg-type]
            environ={},
        )
    assert missing_id_env.closed is True


@pytest.mark.asyncio
async def test_run_managed_session_uses_relax_ids() -> None:
    env = FakeEpisodeEnv(
        _snapshot("Your task is to: wait", ("look",)),
        [_snapshot("Done.", ("look",), done=True)],
    )
    records = await run_managed_session(
        session_input={"messages": [], "metadata": {}},
        chat_client=FakeChatClient([_valid_response("look")]),
        env=env,  # type: ignore[arg-type]
        environ={
            "RELAX_GROUP_ID": "shared-group",
            "RELAX_SESSION_ID": "slot-session",
        },
    )
    metadata = records[0]["metadata"]
    assert metadata["trajectory_id"] == "slot-session"
    assert metadata["rollout_group_id"] == "shared-group"
    assert metadata["turn_id"] == "turn_000"
    assert metadata["row_id"] == stable_row_id(
        task_id=metadata["task_id"],
        rollout_group_id="shared-group",
        trajectory_id="slot-session",
        turn_index=0,
    )


@pytest.mark.asyncio
async def test_run_managed_session_uses_relax_chat_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeDefaultClient(FakeChatClient):
        def __init__(self, **kwargs: object) -> None:
            super().__init__([_valid_response("look")])
            captured.update(kwargs)
            self.was_closed = False

        async def close(self) -> None:
            self.was_closed = True

    monkeypatch.setattr(
        rollout_agent,
        "OpenAICompatibleChatClient",
        FakeDefaultClient,
    )
    env = FakeEpisodeEnv(
        _snapshot("Your task is to: wait", ("look",)),
        [_snapshot("Done.", ("look",), done=True)],
    )
    await run_managed_session(
        session_input={
            "messages": [],
            "metadata": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_tokens": 512,
            },
        },
        env=env,  # type: ignore[arg-type]
        environ={
            "RELAX_GROUP_ID": "shared-group",
            "RELAX_SESSION_ID": "slot-session",
            "RELAX_BASE_URL": "http://relax-chat/v1",
        },
    )
    assert captured["base_url"] == "http://relax-chat/v1"
    assert captured["api_key"] == "slot-session"
    assert captured["temperature"] == 0.8
    assert captured["top_p"] == 0.95
    assert captured["max_tokens"] == 512


@pytest.mark.asyncio
async def test_run_managed_session_allows_explicit_test_overrides() -> None:
    env = FakeEpisodeEnv(
        _snapshot(
            "Your task is to: wait",
            ("look",),
            gamefile=None,
        ),
        [_snapshot("Done.", ("look",), done=True, gamefile=None)],
    )
    records = await run_managed_session(
        session_input={"messages": [], "metadata": {}},
        chat_client=FakeChatClient([_valid_response("look")]),
        env=env,  # type: ignore[arg-type]
        group_id="test-group",
        trajectory_id="test-trajectory",
        task_id="test-task",
        environ={},
    )
    assert records[0]["metadata"]["task_id"] == "test-task"
    assert records[0]["metadata"]["trajectory_id"] == "test-trajectory"
    assert records[0]["metadata"]["rollout_group_id"] == "test-group"


@pytest.mark.asyncio
async def test_same_group_seed_selects_same_fake_task() -> None:
    task_by_seed: dict[int, str] = {}

    def make_env(seed: int) -> FakeEpisodeEnv:
        task = task_by_seed.setdefault(seed, f"task-{seed}")
        return FakeEpisodeEnv(
            _snapshot(
                "Your task is to: wait",
                ("look",),
                gamefile=f"games/{task}/game.tw-pddl",
            ),
            [
                _snapshot(
                    "Done.",
                    ("look",),
                    done=True,
                    gamefile=f"games/{task}/game.tw-pddl",
                )
            ],
        )

    seed_a = stable_group_seed("one-group")
    seed_b = stable_group_seed("one-group")
    records_a = await run_episode(
        chat_client=FakeChatClient([_valid_response("look")]),
        env=make_env(seed_a),  # type: ignore[arg-type]
        trajectory_id="slot-a",
    )
    records_b = await run_episode(
        chat_client=FakeChatClient([_valid_response("look")]),
        env=make_env(seed_b),  # type: ignore[arg-type]
        trajectory_id="slot-b",
    )
    assert records_a[0]["metadata"]["task_id"] == records_b[0]["metadata"]["task_id"]


@pytest.mark.asyncio
async def test_jsonl_output_is_accepted_by_session_output(tmp_path: Path) -> None:
    env = FakeEpisodeEnv(
        _snapshot("Your task is to: wait", ("look",)),
        [_snapshot("Done.", ("look",), done=True)],
    )
    records = await run_episode(
        chat_client=FakeChatClient([_valid_response("look")]),
        env=env,  # type: ignore[arg-type]
        trajectory_id="session-jsonl",
    )
    output_path = tmp_path / "session_output.json"
    write_session_output(output_path, records)

    raw_text = output_path.read_text(encoding="utf-8")
    assert raw_text.startswith("{")
    assert not raw_text.lstrip().startswith("[")
    parsed_records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    assert len(parsed_records) == 1

    try:
        from relax.agentic.pipeline.runtime import SessionOutput
    except ImportError:
        pytest.skip("Relax runtime dependencies are unavailable")
    output = SessionOutput.from_records(parsed_records)
    assert len(output.records) == 1
    assert output.records[0]["name"] == "turn_000"
