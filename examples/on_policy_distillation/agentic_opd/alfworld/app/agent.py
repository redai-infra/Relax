# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""ALFWorld agentic rollout mainline (one process per session).

Reads a session input containing ``metadata['game_file']`` (relative to
``$ALFWORLD_DATA``), drives a multi-turn interaction against a single pinned
ALFWorld game via the Relax OpenAI-compatible endpoint, and writes back a
session output whose ``metadata`` carries the environment outcome (``won`` /
``task_type`` / turn stats). The reward is computed downstream by
``reward_alfworld.reward_func`` from this metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from openai import APIStatusError, AsyncOpenAI

from app.env_alfworld import AlfworldEnv


MAX_TURNS = int(os.environ.get("ALFWORLD_MAX_TURNS", "40"))
HISTORY_LENGTH = int(os.environ.get("ALFWORLD_HISTORY_LENGTH", "2"))
CONFIG_PATH = str(Path(__file__).with_name("config_tw.yaml"))


def read_session_input(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_session_output(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


async def run_alfworld(metadata: dict[str, Any]) -> dict[str, Any]:
    game_file = metadata.get("game_file")
    if not game_file:
        raise ValueError("session metadata is missing required key 'game_file'")

    env = AlfworldEnv(
        game_file=game_file,
        config_path=CONFIG_PATH,
        max_turns=MAX_TURNS,
        history_length=HISTORY_LENGTH,
    )
    first_prompt = env.reset()
    messages: list[dict[str, Any]] = [{"role": "user", "content": first_prompt}]

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=httpx.Timeout(timeout=900.0, connect=30.0),
    )

    won = False
    stop_reason = "max_turns"
    num_valid_actions = 0

    for _turn in range(MAX_TURNS):
        try:
            response = await client.chat.completions.create(model="model", messages=messages)
        except APIStatusError as exc:
            error = exc.response.json().get("error")
            if isinstance(error, dict) and error.get("code") == "context_length_exceeded":
                stop_reason = "finish_length"
                break
            raise

        response_text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": response_text})
        if response.choices[0].finish_reason == "length":
            stop_reason = "finish_length"
            break

        next_prompt, done, info = env.step(response_text)
        num_valid_actions += int(bool(info["is_action_valid"]))
        if info["won"]:
            won = True
        if done:
            stop_reason = "env_done" if info["env_done"] else "max_turns"
            break
        messages.append({"role": "user", "content": next_prompt})

    return {
        "metadata": {
            "won": won,
            "success": 1.0 if won else 0.0,
            "task_type": metadata.get("task_type"),
            "num_turns": env.turn,
            "num_valid_actions": num_valid_actions,
            "stop_reason": stop_reason,
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ALFWorld agent session.")
    parser.add_argument("--input-json", required=True, help="Path to the session input JSON.")
    parser.add_argument("--output-json", required=True, help="Path to write the session output JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_input = read_session_input(args.input_json)
    metadata = session_input.get("metadata") or {}
    session_output = asyncio.run(run_alfworld(metadata=metadata))
    write_session_output(args.output_json, session_output)


if __name__ == "__main__":
    main()
