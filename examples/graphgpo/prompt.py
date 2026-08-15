# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prompt construction matching the frozen ALFWorld history policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from examples.graphgpo.state import (
    TrackerState,
)
from examples.graphgpo.state import (
    format_reference_observation as _format_reference_observation,
)


HISTORY_LENGTH = 2


@dataclass(frozen=True)
class HistoryTurn:
    """One action/observation pair shown in a later prompt."""

    action: str
    observation: str


def format_reference_observation(
    raw_observation: str,
    tracker: TrackerState,
) -> str:
    """Expose the shared frozen-reference display formatter from this
    module."""

    return _format_reference_observation(raw_observation, tracker)


def _visible_commands(admissible_commands: Sequence[str]) -> list[str]:
    if isinstance(admissible_commands, (str, bytes)) or not isinstance(admissible_commands, Sequence):
        raise TypeError("admissible_commands must be a sequence of strings")

    commands: list[str] = []
    for command in admissible_commands:
        if not isinstance(command, str):
            raise TypeError("every admissible command must be a string")
        if command != "help":
            commands.append(command)
    return commands


def _history_context(
    history: Sequence[HistoryTurn],
    *,
    step_index: int,
) -> tuple[str, int]:
    recent_history = history[-HISTORY_LENGTH:]
    if step_index < len(recent_history):
        raise ValueError("step_index cannot be smaller than the visible history")
    start_index = step_index - len(recent_history)
    lines = [
        (
            f"[Observation {start_index + offset + 1}: '{turn.observation}', "
            f"Action {start_index + offset + 1}: '{turn.action}']"
        )
        for offset, turn in enumerate(recent_history)
    ]
    return "\n".join(lines), len(recent_history)


def build_prompt(
    *,
    task_description: str,
    raw_observation: str,
    admissible_commands: Sequence[str],
    history: Sequence[HistoryTurn] = (),
    step_index: int | None = None,
) -> str:
    """Build a per-turn prompt with exactly two recent history entries.

    The initial ALFWorld observation already contains the task description, so
    the first prompt does not duplicate it.  Later prompts include the task,
    current step, the last two action/observation pairs, the current
    observation, and commands in environment order.  ``help`` is hidden only
    from the prompt; the state anchor keeps it.
    """

    if not isinstance(task_description, str):
        raise TypeError("task_description must be a string")
    if not isinstance(raw_observation, str):
        raise TypeError("raw_observation must be a string")
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be a sequence of HistoryTurn values")
    if any(not isinstance(turn, HistoryTurn) for turn in history):
        raise TypeError("every history entry must be a HistoryTurn")

    if step_index is None:
        step_index = len(history)
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a non-negative integer")

    commands = "\n ".join(f"'{command}'" for command in _visible_commands(admissible_commands))
    if history:
        action_history, valid_history_length = _history_context(
            history,
            step_index=step_index,
        )
        return (
            "You are an expert agent operating in the ALFRED Embodied "
            f"Environment. Your task is to: {task_description}\n\n"
            f"Prior to this step, you have already taken {step_index} step(s). "
            f"Below are the most recent {valid_history_length} observations and "
            f"the corresponding actions you took: {action_history}\n"
            f"You are now at step {step_index + 1} and your current observation "
            f"is: {raw_observation}\n\n"
            "Your admissible actions of the current situation are: "
            f"[{commands}].\n\n"
            "Now it's your turn to take an action.\n\n"
            "You should first reason step-by-step about the current situation. "
            "This reasoning process MUST be enclosed within <think> </think> "
            "tags.\n"
            "Once you've finished your reasoning, you should choose an "
            "admissible action for current step and present it within <action> "
            "</action> tags."
        )

    return (
        "You are an expert agent operating in the ALFRED Embodied "
        "Environment.\n\n"
        f"Your current observation is: {raw_observation}\n\n"
        "Your admissible actions of the current situation are: "
        f"[{commands}].\n\n"
        "Now it's your turn to take an action.\n\n"
        "You should first reason step-by-step about the current situation. "
        "This reasoning process MUST be enclosed within <think> </think> "
        "tags.\n"
        "Once you've finished your reasoning, you should choose an admissible "
        "action for current step and present it within <action> </action> tags."
    )
