# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from examples.graphgpo.alfworld_env import (
    AlfWorldTextEnv,
    _canonicalize_game_files,
    _select_game_file,
)


def _batched_info(
    commands: list[str],
    *,
    won: bool = False,
    gamefile: str = "games/task-1/game.tw-pddl",
) -> dict[str, list[object]]:
    return {
        "admissible_commands": [commands],
        "won": [won],
        "extra.gamefile": [gamefile],
    }


class FakeRawAlfWorldEnv:
    def __init__(self, transitions: list[tuple[str, list[str], bool, bool]]) -> None:
        self.transitions = list(transitions)
        self.actions: list[list[str]] = []
        self.closed = False
        self.seed_values: list[int] = []

    def seed(self, seed: int) -> None:
        self.seed_values.append(seed)

    def reset(self):
        return (
            ["Room intro. Your task is to: put the apple in the fridge"],
            _batched_info(["go to cabinet 1", "help"]),
        )

    def step(self, actions: list[str]):
        self.actions.append(actions)
        observation, commands, done, won = self.transitions.pop(0)
        return (
            [observation],
            [10.0 if won else 0.0],
            [done],
            _batched_info(commands, won=won),
        )

    def close(self) -> None:
        self.closed = True


def test_alfworld_game_files_are_canonicalized_before_seeded_reset() -> None:
    class FakeBaseEnv:
        game_files = [
            r"root\z-task\game.tw-pddl",
            "root/a-task/game.tw-pddl",
            r"root\m-task\game.tw-pddl",
        ]
        num_games = 99

    base_env = FakeBaseEnv()
    _canonicalize_game_files(base_env)

    assert base_env.game_files == [
        "root/a-task/game.tw-pddl",
        r"root\m-task\game.tw-pddl",
        r"root\z-task\game.tw-pddl",
    ]
    assert base_env.num_games == 3


def test_alfworld_selected_game_file_uses_manifest_relative_path() -> None:
    class FakeBaseEnv:
        game_files = [
            "/data/alfworld/json_2.1.1/train/task-b/game.tw-pddl",
            "/data/alfworld/json_2.1.1/train/task-a/game.tw-pddl",
        ]
        num_games = 2

    base_env = FakeBaseEnv()
    _select_game_file(
        base_env,
        "json_2.1.1/train/task-a/game.tw-pddl",
        data_root="/data/alfworld",
    )

    assert base_env.game_files == ["/data/alfworld/json_2.1.1/train/task-a/game.tw-pddl"]
    assert base_env.num_games == 1


def test_alfworld_env_tracks_reference_fields_and_nothing_happens() -> None:
    raw_env = FakeRawAlfWorldEnv(
        [
            (
                "You arrive at cabinet 1.",
                ["take apple 1 from cabinet 1"],
                False,
                False,
            ),
            (
                "You take apple 1 from cabinet 1.",
                ["heat apple 1 with microwave 1"],
                False,
                False,
            ),
            (
                "You heat apple 1 with microwave 1.",
                ["go to fridge 1"],
                False,
                False,
            ),
            (
                "Nothing happens.",
                ["look"],
                True,
                False,
            ),
        ]
    )
    env = AlfWorldTextEnv(env=raw_env)

    initial = env.reset()
    assert initial.raw_observation.startswith("Room intro.")
    assert initial.admissible_commands == ("go to cabinet 1", "help")
    assert initial.gamefile == "games/task-1/game.tw-pddl"
    assert initial.tracker.to_mapping() == {
        "location": "middle of a room",
        "holding": "nothing",
        "history_items": {},
        "item_location": {},
    }

    at_cabinet = env.step("go to cabinet 1")
    assert at_cabinet.tracker.to_mapping()["location"] == "cabinet 1"

    holding = env.step("take apple 1 from cabinet 1")
    holding_tracker = holding.tracker.to_mapping()
    assert holding_tracker["holding"] == "apple 1"
    assert holding_tracker["item_location"]["apple 1"] == {
        "old_location": "cabinet 1",
        "new_location": "cabinet 1",
    }

    heated = env.step("heat apple 1 with microwave 1")
    heated_tracker = heated.tracker.to_mapping()
    assert heated_tracker["history_items"]["apple 1"] == {
        "heated": True,
        "cooled": False,
        "cleaned": False,
        "slice": False,
    }

    unchanged = env.step("go to fridge 1")
    assert unchanged.done is True
    assert unchanged.tracker == heated.tracker
    assert raw_env.actions == [
        ["go to cabinet 1"],
        ["take apple 1 from cabinet 1"],
        ["heat apple 1 with microwave 1"],
        ["go to fridge 1"],
    ]

    env.close()
    env.close()
    assert raw_env.closed is True


def test_alfworld_env_is_lazy_and_factory_receives_seed_and_split() -> None:
    calls: list[tuple[str | Path, int, str, str | Path | None]] = []
    raw_env = FakeRawAlfWorldEnv([])

    def factory(
        config_path: str | Path,
        seed: int,
        split: str,
        game_file: str | Path | None,
    ):
        calls.append((config_path, seed, split, game_file))
        return raw_env

    env = AlfWorldTextEnv(
        config_path="config.yaml",
        seed=123,
        train_eval="eval_in_distribution",
        game_file="json_2.1.1/valid_seen/task/game.tw-pddl",
        env_factory=factory,
    )
    assert calls == []
    env.reset()
    assert calls == [
        (
            "config.yaml",
            123,
            "eval_in_distribution",
            "json_2.1.1/valid_seen/task/game.tw-pddl",
        )
    ]
    env.close()


def test_alfworld_env_rejects_step_after_done() -> None:
    raw_env = FakeRawAlfWorldEnv([("Done.", ["look"], True, False)])
    env = AlfWorldTextEnv(env=raw_env)
    env.reset()
    env.step("look")
    with pytest.raises(RuntimeError, match="completed"):
        env.step("look")
    env.close()


def test_alfworld_env_requires_one_element_batches() -> None:
    class BadRawEnv:
        def reset(self):
            return ["one", "two"], _batched_info(["look"])

        def close(self) -> None:
            pass

    env = AlfWorldTextEnv(env=BadRawEnv())
    with pytest.raises(ValueError, match="exactly one"):
        env.reset()
    env.close()
