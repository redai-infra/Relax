# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import unittest

from examples.graphgpo.prompt import (
    HistoryTurn,
    build_prompt,
    format_reference_observation,
)
from examples.graphgpo.state import TrackerState


class PromptTest(unittest.TestCase):
    def test_reference_observation_includes_tracker_context(self) -> None:
        tracker = TrackerState.from_mapping(
            {
                "location": "fridge 1",
                "holding": "apple 1",
                "history_items": {
                    "apple 1": {
                        "heated": True,
                        "cooled": False,
                    },
                    "plate 1": {
                        "cleaned": True,
                    },
                },
                "item_location": {
                    "apple 1": {
                        "old_location": "cabinet 1",
                        "new_location": "fridge 1",
                    },
                    "plate 1": {
                        "old_location": "sink 1",
                        "new_location": "sink 1",
                    },
                },
            }
        )

        observation = format_reference_observation("You arrive.", tracker)

        self.assertIn("Location: fridge 1.", observation)
        self.assertIn(
            "Items in hand (status): apple 1(heated).",
            observation,
        )
        self.assertIn(
            "apple 1(heated) from cabinet 1 move to fridge 1;",
            observation,
        )
        self.assertIn(
            "plate 1(cleaned) from sink 1 move to sink 1;",
            observation,
        )

    def test_first_prompt_uses_initial_observation_without_duplicate_task(
        self,
    ) -> None:
        prompt = build_prompt(
            task_description="Put the apple in the fridge.",
            raw_observation="Your task is to put the apple in the fridge.",
            admissible_commands=["help", "go north"],
        )

        self.assertNotIn("Task: Put the apple", prompt)
        self.assertIn("Your task is to put the apple", prompt)
        self.assertNotIn("'help'", prompt)
        self.assertIn("'go north'", prompt)

    def test_later_prompt_shows_only_last_two_history_turns(self) -> None:
        history = [
            HistoryTurn("old action", "old observation"),
            HistoryTurn("take apple", "You take the apple."),
            HistoryTurn("go north", "You enter the hall."),
        ]
        prompt = build_prompt(
            task_description="Put the apple in the fridge.",
            raw_observation="The fridge is here.",
            admissible_commands=["open fridge", "help", "go south"],
            history=history,
            step_index=3,
        )

        self.assertNotIn("old action", prompt)
        self.assertNotIn("old observation", prompt)
        self.assertIn("take apple", prompt)
        self.assertIn("go north", prompt)
        self.assertIn(
            "Your task is to: Put the apple in the fridge.",
            prompt,
        )
        self.assertIn("already taken 3 step(s)", prompt)
        self.assertIn("now at step 4", prompt)
        self.assertNotIn("'help'", prompt)

    def test_visible_commands_preserve_reference_order(self) -> None:
        prompt = build_prompt(
            task_description="task",
            raw_observation="obs",
            admissible_commands=["z command", "help", "a command"],
        )

        self.assertLess(prompt.index("'z command'"), prompt.index("'a command'"))

    def test_trimmed_history_uses_absolute_step_numbers(self) -> None:
        prompt = build_prompt(
            task_description="task",
            raw_observation="current",
            admissible_commands=["look"],
            history=[
                HistoryTurn("action nine", "observation nine"),
                HistoryTurn("action ten", "observation ten"),
            ],
            step_index=10,
        )

        self.assertIn("Observation 9", prompt)
        self.assertIn("Action 10", prompt)
        self.assertIn("now at step 11", prompt)


if __name__ == "__main__":
    unittest.main()
