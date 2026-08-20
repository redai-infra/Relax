# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import unittest

from examples.graphgpo.state import (
    ALFWORLD_TRACKER_FIELDS,
    TrackerState,
    reference_anchor_v1,
    state_key_v1,
    update_tracker,
)


class StateAnchorTest(unittest.TestCase):
    @staticmethod
    def _tracker(**updates: object) -> TrackerState:
        values: dict[str, object] = {
            "holding": "nothing",
            "location": "kitchen",
            "history_items": {},
            "item_location": {},
        }
        values.update(updates)
        return TrackerState.from_mapping(values)

    def test_anchor_contains_raw_observation_tracker_and_all_sorted_commands(
        self,
    ) -> None:
        tracker = self._tracker(holding="apple")
        anchor = reference_anchor_v1(
            "You see a table.\nRaw line.",
            tracker,
            ["take apple", "help", "go north"],
        )

        self.assertIn("You see a table.\nRaw line.", anchor)
        self.assertIn("Items in hand (status): apple(unprocessed)", anchor)
        self.assertIn("Location: kitchen", anchor)
        self.assertLess(anchor.index("go north"), anchor.index("help"))
        self.assertLess(anchor.index("help"), anchor.index("take apple"))

    def test_nothing_happens_does_not_update_tracker(self) -> None:
        tracker = self._tracker()
        updated = update_tracker(
            tracker,
            raw_observation="Nothing happens.",
            updates={"holding": "apple", "location": "hall"},
        )

        self.assertIs(updated, tracker)

    def test_successful_observation_applies_structured_updates(self) -> None:
        tracker = self._tracker()
        updated = update_tracker(
            tracker,
            raw_observation="You pick up the apple.",
            updates={"holding": "apple"},
        )

        self.assertEqual(
            updated.to_mapping(),
            {
                "location": "kitchen",
                "holding": "apple",
                "history_items": {},
                "item_location": {},
            },
        )

    def test_state_key_is_stable_across_nested_mapping_and_command_order(self) -> None:
        first = state_key_v1(
            "obs",
            self._tracker(
                history_items={"mug": {"heated": True, "cooled": False}},
                item_location={
                    "mug": {
                        "old_location": "kitchen",
                        "new_location": "hall",
                    }
                },
            ),
            ["help", "go north"],
        )
        second = state_key_v1(
            "obs",
            TrackerState.from_mapping(
                {
                    "item_location": {
                        "mug": {
                            "new_location": "hall",
                            "old_location": "kitchen",
                        }
                    },
                    "history_items": {"mug": {"cooled": False, "heated": True}},
                    "holding": "nothing",
                    "location": "kitchen",
                }
            ),
            ["go north", "help"],
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_tracker_fields_are_an_exact_whitelist(self) -> None:
        self.assertEqual(
            ALFWORLD_TRACKER_FIELDS,
            ("location", "holding", "history_items", "item_location"),
        )

        with self.assertRaisesRegex(ValueError, "missing=.*item_location"):
            TrackerState.from_mapping(
                {
                    "location": "kitchen",
                    "holding": "nothing",
                    "history_items": {},
                }
            )
        with self.assertRaisesRegex(ValueError, "unexpected=.*debug_history"):
            TrackerState.from_mapping(
                {
                    "location": "kitchen",
                    "holding": "nothing",
                    "history_items": {},
                    "item_location": {},
                    "debug_history": [],
                }
            )
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            update_tracker(
                self._tracker(),
                raw_observation="You wait.",
                updates={"debug_history": []},
            )

    def test_raw_observation_changes_state_key(self) -> None:
        tracker = self._tracker()

        self.assertNotEqual(
            state_key_v1("first", tracker, ["help"]),
            state_key_v1("second", tracker, ["help"]),
        )

    def test_hidden_internal_history_does_not_split_reference_state(self) -> None:
        base = self._tracker()
        with_hidden_history = self._tracker(
            history_items={"mug": {"cleaned": False}},
            item_location={
                "mug": {
                    "old_location": "kitchen",
                    "new_location": "kitchen",
                }
            },
        )

        self.assertEqual(
            state_key_v1("obs", base, ["look"]),
            state_key_v1("obs", with_hidden_history, ["look"]),
        )

    def test_displayed_tracker_difference_splits_reference_state(self) -> None:
        self.assertNotEqual(
            state_key_v1("obs", self._tracker(location="kitchen"), ["look"]),
            state_key_v1("obs", self._tracker(location="hall"), ["look"]),
        )


if __name__ == "__main__":
    unittest.main()
