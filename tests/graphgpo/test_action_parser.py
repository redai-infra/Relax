# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import unittest

from examples.graphgpo.action_parser import parse_action


class ActionParserTest(unittest.TestCase):
    def test_first_lowercase_action_tag_wins(self) -> None:
        result = parse_action("<think>choose</think><action>go north</action><action>open door</action>")

        self.assertEqual(result.action, "go north")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.source, "tag")

    def test_response_is_lowercased_before_action_extraction(self) -> None:
        response = "<think>Choose</think><ACTION>Go North</ACTION>"
        result = parse_action(response)

        self.assertEqual(result.action, "go north")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.source, "tag")

    def test_missing_tag_fallback_is_always_invalid(self) -> None:
        result = parse_action("<think>x</think>" + "X" * 40)

        self.assertEqual(result.action, "x" * 30)
        self.assertFalse(result.is_valid)

    def test_missing_think_or_chinese_response_is_invalid(self) -> None:
        self.assertFalse(parse_action("<action>go north</action>").is_valid)
        self.assertFalse(parse_action("<think>我来选择</think><action>go north</action>").is_valid)

    def test_multiline_action_is_supported(self) -> None:
        result = parse_action("<think>choose</think><action>\nput apple in fridge\n</action>")

        self.assertEqual(result.action, "put apple in fridge")
        self.assertTrue(result.is_valid)

    def test_closing_tag_before_opening_tag_is_not_a_match(self) -> None:
        response = "<think>choose</think></action>prefix<action>go north</action>"
        result = parse_action(response)

        self.assertEqual(result.action, "go north")
        self.assertTrue(result.is_valid)

    def test_empty_action_matches_reference_validity(self) -> None:
        result = parse_action("<think>choose</think><action> </action>")

        self.assertEqual(result.action, "")
        self.assertTrue(result.is_valid)


if __name__ == "__main__":
    unittest.main()
