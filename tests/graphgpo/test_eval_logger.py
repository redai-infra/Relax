# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from examples.graphgpo.eval_logger import (
    build_eval_metrics,
    episode_metrics,
    log_eval_rollout_data,
)


def _sample(
    trajectory_id: str,
    *,
    success: bool,
    episode_return: float,
    truncated: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "trajectory_id": trajectory_id,
            "success": success,
            "episode_return": episode_return,
            "truncated": truncated,
        }
    )


class EvalLoggerTest(unittest.TestCase):
    def test_episode_metrics_give_long_and_short_trajectories_one_vote_each(self) -> None:
        samples = [
            _sample("long", success=True, episode_return=9.8),
            _sample("long", success=True, episode_return=9.8),
            _sample("long", success=True, episode_return=9.8),
            _sample(
                "short",
                success=False,
                episode_return=-0.1,
                truncated=True,
            ),
        ]

        metrics = episode_metrics(samples)

        self.assertEqual(metrics["episode_count"], 2)
        self.assertEqual(metrics["success_rate"], 0.5)
        self.assertAlmostEqual(metrics["episode_return_mean"], 4.85)
        self.assertEqual(metrics["truncated_rate"], 0.5)

    def test_episode_metrics_reject_inconsistent_turn_metadata(self) -> None:
        samples = [
            _sample("trajectory", success=True, episode_return=10.0),
            _sample("trajectory", success=False, episode_return=10.0),
        ]

        with self.assertRaisesRegex(ValueError, "inconsistent success"):
            episode_metrics(samples)

    def test_eval_hook_namespaces_metrics_and_replaces_default_row_logging(self) -> None:
        data = {
            "alfworld": {
                "samples": [
                    _sample("a", success=True, episode_return=10.0),
                    _sample("a", success=True, episode_return=10.0),
                    _sample("b", success=False, episode_return=0.0, truncated=True),
                ]
            }
        }
        expected = build_eval_metrics(data, {"eval/runtime_s": 1.25})

        with patch("examples.graphgpo.eval_logger._emit_metrics") as emit:
            handled = log_eval_rollout_data(7, object(), data, {"eval/runtime_s": 1.25})

        self.assertTrue(handled)
        emit.assert_called_once_with(7, unittest.mock.ANY, expected)
        self.assertEqual(expected["eval/alfworld/episode_count"], 2)
        self.assertEqual(expected["eval/alfworld/success_rate"], 0.5)
        self.assertEqual(expected["eval/alfworld/truncated_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
