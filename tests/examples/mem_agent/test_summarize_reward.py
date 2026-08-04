# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tests for the durable MemAgent training-reward acceptance artifact."""

from __future__ import annotations

import pytest

from examples.mem_agent.summarize_reward import extract_reward_points, summarize_reward_points


def test_extract_and_summarize_complete_reward_series():
    lines = [
        "unrelated startup output\n",
        "2026-08-04 | INFO | worker perf 0: {'rollout/response_len/mean': 4.0, "
        "'rollout/mem_agent_raw_reward/mean': 0.25}\n",
        "perf 1: {'rollout/mem_agent_raw_reward/mean': 0.5}\n",
        "perf 2: {'rollout/mem_agent_raw_reward/mean': 0.75}\n",
    ]

    points = extract_reward_points(lines)
    summary = summarize_reward_points(points, expected_steps=3, window_size=1)

    assert points == [(0, 0.25), (1, 0.5), (2, 0.75)]
    assert summary["first_window_mean"] == 0.25
    assert summary["last_window_mean"] == 0.75
    assert summary["last_minus_first"] == 0.5
    assert summary["strictly_improved"] is True
    assert summary["peak_rollout_id"] == 2
    assert summary["points"][-1] == {"rollout_id": 2, "reward": 0.75}


def test_identical_replayed_line_is_deduplicated_but_conflict_fails():
    line = "perf 0: {'rollout/mem_agent_raw_reward/mean': 0.5}\n"
    assert extract_reward_points([line, line]) == [(0, 0.5)]

    conflict = "perf 0: {'rollout/mem_agent_raw_reward/mean': 0.75}\n"
    with pytest.raises(ValueError, match="Conflicting reward values"):
        extract_reward_points([line, conflict])


def test_summary_rejects_missing_rollout_and_invalid_reward():
    with pytest.raises(ValueError, match="incomplete"):
        summarize_reward_points([(0, 0.0), (2, 1.0)], expected_steps=3)

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        extract_reward_points(["perf 0: {'rollout/mem_agent_raw_reward/mean': 1.1}\n"])
