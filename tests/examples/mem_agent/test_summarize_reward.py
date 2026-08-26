# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tests for the durable MemAgent training-reward acceptance artifact."""

from __future__ import annotations

import pytest

from examples.mem_agent.summarize_reward import (
    extract_reward_points,
    extract_reward_points_from_rollout_results,
    summarize_reward_points,
    write_reward_csv,
    write_reward_svg,
)


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


def test_summary_accepts_complete_resumed_rollout_range():
    summary = summarize_reward_points(
        [(20, 0.25), (21, 0.5), (22, 0.75)],
        expected_steps=3,
        expected_start=20,
        window_size=1,
    )

    assert summary["first_rollout_id"] == 20
    assert summary["last_rollout_id"] == 22
    with pytest.raises(ValueError, match="incomplete"):
        summarize_reward_points([(20, 0.25), (22, 0.75)], expected_steps=3, expected_start=20)


def test_rollout_result_rewards_are_trajectory_means_and_support_resume(tmp_path):
    rollout_dir = tmp_path / "rollout_result" / "train"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "20.jsonl").write_text(
        "\n".join(
            [
                '{"reward": {"score": 1.0, "mem_agent_raw_reward": 1.0}}',
                '{"reward": {"score": 0.0, "mem_agent_raw_reward": 0.0}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (rollout_dir / "21.jsonl").write_text(
        '{"reward": {"score": 1.0, "mem_agent_raw_reward": 1.0}}\n', encoding="utf-8"
    )

    points = extract_reward_points_from_rollout_results(
        rollout_dir,
        expected_rollout_ids=range(20, 22),
    )

    assert points == [(20, 0.5), (21, 1.0)]
    with pytest.raises(ValueError, match="Missing rollout result"):
        extract_reward_points_from_rollout_results(rollout_dir, expected_rollout_ids=range(20, 23))


def test_reward_csv_and_svg_keep_auditable_points(tmp_path):
    points = [(0, 0.25), (1, 0.75)]
    csv_path = tmp_path / "reward.csv"
    svg_path = tmp_path / "reward.svg"

    write_reward_csv(csv_path, points)
    write_reward_svg(svg_path, points)

    assert csv_path.read_text(encoding="utf-8").splitlines() == ["rollout_id,reward", "0,0.25", "1,0.75"]
    svg = svg_path.read_text(encoding="utf-8")
    assert "MemAgent rollout reward" in svg
    assert '<polyline points="' in svg
    assert svg.count("<circle ") == 2
