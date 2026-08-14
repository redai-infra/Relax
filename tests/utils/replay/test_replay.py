# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay pipeline tests: adapters and divergence detection."""

from __future__ import annotations

import json

import pytest
import torch

from relax.utils.replay.adapters.reward import replay_reward_post_process
from relax.utils.replay.bundle import BundleReader
from relax.utils.replay.report import StageStatus
from relax.utils.replay.runner import replay
from relax.utils.replay.validate import validate_bundle
from tests.utils.replay.helpers import (
    DEFAULT_LOSS,
    NORMALIZED_REWARDS,
    build_grpo_bundle,
)


def test_reward_group_normalization_reference(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    loaded = BundleReader(bundle).load()
    ctx: dict = {}
    result = replay_reward_post_process(loaded, ctx)

    assert result.status == StageStatus.PASS
    assert ctx["reward.post_process"] == pytest.approx(NORMALIZED_REWARDS)


def test_replay_passes_on_valid_bundle(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    report = replay(bundle)

    assert report.passed is True
    assert report.first_divergent_stage is None
    assert all(stage.status != StageStatus.FAIL for stage in report.stages)


def test_reward_pr65_fixture(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", pr65_bug=True)
    report = replay(bundle)

    assert report.first_divergent_stage == "reward.post_process"
    reward_stage = next(stage for stage in report.stages if stage.stage == "reward.post_process")
    assert reward_stage.status == StageStatus.FAIL


def test_loss_ratio_one_reference(tmp_path):
    bundle, _, expected = build_grpo_bundle(tmp_path / "bundle", ratio_one=True)
    report = replay(bundle)
    assert report.passed
    assert expected["loss.policy"]["loss"] == pytest.approx(DEFAULT_LOSS)
    assert expected["loss.policy"]["pg_loss"] == pytest.approx(0.0)
    assert expected["loss.policy"]["entropy_loss"] == pytest.approx(2.0)


def test_loss_ratio_not_one(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", ratio_one=False)
    report = replay(bundle)
    assert report.passed


@pytest.mark.parametrize(
    ("corrupt", "first_stage", "kwargs"),
    [
        ("reward", "reward.raw", {}),
        (
            "mask_token",
            "loss.policy",
            {"old_log_probs": torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5])},
        ),
        ("old_log_probability", "loss.policy", {"kl_coef": 0.0}),
    ],
)
def test_corrupt_input_detected(tmp_path, corrupt, first_stage, kwargs):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", corrupt=corrupt, **kwargs)
    report = replay(bundle)
    assert report.first_divergent_stage == first_stage


def test_corrupt_schema_field_detected(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    index_path = bundle / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["samples"][0]["loss_mask"]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    result = validate_bundle(bundle)
    assert not result.valid
