# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay pipeline tests: adapters, closure and divergence detection."""

from __future__ import annotations

import json

import pytest
import torch

from relax.utils.replay.adapters.reward import replay_reward_post_process
from relax.utils.replay.bundle import BundleReader
from relax.utils.replay.identity import ClosureError, expand_selection
from relax.utils.replay.report import StageStatus
from relax.utils.replay.runner import replay
from relax.utils.replay.validate import validate_bundle
from tests.utils.replay.helpers import (
    ADVANTAGES,
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
    bundle, _, expected = build_grpo_bundle(tmp_path / "bundle")
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


def test_advantage_grpo_layout(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    loaded = BundleReader(bundle).load()
    assert loaded.tensors["advantages"].tolist() == pytest.approx(ADVANTAGES)


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


def test_corrupt_reward_detected(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", corrupt="reward")
    report = replay(bundle)
    assert report.first_divergent_stage == "reward.raw"


def test_corrupt_mask_token_detected(tmp_path):
    # Non-uniform per-token values so masking one token changes the per-sample mean.
    old_log_probs = torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", corrupt="mask_token", old_log_probs=old_log_probs)
    report = replay(bundle)
    assert report.first_divergent_stage == "loss.policy"


def test_corrupt_old_log_probability_detected(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", corrupt="old_log_probability", kl_coef=0.0)
    report = replay(bundle)
    assert report.first_divergent_stage == "loss.policy"


def test_corrupt_schema_field_detected(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    index_path = bundle / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["samples"][0]["loss_mask"]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    result = validate_bundle(bundle)
    assert not result.valid


def test_selection_group_closure(tmp_path):
    bundle, index, _ = build_grpo_bundle(tmp_path / "bundle")
    closure = expand_selection(index, sample_ids=["s-0"])
    assert set(closure.sample_ids) == {"s-0", "s-1"}
    assert closure.group_ids == {"g-0"}


def test_selection_missing_group_fails_closed(tmp_path):
    bundle, index, _ = build_grpo_bundle(tmp_path / "bundle")
    index.samples[0].group_index = None
    with pytest.raises(ClosureError):
        expand_selection(index, sample_ids=["s-0"])
