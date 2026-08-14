# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the PR B production-capture data plane.

These exercise ``CaptureRecord`` / ``build_bundle_from_record`` (the pure
capture→bundle bridge) and ``CaptureManager`` (the async, bounded, failure-
isolated sink) entirely on CPU — the round-trip parity test proves the captured
bundle is replayable by the PR A runner.
"""

from __future__ import annotations

import threading

import pytest
import torch

from relax.utils.replay import capture
from relax.utils.replay.bundle import BundleReader
from relax.utils.replay.capture import (
    CaptureConfig,
    CaptureManager,
    CaptureRecord,
    active,
    begin_rollout,
    begin_step,
    build_bundle_from_record,
    build_manifest_for_record,
    config_from_env,
    current_step,
    disable,
    enable,
    end_rollout,
    end_step,
    maybe_enable_from_env,
    should_capture,
    should_capture_rollout,
)
from relax.utils.replay.runner import replay
from relax.utils.replay.schema import (
    ActorStepId,
    Identity,
    RecomputeConfig,
    StageCapability,
    StageId,
)
from tests.utils.replay.helpers import make_capture_record, make_rollout_capture_record


@pytest.fixture(autouse=True)
def _reset_capture_global():
    yield
    disable()


def test_capture_roundtrip_parity(tmp_path):
    record = make_capture_record("b-capture")
    bundle_path = build_bundle_from_record(record, tmp_path)

    # The manifest must be derived from the frozen V1 capability matrix, not
    # hard-coded by the capture layer.
    manifest = BundleReader(bundle_path).load().manifest
    for stage in (
        StageId.SAMPLE,
        StageId.REWARD_RAW,
        StageId.REWARD_POST_PROCESS,
        StageId.ADVANTAGE_KL,
        StageId.ADVANTAGE_ESTIMATE,
        StageId.LOSS_POLICY,
    ):
        assert manifest.stage_contracts[stage].capability == StageCapability.RECOMPUTE
    assert manifest.stage_contracts[StageId.LOSS_VALUE].capability == StageCapability.UNSUPPORTED

    report = replay(bundle_path)
    assert report.passed
    assert report.first_divergent_stage is None


def test_capture_manifest_reimplemented_marker():
    manifest = build_manifest_for_record(make_capture_record("b-marker"))
    assert manifest.stage_contracts[StageId.REWARD_POST_PROCESS].implementation == "reimplemented"
    assert manifest.stage_contracts[StageId.ADVANTAGE_ESTIMATE].implementation == "reuse"


def test_capture_manager_disabled_noop(tmp_path):
    manager = CaptureManager(CaptureConfig(enabled=False, output_dir=str(tmp_path)))
    assert manager.submit(make_capture_record("b-disabled")) is False
    assert manager.dropped_count == 0
    assert not (tmp_path / "b-disabled").exists()


def test_capture_manager_enabled_writes_bundle(tmp_path):
    manager = CaptureManager(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    assert manager.submit(make_capture_record("b-async")) is True
    manager.flush(wait=True)
    manager.close()
    assert (tmp_path / "b-async" / "COMPLETE").exists()
    assert replay(tmp_path / "b-async").passed


def test_capture_manager_queue_overflow_drops(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def slow_build(record, output_dir):
        entered.set()
        release.wait(timeout=10)
        return build_bundle_from_record(record, output_dir)

    monkeypatch.setattr(capture, "build_bundle_from_record", slow_build)
    manager = CaptureManager(CaptureConfig(enabled=True, output_dir=str(tmp_path), queue_capacity=1))

    assert manager.submit(make_capture_record("b-1")) is True
    assert entered.wait(timeout=5)
    # Writer is blocked inside build, so the queue is empty. Fill it, then
    # overflow it: the third submit must be dropped, never back-pressure.
    assert manager.submit(make_capture_record("b-2")) is True
    assert manager.submit(make_capture_record("b-3")) is False
    assert manager.dropped_count == 1

    release.set()
    manager.close()


def test_capture_failure_isolation(tmp_path, monkeypatch):
    def failing_build(record, output_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(capture, "build_bundle_from_record", failing_build)
    manager = CaptureManager(CaptureConfig(enabled=True, output_dir=str(tmp_path)))

    assert manager.submit(make_capture_record("b-fail")) is True  # never raises
    manager.flush(wait=True)
    assert manager.error_count == 1
    manager.close()


def test_capture_selected_steps(tmp_path):
    manager = CaptureManager(CaptureConfig(enabled=True, output_dir=str(tmp_path), selected_steps={(120, 0)}))
    assert manager.submit(make_capture_record("b-selected")) is True

    other = make_capture_record("b-skipped")
    other.actor_step_id = (121, 0)
    assert manager.submit(other) is False

    manager.flush(wait=True)
    manager.close()
    assert (tmp_path / "b-selected" / "COMPLETE").exists()
    assert not (tmp_path / "b-skipped").exists()


def _step_identity() -> Identity:
    return Identity(actor_step_id=ActorStepId(rollout_id=120, step_id=0), rank={"dp": 0, "tp": 0, "pp": 0, "cp": 1})


def _step_config() -> RecomputeConfig:
    return RecomputeConfig(advantage_estimator="grpo", n_samples_per_prompt=2)


def test_capture_enable_disable_active(tmp_path):
    assert active() is None
    manager = enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    assert active() is manager
    assert manager.enabled
    disable()
    assert active() is None


def test_capture_step_lifecycle(tmp_path):
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    record = make_capture_record("b-step")

    begin_step((120, 0), identity=record.identity, config=record.config, bundle_id="b-step")
    step = current_step()
    assert step is not None and step.actor_step_id == (120, 0)
    step.samples = record.samples
    step.tensors = record.tensors
    step.expected = record.expected
    # end_step drops accumulators that never received a hook payload (stages
    # stays None). The production hooks always set this; the test must too.
    step.stages = {
        StageId.SAMPLE,
        StageId.REWARD_RAW,
        StageId.REWARD_POST_PROCESS,
        StageId.ADVANTAGE_KL,
        StageId.ADVANTAGE_ESTIMATE,
        StageId.LOSS_POLICY,
    }
    end_step()
    disable()

    assert (tmp_path / "b-step" / "COMPLETE").exists()
    assert replay(tmp_path / "b-step").passed


def test_capture_begin_step_disabled_noop():
    disable()
    begin_step((120, 0), identity=_step_identity(), config=_step_config(), bundle_id="b-x")
    assert current_step() is None


def test_capture_begin_step_unselected_noop(tmp_path):
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path), selected_steps={(0, 0)}))
    begin_step((120, 0), identity=_step_identity(), config=_step_config(), bundle_id="b-x")
    assert current_step() is None


def test_end_step_without_payload_does_not_write(tmp_path):
    # Non-last PP ranks open an accumulator then end it with no stages filled.
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    begin_step((120, 0), identity=_step_identity(), config=_step_config(), bundle_id="b-empty")
    end_step()
    disable()
    assert not (tmp_path / "b-empty").exists()


def test_end_rollout_without_payload_does_not_write(tmp_path):
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    begin_rollout(120, identity=Identity(rollout_id=120, rank={"cp": 1}), config=_step_config(), bundle_id="b-empty-r")
    end_rollout()
    disable()
    assert not (tmp_path / "b-empty-r").exists()


def test_capture_should_capture_guard():
    # Disabled -> False (the hot-path short-circuit contract).
    assert should_capture((0, 0)) is False


def test_capture_loss_only_roundtrip(tmp_path):
    # A partial (loss-only) capture, as produced by capture_hooks.capture_policy_loss:
    # positional sample ids built from raw metadata, advantages recorded as a
    # tensor (not recomputed upstream), and only loss.policy declared.
    record = make_capture_record("b-loss-only")
    loss_only = CaptureRecord(
        actor_step_id=record.actor_step_id,
        identity=record.identity,
        samples=[],
        config=record.config,
        tensors={name: record.tensors[name] for name in ("old_log_probs", "log_probs", "entropy", "advantages")},
        expected={StageId.LOSS_POLICY.value: record.expected[StageId.LOSS_POLICY.value]},
        bundle_id="b-loss-only",
        stages={StageId.LOSS_POLICY},
        response_lengths=[sample.response_length for sample in record.samples],
        total_lengths=[sample.total_length for sample in record.samples],
        loss_masks_tensor=torch.tensor(
            [value for sample in record.samples for value in sample.loss_mask], dtype=torch.float32
        ),
    )

    bundle_path = build_bundle_from_record(loss_only, tmp_path)
    loaded = BundleReader(bundle_path).load()
    assert set(loaded.manifest.stage_contracts) == {StageId.LOSS_POLICY}
    assert loaded.index.identity.micro_batch_ids == ["mb-0000"]
    assert all(sample.micro_batch_id == "mb-0000" for sample in loaded.index.samples)

    assert replay(bundle_path).passed
    assert replay(bundle_path, batch_ids=["mb-0000"]).passed


def test_capture_rollout_roundtrip_parity(tmp_path):
    record = make_rollout_capture_record("b-rollout")
    bundle_path = build_bundle_from_record(record, tmp_path)

    loaded = BundleReader(bundle_path).load()
    # A rollout-level bundle declares only the reward → advantage chain; the
    # per-step loss stages are absent because their payloads are not captured.
    assert set(loaded.manifest.stage_contracts) == {
        StageId.REWARD_RAW,
        StageId.REWARD_POST_PROCESS,
        StageId.ADVANTAGE_KL,
        StageId.ADVANTAGE_ESTIMATE,
    }
    for contract in loaded.manifest.stage_contracts.values():
        assert contract.capability == StageCapability.RECOMPUTE
    assert loaded.index.identity.rollout_id == 120
    assert loaded.index.identity.actor_step_id is None

    report = replay(bundle_path)
    assert report.passed
    assert report.first_divergent_stage is None


def test_capture_rollout_lifecycle(tmp_path):
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    record = make_rollout_capture_record("b-rollout-step")

    begin_rollout(120, identity=record.identity, config=record.config, bundle_id="b-rollout-step")
    step = current_step()
    assert step is not None and step.rollout_id == 120 and step.actor_step_id is None
    step.samples = record.samples
    step.tensors = record.tensors
    step.expected = record.expected
    step.stages = record.stages
    step.response_lengths = record.response_lengths
    step.total_lengths = record.total_lengths
    step.loss_masks_tensor = record.loss_masks_tensor
    step.group_indices_tensor = record.group_indices_tensor
    step.raw_rewards_tensor = record.raw_rewards_tensor
    step.rewards_tensor = record.rewards_tensor
    end_rollout()
    disable()

    assert (tmp_path / "b-rollout-step" / "COMPLETE").exists()
    assert replay(tmp_path / "b-rollout-step").passed


def test_capture_selected_rollouts(tmp_path):
    manager = CaptureManager(CaptureConfig(enabled=True, output_dir=str(tmp_path), selected_rollouts={120}))
    assert manager.submit(make_rollout_capture_record("b-rselected")) is True

    other = make_rollout_capture_record("b-rskipped")
    other.rollout_id = 121
    assert manager.submit(other) is False

    manager.flush(wait=True)
    manager.close()
    assert (tmp_path / "b-rselected" / "COMPLETE").exists()
    assert not (tmp_path / "b-rskipped").exists()


def test_capture_should_capture_rollout_guard():
    # Disabled -> False (the rollout hot-path short-circuit contract).
    assert should_capture_rollout(120) is False


def test_capture_begin_rollout_unselected_noop(tmp_path):
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path), selected_rollouts={0}))
    begin_rollout(120, identity=Identity(rollout_id=120, rank={"cp": 1}), config=_step_config(), bundle_id="b-x")
    assert current_step() is None


def test_config_from_env_disabled(monkeypatch):
    monkeypatch.delenv("RELAX_REPLAY_CAPTURE", raising=False)
    monkeypatch.delenv("RELAX_REPLAY_CAPTURE_DIR", raising=False)
    assert config_from_env() is None


def test_config_from_env_missing_dir_stays_disabled(monkeypatch):
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE", "1")
    monkeypatch.delenv("RELAX_REPLAY_CAPTURE_DIR", raising=False)
    assert config_from_env() is None


def test_config_from_env_parses_selectors(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE", "1")
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_STEPS", "0:0, 1:2")
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_ROLLOUTS", "0,3")
    config = config_from_env()
    assert config is not None
    assert config.enabled
    assert config.output_dir == str(tmp_path)
    assert config.selected_steps == {(0, 0), (1, 2)}
    assert config.selected_rollouts == {0, 3}


def test_config_from_env_rejects_bad_steps(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE", "1")
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_STEPS", "not-a-step")
    with pytest.raises(ValueError, match="RELAX_REPLAY_CAPTURE_STEPS"):
        config_from_env()


def test_maybe_enable_from_env_scopes_rank_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE", "1")
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_DIR", str(tmp_path))
    manager = maybe_enable_from_env(rank=3)
    assert manager is not None
    assert manager.enabled
    assert active() is manager
    assert str(tmp_path / "rank-3") == manager.output_dir
