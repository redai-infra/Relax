# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the Task40 behavior-only temperature wrapper."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

# ``examples.algorithms.p3o.rollout`` imports ``relax.engine.rollout.sglang_rollout``,
# which transitively reaches ``megatron.core`` via the checkpoint-service backend.
# CI installs no megatron, so the import is done under the shared stub to keep this
# module collectable; the tested wrapper itself is pure dict/await logic.
sys.path.insert(0, str(REPO_ROOT / "tests" / "backends" / "megatron"))

from _megatron_stub import stubbed_megatron_modules  # noqa: E402


with stubbed_megatron_modules():
    from examples.algorithms.p3o import rollout  # noqa: E402


def test_behavior_sampling_params_overrides_training_copy_only():
    original = {"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 64}

    updated = rollout.behavior_sampling_params(original, evaluation=False)

    assert updated == {"temperature": 1.2, "top_p": 1.0, "max_new_tokens": 64}
    assert original == {"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 64}


def test_behavior_sampling_params_accepts_runtime_temperature(monkeypatch):
    monkeypatch.setenv("TASK40_BEHAVIOR_TEMPERATURE", "2.0")

    updated = rollout.behavior_sampling_params({"temperature": 1.0}, evaluation=False)

    assert updated["temperature"] == 2.0


def test_behavior_sampling_params_rejects_invalid_runtime_temperature(monkeypatch):
    monkeypatch.setenv("TASK40_BEHAVIOR_TEMPERATURE", "nan")

    with pytest.raises(ValueError, match="finite and positive"):
        rollout.behavior_sampling_params({"temperature": 1.0}, evaluation=False)


def test_behavior_sampling_params_preserves_evaluation():
    original = {"temperature": 0.0, "top_p": 0.7, "max_new_tokens": 128}

    updated = rollout.behavior_sampling_params(original, evaluation=True)

    assert updated == original
    assert updated is not original


async def test_generate_delegates_with_isolated_behavior_params(monkeypatch):
    captured = {}
    expected = object()

    async def fake_generate(args, sample, sampling_params, evaluation=False):
        captured.update(
            args=args,
            sample=sample,
            sampling_params=sampling_params,
            evaluation=evaluation,
        )
        return expected

    monkeypatch.setattr(rollout, "_sglang_generate", fake_generate)
    args = SimpleNamespace()
    sample = object()
    original = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 32}

    result = await rollout.generate(args, sample, original, evaluation=False)

    assert result is expected
    assert captured == {
        "args": args,
        "sample": sample,
        "sampling_params": {"temperature": 1.2, "top_p": 1.0, "max_new_tokens": 32},
        "evaluation": False,
    }
    assert original == {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 32}
