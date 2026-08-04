# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the P3O behavior-only temperature wrapper."""

from types import SimpleNamespace

import pytest

from examples.algorithms.p3o import rollout


def test_behavior_sampling_params_overrides_only_temperature(monkeypatch):
    monkeypatch.setenv("P3O_BEHAVIOR_TEMPERATURE", "0.6")
    original = {"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 64}

    updated = rollout.behavior_sampling_params(original, evaluation=False)

    assert updated == {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 64}
    assert original == {"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 64}


@pytest.mark.parametrize(("raw_value", "expected"), [("0.6", 0.6), ("1.2", 1.2), ("2.0", 2.0)])
def test_behavior_sampling_params_accepts_runtime_temperature(monkeypatch, raw_value, expected):
    monkeypatch.setenv("P3O_BEHAVIOR_TEMPERATURE", raw_value)

    updated = rollout.behavior_sampling_params({"temperature": 1.0}, evaluation=False)

    assert updated["temperature"] == expected


def test_behavior_sampling_params_requires_runtime_temperature(monkeypatch):
    monkeypatch.delenv("P3O_BEHAVIOR_TEMPERATURE", raising=False)

    with pytest.raises(ValueError, match="must be set"):
        rollout.behavior_sampling_params({"temperature": 1.0}, evaluation=False)


@pytest.mark.parametrize("raw_value", ["0", "0.0", "-1", "nan", "inf", "-inf"])
def test_behavior_sampling_params_rejects_non_positive_or_nonfinite_temperature(monkeypatch, raw_value):
    monkeypatch.setenv("P3O_BEHAVIOR_TEMPERATURE", raw_value)

    with pytest.raises(ValueError, match="finite and greater than zero"):
        rollout.behavior_sampling_params({"temperature": 1.0}, evaluation=False)


def test_behavior_sampling_params_rejects_nonnumeric_temperature(monkeypatch):
    monkeypatch.setenv("P3O_BEHAVIOR_TEMPERATURE", "warm")

    with pytest.raises(ValueError, match="must be numeric"):
        rollout.behavior_sampling_params({"temperature": 1.0}, evaluation=False)


def test_behavior_sampling_params_preserves_evaluation_without_temperature_env(monkeypatch):
    monkeypatch.delenv("P3O_BEHAVIOR_TEMPERATURE", raising=False)
    original = {"temperature": 0.0, "top_p": 0.7, "max_new_tokens": 128}

    updated = rollout.behavior_sampling_params(original, evaluation=True)

    assert updated == original
    assert updated is not original


async def test_generate_delegates_with_isolated_behavior_params(monkeypatch):
    monkeypatch.setenv("P3O_BEHAVIOR_TEMPERATURE", "1.2")
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
        "sampling_params": {"temperature": 1.2, "top_p": 0.95, "max_new_tokens": 32},
        "evaluation": False,
    }
    assert original == {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 32}
