# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Behavior-logprob pairing-mask tests."""

from types import SimpleNamespace

import pytest

from relax.utils.training.data_fields import build_data_fields
from relax.utils.types import Sample
from relax.utils.utils import convert_samples_to_train_data


def _args(**overrides):
    values = {
        "advantage_estimator": "p3o",
        "agentic_custom_advantage_path": None,
        "custom_reward_post_process_path": None,
        "debug_train_only": True,
        "grpo_std_normalization": True,
        "loss_type": "policy_loss",
        "multimodal_keys": None,
        "n_samples_per_prompt": 1,
        "reward_key": None,
        "rewards_normalization": False,
        "use_opd": False,
        "use_rollout_logprobs": True,
        "use_rollout_routing_replay": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(*, pairing_mask=None, rollout_log_probs=None, loss_mask=None):
    return Sample(
        tokens=[100, 1, 2, 3],
        response_length=3,
        reward=1.0,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        rollout_log_probs_mask=pairing_mask,
    )


def test_pairing_mask_is_carried_and_intersected_with_loss_mask() -> None:
    sample = _sample(
        pairing_mask=[True, False, True],
        rollout_log_probs=[-0.1, -0.2, -0.3],
        loss_mask=[1, 1, 0],
    )

    train_data = convert_samples_to_train_data(_args(), [sample])

    assert train_data["rollout_log_probs_mask"] == [[True, False, True]]
    assert train_data["loss_masks"] == [[1, 0, 0]]
    assert "rollout_log_probs_mask" in build_data_fields(_args())


def test_missing_pairing_mask_defaults_to_all_true_without_changing_loss_mask() -> None:
    sample = _sample(rollout_log_probs=[-0.1, -0.2, -0.3], loss_mask=[1, 0, 1])

    train_data = convert_samples_to_train_data(_args(), [sample])

    assert train_data["rollout_log_probs_mask"] == [[True, True, True]]
    assert train_data["loss_masks"] == [[1, 0, 1]]


@pytest.mark.parametrize(
    ("rollout_log_probs", "pairing_mask", "match"),
    [
        ([-0.1, -0.2], None, "rollout log-prob length"),
        ([-0.1, -0.2, -0.3], [True, False], "rollout log-prob mask length"),
    ],
)
def test_pairing_alignment_mismatch_is_rejected(rollout_log_probs, pairing_mask, match) -> None:
    sample = _sample(rollout_log_probs=rollout_log_probs, pairing_mask=pairing_mask)

    with pytest.raises(ValueError, match=match):
        convert_samples_to_train_data(_args(), [sample])


def test_requested_behavior_logprobs_cannot_be_missing() -> None:
    with pytest.raises(ValueError, match="requires behavior log-probs"):
        convert_samples_to_train_data(_args(), [_sample()])
