# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the RLOO branch of group reward normalization.

Covers the path a real run takes: ``post_process_rewards`` groups samples by
``group_index``, centers each group, and then applies the estimator-specific
scaling. These tests stay on CPU and build ``Sample`` objects directly, so no
rollout engine, GPU, or Megatron process group is involved.
"""

from argparse import Namespace

import pytest
import torch

from relax.utils.types import Sample


def _args(estimator: str, n_samples: int, **overrides) -> Namespace:
    args = Namespace(
        advantage_estimator=estimator,
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=n_samples,
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        rm_type="math",
        reward_key=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _samples(rewards, group_index=0):
    out = []
    for i, reward in enumerate(rewards):
        sample = Sample(index=i, group_index=group_index)
        sample.reward = float(reward)
        out.append(sample)
    return out


def _post_process(args, samples):
    from relax.utils.utils import post_process_rewards

    return post_process_rewards(args, samples)


def test_rloo_group_norm_matches_leave_one_out():
    """The RLOO branch reproduces A_i = R_i - mean(R_j, j!=i) end to end."""
    from relax.utils.training.ppo_utils import get_rloo_advantages

    rewards = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    raw, normalized = _post_process(_args("rloo", len(rewards)), _samples(rewards))

    assert raw == rewards, "raw rewards must pass through unchanged"
    expected = get_rloo_advantages(torch.tensor(rewards, dtype=torch.float))
    assert torch.allclose(torch.tensor(normalized), expected, atol=1e-6), (
        f"expected {expected.tolist()}, got {normalized}"
    )


def test_rloo_skips_std_normalization_even_when_flag_is_on():
    """The ``grpo_std_normalization`` flag must not affect RLOO."""
    rewards = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    _, rloo = _post_process(_args("rloo", len(rewards), grpo_std_normalization=True), _samples(rewards))
    _, grpo = _post_process(_args("grpo", len(rewards), grpo_std_normalization=True), _samples(rewards))

    centered = torch.tensor(rewards) - torch.tensor(rewards).mean()
    ratio = torch.tensor(rloo) / centered
    assert torch.allclose(ratio, torch.full_like(ratio, 8 / 7), atol=1e-5), (
        "RLOO must scale centered rewards by exactly k/(k-1), independent of the std flag"
    )
    assert not torch.allclose(torch.tensor(rloo), torch.tensor(grpo), atol=1e-3), (
        "RLOO and GRPO must differ under the same input"
    )


def test_multiple_groups_are_normalized_independently():
    """Each prompt group gets its own baseline, with no cross-talk."""
    samples = _samples([1.0, 1.0, 0.0, 0.0], group_index=0) + _samples([1.0, 0.0, 0.0, 0.0], group_index=1)
    for i, sample in enumerate(samples):
        sample.index = i
    _, normalized = _post_process(_args("rloo", 4), samples)

    first, second = torch.tensor(normalized[:4]), torch.tensor(normalized[4:])
    assert abs(float(first.sum())) < 1e-5, "group 0 must sum to zero"
    assert abs(float(second.sum())) < 1e-5, "group 1 must sum to zero"
    assert not torch.allclose(first, second, atol=1e-3), "groups with different rewards must differ"


def test_rloo_equals_grpo_without_std_up_to_k_over_k_minus_1():
    """RLOO is `--disable-grpo-std-normalization` times the constant k/(k-1).

    Pinned deliberately: this is the closest existing behaviour to RLOO, and
    the factor is a global constant because every group is required to have
    exactly ``--n-samples-per-prompt`` samples. If a future change makes RLOO
    diverge from this identity, that is a behaviour change worth noticing here.
    """
    rewards = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    k = len(rewards)
    _, rloo = _post_process(_args("rloo", k), _samples(rewards))
    _, grpo_no_std = _post_process(_args("grpo", k, grpo_std_normalization=False), _samples(rewards))

    expected = torch.tensor(grpo_no_std) * (k / (k - 1))
    assert torch.allclose(torch.tensor(rloo), expected, atol=1e-6), f"expected {expected.tolist()}, got {rloo}"


def test_all_equal_rewards_give_zero_advantages():
    """An all-equal group contributes no gradient signal."""
    rewards = [1.0] * 8
    _, normalized = _post_process(_args("rloo", 8), _samples(rewards))
    assert max(abs(v) for v in normalized) < 1e-6, f"expected all-zero advantages, got {normalized}"


def test_group_size_mismatch_raises():
    """A size disagreeing with --n-samples-per-prompt must not be silent."""
    with pytest.raises(ValueError, match="expected"):
        _post_process(_args("rloo", 8), _samples([1.0, 0.0, 0.0]))


def test_missing_group_index_raises():
    """group_index is required to build the leave-one-out baseline."""
    samples = _samples([1.0, 0.0])
    samples[0].group_index = None
    with pytest.raises(ValueError, match="group_index"):
        _post_process(_args("rloo", 2), samples)
