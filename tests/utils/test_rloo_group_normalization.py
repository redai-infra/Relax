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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_reward_is_rejected_with_the_offending_position(bad):
    """RLOO rejects non-finite rewards instead of propagating them.

    The baseline is a whole-group reduction, so one bad reward poisons every
    advantage in the group. Failing with the sample position is actionable;
    letting NaN reach the optimizer is not.
    """
    rewards = [1.0, 0.0, bad, 0.0]
    with pytest.raises(ValueError, match=r"non-finite rewards at sample positions \[2\]"):
        _post_process(_args("rloo", len(rewards)), _samples(rewards))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_grpo_behaviour_on_non_finite_rewards_is_unchanged(bad):
    """The rejection is scoped to RLOO; GRPO must behave exactly as before.

    Guards the PR's central promise that no existing estimator changes.
    """
    rewards = [1.0, 0.0, bad, 0.0]
    _, normalized = _post_process(_args("grpo", len(rewards)), _samples(rewards))
    assert not all(v == v for v in normalized) or any(abs(v) == float("inf") for v in normalized), (
        "GRPO should still propagate the non-finite value rather than raise"
    )


def test_advantages_do_not_depend_on_how_samples_are_grouped_into_dp_ranks():
    """DP splitting cannot change RLOO's advantages, by construction.

    Acceptance criterion 3 covers DP as well as CP. For RLOO the DP half is
    structural rather than something the loss path has to get right:
    ``post_process_rewards`` runs inside ``convert_samples_to_train_data`` on the
    rollout side, *before* the batch enters the TransferQueue and is handed to
    data-parallel ranks. So the per-sample advantage is fixed before any DP split
    exists, and no partitioning can alter it.

    This pins that property directly: normalizing the whole batch at once must
    give exactly what normalizing each would-be DP shard separately gives, as long
    as groups stay intact -- which the group-size assertion already enforces.
    """
    k = 4
    # Two prompt groups that would land on two different DP ranks.
    group_a = _samples([1.0, 0.0, 0.0, 1.0], group_index=0)
    group_b = _samples([1.0, 1.0, 0.0, 0.0], group_index=1)
    for i, sample in enumerate(group_a + group_b):
        sample.index = i

    _, whole_batch = _post_process(_args("rloo", k), group_a + group_b)
    _, shard_0 = _post_process(_args("rloo", k), group_a)
    _, shard_1 = _post_process(_args("rloo", k), group_b)

    assert torch.allclose(torch.tensor(whole_batch), torch.tensor(shard_0 + shard_1), atol=1e-12), (
        f"whole batch gave {whole_batch}, per-shard gave {shard_0 + shard_1}"
    )


def test_group_order_within_the_batch_does_not_change_advantages():
    """Reordering groups -- as a different DP assignment would -- changes
    nothing.

    Guards against an implementation that accidentally normalized across group
    boundaries: that would make the result depend on batch composition, and hence
    on how many DP ranks the batch was spread over.
    """
    k = 4
    forward = _samples([1.0, 0.0, 0.0, 1.0], group_index=0) + _samples([1.0, 1.0, 0.0, 0.0], group_index=1)
    for i, sample in enumerate(forward):
        sample.index = i
    _, straight = _post_process(_args("rloo", k), forward)

    reversed_batch = _samples([1.0, 1.0, 0.0, 0.0], group_index=1) + _samples([1.0, 0.0, 0.0, 1.0], group_index=0)
    for i, sample in enumerate(reversed_batch):
        sample.index = i
    _, swapped = _post_process(_args("rloo", k), reversed_batch)

    assert torch.allclose(torch.tensor(straight[:k]), torch.tensor(swapped[k:]), atol=1e-12)
    assert torch.allclose(torch.tensor(straight[k:]), torch.tensor(swapped[:k]), atol=1e-12)
