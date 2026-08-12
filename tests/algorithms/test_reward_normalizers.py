# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bit-exact characterization tests for the reward normalizers.

``_legacy_post_process_rewards`` below is a frozen copy of the body of
``relax.utils.utils.post_process_rewards`` as of main@039ce87.  It exists so
the refactor can be proven to change nothing: any difference in a single float
bit fails these tests.  Do not "clean it up" — its value is that it is stale.
"""

import random
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms import get_algorithm  # noqa: E402
from relax.algorithms.rewards import REWARD_NORMALIZERS  # noqa: E402


_LEGACY_GROUP_NORM_ESTIMATORS = ["grpo", "gspo", "sapo", "cispo", "reinforce_plus_plus_baseline"]
_LEGACY_STD_NORM_ESTIMATORS = ["grpo", "gspo", "sapo", "cispo"]

ALL_ESTIMATORS = [
    "grpo",
    "gspo",
    "sapo",
    "cispo",
    "ppo",
    "reinforce_plus_plus",
    "reinforce_plus_plus_baseline",
]

CONTIGUOUS_GROUPS = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
INTERLEAVED_GROUPS = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]


def _legacy_post_process_rewards(args, samples, raw_rewards):
    """Frozen pre-refactor logic.

    Mirrors relax/utils/utils.py:181-207.
    """
    if args.advantage_estimator in _LEGACY_GROUP_NORM_ESTIMATORS and args.rewards_normalization:
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        positions_by_group: dict[int, list[int]] = {}
        for position, sample in enumerate(samples):
            if sample.group_index is None:
                raise ValueError("Sample.group_index is required for group reward normalization.")
            if sample.group_index not in positions_by_group:
                positions_by_group[sample.group_index] = []
            positions_by_group[sample.group_index].append(position)

        normalized_rewards = torch.empty_like(rewards)
        for group_index, positions in positions_by_group.items():
            if len(positions) != args.n_samples_per_prompt:
                raise ValueError(
                    f"Reward group {group_index} has {len(positions)} samples, expected {args.n_samples_per_prompt}."
                )
            group_rewards = rewards[positions]
            group_rewards = group_rewards - group_rewards.mean()
            if args.advantage_estimator in _LEGACY_STD_NORM_ESTIMATORS and args.grpo_std_normalization:
                group_rewards = group_rewards / (group_rewards.std() + 1e-6)
            normalized_rewards[positions] = group_rewards

        return normalized_rewards.tolist()

    return raw_rewards


def _new_normalize(args, samples, raw_rewards):
    """Mirrors the refactored dispatch in
    relax.utils.utils.post_process_rewards."""
    if not args.rewards_normalization:
        return raw_rewards
    spec = get_algorithm(args.advantage_estimator)
    return REWARD_NORMALIZERS[spec.reward_normalizer](args, samples, raw_rewards)


def _args(estimator, *, n=4, rewards_normalization=True, grpo_std_normalization=True):
    return SimpleNamespace(
        advantage_estimator=estimator,
        n_samples_per_prompt=n,
        rewards_normalization=rewards_normalization,
        grpo_std_normalization=grpo_std_normalization,
    )


def _samples(group_indices):
    return [SimpleNamespace(group_index=g) for g in group_indices]


def _assert_bitwise_equal(left, right):
    left_t = torch.tensor(left, dtype=torch.float32)
    right_t = torch.tensor(right, dtype=torch.float32)
    assert left_t.shape == right_t.shape
    assert torch.equal(left_t.view(torch.int32), right_t.view(torch.int32)), f"{left} != {right}"


def _reward_fixtures():
    rng = random.Random(20260725)
    return {
        "normal": [rng.uniform(-3, 3) for _ in range(12)],
        "binary": [float(rng.randint(0, 1)) for _ in range(12)],
        "all_equal": [0.7] * 12,
        "one_group_collapsed": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.25, -1.0, 2.0, -0.5, 3.0],
        "negatives": [-rng.uniform(0, 5) for _ in range(12)],
        "duplicates": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        "large": [1e6, 1e6 + 1, 1e6 - 1, 1e6, 2e6, 2e6, 2e6, 2e6, 0.0, 1.0, 2.0, 3.0],
    }


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
@pytest.mark.parametrize("rewards_normalization", [True, False])
@pytest.mark.parametrize("grpo_std_normalization", [True, False])
@pytest.mark.parametrize("fixture_name", sorted(_reward_fixtures()))
@pytest.mark.parametrize("groups", [CONTIGUOUS_GROUPS, INTERLEAVED_GROUPS])
def test_normalizer_is_bitwise_identical_to_legacy(
    estimator, rewards_normalization, grpo_std_normalization, fixture_name, groups
):
    raw = _reward_fixtures()[fixture_name]
    args = _args(
        estimator,
        rewards_normalization=rewards_normalization,
        grpo_std_normalization=grpo_std_normalization,
    )
    samples = _samples(groups)

    expected = _legacy_post_process_rewards(args, samples, raw)
    actual = _new_normalize(args, samples, raw)

    _assert_bitwise_equal(expected, actual)


def test_identity_normalizer_returns_the_same_list_object():
    """Non-normalising estimators must not copy — legacy returned raw_rewards
    itself."""
    raw = [1.0, 2.0, 3.0, 4.0]
    args = _args("reinforce_plus_plus")
    samples = _samples([0, 0, 0, 0])
    assert _new_normalize(args, samples, raw) is raw


def test_missing_group_index_raises():
    args = _args("grpo")
    samples = [SimpleNamespace(group_index=None) for _ in range(4)]
    with pytest.raises(ValueError, match="group_index is required"):
        _new_normalize(args, samples, [1.0, 2.0, 3.0, 4.0])


def test_wrong_group_size_raises():
    args = _args("grpo", n=4)
    samples = _samples([0, 0, 1, 1])
    with pytest.raises(ValueError, match="expected 4"):
        _new_normalize(args, samples, [1.0, 2.0, 3.0, 4.0])


def test_group_mean_normalizer_never_divides_by_std():
    """reinforce_plus_plus_baseline only centres, even with std normalisation
    on."""
    args = _args("reinforce_plus_plus_baseline", n=4, grpo_std_normalization=True)
    samples = _samples([0, 0, 0, 0])
    out = _new_normalize(args, samples, [0.0, 1.0, 2.0, 3.0])
    _assert_bitwise_equal(out, [-1.5, -0.5, 0.5, 1.5])


def test_group_mean_std_respects_the_dr_grpo_switch():
    args_on = _args("grpo", n=4, grpo_std_normalization=True)
    args_off = _args("grpo", n=4, grpo_std_normalization=False)
    samples = _samples([0, 0, 0, 0])
    raw = [0.0, 1.0, 2.0, 3.0]
    assert _new_normalize(args_on, samples, raw) != _new_normalize(args_off, samples, raw)
    _assert_bitwise_equal(_new_normalize(args_off, samples, raw), [-1.5, -0.5, 0.5, 1.5])


def test_grouping_is_driven_by_group_index_not_position():
    """Group membership follows ``group_index``, not batch position.

    Interleaving the same rewards into different groups therefore changes the
    group statistics and so the normalized values -- which is what the
    assertion below checks. (An earlier version of this docstring claimed the
    opposite.)
    """
    args = _args("grpo", n=4)
    raw = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]

    contiguous = _new_normalize(args, _samples([0, 0, 0, 0, 1, 1, 1, 1]), raw)
    interleaved = _new_normalize(args, _samples([0, 1, 0, 1, 0, 1, 0, 1]), raw)

    assert sorted(round(v, 5) for v in contiguous) != sorted(round(v, 5) for v in interleaved)
    # group 0 of the interleaved layout holds raw[0], raw[2], raw[4], raw[6]
    group0 = [interleaved[i] for i in (0, 2, 4, 6)]
    assert abs(sum(group0)) < 1e-5
