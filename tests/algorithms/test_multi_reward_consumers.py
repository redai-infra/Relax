# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Consumers upstream of the advantage stage must ask the algorithm's question.

The dynamic-sampling filter and the zero-std metrics both decide whether a
prompt group "carries signal". They answered that with the single
``--reward-key`` scalar, which is only the right question for an algorithm that
consumes that scalar.

For a multi-reward algorithm it is the wrong one, and wrong in the direction
that undoes the algorithm: a group where ``correctness`` is flat but ``format``
still varies looks dead by the summed scalar, gets dropped by the filter, and
never reaches training -- while GDPO would have extracted signal from the
component that did vary. Keeping those groups is the entire point.
"""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.rewards import group_carries_reward_signal  # noqa: E402


class _S:
    """Minimal stand-in for Sample with the same reward contract."""

    def __init__(self, group_index, reward):
        self.group_index = group_index
        self.reward = reward

    def get_reward_value(self, args):
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    def get_reward_components(self, keys):
        return [self.reward[key] for key in keys]


def _gdpo_args(n=4):
    return SimpleNamespace(
        advantage_estimator="gdpo",
        n_samples_per_prompt=n,
        reward_key="score",
        gdpo_reward_keys=["correctness", "format"],
        gdpo_reward_weights=None,
    )


def _grpo_args():
    return SimpleNamespace(advantage_estimator="grpo", n_samples_per_prompt=4, reward_key="score")


def _group(correctness, fmt):
    """One prompt group; `score` is the summed scalar --reward-key selects."""
    return [_S(0, {"correctness": c, "format": f, "score": c + f}) for c, f in zip(correctness, fmt, strict=True)]


# ---------------- the case the whole feature exists for ----------------


def test_group_flat_in_one_component_still_carries_signal():
    """`correctness` constant, `format` varying: one live component is enough.

    The scalar happens to vary here too, so this case alone does not separate
    the two tests -- it pins the weaker property that a partially collapsed
    group is not treated as dead. The case that does separate them is below.
    """
    group = _group(correctness=[1.0, 1.0, 1.0, 1.0], fmt=[0.0, 0.5, 1.0, 0.5])
    assert group_carries_reward_signal(_gdpo_args(), group) is True


def test_equal_sums_from_different_components_carry_signal():
    """(1,0) and (0,1) both sum to 1 -- the paper's motivating case.

    Every sample's --reward-key scalar is identical, so this is precisely the
    group a scalar-based filter throws away.
    """
    group = _group(correctness=[1.0, 0.0, 1.0, 0.0], fmt=[0.0, 1.0, 0.0, 1.0])
    scalars = {s.get_reward_value(_gdpo_args()) for s in group}
    assert scalars == {1.0}, "precondition: the scalar really is constant"

    assert group_carries_reward_signal(_gdpo_args(), group) is True


def test_group_flat_in_every_component_is_dead():
    """GDPO does not invent signal: all components constant means no signal."""
    group = _group(correctness=[1.0, 1.0, 1.0, 1.0], fmt=[0.5, 0.5, 0.5, 0.5])
    assert group_carries_reward_signal(_gdpo_args(), group) is False


# ---------------- single-reward algorithms keep their old answer ----------------


def test_single_reward_algorithm_reads_only_the_reward_key():
    varying = _group(correctness=[1.0, 0.0, 1.0, 0.0], fmt=[0.0, 1.0, 0.0, 1.0])
    # Components vary, but the scalar GRPO consumes does not.
    assert group_carries_reward_signal(_grpo_args(), varying) is False

    real = _group(correctness=[1.0, 0.0, 1.0, 0.0], fmt=[1.0, 0.0, 1.0, 0.0])
    assert group_carries_reward_signal(_grpo_args(), real) is True


def test_empty_group_carries_nothing():
    assert group_carries_reward_signal(_gdpo_args(), []) is False
    assert group_carries_reward_signal(_grpo_args(), []) is False


# ---------------- the built-in filter is wired to the same question ----------------


def test_builtin_filter_keeps_a_group_only_one_component_varies_in():
    from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std

    group = _group(correctness=[1.0, 0.0, 1.0, 0.0], fmt=[0.0, 1.0, 0.0, 1.0])
    assert check_reward_nonzero_std(_gdpo_args(), group).keep is True


def test_builtin_filter_drops_a_fully_flat_group():
    from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std

    group = _group(correctness=[1.0, 1.0, 1.0, 1.0], fmt=[0.5, 0.5, 0.5, 0.5])
    out = check_reward_nonzero_std(_gdpo_args(), group)
    assert out.keep is False
    assert out.reason.startswith("zero_std_")


def test_builtin_filter_is_unchanged_for_single_reward_algorithms():
    from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std

    flat = _group(correctness=[1.0, 1.0, 1.0, 1.0], fmt=[0.0, 0.0, 0.0, 0.0])
    assert check_reward_nonzero_std(_grpo_args(), flat).keep is False

    varied = _group(correctness=[1.0, 0.0, 1.0, 0.0], fmt=[1.0, 0.0, 1.0, 0.0])
    assert check_reward_nonzero_std(_grpo_args(), varied).keep is True
