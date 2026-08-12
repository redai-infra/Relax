# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Consumers upstream of the advantage stage must ask the algorithm's question.

The dynamic-sampling filter and the zero-std metrics both decide whether a
prompt group "carries signal". They answered that with the single
``--reward-key`` scalar, which is the right question only for an algorithm that
consumes that scalar.

For a multi-reward algorithm the right question is narrower than "did any
component vary" and wider than "did the scalar vary": it is whether the
*combined* advantage GDPO will actually produce is non-zero. The three differ.
A zero weight mutes a varying component; two components whose standardised
values are exact opposites cancel; and a group can be flat in the scalar while
alive in the components. Only the combined value predicts whether the group
contributes a gradient, so that is what these consumers compute.

The tests below deliberately include the case that motivated all of this and
turned out to be wrong -- equal sums under equal weights cancel to zero -- so
that the mistake cannot be reintroduced as a "fix".
"""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.rewards import group_carries_reward_signal, normalize_gdpo_decoupled  # noqa: E402


def _combined(args, group):
    """The scalar GDPO steps 1+2 actually hand to the advantage stage."""
    return normalize_gdpo_decoupled(args, group, [0.0] * len(group))


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


def test_equal_sums_cancel_exactly_under_equal_weights():
    """Equal sums do NOT survive GDPO under equal weights. Pinning the maths.

    This was written the other way round first, and the mistake is worth a
    test of its own: if the components sum to a constant then ``r2 = C - r1``,
    so ``std(r2) == std(r1)`` and ``z2 == -z1``. Equal weights cancel them to
    exactly zero -- the same answer GRPO gives. Nothing about GDPO rescues
    this group, and a filter that claims otherwise keeps a group that then
    contributes no gradient.
    """
    group = _group(correctness=[1.0, 0.0, 1.0, 0.0], fmt=[0.0, 1.0, 0.0, 1.0])
    scalars = {s.get_reward_value(_gdpo_args()) for s in group}
    assert scalars == {1.0}, "precondition: the summed scalar really is constant"

    combined = _combined(_gdpo_args(), group)
    assert combined == [0.0, 0.0, 0.0, 0.0], combined
    assert group_carries_reward_signal(_gdpo_args(), group) is False


def test_unequal_weights_break_the_cancellation():
    """The tie is broken by weights, not by the standardisation itself."""
    args = _gdpo_args()
    args.gdpo_reward_weights = [2.0, 1.0]
    group = _group(correctness=[1.0, 0.0, 1.0, 0.0], fmt=[0.0, 1.0, 0.0, 1.0])

    assert any(v != 0.0 for v in _combined(args, group))
    assert group_carries_reward_signal(args, group) is True


def test_scale_disparity_is_what_separates_gdpo_from_grpo():
    """The real motivation: components whose spreads differ by orders of
    magnitude.

    `correctness` in {0,1} against a `format` reward in the hundreds, ranked
    the opposite way. GRPO standardises the *sum*, so the large-spread
    component decides the direction and even reverses the sign for the correct
    samples. GDPO gives each component unit variance first, so both get a say.
    """
    args = _gdpo_args()
    group = _group(correctness=[1.0, 1.0, 0.0, 0.0], fmt=[0.0, 100.0, 200.0, 300.0])

    summed = torch.tensor([s.get_reward_value(args) for s in group])
    grpo = (summed - summed.mean()) / (summed.std() + 1e-6)
    gdpo = torch.tensor(_combined(args, group))

    # GRPO ranks the two correct samples *lowest*, dragged there by `format`.
    assert grpo[0] < grpo[2] and grpo[1] < grpo[3]
    # GDPO does not: correctness pulls them back up, and the signs disagree.
    assert not torch.equal(grpo.sign(), gdpo.sign())


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
    """`correctness` flat, `format` varying: one live component is enough."""
    from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std

    group = _group(correctness=[1.0, 1.0, 1.0, 1.0], fmt=[0.0, 0.5, 1.0, 0.5])
    assert check_reward_nonzero_std(_gdpo_args(), group).keep is True


def test_builtin_filter_agrees_with_the_advantage_it_will_produce():
    """The filter's verdict must match what the reward stage actually outputs.

    Cheaper proxies ("did any raw component vary?") disagree with the real
    answer whenever a weight mutes a component or two standardised components
    cancel -- and disagreeing means keeping groups with no gradient.
    """
    from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std

    muted = _gdpo_args()
    muted.gdpo_reward_weights = [0.0, 1.0]
    cases = [
        (_gdpo_args(), _group([1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0])),  # cancels
        (_gdpo_args(), _group([1.0, 1.0, 0.0, 0.0], [0.0, 100.0, 200.0, 300.0])),  # real signal
        (_gdpo_args(), _group([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5])),  # fully flat
        (muted, _group([1.0, 0.0, 1.0, 0.0], [2.0, 2.0, 2.0, 2.0])),  # only muted one varies
    ]
    for args, group in cases:
        verdict = check_reward_nonzero_std(args, group).keep
        actually_moves = any(v != 0.0 for v in _combined(args, group))
        assert verdict is actually_moves, (verdict, _combined(args, group))


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
