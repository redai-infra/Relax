# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GDPO: per-reward group standardisation, weighted sum, batch whitening.

The reference in ``_manual_gdpo`` implements arXiv 2601.05242 Eq. 4 and Eq. 7
directly in plain Python, independently of the tensor implementation, so a
mistake in one is unlikely to be mirrored in the other.
"""

import math
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms import get_algorithm  # noqa: E402
from relax.algorithms.advantages import whiten_scalar  # noqa: E402
from relax.algorithms.rewards import (  # noqa: E402
    REWARD_NORMALIZERS,
    combine_group,
    component_noise_scale,
    extract_reward_components,
    group_carries_reward_signal,
    standardize_group_components,
)


_UNSET = object()


def _args(keys=("correctness", "format"), weights=None, n=4):
    return SimpleNamespace(
        advantage_estimator="gdpo",
        n_samples_per_prompt=n,
        rewards_normalization=True,
        grpo_std_normalization=True,
        gdpo_reward_keys=list(keys) if keys is not None else None,
        gdpo_reward_weights=weights,
    )


class _S:
    """Minimal stand-in for Sample with the same reward-component contract."""

    def __init__(self, group_index, reward):
        self.group_index = group_index
        self.reward = reward

    def get_reward_components(self, keys):
        if not isinstance(self.reward, dict):
            raise ValueError("Sample.reward must be a dict")
        values = []
        for key in keys:
            if key not in self.reward:
                raise ValueError(f"Reward key {key!r} missing from sample reward")
            values.append(self.reward[key])
        return values


def _normalize(args, samples):
    spec = get_algorithm(args.advantage_estimator)
    return REWARD_NORMALIZERS[spec.reward_normalizer](args, samples, [0.0] * len(samples))


def _mk(groups, correctness, fmt):
    return [_S(g, {"correctness": c, "format": f}) for g, c, f in zip(groups, correctness, fmt, strict=True)]


def _manual_gdpo(correctness, fmt, groups, weights=(1.0, 1.0)):
    """Plain-Python reference for Eq.

    4 + Eq. 7.
    """
    per_key = []
    for column in (correctness, fmt):
        out = [0.0] * len(column)
        for g in sorted(set(groups)):
            idx = [i for i, gg in enumerate(groups) if gg == g]
            vals = [column[i] for i in idx]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
            # Exact equality, matching the implementation. A relative tolerance
            # here would make the oracle disagree with the code on precisely the
            # inputs where the choice matters -- a narrow continuous column with
            # std near 1e-6 * scale, which the oracle would zero and the
            # implementation would keep. The existing cases are binary rewards,
            # whose std is either exactly 0 or >= 0.5, so the two criteria never
            # diverge there and the mismatch would go unnoticed until someone
            # added a continuous case and mistrusted the wrong side.
            collapsed = max(vals) == min(vals)
            # 1e-4, hardcoded on purpose: this oracle exists to pin parity with
            # the reference implementation (trl grpo_trainer.py's scale_rewards
            # GDPO path divides by std + 1e-4 at both steps), so importing our
            # own constant here would make the test agree with itself.
            for i in idx:
                out[i] = 0.0 if collapsed else (column[i] - mean) / (std + 1e-4)
        per_key.append(out)
    return [weights[0] * a + weights[1] * b for a, b in zip(per_key[0], per_key[1], strict=True)]


# ---------------- registration ----------------


def test_gdpo_is_registered():
    spec = get_algorithm("gdpo")
    assert spec.reward_normalizer == "gdpo_decoupled"
    assert spec.advantage_fn == "gdpo"
    assert spec.policy_loss_fn == "ppo_clip"
    assert spec.kl_level == "token"


def test_gdpo_spec_guards():
    spec = get_algorithm("gdpo")
    assert spec.min_group_size == 2
    assert spec.allows_reward_post_process_hooks is False
    assert spec.forbids_normalize_advantages is True
    assert spec.requires_rewards_normalization is True
    assert spec.uses_reward_components is True


# ---------------- step 1 + 2 ----------------


def test_matches_hand_computed_reference():
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    correctness = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    fmt = [1.0, 1.0, 0.0, 0.0, 0.5, 0.25, 0.75, 1.0]
    samples = _mk(groups, correctness, fmt)

    got = _normalize(_args(), samples)
    want = _manual_gdpo(correctness, fmt, groups)

    assert torch.allclose(torch.tensor(got), torch.tensor(want), atol=1e-6)


def test_weights_apply_to_normalized_advantages_not_raw_rewards():
    groups = [0, 0, 0, 0]
    correctness = [1.0, 0.0, 1.0, 0.0]
    fmt = [0.0, 0.0, 10.0, 10.0]
    samples = _mk(groups, correctness, fmt)

    got = _normalize(_args(weights=[2.0, 0.5]), samples)
    want = _manual_gdpo(correctness, fmt, groups, weights=(2.0, 0.5))

    assert torch.allclose(torch.tensor(got), torch.tensor(want), atol=1e-6)


def test_component_scale_does_not_leak_into_the_combination():
    """After step 1 every component is unit-variance, so rescaling one is a no-
    op."""
    groups = [0, 0, 0, 0]
    correctness = [1.0, 0.0, 1.0, 0.0]
    unit = [1.0, 2.0, 3.0, 4.0]
    thousandfold = [1000.0, 2000.0, 3000.0, 4000.0]

    with_unit = _normalize(_args(), _mk(groups, correctness, unit))
    with_large = _normalize(_args(), _mk(groups, correctness, thousandfold))

    # Not exactly a no-op: the additive epsilon does not scale with the data, so
    # a 1000x rescale leaks about eps/std = 1e-4/1.291 = 7.7e-5. Measured 9.0e-5.
    assert torch.allclose(torch.tensor(with_unit), torch.tensor(with_large), atol=1e-4)


def test_eps_makes_scale_invariance_approximate_for_tiny_rewards():
    """Documented limitation of the additive epsilon, shared with the GRPO
    path.

    Dividing by ``std + eps`` is only scale-free while ``std >> eps``, and GDPO
    uses ``eps = 1e-4`` to match the reference implementation rather than the
    ``1e-6`` the GRPO path uses. That choice is not free: a component whose
    spread is ~1e-3 is shrunk by roughly eps/std, which at 1e-4 is **7.2%**
    against 0.08% at 1e-6.

    So a continuous reward with a very narrow spread -- the paper's maths setup
    scores response length -- is damped noticeably more here than a reader
    coming from the GRPO path would expect. Pinning the number is the point;
    if it ever needs to be configurable, this is the test that says why.
    """
    groups = [0, 0, 0, 0]
    correctness = [1.0, 0.0, 1.0, 0.0]

    normal = _normalize(_args(), _mk(groups, correctness, [1.0, 2.0, 3.0, 4.0]))
    tiny = _normalize(_args(), _mk(groups, correctness, [0.001, 0.002, 0.003, 0.004]))

    deviation = (torch.tensor(normal) - torch.tensor(tiny)).abs().max()
    assert 0.07 < float(deviation) < 0.09, f"expected ~7% damping from eps=1e-4, got {float(deviation)}"


def test_default_weights_are_all_ones():
    groups = [0, 0, 0, 0]
    samples = _mk(groups, [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    assert _normalize(_args(weights=None), samples) == _normalize(_args(weights=[1.0, 1.0]), samples)


def test_grouping_follows_group_index_not_position():
    """Interleave two groups so position and group_index disagree.

    The previous version of this test only ever built the contiguous layout
    [0,0,0,0,1,1,1,1], which an implementation that chopped the batch into
    fixed-size runs would satisfy just as well -- it asserted nothing its own
    name claimed. Here sample i belongs to group i % 2, so any position-based
    grouping produces different numbers.
    """
    groups = [0, 1, 0, 1, 0, 1, 0, 1]
    correctness = [1.0, 5.0, 0.0, 5.0, 1.0, 5.0, 0.0, 5.0]
    fmt = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    interleaved = _normalize(_args(n=4), _mk(groups, correctness, fmt))

    # Group 1 (odd positions) has collapsed correctness, so only format acts there.
    odd = [interleaved[i] for i in (1, 3, 5, 7)]
    assert abs(sum(odd)) < 1e-5
    # Reordering the samples so each group is contiguous must not change any
    # sample's own advantage -- which is only true if group_index decides.
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    contiguous = _normalize(
        _args(n=4),
        _mk([groups[i] for i in order], [correctness[i] for i in order], [fmt[i] for i in order]),
    )
    for new_position, original in enumerate(order):
        assert abs(contiguous[new_position] - interleaved[original]) < 1e-5


def test_collapsed_component_in_one_group_only():
    correctness = [1.0, 0.0, 1.0, 0.0, 5.0, 5.0, 5.0, 5.0]
    fmt = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    contiguous = _normalize(_args(), _mk([0, 0, 0, 0, 1, 1, 1, 1], correctness, fmt))
    # group 1 has collapsed correctness, so only format contributes there
    assert abs(sum(contiguous[4:])) < 1e-5


# ---------------- reward collapse ----------------


def test_collapsed_component_contributes_zero_but_others_keep_signal():
    """This is GDPO's core benefit over GRPO."""
    groups = [0, 0, 0, 0]
    correctness = [1.0, 1.0, 1.0, 1.0]  # collapsed
    fmt = [1.0, 0.0, 1.0, 0.0]
    samples = _mk(groups, correctness, fmt)

    got = _normalize(_args(), samples)
    fmt_only = _manual_gdpo([0.0] * 4, fmt, groups)

    assert torch.allclose(torch.tensor(got), torch.tensor(fmt_only), atol=1e-6)
    assert max(abs(v) for v in got) > 0.5


def test_fully_collapsed_group_yields_exact_zeros_without_nan():
    samples = _mk([0, 0, 0, 0], [0.7] * 4, [0.7] * 4)
    got = _normalize(_args(), samples)
    assert got == [0.0, 0.0, 0.0, 0.0]


def test_collapsed_group_does_not_leak_fp32_residual():
    """0.7 repeated 7x produces a nonzero residual if you rely on `/(std + eps)` alone."""
    samples = _mk([0] * 7, [0.7] * 7, [0.7] * 7)
    got = _normalize(_args(n=7), samples)
    assert got == [0.0] * 7


@pytest.mark.parametrize("value", [0.0, 0.1, 0.7, 1.0, 1000.0, -3.5])
def test_collapse_is_exact_at_any_magnitude(value):
    samples = _mk([0] * 5, [value] * 5, [value] * 5)
    assert _normalize(_args(n=5), samples) == [0.0] * 5


def _grpo_reference(summed, groups):
    """What GRPO does: sum the components first, then standardise once."""
    out = [0.0] * len(summed)
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        vals = [summed[i] for i in idx]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(var)
        for i in idx:
            out[i] = 0.0 if std == 0 else (summed[i] - mean) / (std + 1e-4)
    return out


def test_gdpo_distinguishes_reward_patterns_that_grpo_flattens():
    """The paper's core argument (arXiv 2601.05242, Sec. 3.1).

    With G=2, standardising collapses any two distinct values to +-1/sqrt(2),
    so GRPO maps "one component fires" and "both components fire" to the exact
    same advantage. Normalising per component first keeps the two apart.
    """
    groups = [0, 0]
    one_fires = _mk(groups, [0.0, 1.0], [0.0, 0.0])
    both_fire = _mk(groups, [0.0, 1.0], [0.0, 1.0])

    grpo_one = _grpo_reference([0.0, 1.0], groups)
    grpo_both = _grpo_reference([0.0, 2.0], groups)
    assert torch.allclose(torch.tensor(grpo_one), torch.tensor(grpo_both), atol=1e-4), (
        "GRPO should be unable to tell these apart"
    )

    gdpo_one = _normalize(_args(n=2), one_fires)
    gdpo_both = _normalize(_args(n=2), both_fire)
    assert not torch.allclose(torch.tensor(gdpo_one), torch.tensor(gdpo_both), atol=1e-2)
    # both_fire carries twice the signal: two components at +-1/sqrt(2) each
    assert math.isclose(gdpo_both[1], 2 * gdpo_one[1], rel_tol=1e-4)


# ---------------- error contracts ----------------


def test_missing_reward_key_raises():
    samples = [_S(0, {"correctness": 1.0}) for _ in range(4)]
    with pytest.raises(ValueError, match="format"):
        _normalize(_args(), samples)


def test_non_dict_reward_raises():
    samples = [_S(0, 1.0) for _ in range(4)]
    with pytest.raises(ValueError, match="must be a dict"):
        _normalize(_args(), samples)


def test_non_numeric_reward_raises():
    samples = [_S(0, {"correctness": 1.0, "format": "good"}) for _ in range(4)]
    with pytest.raises(TypeError, match="must be a real number"):
        _normalize(_args(), samples)


def test_bool_reward_raises():
    """bool subclasses int, so it would slip through a naive isinstance
    check."""
    samples = [_S(0, {"correctness": True, "format": 1.0}) for _ in range(4)]
    with pytest.raises(TypeError, match="must be a real number"):
        _normalize(_args(), samples)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_or_inf_reward_raises(bad):
    samples = [_S(0, {"correctness": bad, "format": 1.0}) for _ in range(4)]
    with pytest.raises(ValueError, match="not finite"):
        _normalize(_args(), samples)


@pytest.mark.parametrize("huge", [1e300, -1e300, 3.5e38])
def test_reward_that_overflows_float32_raises_rather_than_becoming_inf(huge):
    """Finite in float64, `inf` after the cast — the gap `isfinite` misses.

    Left unchecked this is silent, not loud: the cast produces `inf`, the group
    std reads as non-finite one stage later, `whiten_scalar` returns zeros, and
    the run trains on no signal while exiting cleanly.
    """
    samples = [_S(0, {"correctness": huge, "format": 1.0}) for _ in range(4)]
    with pytest.raises(ValueError, match="overflows float32"):
        _normalize(_args(), samples)


def test_a_large_but_representable_reward_is_accepted():
    """The bound is float32's range, not an opinion about reward magnitude."""
    samples = _mk([0, 0, 0, 0], [1e30, 2e30, 3e30, 4e30], [1.0, 0.0, 1.0, 0.0])
    out = _normalize(_args(), samples)
    assert all(math.isfinite(v) for v in out), out


def test_fewer_than_two_keys_raises():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="at least two"):
        _normalize(_args(keys=("correctness",)), samples)


def test_no_keys_raises():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="at least two"):
        _normalize(_args(keys=None), samples)


def test_duplicate_keys_raise():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="duplicates"):
        _normalize(_args(keys=("correctness", "correctness")), samples)


def test_weight_count_mismatch_raises():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="gdpo-reward-weights"):
        _normalize(_args(weights=[1.0]), samples)


def test_wrong_group_size_raises():
    samples = _mk([0, 0, 1, 1], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="expected 4"):
        _normalize(_args(n=4), samples)


# ---------------- numerical properties ----------------


def test_group_of_two_always_normalizes_to_plus_minus_one_over_sqrt_two():
    """With G=2 an unbiased std absorbs the magnitude entirely."""
    samples = _mk([0, 0], [0.0, 100.0], [0.0, 1.0])
    got = _normalize(_args(n=2), samples)
    expected = 2 * (1.0 / math.sqrt(2.0))
    assert math.isclose(got[1], expected, rel_tol=1e-4)
    assert math.isclose(got[0], -expected, rel_tol=1e-4)


def test_extract_reward_components_shape_and_dtype():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    out = extract_reward_components(samples, ["correctness", "format"])
    assert out.shape == (4, 2)
    assert out.dtype == torch.float64


def test_extract_reward_components_preserves_key_order():
    samples = _mk([0, 0, 0, 0], [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    forward = extract_reward_components(samples, ["correctness", "format"])
    reversed_ = extract_reward_components(samples, ["format", "correctness"])
    assert torch.equal(forward, reversed_.flip(dims=[1]))


def test_extract_accepts_python_ints():
    samples = [_S(0, {"correctness": 1, "format": 0}) for _ in range(2)]
    out = extract_reward_components(samples, ["correctness", "format"])
    assert out.dtype == torch.float64


def test_single_reward_gdpo_is_a_constant_positive_multiple_of_grpo():
    """Documented deviation: GDPO does NOT reduce to GRPO for one reward."""
    torch.manual_seed(0)
    b, g = 16, 4
    rewards = (torch.rand(b * g) < 0.5).float().view(b, g)
    grpo = ((rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)).flatten()
    gdpo = whiten_scalar(grpo)

    mask = grpo.abs() > 1e-6
    ratio = gdpo[mask] / grpo[mask]
    assert ratio.min() > 0
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-4)
    assert not torch.allclose(gdpo, grpo, atol=1e-3)


def test_sample_get_reward_components_matches_the_test_double():
    """The real Sample must honour the same contract as _S above."""
    from relax.utils.types import Sample

    sample = Sample(group_index=0, reward={"correctness": 1.0, "format": 0.5})
    assert sample.get_reward_components(["format", "correctness"]) == [0.5, 1.0]

    with pytest.raises(ValueError, match="missing from sample reward"):
        sample.get_reward_components(["nope"])

    scalar = Sample(group_index=0, reward=1.0)
    with pytest.raises(ValueError, match="must be a dict"):
        scalar.get_reward_components(["correctness"])


def test_warns_once_when_every_component_collapses_in_every_group(caplog):
    """A batch that produces no gradient at all must not be silent."""
    import logging

    samples = _mk([0, 0, 0, 0], [1.0] * 4, [0.5] * 4)
    with caplog.at_level(logging.WARNING):
        out = _normalize(_args(), samples)

    assert out == [0.0] * 4
    assert sum("combined to exactly zero" in r.message for r in caplog.records) == 1


def test_does_not_warn_when_some_signal_survives(caplog):
    import logging

    samples = _mk([0, 0, 0, 0], [1.0] * 4, [1.0, 0.0, 1.0, 0.0])
    with caplog.at_level(logging.WARNING):
        _normalize(_args(), samples)

    assert not any("combined to exactly zero" in r.message for r in caplog.records)


def test_does_not_warn_when_only_some_groups_collapse(caplog):
    import logging

    samples = _mk(
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0, 0.0],
    )
    with caplog.at_level(logging.WARNING):
        _normalize(_args(), samples)

    assert not any("combined to exactly zero" in r.message for r in caplog.records)


# ---------------- reward value contract ----------------


def test_numpy_scalars_are_accepted():
    """Reward functions routinely return numpy scalars; only float64 subclasses
    float."""
    np = pytest.importorskip("numpy")

    for dtype in (np.float64, np.float32, np.int64, np.int32, np.float16):
        samples = [
            _S(0, {"correctness": dtype(1), "format": dtype(0)}),
            _S(0, {"correctness": dtype(0), "format": dtype(1)}),
        ]
        out = extract_reward_components(samples, ["correctness", "format"])
        assert out.dtype == torch.float64, dtype
        assert out.shape == (2, 2), dtype


def test_numpy_bool_is_still_rejected():
    np = pytest.importorskip("numpy")

    samples = [_S(0, {"correctness": np.bool_(True), "format": 1.0}) for _ in range(4)]
    with pytest.raises(TypeError, match="must be a real number"):
        _normalize(_args(), samples)


def test_numpy_nan_is_still_rejected():
    np = pytest.importorskip("numpy")

    samples = [_S(0, {"correctness": np.float32("nan"), "format": 1.0}) for _ in range(4)]
    with pytest.raises(ValueError, match="not finite"):
        _normalize(_args(), samples)


# ---------------- weight contract ----------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weights_raise_instead_of_zeroing_the_batch(bad):
    """Left unchecked these produce an all-zero batch with no error at all."""
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="not finite"):
        _normalize(_args(weights=[bad, 1.0]), samples)


# ---------------- collapse detection is exact ----------------


def test_component_with_small_relative_spread_is_kept():
    """A relative tolerance would have erased this component entirely."""
    groups = [0, 0, 0, 0]
    correctness = [10000.0, 10000.005, 10000.010, 10000.015]
    fmt = [1.0, 0.0, 1.0, 0.0]
    got = _normalize(_args(), _mk(groups, correctness, fmt))

    fmt_only = _manual_gdpo([0.0] * 4, fmt, groups)
    assert not torch.allclose(torch.tensor(got), torch.tensor(fmt_only), atol=1e-3)


# ---------------- batch statistics must survive a large offset ----------------
#
# distributed_mean_std used the one-pass E[x^2] - E[x]^2 form in the caller's
# float32. That subtracts two nearly equal large numbers whenever the values sit
# far from zero, and it failed silently in two different directions. Neither is
# hypothetical: the GDPO paper's maths setup uses a length reward, and token
# counts live exactly in this range.


def test_batch_std_survives_a_large_offset():
    """One-pass returned exactly 0 here, which reads as a collapsed batch."""
    from relax.algorithms.numerics import distributed_mean_std

    values = torch.tensor([1000.0, 1000.01, 1000.02, 1000.03])
    _mean, std = distributed_mean_std(values)

    assert std > 0, "a batch with real spread was reported as collapsed"
    # rtol at float32 resolution, not float64: the statistic is computed in
    # float64 but handed back in the caller's dtype.
    torch.testing.assert_close(std.double(), values.double().std(), rtol=1e-6, atol=0)


def test_batch_std_is_not_inflated_by_a_large_offset():
    """One-pass returned 4.6 here against a true std of 1.3e-3."""
    from relax.algorithms.numerics import distributed_mean_std

    values = torch.tensor([10000.0, 10000.001, 10000.002, 10000.003])
    _mean, std = distributed_mean_std(values)

    torch.testing.assert_close(std.double(), values.double().std(), rtol=1e-6, atol=0)
    assert std < 1e-2, f"std inflated to {float(std)}; advantages would be rescaled by ~1/{float(std):.3g}"


@pytest.mark.parametrize("offset", [0.0, 1e2, 1e3, 1e4])
def test_whitening_is_shift_invariant(offset):
    """Whitening centres before scaling, so adding a constant to every value
    must not change the result.

    One-pass broke this well before float32 ran out of significand.
    """
    from relax.algorithms.advantages import whiten_scalar

    base = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    torch.testing.assert_close(whiten_scalar(base + offset), whiten_scalar(base), rtol=1e-5, atol=1e-6)


def test_batch_std_matches_torch_std_on_ordinary_input():
    """The fix must not move the numbers the existing algorithms already
    produce."""
    from relax.algorithms.numerics import distributed_mean_std

    torch.manual_seed(20260726)
    for n in (2, 4, 8, 64):
        values = torch.randn(n)
        mean, std = distributed_mean_std(values)
        torch.testing.assert_close(mean, values.mean(), rtol=1e-6, atol=0)
        torch.testing.assert_close(std, values.std(), rtol=1e-6, atol=0)


# ---------------- what step 3's boundary actually is ----------------


def test_whitening_scope_is_whatever_the_caller_passes_not_a_training_batch():
    """Step 3 has no notion of a batch boundary; it whitens its argument.

    This is worth pinning because the surrounding docs used to claim the
    statistics describe one training batch "matching the paper", and they do not.
    The Megatron caller merges `num_rollout_minis` windows before calling
    (actor.py: concat_rollout_batches, under the comment "we may need normalize
    the whole rollout"), so whether one whitening spans one optimizer step or
    several is the caller's business, not this function's. The shipped example
    runs 4 * 8 against a --global-batch-size of 32, i.e. one window; halving
    that to 16 gives two, which is what `mini_batch_sizes` then separates.

    Concretely: whitening two batches together is not the same as whitening each.
    """
    from relax.algorithms.advantages import whiten_scalar

    first = torch.tensor([1.0, 2.0, 3.0, 4.0])
    second = torch.tensor([101.0, 102.0, 103.0, 104.0])

    merged = whiten_scalar(torch.cat([first, second]))
    separate = torch.cat([whiten_scalar(first), whiten_scalar(second)])

    assert not torch.allclose(merged, separate, atol=1e-3), (
        "if these agreed, the scope of step 3 would not matter and this test would be pointless"
    )
    # The merged form is dominated by the between-batch offset, which is exactly
    # the deviation from Eq. 6 the docs now describe.
    assert merged[:4].max() < 0, "merged whitening puts the whole first batch below the mean"
    assert torch.allclose(separate[:4], separate[4:], atol=1e-5), "per-batch whitening treats them alike"


# ---------------- step 3 now normalises per training batch ----------------


def _gdpo_adv(rewards, mini_batch_sizes=_UNSET):
    """`mini_batch_sizes` defaults to one segment covering everything.

    Not `None`: that is now rejected outright, because a caller that fails to
    say how the rollout was split is asking for a different objective without
    knowing it. Tests that want the merged behaviour ask for it explicitly.
    """
    from relax.algorithms.advantages import compute_advantages_and_returns

    if mini_batch_sizes is _UNSET:
        mini_batch_sizes = [len(rewards)]
    kl = [torch.zeros(1) for _ in rewards]
    adv, _ = compute_advantages_and_returns(
        SimpleNamespace(advantage_estimator="gdpo", kl_coef=0.0),
        rewards=list(rewards),
        kl=kl,
        mini_batch_sizes=mini_batch_sizes,
    )
    return torch.cat(adv)


def test_step_three_normalises_each_training_batch_separately():
    """Eq.

    6's boundary. The caller merges num_rollout_minis batches before the
    advantage stage, so without the counts one whitening covered all of them.
    """
    from relax.algorithms.advantages import whiten_scalar

    first, second = [0.9, 1.1, 0.8, 1.2], [-1.2, -0.8, -1.1, -0.9]
    got = _gdpo_adv(first + second, mini_batch_sizes=[4, 4])
    want = torch.cat([whiten_scalar(torch.tensor(first)), whiten_scalar(torch.tensor(second))])
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_merging_the_batches_would_flip_signs_not_just_rescale():
    """Why the boundary matters: it decides which samples are reinforced.

    Whitening the two batches together centres both on the pooled mean, so
    samples that were below their own batch's mean come out positive. Four of
    these eight change sign -- this is a different objective, not a precision
    difference.
    """
    first, second = [0.9, 1.1, 0.8, 1.2], [-1.2, -0.8, -1.1, -0.9]
    per_batch = _gdpo_adv(first + second, mini_batch_sizes=[4, 4])
    # `[8]`, not `None`: one segment spanning the merged rollout is exactly the
    # behaviour a missing count would have produced, and it is now the only way
    # to ask for it -- which is the point of this test.
    merged = _gdpo_adv(first + second, mini_batch_sizes=[8])

    assert int((per_batch.sign() != merged.sign()).sum()) == 4


def test_a_single_batch_is_unchanged_by_the_counts():
    """num_rollout_minis == 1 is the common case and must not move."""
    rewards = [1.0, 2.0, 3.0, 4.0]
    torch.testing.assert_close(_gdpo_adv(rewards, [4]), _gdpo_adv(rewards, [len(rewards)]))


def test_absent_counts_are_rejected_rather_than_merged():
    """The caller must say how the rollout was split; there is no default.

    Falling back to one window looks like a conservative default and is not:
    `test_merging_the_batches_would_flip_signs_not_just_rescale` shows it
    changing the sign of half the advantages. A caller that forgot the metadata
    would have trained on a different objective with every metric finite.
    """
    with pytest.raises(ValueError, match="mini_batch_sizes is None"):
        _gdpo_adv([1.0, 2.0, 3.0, 4.0], mini_batch_sizes=None)


def test_counts_that_do_not_cover_the_shard_are_rejected():
    """Silently whitening the wrong window is the failure this replaces."""
    with pytest.raises(ValueError, match="sum to"):
        _gdpo_adv([1.0, 2.0, 3.0, 4.0], mini_batch_sizes=[3, 3])


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "sapo", "cispo"])
def test_other_estimators_ignore_the_batch_counts(estimator):
    """They absorb it in **_unused, so passing it must be bit-identical."""
    from relax.algorithms.advantages import compute_advantages_and_returns

    args = SimpleNamespace(advantage_estimator=estimator, kl_coef=0.0)
    inputs = dict(rewards=[1.0, -1.0, 0.5, -0.5], kl=[torch.zeros(2) for _ in range(4)])

    without, _ = compute_advantages_and_returns(args, **inputs)
    with_counts, _ = compute_advantages_and_returns(args, mini_batch_sizes=[2, 2], **inputs)

    for left, right in zip(without, with_counts, strict=True):
        assert torch.equal(left, right)


def test_all_zero_weights_are_rejected():
    """Every component times zero is a batch of zero advantages and a clean
    exit."""
    from relax.algorithms.rewards import resolve_gdpo_weights

    with pytest.raises(ValueError, match="all zero"):
        resolve_gdpo_weights(_args(weights=[0.0, 0.0]), ["correctness", "format"])


def test_one_zero_weight_is_allowed():
    """Muting a component is a legitimate configuration; only muting all is
    not."""
    from relax.algorithms.rewards import resolve_gdpo_weights

    assert resolve_gdpo_weights(_args(weights=[1.0, 0.0]), ["correctness", "format"]) == [1.0, 0.0]


@pytest.mark.parametrize("bad", [[], [0, 4], [2.5, 1.5], [-1, 5]])
def test_malformed_batch_sizes_raise_rather_than_falling_back(bad):
    """An empty or malformed list must not quietly restore merged whitening."""
    with pytest.raises(ValueError, match="positive ints"):
        _gdpo_adv([1.0, 2.0, 3.0, 4.0], mini_batch_sizes=bad)


def test_a_single_segment_is_still_size_checked():
    """[4] on a 5-sample shard is a real mismatch, not a request to use one
    window."""
    with pytest.raises(ValueError, match="sum to"):
        _gdpo_adv([1.0, 2.0, 3.0, 4.0, 5.0], mini_batch_sizes=[4])


# ---------------- the noise floor (see combine_group) ----------------


_CONSTANT_SUM_C = 308.95172119140625
_CONSTANT_SUM_R = [-75.74329876632146, -75.74330217449115, -75.7432989215475]
"""Three samples whose two components sum to exactly the same float32 value.

Found by search, not by hand: the pair has to round to an identical sum while
the individual values still differ, which needs the components to sit far
enough from zero that their ulp exceeds the spread being represented.
"""


def _constant_sum_group():
    return torch.tensor([[r, _CONSTANT_SUM_C - r] for r in _CONSTANT_SUM_R], dtype=torch.float64)


def test_components_that_cancel_produce_exactly_zero_not_amplified_rounding():
    """Equal weights on two components summing to a constant must cancel.

    In float32 this group came out of step 2 at [-0.086, 0.173, -0.086] and out
    of step 3 -- which divides by the std of that very residue -- at [-0.577,
    1.154, -0.577]. Finite, plausible, and pointing wherever the last bits of
    the reward happened to fall.
    """
    combined = combine_group(_constant_sum_group(), torch.tensor([0.5, 0.5]))
    assert combined.tolist() == [0.0, 0.0, 0.0]
    assert whiten_scalar(combined).tolist() == [0.0, 0.0, 0.0]


def test_float32_columns_are_what_made_the_residue_dangerous():
    """The dtype is the half that stops a full-scale fake advantage.

    Pins the claim `combine_group` makes about the split of responsibility. If
    someone narrows `extract_reward_components` back to float32 on the grounds
    that "the floor handles it", the residue returns to being larger than the
    signal and this fails.
    """
    weights = torch.tensor([0.5, 0.5])
    narrow = _constant_sum_group().float().double()

    residue = (standardize_group_components(narrow).double() * weights.double()).sum(dim=1)
    assert residue.abs().amax() > 1e-3
    assert whiten_scalar(residue).abs().amax() > 0.5


def test_float64_alone_leaves_a_residue_small_but_not_zero():
    """The floor's job is exact detectability, not magnitude.

    Also pins that it is `GDPO_EPS` capping step 3, not luck: without it the
    residue would be divided by its own std and come back to order 1.
    """
    group = _constant_sum_group()
    weights = torch.tensor([0.5, 0.5])
    unfloored = (standardize_group_components(group).double() * weights.double()).sum(dim=1)

    assert 0.0 < float(unfloored.abs().amax()) < 1e-8
    assert float(whiten_scalar(unfloored).abs().amax()) < 1e-4


def test_the_floor_leaves_a_well_conditioned_group_alone():
    """The margin is fifteen orders of magnitude, so nothing real is
    suppressed."""
    group = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    weights = torch.tensor([0.5, 0.5])

    combined = combine_group(group, weights)
    floor = component_noise_scale(group)

    assert combined.abs().amax() > 0.1
    assert float(floor.amax()) < 1e-14


def test_noise_scale_is_zero_for_a_collapsed_column():
    """A collapsed column contributes exact zeros, so it contributes no
    error."""
    group = torch.tensor([[1.0, 5.0], [0.0, 5.0], [1.0, 5.0]], dtype=torch.float64)
    assert component_noise_scale(group)[1].item() == 0.0


def test_the_filter_agrees_with_the_normalizer_on_a_cancelling_group():
    """Both go through `combine_group`, so they cannot disagree about this."""
    samples = [_S(0, {"correctness": r, "format": _CONSTANT_SUM_C - r}) for r in _CONSTANT_SUM_R]
    args = _args(weights=[0.5, 0.5], n=3)

    assert group_carries_reward_signal(args, samples) is False
    assert REWARD_NORMALIZERS["gdpo_decoupled"](args, samples, [0.0] * 3) == [0.0, 0.0, 0.0]


# ---------------- where the floor still swallows signal (a known limit) ----
#
# A guard that raised above a fixed per-column noise level used to live in
# `combine_group`. It was removed because it could not be made to hold, and
# these tests pin the measurements that showed why: each one is a group the
# guard would have let through while the floor zeroes a real signal. They are
# characterisation tests -- they assert the limitation, so that anyone who
# fixes it sees them fail rather than discovering the behaviour from a run.


def _offset_group(base, delta):
    """A component varying by `delta` around `base`, plus a well-conditioned
    one."""
    return torch.tensor([[base + i * delta, float(i)] for i in range(3)], dtype=torch.float64)


def test_the_floor_zeroes_a_real_signal_two_columns_are_enough():
    """`noise` under 1% per column, and the floor still takes the group.

    This is the measurement that retired the per-column guard: any threshold
    high enough to leave the cancelling group alone sits above this one.
    """
    base = 4.05e13
    # float64 throughout: `extract_reward_components` builds float64 precisely
    # so this offset survives, and in float32 `base + 1.0` would not move at all.
    group = torch.tensor([[base, base + 2.0], [base + 1.0, base + 1.0], [base + 2.0, base + 0.1]], dtype=torch.float64)
    weights = torch.tensor([1.0, 1.0])

    noise = component_noise_scale(group)
    unfloored = (standardize_group_components(group).double() * weights.double()).sum(dim=1)

    assert float(noise.amax()) < 0.01  # every column reads as clean...
    assert float(unfloored.abs().amax()) > 0.03  # ...and the signal is real...
    assert combine_group(group, weights).tolist() == [0.0, 0.0, 0.0]  # ...and lost anyway


def test_the_floor_grows_with_the_component_count():
    """The floor is a weighted *sum*, so more components raise it.

    No per-column bound can see this: the columns stay equally clean as the
    floor climbs past the signal.
    """
    base = 2.9e13
    rows = []
    for i in range(3):
        up = [base + float(i)] * 8
        down = [base + float(2 - i)] * 8
        rows.append(up + down)
    group = torch.tensor(rows, dtype=torch.float64)
    group[2, 0] += 0.1  # a real difference, 12% of that column's spread
    weights = torch.ones(16)

    noise = component_noise_scale(group)
    floor = 8.0 * float((weights.double().abs() * noise).sum())

    assert float(noise.amax()) < 0.01  # per column: clean
    assert floor > 0.5  # summed over 16 columns: not
    assert combine_group(group, weights).tolist() == [0.0, 0.0, 0.0]


def test_the_noise_estimate_assumes_float64_provenance():
    """A reward computed in float32 carries ~5e8x more error than this bound.

    The estimate uses float64's eps, so a column whose variation is entirely
    float32 rounding measures as clean and standardises to full scale.
    """
    base = 1e9
    ulp32 = float(torch.tensor(base, dtype=torch.float32).item() * 2**-23)
    group = torch.tensor([[base + i * ulp32, float(i)] for i in range(3)], dtype=torch.float64)

    assert float(component_noise_scale(group).amax()) < 1e-8  # reads as pristine
    assert float(combine_group(group, torch.tensor([1.0, 1.0])).abs().amax()) > 1.0  # full-scale garbage


def test_the_noise_estimate_is_a_fraction_only_above_gdpo_eps():
    """Below GDPO_EPS the denominator is clamped, so `noise` understates.

    Reading `component_noise_scale` as "what fraction of the standardised value
    is rounding" is only valid while std dominates GDPO_EPS.
    """
    group = torch.tensor([[1e9 + i * 1e-6, float(i)] for i in range(3)], dtype=torch.float64)

    noise = float(component_noise_scale(group).amax())
    standardized = standardize_group_components(group)
    actual = noise / float(standardized[:, 0].abs().amax())

    assert noise < 0.01  # the estimate says "clean"
    assert actual > 0.15  # the real contamination is over 15%


def test_the_cancelling_group_still_reaches_the_floor():
    """The group the floor exists for still comes out as exactly zero."""
    assert combine_group(_constant_sum_group(), torch.tensor([0.5, 0.5])).tolist() == [0.0, 0.0, 0.0]


def test_a_large_but_readable_reward_keeps_its_signal():
    """1e9 with a spread of 0.1 still carries eight significant digits."""
    combined = combine_group(_offset_group(1e9, 0.1), torch.tensor([0.5, 0.5]))
    assert float(combined.abs().amax()) > 0.1
