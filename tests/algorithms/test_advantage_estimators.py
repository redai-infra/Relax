# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Golden-value tests for the extracted advantage estimators."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.advantages import (  # noqa: E402
    ADVANTAGE_FNS,
    compute_advantages_and_returns,
    whiten_scalar,
)


def _args(estimator, **overrides):
    base = dict(advantage_estimator=estimator, kl_coef=0.0, gamma=1.0, lambd=1.0)
    base.update(overrides)
    return SimpleNamespace(**base)


def _inputs(lengths=(3, 2)):
    return dict(
        kl=[torch.zeros(n, dtype=torch.float32) for n in lengths],
        loss_masks=[torch.ones(n, dtype=torch.float32) for n in lengths],
        response_lengths=list(lengths),
        total_lengths=[n + 2 for n in lengths],
        values=None,
        # One segment covering the whole shard. Estimators that do not whiten
        # per batch absorb this in `**_unused`; GDPO requires it, because a
        # caller that cannot say how the rollout was split cannot be given a
        # default without silently changing which objective it optimises.
        mini_batch_sizes=[len(lengths)],
    )


# ---------------- dispatch ----------------


def test_registry_covers_every_spec_advantage_id():
    from relax.algorithms import list_algorithm_names
    from relax.algorithms.spec import get_algorithm

    for name in list_algorithm_names():
        assert get_algorithm(name).advantage_fn in ADVANTAGE_FNS


def test_unknown_advantage_fn_is_not_silently_tolerated():
    with pytest.raises(KeyError):
        ADVANTAGE_FNS["not_a_real_fn"]


# ---------------- grpo family ----------------


def test_grpo_broadcast_repeats_the_scalar_over_tokens():
    adv, ret = compute_advantages_and_returns(_args("grpo"), rewards=[1.5, -2.0], **_inputs())
    assert torch.equal(adv[0], torch.full((3,), 1.5))
    assert torch.equal(adv[1], torch.full((2,), -2.0))
    assert torch.equal(ret[0], adv[0])


def test_grpo_accepts_a_tensor_as_well_as_a_list():
    from_list = compute_advantages_and_returns(_args("grpo"), rewards=[1.5, -2.0], **_inputs())[0]
    from_tensor = compute_advantages_and_returns(_args("grpo"), rewards=torch.tensor([1.5, -2.0]), **_inputs())[0]
    assert torch.equal(from_list[0], from_tensor[0])


def test_grpo_advantages_is_a_distinct_list_from_returns():
    """Legacy did `advantages = list(returns)`; rebinding one must not touch the other."""
    adv, ret = compute_advantages_and_returns(_args("grpo"), rewards=[1.0, 1.0], **_inputs())
    assert adv is not ret
    adv[0] = torch.zeros(3)
    assert not torch.equal(ret[0], adv[0])


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "sapo", "cispo"])
def test_grpo_family_produces_identical_advantages(estimator):
    baseline, _ = compute_advantages_and_returns(_args("grpo"), rewards=[0.5, -0.5], **_inputs())
    actual, _ = compute_advantages_and_returns(_args(estimator), rewards=[0.5, -0.5], **_inputs())
    for left, right in zip(baseline, actual, strict=True):
        assert torch.equal(left, right)


# ---------------- reinforce++ baseline ----------------


def test_reinforce_plus_plus_baseline_aliases_returns_to_advantages():
    """Legacy did `returns = advantages` (the same list object). Preserve it."""
    adv, ret = compute_advantages_and_returns(_args("reinforce_plus_plus_baseline"), rewards=[1.0, 1.0], **_inputs())
    assert ret is adv


def test_reinforce_plus_plus_baseline_keeps_kl_out_of_the_advantage():
    """The baseline variant regularises via a separate k2 loss, not via the
    advantage, so the KL tensor only supplies the per-token shape."""
    inputs = _inputs(lengths=(2,))
    inputs["kl"] = [torch.tensor([2.0, 4.0])]
    inputs["loss_masks"] = [torch.ones(2)]
    adv, _ = compute_advantages_and_returns(
        _args("reinforce_plus_plus_baseline", kl_coef=0.5), rewards=[3.0], **inputs
    )
    # The scalar reward is broadcast to every unmasked token; kl is not subtracted
    # even though kl_coef is non-zero (argument validation forbids that combination).
    assert torch.equal(adv[0], torch.tensor([3.0, 3.0]))


def test_reinforce_plus_plus_baseline_zeroes_masked_tokens():
    inputs = _inputs(lengths=(3,))
    inputs["kl"] = [torch.zeros(3)]
    inputs["loss_masks"] = [torch.tensor([1.0, 0.0, 1.0])]
    adv, _ = compute_advantages_and_returns(_args("reinforce_plus_plus_baseline"), rewards=[2.0], **inputs)
    assert torch.equal(adv[0], torch.tensor([2.0, 0.0, 2.0]))


# ---------------- gdpo step 3 ----------------


def test_whiten_scalar_produces_zero_mean_unit_std():
    out = whiten_scalar(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert abs(out.mean().item()) < 1e-6
    assert abs(out.std().item() - 1.0) < 1e-3


def test_whiten_scalar_zero_variance_returns_zeros_not_noise():
    """A collapsed batch must give exactly 0, not the fp32 residual over
    eps."""
    assert torch.equal(whiten_scalar(torch.full((7,), 0.7)), torch.zeros(7))


def test_whiten_scalar_single_element_returns_zero():
    """std of one element is NaN under Bessel correction."""
    assert torch.equal(whiten_scalar(torch.tensor([3.0])), torch.zeros(1))


def test_whiten_scalar_empty_returns_empty():
    assert whiten_scalar(torch.tensor([])).numel() == 0


def test_whiten_scalar_preserves_ordering():
    values = torch.tensor([-3.0, 0.5, 0.0, 9.0])
    out = whiten_scalar(values)
    assert torch.equal(values.argsort(), out.argsort())


def test_gdpo_advantage_whitens_before_broadcasting():
    rewards = [2.0, -2.0]
    adv, _ = compute_advantages_and_returns(_args("gdpo"), rewards=rewards, **_inputs())
    expected = whiten_scalar(torch.tensor(rewards, dtype=torch.float32))
    assert torch.allclose(adv[0], expected[0].expand(3))
    assert torch.allclose(adv[1], expected[1].expand(2))


def test_gdpo_differs_from_grpo_by_a_positive_scalar():
    rewards = [1.0, -1.0, 0.5, -0.5]
    inputs = _inputs(lengths=(1, 1, 1, 1))
    grpo, _ = compute_advantages_and_returns(_args("grpo"), rewards=rewards, **inputs)
    gdpo, _ = compute_advantages_and_returns(_args("gdpo"), rewards=rewards, **inputs)

    grpo_flat = torch.cat(grpo)
    gdpo_flat = torch.cat(gdpo)
    ratio = gdpo_flat / grpo_flat
    assert (ratio > 0).all()
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-4)
    assert not torch.allclose(gdpo_flat, grpo_flat, atol=1e-3)


def test_gdpo_collapsed_batch_gives_zero_advantages():
    adv, _ = compute_advantages_and_returns(_args("gdpo"), rewards=[0.7, 0.7], **_inputs())
    assert torch.equal(adv[0], torch.zeros(3))
    assert torch.equal(adv[1], torch.zeros(2))


# ---------------- collapse detection is exact ----------------


def test_collapse_uses_exact_equality_not_a_relative_tolerance():
    """A relative tolerance erased this perfectly informative batch."""
    values = torch.tensor([10000.0, 10000.005, 10000.010, 10000.015])
    out = whiten_scalar(values)
    assert not torch.equal(out, torch.zeros_like(out))
    assert torch.equal(values.argsort(), out.argsort())


@pytest.mark.parametrize("magnitude", [0.0, 0.1, 0.7, 1.0, 1e6, -3.5])
def test_identical_values_collapse_exactly_at_any_magnitude(magnitude):
    out = whiten_scalar(torch.full((7,), magnitude))
    assert torch.equal(out, torch.zeros(7))


def test_two_values_differing_by_one_ulp_are_not_collapsed():
    base = torch.tensor(1.0)
    values = torch.stack([base, torch.nextafter(base, torch.tensor(2.0))] * 2)
    out = whiten_scalar(values)
    # The name's claim is the > 0: is_collapsed tests exact equality, so a
    # one-ULP spread is real signal and must survive. An earlier version of this
    # test asserted only the upper bound, which an all-zero output also passes --
    # i.e. it did not test the thing it was named after.
    assert out.abs().max() > 0.0
    # eps damps it towards zero rather than amplifying it to unit scale.
    assert out.abs().max() < 1.0


def test_empty_batch_is_treated_as_collapsed():
    assert whiten_scalar(torch.tensor([])).numel() == 0


# ---------------- distributed statistics ----------------


def test_distributed_mean_std_without_a_group_matches_torch():
    from relax.algorithms.numerics import distributed_mean_std

    values = torch.tensor([1.0, 2.0, 4.0, 8.0])
    mean, std = distributed_mean_std(values)
    assert torch.allclose(mean, values.mean())
    assert torch.allclose(std, values.std(), atol=1e-5)


def test_whiten_scalar_matches_manual_formula_without_a_group():
    values = torch.tensor([1.0, 2.0, 4.0, 8.0])
    # 1e-4, not the GRPO path's 1e-6: whiten_scalar is GDPO step 3 and matches
    # the reference implementation's epsilon. Hardcoded rather than imported so
    # the test would notice the constant moving.
    expected = (values - values.mean()) / (values.std() + 1e-4)
    assert torch.allclose(whiten_scalar(values), expected, atol=1e-6)


def test_sharded_whitening_differs_from_local_whitening():
    """Why the process group matters: local stats give each shard its own
    scale.

    Simulates two DP ranks by whitening each shard alone and comparing against
    whitening the concatenated batch.
    """
    shard_a = torch.tensor([-0.7, 0.7])
    shard_b = torch.tensor([-1.4, 1.4])

    local = torch.cat([whiten_scalar(shard_a), whiten_scalar(shard_b)])
    joint = whiten_scalar(torch.cat([shard_a, shard_b]))

    # Local whitening flattens the two shards onto the same amplitude; the joint
    # statistics keep shard_b's larger relative contribution.
    assert torch.allclose(local[:2].abs(), local[2:].abs(), atol=1e-4)
    assert not torch.allclose(joint[:2].abs(), joint[2:].abs(), atol=1e-2)


def test_compute_advantages_and_returns_accepts_a_process_group_kwarg():
    """Both call sites pass it; every estimator must tolerate it."""
    from relax.algorithms import list_algorithm_names
    from relax.algorithms.spec import get_algorithm

    # "gae" needs a critic and "reinforce_plus_plus" imports megatron.core for the
    # CP world size; neither is available on a CPU-only runner.
    needs_megatron = {"gae", "reinforce_plus_plus"}
    for name in list_algorithm_names():
        if get_algorithm(name).advantage_fn in needs_megatron:
            continue
        args = _args(name, kl_coef=0.0)
        adv, _ = compute_advantages_and_returns(args, rewards=[0.5, -0.5], process_group=None, **_inputs())
        assert len(adv) == 2


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_value_raises_rather_than_reading_as_zero_variance(bad):
    """This used to return zeros, and the zeros were the bug.

    A single inf or nan among the rewards makes the batch std non-finite, and
    the old guard read that as "no spread" and returned an all-zero batch. The
    two states are opposites: one says every sample scored the same, the other
    says the arithmetic broke. Reporting the second as the first is how an
    overflow at the float32 hand-off became a run that trained on nothing and
    logged nothing -- the combined reward is verified finite in float64 by the
    reward stage and can still be `inf` by the time it is used.

    Reward functions are user code, so this is reachable input.
    """
    values = torch.tensor([1.0, bad, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-finite advantage"):
        whiten_scalar(values)


def test_a_huge_but_finite_spread_is_still_whitened():
    """The check must catch non-finite, not merely large -- otherwise it would
    reject batches it should scale."""
    out = whiten_scalar(torch.tensor([1e38, -1e38, 0.0]))
    assert not torch.equal(out, torch.zeros_like(out))
    assert torch.isfinite(out).all()
