# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Element-wise parity tests for the P3O primitives.

The golden values come from running the reference implementation (FeynRL
``algs/P3O/p3o.py``) over one logical batch. The same element-wise oracle is
used by the default micro-batch scope and the optional optimizer-step scope.
"""

import math

import pytest
import torch

from relax.utils.training.p3o_utils import (
    P3OStepContext,
    P3OSufficientStats,
    compute_p3o_behavior_kl_proxy,
    compute_p3o_exact_kl,
    compute_p3o_sufficient_stats,
    compute_p3o_token_terms,
    finalize_p3o_step_context,
)


# Golden case: ratios [1.0, 2.0, 0.5, 4.0] laid out as two sequences of three
# tokens each, with the third token of every sequence invalid (padding).
GOLDEN_RATIOS = [1.0, 2.0, 0.5, 4.0]
GOLDEN_S1 = 7.5
GOLDEN_S2 = 21.25
GOLDEN_N = 4
GOLDEN_ESS = 0.6617647055709343
GOLDEN_LOSS_MEAN = 0.8332794905
GOLDEN_GRAD = [
    [-0.6617646813, 0.8308823705, 0.0],
    [-1.3382353783, 0.5845587850, 0.0],
]

# pytest.approx uses rel/abs; torch.testing.assert_close uses rtol/atol.
TOL = dict(rel=1e-6, abs=1e-6)
TENSOR_TOL = dict(rtol=1e-6, atol=1e-6)


GOLDEN_BEHAVIOR_LOG_PROB = -2.0
GOLDEN_ADVANTAGES = [[1.0, -1.0, 0.0], [2.0, -0.5, 0.0]]
GOLDEN_COEFFICIENTS = [[GOLDEN_ESS, GOLDEN_ESS, 0.0], [0.5, GOLDEN_ESS, 0.0]]
GOLDEN_TOKEN_TOTALS = [
    [1.3235294111, -0.7994998778, 0.0],
    [2.7969356343, 0.0121528449, 0.0],
]


def _golden_batch(requires_grad: bool = False):
    """Build the golden 2x3 batch: ratios above, pad in column 2.

    The behavior log-prob level and the advantages are part of the frozen golden
    case: the loss value pins the log-prob level (the score term is
    ``-coef * log_prob * A``), while the four gradients pin the advantages.
    """
    behavior_log_probs = torch.full((2, 3), GOLDEN_BEHAVIOR_LOG_PROB, dtype=torch.float32)
    log_ratio = torch.tensor(
        [[math.log(1.0), math.log(2.0), 0.0], [math.log(0.5), math.log(4.0), 0.0]],
        dtype=torch.float32,
    )
    log_probs = (behavior_log_probs + log_ratio).clone()
    log_probs.requires_grad_(requires_grad)
    advantages = torch.tensor(GOLDEN_ADVANTAGES, dtype=torch.float32)
    valid_mask = torch.tensor([[True, True, False], [True, True, False]])
    return log_probs, behavior_log_probs, advantages, valid_mask


def _mean_loss(terms, valid_mask):
    """Token-sum of the full objective normalized by the global valid count."""
    total = (terms.score_loss + terms.adaptive_kl_loss).sum()
    return total / valid_mask.sum()


def test_p3o_utils_sufficient_stats_match_reference_moments():
    log_probs, behavior_log_probs, _, valid_mask = _golden_batch()
    stats = compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask)

    assert float(stats.sum_ratio) == pytest.approx(GOLDEN_S1, **TOL)
    assert float(stats.sum_ratio_sq) == pytest.approx(GOLDEN_S2, **TOL)
    assert float(stats.valid_token_count) == GOLDEN_N


def test_p3o_utils_normalized_ess_matches_reference():
    stats = P3OSufficientStats(
        sum_ratio=torch.tensor(GOLDEN_S1, dtype=torch.float64),
        sum_ratio_sq=torch.tensor(GOLDEN_S2, dtype=torch.float64),
        valid_token_count=torch.tensor(float(GOLDEN_N), dtype=torch.float64),
    )
    ctx = finalize_p3o_step_context(stats)

    assert float(ctx.normalized_ess) == pytest.approx(GOLDEN_ESS, **TOL)
    assert float(ctx.adaptive_cap) == pytest.approx(GOLDEN_ESS, **TOL)
    assert float(ctx.valid_token_count) == GOLDEN_N
    assert float(ctx.ratio_mean) == pytest.approx(GOLDEN_S1 / GOLDEN_N, **TOL)
    assert ctx.clamp_events == 0


def test_p3o_utils_total_loss_matches_reference_golden_value():
    log_probs, behavior_log_probs, advantages, valid_mask = _golden_batch()
    stats = compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask)
    ctx = finalize_p3o_step_context(stats)
    terms = compute_p3o_token_terms(log_probs, behavior_log_probs, advantages, valid_mask, ctx)

    assert float(_mean_loss(terms, valid_mask)) == pytest.approx(GOLDEN_LOSS_MEAN, **TOL)


def test_p3o_reference_oracle_matches_ess_cap_and_token_loss():
    """Expose the complete FeynRL formula oracle in one elementwise check."""
    log_probs, behavior_log_probs, advantages, valid_mask = _golden_batch()
    context = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))
    terms = compute_p3o_token_terms(log_probs, behavior_log_probs, advantages, valid_mask, context)

    expected_coefficients = torch.tensor(GOLDEN_COEFFICIENTS, dtype=torch.float32)
    expected_token_totals = torch.tensor(GOLDEN_TOKEN_TOTALS, dtype=torch.float32)

    assert float(context.normalized_ess) == pytest.approx(GOLDEN_ESS, **TOL)
    assert float(context.adaptive_cap) == pytest.approx(GOLDEN_ESS, **TOL)
    torch.testing.assert_close(
        torch.minimum(terms.ratio, context.adaptive_cap.float()),
        expected_coefficients,
        **TENSOR_TOL,
    )
    torch.testing.assert_close(
        terms.score_loss + terms.adaptive_kl_loss,
        expected_token_totals,
        **TENSOR_TOL,
    )
    assert float(_mean_loss(terms, valid_mask)) == pytest.approx(GOLDEN_LOSS_MEAN, **TOL)


def test_p3o_utils_gradient_matches_reference_golden_value():
    log_probs, behavior_log_probs, advantages, valid_mask = _golden_batch(requires_grad=True)
    stats = compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask)
    ctx = finalize_p3o_step_context(stats)
    terms = compute_p3o_token_terms(log_probs, behavior_log_probs, advantages, valid_mask, ctx)

    (terms.score_loss + terms.adaptive_kl_loss).sum().backward()

    expected = torch.tensor(GOLDEN_GRAD, dtype=torch.float32)
    torch.testing.assert_close(log_probs.grad, expected, **TENSOR_TOL)


def test_p3o_utils_ess_invariant_to_token_partitioning():
    """Splitting the same tokens across micro-batches must not move the cap."""
    log_probs, behavior_log_probs, _, valid_mask = _golden_batch()

    whole = compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask)
    accumulated = P3OSufficientStats.zeros()
    for row in range(log_probs.shape[0]):
        accumulated = accumulated + compute_p3o_sufficient_stats(
            log_probs[row : row + 1], behavior_log_probs[row : row + 1], valid_mask[row : row + 1]
        )

    whole_ess = float(finalize_p3o_step_context(whole).normalized_ess)
    split_ess = float(finalize_p3o_step_context(accumulated).normalized_ess)
    assert whole_ess == pytest.approx(split_ess, **TOL)
    assert whole_ess == pytest.approx(GOLDEN_ESS, **TOL)


def test_p3o_utils_dp_cp_stat_reduction_matches_single_rank():
    """Per-rank shards summed elementwise reproduce the single-rank moments."""
    rank0 = P3OSufficientStats(
        sum_ratio=torch.tensor(3.0, dtype=torch.float64),
        sum_ratio_sq=torch.tensor(5.0, dtype=torch.float64),
        valid_token_count=torch.tensor(2.0, dtype=torch.float64),
    )
    rank1 = P3OSufficientStats(
        sum_ratio=torch.tensor(4.5, dtype=torch.float64),
        sum_ratio_sq=torch.tensor(16.25, dtype=torch.float64),
        valid_token_count=torch.tensor(2.0, dtype=torch.float64),
    )
    reduced = P3OSufficientStats.from_vector(rank0.as_vector() + rank1.as_vector())

    assert float(reduced.sum_ratio) == pytest.approx(GOLDEN_S1, **TOL)
    assert float(reduced.sum_ratio_sq) == pytest.approx(GOLDEN_S2, **TOL)
    assert float(reduced.valid_token_count) == GOLDEN_N
    assert float(finalize_p3o_step_context(reduced).normalized_ess) == pytest.approx(GOLDEN_ESS, **TOL)


def test_p3o_utils_on_policy_degenerates_to_vanilla_policy_gradient():
    """rho == 1 everywhere => cap == 1, adaptive KL == 0, gradient == PG."""
    behavior_log_probs = torch.full((2, 4), -0.5, dtype=torch.float32)
    log_probs = behavior_log_probs.clone().requires_grad_(True)
    advantages = torch.tensor([[1.0, -2.0, 0.5, 1.5], [-1.0, 2.0, -0.5, 0.25]], dtype=torch.float32)
    valid_mask = torch.ones(2, 4, dtype=torch.bool)

    ctx = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))
    assert float(ctx.normalized_ess) == pytest.approx(1.0, **TOL)

    terms = compute_p3o_token_terms(log_probs, behavior_log_probs, advantages, valid_mask, ctx)
    torch.testing.assert_close(terms.adaptive_kl_loss, torch.zeros_like(terms.adaptive_kl_loss), **TENSOR_TOL)

    (terms.score_loss + terms.adaptive_kl_loss).sum().backward()
    torch.testing.assert_close(log_probs.grad, -advantages, **TENSOR_TOL)


def test_p3o_utils_uniform_ratio_offset_leaves_ess_near_one():
    """ESS measures concentration, so a constant shift is not mismatch."""
    behavior_log_probs = torch.zeros(2, 4, dtype=torch.float32)
    log_probs = behavior_log_probs + 0.75
    valid_mask = torch.ones(2, 4, dtype=torch.bool)

    ctx = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))
    assert float(ctx.normalized_ess) == pytest.approx(1.0, **TOL)


def test_p3o_utils_dominant_ratio_drives_ess_toward_one_over_n():
    """One huge ratio among N tokens collapses ESS to roughly 1/N."""
    behavior_log_probs = torch.zeros(1, 4, dtype=torch.float32)
    log_probs = torch.tensor([[math.log(1e6), 0.0, 0.0, 0.0]], dtype=torch.float32)
    valid_mask = torch.ones(1, 4, dtype=torch.bool)

    ctx = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))
    assert float(ctx.normalized_ess) == pytest.approx(0.25, rel=1e-3)


def test_p3o_utils_single_valid_token_gives_full_ess():
    behavior_log_probs = torch.zeros(1, 3, dtype=torch.float32)
    log_probs = torch.tensor([[math.log(3.0), 0.0, 0.0]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, False, False]])

    ctx = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))
    assert float(ctx.normalized_ess) == pytest.approx(1.0, **TOL)
    assert float(ctx.valid_token_count) == 1


def test_p3o_utils_masked_positions_tolerate_non_finite_values():
    """NaN/Inf in prompt or padding slots must not leak into the stats."""
    log_probs, behavior_log_probs, advantages, valid_mask = _golden_batch()
    log_probs, behavior_log_probs = log_probs.clone(), behavior_log_probs.clone()
    advantages = advantages.clone()
    for tensor, poison in ((log_probs, float("nan")), (behavior_log_probs, float("inf")), (advantages, 1e30)):
        tensor[0, 2] = poison
        tensor[1, 2] = -poison if poison == 1e30 else float("nan")

    stats = compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask)
    ctx = finalize_p3o_step_context(stats)
    assert float(ctx.normalized_ess) == pytest.approx(GOLDEN_ESS, **TOL)

    terms = compute_p3o_token_terms(log_probs, behavior_log_probs, advantages, valid_mask, ctx)
    assert torch.isfinite(terms.score_loss).all()
    assert float(_mean_loss(terms, valid_mask)) == pytest.approx(GOLDEN_LOSS_MEAN, **TOL)


@pytest.mark.parametrize(
    ("log_prob", "behavior_log_prob"),
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (0.0, float("inf")),
        (0.0, float("-inf")),
    ],
)
def test_p3o_utils_non_finite_valid_token_raises(log_prob, behavior_log_prob):
    behavior_log_probs = torch.tensor([[behavior_log_prob, 0.0]], dtype=torch.float32)
    log_probs = torch.tensor([[log_prob, 0.0]], dtype=torch.float32)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="non-finite importance ratio"):
        compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask)


def test_p3o_utils_all_masked_poison_produces_fp64_zero_stats():
    log_probs = torch.tensor([[float("nan"), float("inf")]], dtype=torch.float32)
    behavior_log_probs = torch.tensor([[float("-inf"), float("nan")]], dtype=torch.float32)
    valid_mask = torch.zeros(1, 2, dtype=torch.bool)

    stats = compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask)

    for value in (stats.sum_ratio, stats.sum_ratio_sq, stats.valid_token_count):
        assert value.dtype == torch.float64
        assert torch.equal(value, torch.zeros((), dtype=torch.float64))


def test_p3o_utils_empty_global_batch_falls_back_to_full_ess():
    stats = P3OSufficientStats.zeros()
    context = finalize_p3o_step_context(stats)

    assert float(context.normalized_ess) == 1.0
    assert float(context.adaptive_cap) == 1.0
    assert float(context.valid_token_count) == 0.0
    assert float(context.ratio_mean) == 1.0
    assert float(context.ratio_std) == 0.0


@pytest.mark.parametrize("mismatched", ["behavior", "mask"])
def test_p3o_utils_sufficient_stats_reject_shape_mismatch(mismatched):
    log_probs = torch.zeros(2, 3)
    behavior_log_probs = torch.zeros(2, 2) if mismatched == "behavior" else torch.zeros(2, 3)
    valid_mask = torch.ones(2, 2, dtype=torch.bool) if mismatched == "mask" else torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="identical shapes"):
        compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask)


def test_p3o_utils_token_terms_reject_advantage_shape_mismatch():
    log_probs, behavior_log_probs, _, valid_mask = _golden_batch()
    context = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))

    with pytest.raises(ValueError, match="advantages"):
        compute_p3o_token_terms(
            log_probs,
            behavior_log_probs,
            torch.zeros(2, 1),
            valid_mask,
            context,
        )


def test_p3o_utils_cap_hits_track_ratios_above_cap():
    log_probs, behavior_log_probs, advantages, valid_mask = _golden_batch()
    ctx = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))
    terms = compute_p3o_token_terms(log_probs, behavior_log_probs, advantages, valid_mask, ctx)

    # ratios 1.0, 2.0, 4.0 exceed cap 0.6617...; ratio 0.5 does not; pads never count.
    expected = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(terms.cap_hits, expected)
    assert float(terms.cap_hits.sum() / ctx.valid_token_count) == pytest.approx(0.75, **TOL)


def test_p3o_utils_behavior_kl_proxy_is_non_negative_and_directional():
    behavior_log_probs = torch.zeros(1, 3, dtype=torch.float32)
    log_probs = torch.tensor([[math.log(2.0), math.log(0.5), 0.0]], dtype=torch.float32)
    valid_mask = torch.ones(1, 3, dtype=torch.bool)

    kl = compute_p3o_behavior_kl_proxy(log_probs, behavior_log_probs, valid_mask)

    assert (kl >= -1e-7).all()
    assert float(kl[0, 2]) == pytest.approx(0.0, abs=1e-7)
    # k3 form: l + exp(-l) - 1
    assert float(kl[0, 0]) == pytest.approx(math.log(2.0) + 0.5 - 1.0, **TOL)
    assert float(kl[0, 1]) == pytest.approx(math.log(0.5) + 2.0 - 1.0, **TOL)


def test_p3o_utils_behavior_kl_proxy_clamps_extreme_divergence():
    behavior_log_probs = torch.zeros(1, 1, dtype=torch.float32)
    log_probs = torch.tensor([[-50.0]], dtype=torch.float32)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)

    kl = compute_p3o_behavior_kl_proxy(log_probs, behavior_log_probs, valid_mask)
    assert float(kl[0, 0]) == pytest.approx(-50.0 + math.exp(10.0) - 1.0, rel=1e-6)


def test_p3o_utils_proxy_safe_matches_proxy_forward_and_has_correct_gradient_sign():
    behavior_log_probs = torch.zeros(121, dtype=torch.float32)
    proxy_log_probs = torch.linspace(-30.0, 30.0, 121, requires_grad=True)
    safe_log_probs = proxy_log_probs.detach().clone().requires_grad_(True)
    valid_mask = torch.ones_like(proxy_log_probs, dtype=torch.bool)

    proxy = compute_p3o_behavior_kl_proxy(proxy_log_probs, behavior_log_probs, valid_mask, mode="proxy")
    proxy_safe = compute_p3o_behavior_kl_proxy(safe_log_probs, behavior_log_probs, valid_mask, mode="proxy_safe")

    torch.testing.assert_close(proxy_safe, proxy, rtol=0.0, atol=0.0)
    proxy.sum().backward()
    proxy_safe.sum().backward()

    negative = safe_log_probs.detach() < 0
    positive = safe_log_probs.detach() > 0
    assert torch.all(safe_log_probs.grad[negative] <= 0)
    assert torch.all(safe_log_probs.grad[positive] >= 0)
    assert float(safe_log_probs.grad.abs().max()) <= math.exp(10.0)
    assert float(proxy_log_probs.grad[0]) > 0
    assert float(safe_log_probs.grad[0]) < 0


def test_p3o_utils_exact_kl_matches_manual_small_vocabulary_oracle():
    policy_logits = torch.tensor([[[1.0, 0.0, -1.0], [float("nan"), 2.0, 1.0]]], requires_grad=True)
    behavior_logits = torch.tensor([[[0.0, 0.5, -0.5], [float("inf"), 0.0, 0.0]]], requires_grad=True)
    valid_mask = torch.tensor([[True, False]])

    exact = compute_p3o_exact_kl(policy_logits, behavior_logits, valid_mask)
    policy_log_probs = torch.log_softmax(policy_logits[0, 0], dim=-1)
    behavior_log_probs = torch.log_softmax(behavior_logits[0, 0].detach(), dim=-1)
    expected = (policy_log_probs.exp() * (policy_log_probs - behavior_log_probs)).sum()

    torch.testing.assert_close(exact[0, 0], expected)
    assert float(exact[0, 1].detach()) == 0.0
    exact.sum().backward()
    assert policy_logits.grad is not None
    assert behavior_logits.grad is None


def test_p3o_utils_exact_training_mode_requires_behavior_logits():
    log_probs, behavior_log_probs, advantages, valid_mask = _golden_batch()
    context = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))

    with pytest.raises(ValueError, match="full-vocabulary behavior logits"):
        compute_p3o_token_terms(
            log_probs,
            behavior_log_probs,
            advantages,
            valid_mask,
            context,
            kl_mode="exact",
        )


def test_p3o_utils_advantage_and_cap_are_stop_gradient():
    log_probs, behavior_log_probs, advantages, valid_mask = _golden_batch(requires_grad=True)
    advantages = advantages.clone().requires_grad_(True)
    behavior_log_probs = behavior_log_probs.clone().requires_grad_(True)

    ctx = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))
    terms = compute_p3o_token_terms(log_probs, behavior_log_probs, advantages, valid_mask, ctx)
    (terms.score_loss + terms.adaptive_kl_loss).sum().backward()

    assert advantages.grad is None
    assert behavior_log_probs.grad is None
    assert log_probs.grad is not None
    assert not ctx.normalized_ess.requires_grad


def test_p3o_utils_entire_adaptive_coefficient_is_stop_gradient():
    log_probs = torch.tensor([math.log(2.0)], dtype=torch.float32, requires_grad=True)
    behavior_log_probs = torch.zeros(1, dtype=torch.float32)
    advantages = torch.tensor([2.0], dtype=torch.float32)
    valid_mask = torch.ones(1, dtype=torch.bool)
    adaptive_cap = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
    ctx = finalize_p3o_step_context(
        P3OSufficientStats(
            sum_ratio=torch.tensor(1.0, dtype=torch.float64),
            sum_ratio_sq=torch.tensor(1.0, dtype=torch.float64),
            valid_token_count=torch.tensor(1.0, dtype=torch.float64),
        )
    )
    ctx = type(ctx)(
        normalized_ess=ctx.normalized_ess,
        adaptive_cap=adaptive_cap,
        valid_token_count=ctx.valid_token_count,
        ratio_mean=ctx.ratio_mean,
        ratio_std=ctx.ratio_std,
    )

    terms = compute_p3o_token_terms(log_probs, behavior_log_probs, advantages, valid_mask, ctx)
    terms.score_loss.sum().backward()

    torch.testing.assert_close(log_probs.grad, torch.tensor([-1.5]))
    assert adaptive_cap.grad is None


def test_p3o_utils_clip_hits_use_monitoring_interval_not_adaptive_cap():
    behavior_log_probs = torch.zeros(1, 5)
    ratios = torch.tensor([[0.79, 0.8, 1.0, 1.2, 1.21]])
    log_probs = ratios.log()
    valid_mask = torch.ones_like(log_probs, dtype=torch.bool)
    context = finalize_p3o_step_context(compute_p3o_sufficient_stats(log_probs, behavior_log_probs, valid_mask))

    terms = compute_p3o_token_terms(
        log_probs,
        behavior_log_probs,
        torch.ones_like(log_probs),
        valid_mask,
        context,
        clip_low=0.2,
        clip_high=0.2,
    )

    torch.testing.assert_close(terms.clip_hits, torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]]))


def test_p3o_utils_token_terms_keep_adaptive_cap_on_device(monkeypatch):
    """The per-micro-batch loss must not convert the GPU cap to a scalar."""
    adaptive_cap = torch.tensor(0.75, dtype=torch.float64)
    context = P3OStepContext(
        normalized_ess=adaptive_cap,
        adaptive_cap=adaptive_cap,
        valid_token_count=torch.tensor(1.0, dtype=torch.float64),
        ratio_mean=torch.tensor(2.0, dtype=torch.float64),
        ratio_std=torch.tensor(0.0, dtype=torch.float64),
    )
    log_probs = torch.tensor([math.log(2.0)], dtype=torch.float32, requires_grad=True)

    def fail_on_scalar_conversion(tensor):
        raise AssertionError(f"unexpected Tensor.__float__ for {tensor}")

    monkeypatch.setattr(torch.Tensor, "__float__", fail_on_scalar_conversion)
    terms = compute_p3o_token_terms(
        log_probs=log_probs,
        behavior_log_probs=torch.zeros_like(log_probs),
        advantages=torch.ones_like(log_probs),
        valid_mask=torch.ones_like(log_probs, dtype=torch.bool),
        step_context=context,
    )

    torch.testing.assert_close(terms.score_loss, -adaptive_cap.float() * log_probs.detach())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_p3o_utils_stats_stable_across_input_dtypes(dtype):
    log_probs, behavior_log_probs, _, valid_mask = _golden_batch()
    stats = compute_p3o_sufficient_stats(log_probs.to(dtype), behavior_log_probs.to(dtype), valid_mask)
    ess = float(finalize_p3o_step_context(stats).normalized_ess)

    assert stats.as_vector().dtype == torch.float64
    tol = 5e-3 if dtype is torch.bfloat16 else 1e-6
    assert ess == pytest.approx(GOLDEN_ESS, rel=tol, abs=tol)
