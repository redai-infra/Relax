# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure-PyTorch primitives for P3O (adaptive policy optimization).

P3O replaces PPO/GRPO's fixed clip range with a one-sided cap derived from the
normalized Effective Sample Size (ESS) of the token-level importance ratios,
and adds an adaptive trust region weighted by ``(1 - ESS)``.

Reference: Fakoor et al., "Trust the Batch, On- or Off-Policy: Adaptive Policy
Optimization for RL Post-Training" (arXiv:2605.12380), Eq. (7), (11), (12) and
Appendix Algorithm 2.

This module is deliberately free of any Megatron / ``mpu`` dependency: it owns
the formulas, the masking discipline and the stop-gradient boundaries, while
collectives and step lifecycle live in the Megatron backend. The ESS scope in
Relax is one *optimizer* step (not one micro-batch), so the sufficient
statistics are produced here and reduced by the caller before being frozen into
a :class:`P3OStepContext`.
"""

import math
from dataclasses import dataclass

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Epsilon placed in the ESS denominator. Kept bit-compatible with the reference
# implementation (FeynRL ``algs/P3O/p3o.py::calculate_ess``) so golden-value
# parity holds; intentionally not exposed as a CLI hyper-parameter.
ESS_DENOM_EPS = 1e-8

# Clamp applied to the exponent of the behavior-KL proxy, matching the reference
# (FeynRL ``algs/RL/common.py::compute_kl_distance``).
BEHAVIOR_KL_EXP_CLAMP = 10.0

# Shared by the checked and unchecked sufficient-statistics paths so the message
# a user sees does not depend on which one detected the non-finite ratio.
NONFINITE_RATIO_MESSAGE = (
    "P3O: non-finite importance ratio at a valid response token; refusing to "
    "silently fall back to ESS=1. Check rollout log-probs and mask alignment."
)


@dataclass(frozen=True)
class P3OSufficientStats:
    """Local (this-rank, this-micro-batch) ESS sufficient statistics.

    All three fields are ``float64`` scalar tensors so they can be stacked and
    summed by a single collective without precision loss.

    Attributes:
        sum_ratio: ``S1 = sum(rho_i)`` over valid response tokens.
        sum_ratio_sq: ``S2 = sum(rho_i ** 2)`` over valid response tokens.
        valid_token_count: ``N``, the number of valid response tokens.
    """

    sum_ratio: torch.Tensor
    sum_ratio_sq: torch.Tensor
    valid_token_count: torch.Tensor

    def as_vector(self) -> torch.Tensor:
        """Stack the statistics into a ``[3]`` float64 tensor for reduction."""
        return torch.stack([self.sum_ratio, self.sum_ratio_sq, self.valid_token_count])

    @classmethod
    def zeros(cls, device: torch.device | str = "cpu") -> "P3OSufficientStats":
        """Return all-zero statistics, used for dummy micro-batches."""
        zero = torch.zeros((), dtype=torch.float64, device=device)
        return cls(sum_ratio=zero.clone(), sum_ratio_sq=zero.clone(), valid_token_count=zero.clone())

    @classmethod
    def from_vector(cls, vector: torch.Tensor) -> "P3OSufficientStats":
        """Rebuild statistics from a reduced ``[3]`` tensor."""
        assert vector.numel() == 3, f"expected a 3-element stat vector, got {tuple(vector.shape)}"
        flat = vector.reshape(3).to(torch.float64)
        return cls(sum_ratio=flat[0], sum_ratio_sq=flat[1], valid_token_count=flat[2])

    def __add__(self, other: "P3OSufficientStats") -> "P3OSufficientStats":
        """Accumulate statistics across micro-batches on the same rank."""
        return P3OSufficientStats(
            sum_ratio=self.sum_ratio + other.sum_ratio,
            sum_ratio_sq=self.sum_ratio_sq + other.sum_ratio_sq,
            valid_token_count=self.valid_token_count + other.valid_token_count,
        )


@dataclass(frozen=True)
class P3OStepContext:
    """Immutable per-optimizer-step P3O state shared by every micro-batch.

    Attributes:
        normalized_ess: Global normalized ESS in ``[0, 1]``.
        adaptive_cap: The ratio cap. Numerically equal to ``normalized_ess`` but
            kept separate because it plays a different role in the objective.
        valid_token_count: Global valid response-token count ``N``.
        ratio_mean: ``S1 / N``.
        ratio_std: Population std derived from the global moments.
        clamp_events: Number of ``[0, 1]`` round-off corrections applied to ESS.
    """

    normalized_ess: torch.Tensor
    adaptive_cap: torch.Tensor
    valid_token_count: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_std: torch.Tensor
    clamp_events: int = 0


@dataclass(frozen=True)
class P3OTokenTerms:
    """Element-wise P3O loss terms for one micro-batch.

    Every tensor has the shape of the concatenated response tokens and carries
    no reduction, so the caller applies its own masking / normalization.

    Attributes:
        ratio: ``rho_i``, detached.
        score_loss: ``-sg(min(rho_i, cap)) * log_prob_i * sg(A_i)``.
        behavior_kl_proxy: k3-style sampled-token KL against the behavior
            policy, *not* multiplied by ``(1 - ESS)``. Keeps gradient.
        adaptive_kl_loss: ``(1 - ESS) * behavior_kl_proxy``.
        cap_hits: 1.0 where ``rho_i > cap``, else 0.0.
    """

    ratio: torch.Tensor
    score_loss: torch.Tensor
    behavior_kl_proxy: torch.Tensor
    adaptive_kl_loss: torch.Tensor
    cap_hits: torch.Tensor


def compute_p3o_log_ratio(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the masked log importance ratio ``l_i``.

    Invalid positions are zeroed *before* any exponentiation so that padded
    entries holding ``inf`` / ``NaN`` cannot poison the statistics via
    ``inf * 0 -> NaN``.

    Args:
        log_probs: Current-policy log-probs of the sampled tokens.
        behavior_log_probs: Log-probs under the policy that actually generated
            the tokens (rollout log-probs), already detached by the caller.
        valid_mask: Boolean mask selecting valid response tokens.

    Returns:
        ``l_i = log pi_theta - log pi_b`` in float32, zero at invalid positions.
    """
    log_ratio = log_probs.float() - behavior_log_probs.float()
    return torch.where(valid_mask, log_ratio, torch.zeros_like(log_ratio))


def compute_p3o_sufficient_stats(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> P3OSufficientStats:
    """Accumulate this micro-batch's contribution to the global ESS.

    The statistics are computed in float64 and fully detached: ESS is a
    stop-gradient quantity in the P3O objective.

    Args:
        log_probs: Current-policy log-probs of the sampled tokens.
        behavior_log_probs: Behavior-policy (rollout) log-probs.
        valid_mask: Boolean mask selecting valid response tokens. Prompt,
            padding, CP padding and masked tokens must already be excluded.

    Returns:
        Local :class:`P3OSufficientStats` in float64.

    Raises:
        ValueError: If a valid position produced a non-finite ratio.
    """
    stats, invalid_flag = compute_p3o_sufficient_stats_unchecked(log_probs, behavior_log_probs, valid_mask)
    # This is the sync-ing convenience wrapper: it materializes the flag to host
    # memory so callers outside the micro-batch loop (tests, single-batch CPU
    # use) still get an eager ValueError. Hot-path callers must use the
    # unchecked variant and reduce the flag with the stats.
    if bool(invalid_flag > 0):
        raise ValueError(NONFINITE_RATIO_MESSAGE)
    return stats


def compute_p3o_sufficient_stats_unchecked(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[P3OSufficientStats, torch.Tensor]:
    """Sync-free variant: report non-finite ratios as a device-resident flag.

    Identical arithmetic to :func:`compute_p3o_sufficient_stats`, but the
    finiteness verdict is returned as a ``float64`` scalar tensor instead of
    being tested on the host. This is what the ESS pre-pass calls: it runs once
    per micro-batch, and a ``bool()`` there would stall the GPU pipeline
    ``num_microbatches`` times per optimizer step. The flag rides along with
    ``S1/S2/N`` through the step's single all-reduce, so the error still
    surfaces on every rank -- just one collective later.

    Args:
        log_probs: Current-policy log-probs of the sampled tokens.
        behavior_log_probs: Behavior-policy (rollout) log-probs.
        valid_mask: Boolean mask selecting valid response tokens.

    Returns:
        ``(stats, invalid_flag)``. ``invalid_flag`` is ``1.0`` when any valid
        position produced a non-finite ratio, else ``0.0``. When it is set, the
        statistics are zeroed so a caller that defers the check cannot poison
        ``S1/S2`` with ``inf``/``nan`` in the meantime.
    """
    with torch.no_grad():
        mask_bool = valid_mask.bool()
        log_ratio = compute_p3o_log_ratio(log_probs.detach(), behavior_log_probs.detach(), mask_bool)

        ratio = torch.exp(log_ratio.to(torch.float64))
        ratio = torch.where(mask_bool, ratio, torch.zeros_like(ratio))

        # Both checks stay on device. log_ratio is already zeroed outside the
        # mask, so a global isfinite() over it is equivalent to masking first.
        invalid_flag = (~(torch.isfinite(log_ratio).all() & torch.isfinite(ratio).all())).to(torch.float64)

        # Zero the contribution when invalid, so deferring the host-side check
        # cannot let inf/nan reach the reduced moments.
        keep = 1.0 - invalid_flag
        return (
            P3OSufficientStats(
                sum_ratio=ratio.sum() * keep,
                sum_ratio_sq=ratio.pow(2).sum() * keep,
                valid_token_count=mask_bool.sum().to(torch.float64) * keep,
            ),
            invalid_flag,
        )


def finalize_p3o_step_context(stats: P3OSufficientStats) -> P3OStepContext:
    """Turn globally reduced sufficient statistics into a frozen step context.

    Implements the paper's ``e = sg(S1^2 / (N * S2))`` with the reference
    implementation's epsilon placement, i.e. ``S1^2 / (N * (S2 + eps))``.

    Args:
        stats: Sufficient statistics already summed across DP x CP.

    Returns:
        Immutable :class:`P3OStepContext` reused by every micro-batch of the
        current optimizer step.

    Raises:
        ValueError: If the global valid-token count is zero, or if the reduced
            statistics are non-finite. Both are hard errors rather than a silent
            ``ESS = 1`` fallback, so a broken step fails loudly on all ranks.
    """
    sum_ratio = stats.sum_ratio.to(torch.float64)
    sum_ratio_sq = stats.sum_ratio_sq.to(torch.float64)
    count = stats.valid_token_count.to(torch.float64)

    if not (math.isfinite(float(sum_ratio)) and math.isfinite(float(sum_ratio_sq)) and math.isfinite(float(count))):
        raise ValueError(
            f"P3O: non-finite global ESS statistics (S1={float(sum_ratio)}, "
            f"S2={float(sum_ratio_sq)}, N={float(count)})."
        )

    if float(count) < 0.5:
        raise ValueError(
            "P3O: global valid response-token count is zero for this optimizer step. "
            "The step cannot be normalized; skip or abort instead of assuming ESS=1."
        )

    raw_ess = sum_ratio.pow(2) / (count * (sum_ratio_sq + ESS_DENOM_EPS))

    # Only float round-off should ever push ESS outside [0, 1]; record how often
    # it happens rather than clamping silently.
    clamp_events = 0
    if float(raw_ess) < 0.0 or float(raw_ess) > 1.0:
        clamp_events = 1
        logger.warning(
            "P3O: normalized ESS %.12f outside [0, 1]; clamping round-off (S1=%.6f, S2=%.6f, N=%.0f)",
            float(raw_ess),
            float(sum_ratio),
            float(sum_ratio_sq),
            float(count),
        )
    ess = raw_ess.clamp(min=0.0, max=1.0)

    ratio_mean = sum_ratio / count
    variance = (sum_ratio_sq / count) - ratio_mean.pow(2)
    ratio_std = variance.clamp(min=0.0).sqrt()

    return P3OStepContext(
        normalized_ess=ess,
        adaptive_cap=ess.clone(),
        valid_token_count=count,
        ratio_mean=ratio_mean,
        ratio_std=ratio_std,
        clamp_events=clamp_events,
    )


def compute_p3o_behavior_kl_proxy(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Sampled-token k3 proxy for ``KL(pi_theta || pi_b)``.

    ``K_i = l_i + exp(clip(-l_i, -C, C)) - 1`` with ``l_i`` the log ratio and
    ``C = BEHAVIOR_KL_EXP_CLAMP`` (currently 10). When ``|l_i| > C`` the
    exponent saturates: for ``l_i > C`` the exp term floors at ``exp(-C)`` so
    the gradient of the kl term w.r.t. ``log_probs`` approaches 1 (only the
    ``l_i`` addend contributes); for ``l_i < -C`` it caps at ``exp(C)``
    preventing numerical overflow.
    Gradient flows through ``log_probs``, which is what makes this an adaptive
    trust region rather than a diagnostic.

    This is a *proxy*: replay only stores the sampled token's log-prob, so the
    full-vocabulary KL of the paper is not recoverable here. Do not report it as
    the exact paper quantity.

    Args:
        log_probs: Current-policy log-probs of the sampled tokens.
        behavior_log_probs: Behavior-policy (rollout) log-probs, detached.
        valid_mask: Boolean mask selecting valid response tokens.

    Returns:
        Element-wise KL proxy, zero at invalid positions.
    """
    mask_bool = valid_mask.bool()
    log_ratio = compute_p3o_log_ratio(log_probs, behavior_log_probs, mask_bool)
    exponent = torch.clamp(-log_ratio, min=-BEHAVIOR_KL_EXP_CLAMP, max=BEHAVIOR_KL_EXP_CLAMP)
    kl = log_ratio + torch.exp(exponent) - 1.0
    return torch.where(mask_bool, kl, torch.zeros_like(kl))


def compute_p3o_token_terms(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    step_context: P3OStepContext,
) -> P3OTokenTerms:
    """Compute the element-wise P3O loss terms for one micro-batch.

    The score-function term is ``-sg(min(rho_i, cap)) * log pi_theta * sg(A_i)``.
    The *entire* ``min(rho, cap)`` factor is detached, not just the cap: P3O is a
    REINFORCE-style update whose only gradient path is ``log_probs``. There is no
    lower cap and no advantage-sign-dependent branch, so ``eps_clip`` plays no
    part in the objective.

    Args:
        log_probs: Current-policy log-probs of the sampled tokens (gradient
            source).
        behavior_log_probs: Behavior-policy (rollout) log-probs.
        advantages: GRPO group-relative advantages broadcast to response tokens.
        valid_mask: Boolean mask selecting valid response tokens.
        step_context: Frozen context carrying this optimizer step's global cap.

    Returns:
        :class:`P3OTokenTerms` with no reduction applied.
    """
    mask_bool = valid_mask.bool()
    behavior_log_probs = behavior_log_probs.detach()
    cap = step_context.adaptive_cap.to(dtype=torch.float32, device=log_probs.device)
    ess = step_context.normalized_ess.to(dtype=torch.float32, device=log_probs.device)

    with torch.no_grad():
        log_ratio_detached = compute_p3o_log_ratio(log_probs.detach(), behavior_log_probs, mask_bool)
        ratio = torch.exp(log_ratio_detached)
        ratio = torch.where(mask_bool, ratio, torch.zeros_like(ratio))
        # Full stop-gradient on min(ratio, cap): the coefficient must not
        # contribute a gradient path of its own.
        # Keep the cap on device. Converting it with ``float(cap)`` would add a
        # GPU-to-CPU synchronization in every training micro-batch.
        coefficient = torch.minimum(ratio, cap)
        cap_hits = (mask_bool & (ratio > cap)).to(dtype=torch.float32)

    score_loss = -(coefficient * log_probs.float() * advantages.detach().float())
    score_loss = torch.where(mask_bool, score_loss, torch.zeros_like(score_loss))

    behavior_kl_proxy = compute_p3o_behavior_kl_proxy(log_probs, behavior_log_probs, mask_bool)
    adaptive_kl_loss = (1.0 - ess) * behavior_kl_proxy

    return P3OTokenTerms(
        ratio=ratio,
        score_loss=score_loss,
        behavior_kl_proxy=behavior_kl_proxy,
        adaptive_kl_loss=adaptive_kl_loss,
        cap_hits=cap_hits,
    )
