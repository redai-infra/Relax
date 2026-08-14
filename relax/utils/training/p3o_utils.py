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
collectives and lifecycle live in the Megatron backend. The same sufficient
statistics support both paper-compatible micro-batch ESS and Relax's optional
optimizer-step ESS.
"""

from dataclasses import dataclass

import torch


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
    "P3O: non-finite importance ratio or unrepresentable squared ratio at a valid response token; refusing to "
    "silently fall back to ESS=1. Check rollout log-probs and mask alignment."
)


def _require_identical_shapes(**tensors: torch.Tensor) -> None:
    """Reject broadcasting between token-aligned P3O inputs."""
    shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
    if len(set(shapes.values())) != 1:
        formatted = ", ".join(f"{name}={shape}" for name, shape in shapes.items())
        raise ValueError(f"P3O token tensors must have identical shapes; got {formatted}")


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
        if vector.numel() != 3:
            raise ValueError(f"expected a 3-element stat vector, got shape {tuple(vector.shape)}")
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
        clamp_events: Compatibility field. ESS is clamped on-device without a
            host synchronization, so this remains zero.
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
        clip_hits: 1.0 where ``rho_i`` is outside the monitoring interval.
    """

    ratio: torch.Tensor
    score_loss: torch.Tensor
    behavior_kl_proxy: torch.Tensor
    adaptive_kl_loss: torch.Tensor
    cap_hits: torch.Tensor
    clip_hits: torch.Tensor


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
    _require_identical_shapes(
        log_probs=log_probs,
        behavior_log_probs=behavior_log_probs,
        valid_mask=valid_mask,
    )
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
        ValueError: If a valid position produced a non-finite ratio or a
            squared ratio that overflowed or underflowed in float64.
    """
    stats, invalid_flag = compute_p3o_sufficient_stats_unchecked(log_probs, behavior_log_probs, valid_mask)
    # This convenience wrapper is used outside the micro-batch hot path, so an
    # eager host check gives callers a deterministic error. Training uses the
    # unchecked variant and reduces the device flag with the ESS moments.
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
        position produced a non-finite ratio or a squared ratio that cannot be
        represented in float64, else ``0.0``. When it is set, the statistics
        are zeroed so a caller that defers the check cannot poison ``S1/S2``
        with ``inf``/``nan`` in the meantime.
    """
    with torch.no_grad():
        mask_bool = valid_mask.bool()
        log_ratio = compute_p3o_log_ratio(log_probs.detach(), behavior_log_probs.detach(), mask_bool)

        ratio = torch.exp(log_ratio.to(torch.float64))
        ratio = torch.where(mask_bool, ratio, torch.zeros_like(ratio))
        ratio_sq = ratio.square()

        # All checks stay on device. A finite ratio can still overflow when
        # squared (for example exp(500)), while a very small finite ratio can
        # produce an exact-zero square (for example exp(-500)). Either would
        # corrupt S2 and must fail through the synchronized invalid flag. Zero
        # squares outside the valid-token mask remain harmless.
        ratio_sq_representable = torch.where(mask_bool, ratio_sq > 0.0, torch.ones_like(mask_bool)).all()
        token_invalid = (
            ~(
                torch.isfinite(log_ratio).all()
                & torch.isfinite(ratio).all()
                & torch.isfinite(ratio_sq).all()
                & ratio_sq_representable
            )
        ).to(torch.float64)

        # Per-token finiteness is not sufficient: summing several large but
        # finite squares can overflow locally before the collective.
        local_sums = torch.stack(
            (
                ratio.sum(),
                ratio_sq.sum(),
                mask_bool.sum().to(torch.float64),
            )
        )
        invalid_flag = torch.maximum(token_invalid, (~torch.isfinite(local_sums).all()).to(torch.float64))

        # Zero the contribution when invalid, so deferring the host-side check
        # cannot let inf/nan reach the reduced moments. torch.where is required
        # here: multiplying an infinite S2 by zero would still produce NaN.
        keep = invalid_flag <= 0.0
        safe_local_sums = torch.where(keep, local_sums, torch.zeros_like(local_sums))
        return (
            P3OSufficientStats(
                sum_ratio=safe_local_sums[0],
                sum_ratio_sq=safe_local_sums[1],
                valid_token_count=safe_local_sums[2],
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

    Non-finite statistics and an empty valid-token set use the reference
    implementation's neutral fallback: ``ESS=cap=1``, ratio mean 1 and ratio
    std 0. This stays device-resident and does not synchronize a CUDA hot path.
    """
    sum_ratio = stats.sum_ratio.to(torch.float64)
    sum_ratio_sq = stats.sum_ratio_sq.to(torch.float64)
    count = stats.valid_token_count.to(torch.float64)

    valid = torch.stack((sum_ratio, sum_ratio_sq, count)).isfinite().all() & (count >= 0.5)
    one = torch.ones((), dtype=torch.float64, device=count.device)
    zero = torch.zeros((), dtype=torch.float64, device=count.device)
    safe_sum_ratio = torch.where(valid, sum_ratio, one)
    safe_sum_ratio_sq = torch.where(valid, sum_ratio_sq, one)
    safe_count = torch.where(valid, count, one)

    raw_ess = safe_sum_ratio.pow(2) / (safe_count * (safe_sum_ratio_sq + ESS_DENOM_EPS))
    ess = torch.where(valid, raw_ess.clamp(min=0.0, max=1.0), one)

    ratio_mean = torch.where(valid, safe_sum_ratio / safe_count, one)
    variance = (safe_sum_ratio_sq / safe_count) - ratio_mean.pow(2)
    ratio_std = torch.where(valid, variance.clamp(min=0.0).sqrt(), zero)
    valid_token_count = torch.where(torch.isfinite(count) & (count >= 0.0), count, zero)

    return P3OStepContext(
        normalized_ess=ess,
        adaptive_cap=ess.clone(),
        valid_token_count=valid_token_count,
        ratio_mean=ratio_mean,
        ratio_std=ratio_std,
        clamp_events=0,
    )


class _P3OProxySafeK3(torch.autograd.Function):
    """FeynRL k3 forward with a bounded, sign-correct extreme backward."""

    @staticmethod
    def forward(ctx, log_ratio: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(log_ratio)
        exponent = torch.clamp(-log_ratio, min=-BEHAVIOR_KL_EXP_CLAMP, max=BEHAVIOR_KL_EXP_CLAMP)
        return log_ratio + torch.exp(exponent) - 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (log_ratio,) = ctx.saved_tensors
        exponent = torch.clamp(-log_ratio, min=-BEHAVIOR_KL_EXP_CLAMP, max=BEHAVIOR_KL_EXP_CLAMP)
        gradient = 1.0 - torch.exp(exponent)
        return (grad_output * gradient,)


def compute_p3o_behavior_kl_proxy(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str = "proxy",
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

        mode: ``proxy`` preserves the FeynRL autograd behavior. ``proxy_safe``
            preserves the exact forward values but corrects the saturated
            negative-log-ratio gradient direction.

    Returns:
        Element-wise KL proxy, zero at invalid positions.
    """
    if mode not in {"proxy", "proxy_safe"}:
        raise ValueError(f"P3O sampled-token KL mode must be proxy or proxy_safe, got {mode!r}")
    mask_bool = valid_mask.bool()
    log_ratio = compute_p3o_log_ratio(log_probs, behavior_log_probs, mask_bool)
    if mode == "proxy_safe":
        kl = _P3OProxySafeK3.apply(log_ratio)
    else:
        exponent = torch.clamp(-log_ratio, min=-BEHAVIOR_KL_EXP_CLAMP, max=BEHAVIOR_KL_EXP_CLAMP)
        kl = log_ratio + torch.exp(exponent) - 1.0
    return torch.where(mask_bool, kl, torch.zeros_like(kl))


def compute_p3o_exact_kl(
    policy_logits: torch.Tensor,
    behavior_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the exact forward KL over a full vocabulary.

    This pure helper is intended for small-vocabulary verification and for a
    future training path that carries behavior logits. Production rollout data
    currently stores only selected-token log-probs, so the loss integration
    rejects ``exact`` mode with an explicit error.
    """
    if policy_logits.shape != behavior_logits.shape:
        raise ValueError(
            "P3O exact-KL logits must have identical shapes; "
            f"got policy={tuple(policy_logits.shape)}, behavior={tuple(behavior_logits.shape)}"
        )
    if policy_logits.ndim < 1 or tuple(policy_logits.shape[:-1]) != tuple(valid_mask.shape):
        raise ValueError(
            "P3O exact-KL valid_mask must match the logits token dimensions; "
            f"got logits={tuple(policy_logits.shape)}, mask={tuple(valid_mask.shape)}"
        )

    mask_bool = valid_mask.bool()
    expanded_mask = mask_bool.unsqueeze(-1)
    safe_policy_logits = torch.where(expanded_mask, policy_logits.float(), torch.zeros_like(policy_logits.float()))
    safe_behavior_logits = torch.where(
        expanded_mask,
        behavior_logits.detach().float(),
        torch.zeros_like(behavior_logits.detach().float()),
    )
    policy_log_probs = torch.log_softmax(safe_policy_logits, dim=-1)
    behavior_log_probs = torch.log_softmax(safe_behavior_logits, dim=-1)
    exact_kl = (policy_log_probs.exp() * (policy_log_probs - behavior_log_probs)).sum(dim=-1)
    return torch.where(mask_bool, exact_kl, torch.zeros_like(exact_kl))


def compute_p3o_token_terms(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    step_context: P3OStepContext,
    kl_mode: str = "proxy",
    clip_low: float = 0.2,
    clip_high: float = 0.2,
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
        kl_mode: Sampled-token behavioral KL implementation. ``exact`` is
            rejected because this function receives no behavior logits.
        clip_low: Lower monitoring margin around ratio 1.
        clip_high: Upper monitoring margin around ratio 1.

    Returns:
        :class:`P3OTokenTerms` with no reduction applied.
    """
    if kl_mode == "exact":
        raise ValueError(
            "P3O exact KL requires full-vocabulary behavior logits; rollout data currently stores only "
            "selected-token log-probs. Use proxy/proxy_safe for training."
        )
    if clip_low < 0.0 or clip_high < 0.0:
        raise ValueError(f"P3O clip monitoring margins must be non-negative, got {clip_low}, {clip_high}")

    _require_identical_shapes(
        log_probs=log_probs,
        behavior_log_probs=behavior_log_probs,
        advantages=advantages,
        valid_mask=valid_mask,
    )
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
        clip_hits = (mask_bool & ((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high))).to(dtype=torch.float32)

    score_loss = -(coefficient * log_probs.float() * advantages.detach().float())
    score_loss = torch.where(mask_bool, score_loss, torch.zeros_like(score_loss))

    behavior_kl_proxy = compute_p3o_behavior_kl_proxy(log_probs, behavior_log_probs, mask_bool, mode=kl_mode)
    adaptive_kl_loss = (1.0 - ess) * behavior_kl_proxy

    return P3OTokenTerms(
        ratio=ratio,
        score_loss=score_loss,
        behavior_kl_proxy=behavior_kl_proxy,
        adaptive_kl_loss=adaptive_kl_loss,
        cap_hits=cap_hits,
        clip_hits=clip_hits,
    )
