# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward normalisation strategies, one per algorithm family.

These run on the rollout side (CPU) from
``relax.utils.utils.post_process_rewards``.  Every normaliser takes the raw
per-sample scalar rewards and returns the per-sample scalars written into the
TransferQueue ``rewards`` column, so adding a strategy never changes the data
schema — even for multi-reward algorithms, which collapse their components to a
single scalar here.
"""

import math
from numbers import Real
from typing import Any, Callable

import torch

from relax.algorithms.numerics import GDPO_EPS, STD_EPS, collapsed_columns
from relax.algorithms.spec import get_algorithm
from relax.utils.logging_utils import get_logger
from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards


logger = get_logger(__name__)

GROUP_EPS = STD_EPS

_FLOAT32_MAX = float(torch.finfo(torch.float32).max)
"""Largest finite float32. Rewards are carried as float32 from here on, so a
value above this is an overflow waiting to happen rather than a large reward."""

_NOISE_FLOOR_SAFETY = 8.0
"""Headroom on :func:`component_noise_scale`'s first-order error bound.

The bound is an estimate, not a proof, so it gets an order of magnitude. It can
afford to: on well-conditioned rewards the bound sits fifteen orders of
magnitude below the signal, so any value in this neighbourhood suppresses
exactly the same things."""


def group_positions(samples: list[Any], expected_size: int) -> dict[int, list[int]]:
    """Map ``Sample.group_index`` to the positions it occupies in ``samples``.

    Grouping follows ``group_index`` rather than position, so how the caller
    orders the batch does not affect the result.
    """
    positions_by_group: dict[int, list[int]] = {}
    for position, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("Sample.group_index is required for group reward normalization.")
        if sample.group_index not in positions_by_group:
            positions_by_group[sample.group_index] = []
        positions_by_group[sample.group_index].append(position)

    for group_index, positions in positions_by_group.items():
        if len(positions) != expected_size:
            raise ValueError(f"Reward group {group_index} has {len(positions)} samples, expected {expected_size}.")
    return positions_by_group


def _group_normalize(args: Any, samples: list[Any], raw_rewards: list[float], *, use_std: bool) -> list[float]:
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)

    normalized_rewards = torch.empty_like(rewards)
    for positions in positions_by_group.values():
        group_rewards = rewards[positions]
        group_rewards = group_rewards - group_rewards.mean()
        if use_std:
            group_rewards = group_rewards / (group_rewards.std() + GROUP_EPS)
        normalized_rewards[positions] = group_rewards

    return normalized_rewards.tolist()


def normalize_none(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """No normalisation — the estimator consumes raw rewards (REINFORCE++,
    PPO)."""
    return raw_rewards


def normalize_group_mean(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """Subtract the group mean only (REINFORCE++ baseline)."""
    return _group_normalize(args, samples, raw_rewards, use_std=False)


def normalize_group_mean_std(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """Subtract the group mean, then optionally divide by the group std.

    ``--disable-grpo-std-normalization`` (Dr.GRPO) turns the division off.
    """
    return _group_normalize(args, samples, raw_rewards, use_std=args.grpo_std_normalization)


def _reject_non_finite_group(group_index: int, positions: list[int], group_rewards: torch.Tensor) -> None:
    """Raise if any reward in one group is NaN or infinite, naming the sample.

    The other normalisers do not check: an infinite reward there poisons only
    the sample that carries it (``group_mean_std`` divides it away, and the
    caller sees one bad advantage). A leave-one-out baseline averages the
    *other* samples, so one bad value silently propagates into every advantage
    in the group -- which is why this reports the offender's position instead
    of letting the arithmetic swallow it.
    """
    finite_mask = torch.isfinite(group_rewards)
    if finite_mask.all():
        return
    invalid_group_positions = (~finite_mask).nonzero(as_tuple=False).flatten().tolist()
    invalid_sample_positions = [positions[position] for position in invalid_group_positions]
    invalid_values = group_rewards[~finite_mask].tolist()
    raise ValueError(
        f"RLOO group_index={group_index} contains non-finite reward(s) at "
        f"group position(s) {invalid_group_positions}, sample position(s) "
        f"{invalid_sample_positions}: {invalid_values}."
    )


def normalize_group_leave_one_out(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """RLOO's leave-one-out baseline (Ahmadian et al. 2024, arXiv:2402.14740).

    Each sample is centred on the mean of the *others* in its prompt group::

        A_i = r_i - mean_{j != i}(r_j) = G / (G - 1) * (r_i - mean(r))

    Unlike :func:`normalize_group_mean_std` there is no division by the group
    standard deviation, so the advantage keeps the reward's scale. The
    right-hand identity is what
    :func:`~relax.utils.training.ppo_utils.compute_rloo_leave_one_out_rewards`
    computes; that helper also owns the ``G >= 2`` and finiteness contracts,
    and it is shared with the rollout diagnostics so the numbers reported and
    the numbers trained on cannot drift apart.
    """
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)

    normalized_rewards = torch.empty_like(rewards)
    for group_index, positions in positions_by_group.items():
        group_rewards = rewards[positions]
        _reject_non_finite_group(group_index, positions, group_rewards)
        normalized_rewards[positions] = compute_rloo_leave_one_out_rewards(group_rewards)

    return normalized_rewards.tolist()

def resolve_gdpo_keys(args: Any) -> list[str]:
    """The reward components GDPO normalises independently."""
    keys = list(getattr(args, "gdpo_reward_keys", None) or [])
    if len(keys) < 2:
        raise ValueError(f"--gdpo-reward-keys needs at least two reward keys, got {keys}.")
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise ValueError(f"--gdpo-reward-keys contains duplicates: {sorted(duplicates)}.")
    return keys


def resolve_gdpo_weights(args: Any, keys: list[str]) -> list[float]:
    """Per-component weights, defaulting to 1.0 each.

    The weights multiply the *normalised* advantages (arXiv 2601.05242, Eq. 7),
    not the raw rewards.  That is the point: after step 1 every component is on
    the same scale, so a weight expresses relative importance rather than
    accidentally encoding the component's units.
    """
    weights = getattr(args, "gdpo_reward_weights", None)
    if weights is None:
        return [1.0] * len(keys)
    if len(weights) != len(keys):
        raise ValueError(f"--gdpo-reward-weights has {len(weights)} entries but --gdpo-reward-keys has {len(keys)}.")
    resolved = [float(w) for w in weights]
    for key, weight in zip(keys, resolved, strict=True):
        # argparse happily parses "nan" and "inf" for a float option. Unchecked,
        # the weighted sum turns non-finite, whiten_scalar reads a non-finite std
        # as a collapse, and the batch silently produces zero advantages.
        if not math.isfinite(weight):
            raise ValueError(f"--gdpo-reward-weights for {key!r} is {weight}, which is not finite.")
    if all(weight == 0.0 for weight in resolved):
        # Every component gets multiplied by zero, so the combined advantage is
        # identically zero: the run trains on no signal and still exits cleanly.
        # The same shape of failure as the non-finite case above, one line later.
        raise ValueError(f"--gdpo-reward-weights are all zero ({resolved}); the combined advantage would be 0.")
    return resolved


def extract_reward_components(samples: list[Any], keys: list[str]) -> torch.Tensor:
    """Build the ``[B, K]`` component matrix, rejecting malformed rewards.

    Contract violations raise instead of defaulting to 0.0.  A silently zeroed
    component is indistinguishable from a genuinely collapsed one, so falling
    back would hide a broken reward function behind plausible-looking training.
    """
    rows: list[list[float]] = []
    for position, sample in enumerate(samples):
        values = sample.get_reward_components(keys)
        row: list[float] = []
        for key, value in zip(keys, values, strict=True):
            # `numbers.Real` rather than `(int, float)`: reward functions routinely
            # return numpy scalars, and only np.float64 happens to subclass float —
            # np.float32/np.int64 would be rejected while np.float64 sailed through.
            # bool and np.bool_ stay rejected (bool subclasses int; np.bool_ is not
            # a Real), because a boolean reward is almost always a mistake.
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"Reward {key!r} of sample {position} must be a real number, "
                    f"got {value!r} ({type(value).__name__})."
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Reward {key!r} of sample {position} is {numeric}, which is not finite.")
            # Finite in float64 is not enough: the tensor below is float32, whose
            # largest value is ~3.4e38. A reward of 1e300 passes `isfinite` here,
            # becomes `inf` on cast, and then reads as a non-finite std one stage
            # later -- where the batch is zeroed and the run trains on no signal
            # without ever failing. Catching the overflow at the boundary keeps
            # the diagnosis at the reward function that produced it.
            if abs(numeric) > _FLOAT32_MAX:
                raise ValueError(
                    f"Reward {key!r} of sample {position} is {numeric!r}, which overflows float32 "
                    f"(max {_FLOAT32_MAX:.6g}). Rescale the reward; casting it would silently produce inf."
                )
            row.append(numeric)
        rows.append(row)

    # float64, not float32. The rewards arrive as Python floats and the
    # standardisation that follows is ill-conditioned exactly when two
    # components cancel: `C - r` loses the low bits on a float32 cast, and
    # dividing what is left by a near-zero std turns those lost bits into a
    # full-scale advantage. Keeping the width here is what makes the residue
    # small enough for `combine_group`'s noise floor to be able to tell signal
    # from rounding at all -- in float32 the residue is larger than the signal.
    # The float32 range check above still stands: the combined reward is cast
    # back down further along, so a value that cannot survive that cast is
    # still a bug worth reporting at the reward function that produced it.
    components = torch.tensor(rows, dtype=torch.float64)
    # Belt and braces: the per-value check above is exact, but it only sees what
    # `float()` returned. Anything that slips past it must not reach the
    # normaliser, where non-finite input is indistinguishable from a collapse.
    if not torch.isfinite(components).all():
        bad = (~torch.isfinite(components)).nonzero()[0].tolist()
        raise ValueError(f"Reward {keys[bad[1]]!r} of sample {bad[0]} is not finite as a tensor.")
    return components


def standardize_group_components(group: torch.Tensor) -> torch.Tensor:
    """GDPO step 1 on one ``[G, K]`` prompt group (arXiv 2601.05242, Eq. 4).

    Each column is standardised on its own. A column that is flat across the
    group contributes exactly zero rather than ``0 / (0 + eps)`` noise, which
    is what lets the remaining columns keep their signal.

    The arithmetic runs in float64, for the same reason
    :func:`relax.algorithms.numerics.distributed_mean_std` does: the means and
    stds are the part where cancellation bites, and widening them is cheap.

    Standardising is ill-conditioned when a column barely varies: it divides
    by a std that is near zero, so the *relative* error in the result grows
    like ``max|x| / std``. That is not a corner case here -- it is precisely
    what two columns summing to a constant look like, and it is why
    :func:`component_noise_scale` exists and why the columns now arrive in
    float64. See :func:`combine_group` for what is done with the estimate.
    """
    work = group.double()
    centered = work - work.mean(dim=0, keepdim=True)
    std = work.std(dim=0)
    collapsed = collapsed_columns(group, dim=0)
    scaled = centered / (std + GDPO_EPS)
    scaled = torch.where(collapsed.unsqueeze(0), torch.zeros_like(scaled), scaled)
    return scaled.to(group.dtype)


def component_noise_scale(group: torch.Tensor) -> torch.Tensor:
    """Per-column bound on how much of :func:`standardize_group_components` is
    rounding.

    Returns one value per column: the magnitude below which that column's
    standardised values are indistinguishable from the arithmetic that
    produced them.

    The bound is the condition number of the standardisation. Each input
    carries a relative representation error of about ``eps``, i.e. an absolute
    error of ``eps * max|x|``. Dividing by ``std + GDPO_EPS`` scales that error
    up by the same factor it scales the signal, so the error in the output is
    ``eps * max|x| / (std + GDPO_EPS)``.

    Two properties make this usable rather than a tuning knob:

    * On a well-conditioned column it is negligible. Rewards around 1.0 with a
      std of 0.1 give ``2.2e-16 * 1 / 0.1 ~ 2e-15`` against signal of order 1
      -- fifteen orders of margin, so nothing real is ever suppressed.
    * On the pathological column it matches what actually happens. For the
      constant-sum group in ``test_gdpo.py`` the bound comes to ~5.1e-10 and
      the residue measures 4.1e-10.

    A collapsed column contributes exactly zero (not ``0/eps`` noise), so it
    contributes no error either.
    """
    work = group.double()
    eps = torch.finfo(work.dtype).eps
    std = work.std(dim=0)
    scale = work.abs().amax(dim=0)
    noise = eps * scale / (std + GDPO_EPS)
    return torch.where(collapsed_columns(group, dim=0), torch.zeros_like(noise), noise)


def combine_group(group: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """GDPO steps 1 and 2 for one ``[G, K]`` prompt group, with a noise floor.

    Two components summing to a constant should standardise to exact opposites
    and cancel. What is left instead is the rounding error of an
    ill-conditioned division, and it does not stay small: step 3 divides by the
    std of that very residue.

    Two separate things keep that from becoming a gradient, and it is worth
    being exact about which does what, because the tempting summary -- "the
    floor stops a full-scale fake advantage" -- is not true:

    * **The float64 columns do the heavy lifting.** In float32 the residue was
      of order 1e-1, larger than the signal itself, and step 3 returned
      advantages of order 1. In float64 it is of order 1e-10, and ``GDPO_EPS``
      in step 3's denominator caps the amplification, so the worst case is
      already down to ~1e-6. No floor is needed to avoid a fake gradient.
    * **The floor makes the cancellation exactly detectable.** 1e-6 is not
      zero, and several consumers test for zero:
      :func:`group_carries_reward_signal` would keep a group contributing
      nothing, and the all-groups-silent warning would never fire. Rounding a
      residue to the zero it mathematically is turns a "too small to matter"
      into a "recognisably absent".

    The comparison is per group and all-or-nothing, matching how a collapsed
    column is already handled. ``_NOISE_FLOOR_SAFETY`` is the one judgement
    call; given the fifteen orders of margin on well-conditioned input, its
    exact value changes nothing that is not already noise.
    """
    standardized = standardize_group_components(group)
    combined = (standardized.double() * weights.double()).sum(dim=1)

    floor = _NOISE_FLOOR_SAFETY * float((weights.double().abs() * component_noise_scale(group)).sum())
    if float(combined.abs().amax()) <= floor:
        return torch.zeros_like(combined)
    return combined


def normalize_gdpo_decoupled(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """GDPO steps 1 and 2 (arXiv 2601.05242, Eq. 4 and Eq. 7).

    Step 1 standardises each reward component within its prompt group; step 2
    combines them with the configured weights.  The result is one scalar per
    sample, so it travels through the existing ``rewards`` column and needs no
    TransferQueue schema change.  Step 3 (batch whitening) runs later, in
    :func:`relax.algorithms.advantages.advantage_gdpo`.

    What standardising per component actually buys, stated carefully because
    it is easy to overclaim in two opposite directions (this docstring has
    managed both):

    The combined advantage is ``sum_k w_k * z_k`` with each ``z_k`` at unit
    variance, so its variance is ``sum_k w_k^2 + 2 sum_{i<j} w_i w_j rho_ij``
    -- ``2 + 2*rho`` for two equal weights. GRPO standardises ``sum_k r_k``
    instead and lands on unit variance for every group regardless. So what
    GDPO preserves is the **correlation structure**: corroborating components
    (``rho -> +1``) amplify, contradicting ones (``rho -> -1``) attenuate and
    at the limit cancel.

    Both claims this file got wrong earlier are special cases of that one
    formula. 'Equal sums are rescued by GDPO' is ``rho = -1`` -- they cancel,
    GDPO included. 'Two varying components mean a stronger signal' is
    ``rho = +1`` -- true there, false at ``rho = -0.8``, where the combination
    measures 0.63x a single component.

    Separately and unconditionally: GRPO's direction is dominated by whichever
    component has the largest spread, because it standardises the raw sum. A
    correctness reward in ``{0, 1}`` added to a length reward in the hundreds
    is decided almost entirely by length under GRPO, and half by each here.
    """
    keys = resolve_gdpo_keys(args)
    weights = resolve_gdpo_weights(args, keys)

    components = extract_reward_components(samples, keys)
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if not torch.isfinite(weight_tensor).all():
        # `resolve_gdpo_weights` checked `math.isfinite` on the Python floats.
        # That is a different question: 1e300 is finite in float64 and `inf`
        # once it lands in this tensor.
        raise ValueError(
            f"--gdpo-reward-weights {weights} contains a value that overflows float32; "
            f"the largest representable is {_FLOAT32_MAX:.6g}."
        )
    if float(weight_tensor.abs().sum()) == 0.0:
        # Same gap in the other direction: 1e-50 is a nonzero Python float that
        # flushes to zero in float32, so a weight vector that passed the
        # "not all zero" check can still be all zeros by the time it multiplies.
        raise ValueError(
            f"--gdpo-reward-weights {weights} are all zero once cast to float32; "
            "the combined advantage would be identically 0."
        )

    combined = torch.zeros(len(samples), dtype=torch.float64)
    silent_groups = 0
    for positions in positions_by_group.values():
        combined[positions] = combine_group(components[positions], weight_tensor)
        if not bool(combined[positions].any()):
            silent_groups += 1

    if silent_groups == len(positions_by_group):
        # Every group came out at exactly zero, so this batch produces no
        # gradient at all. Usually the reward function is constant for these
        # prompts (e.g. a format reward when nothing in the prompt asks for a
        # format); the other way in is components that cancel. Worth one line,
        # because the symptom downstream is simply "loss does not move".
        logger.warning(
            "GDPO: all %d groups of this batch combined to exactly zero (keys=%s); the batch "
            "contributes no gradient. Either these rewards do not vary across rollouts of the "
            "same prompt, or they cancel under the configured weights (weights=%s).",
            silent_groups,
            keys,
            weights,
        )

    if not torch.isfinite(combined).all():
        # Every individual reward fit in float32 (checked in
        # `extract_reward_components`), but the arithmetic between them need
        # not. The mean itself no longer overflows -- the columns and the
        # standardisation are float64 now, so [3e38, 2e38, 1e38, 0] averages
        # fine, and the earlier version of this comment claiming otherwise is
        # out of date. What can still reach infinity is the weighting: float64
        # weights near its own maximum, or a reward near float64's range rather
        # than float32's. Left unchecked the `inf` reaches `whiten_scalar`,
        # reads as a non-finite std, and the batch is silently zeroed -- the
        # exact failure the per-value check exists to prevent, one stage later.
        raise ValueError(
            "GDPO produced a non-finite combined advantage from finite inputs; the "
            f"arithmetic overflowed. Rescale the rewards (keys={keys}) or their weights."
        )
    return combined.tolist()


def group_carries_reward_signal(args: Any, samples: list[Any]) -> bool:
    """Whether one prompt group still carries signal for *this* algorithm.

    Consumers upstream of the advantage stage -- the dynamic-sampling filter,
    the zero-std metrics -- ask "did this group vary?" to decide whether it is
    worth keeping or worth reporting. They have always answered it with the
    single scalar ``--reward-key`` selects, which is the right question only
    for an algorithm that consumes that scalar.

    For a multi-reward algorithm the honest test is not "did any component
    vary" but "is the combined advantage this algorithm will actually compute
    non-zero". Those differ: a zero weight mutes a varying component, and two
    components whose standardised values are exact opposites cancel. Asking
    the cheaper question keeps groups that then contribute no gradient, and
    under-reports the zero-std metrics. Steps 1 and 2 are per-group and cheap,
    so this runs them.

    The single-reward test is deliberately performed **in float32**, on a
    tensor, rather than on the Python floats. That is the precision the reward
    actually reaches training in (:func:`_group_normalize` casts to float32,
    and so does the GDPO path), so it is the precision that decides whether a
    group will still carry signal by the time it matters.

    Comparing the float64 values instead is not a harmless tightening: it flips
    the verdict for any group whose spread survives in float64 but collapses on
    cast -- ``[0.1 + 0.2, 0.3, ...]`` is the everyday example, and a group of
    NaNs is the pathological one (``nan != nan`` reads as signal). Both would
    be kept and then contribute a zero gradient. Judging on the tensor keeps
    this identical to the ``std > 0`` test it replaces: ``min == max`` and
    ``std == 0`` agree on every float32 input.
    """
    if not samples:
        return False

    spec = get_algorithm(args.advantage_estimator)
    if not spec.uses_reward_components:
        rewards = torch.tensor([sample.get_reward_value(args) for sample in samples], dtype=torch.float32)
        if not torch.isfinite(rewards).all():
            # A group containing NaN or inf reads as "varying" under any
            # inequality test (`nan != nan` is True), which would forward a
            # broken reward into training. `std > 0` answered False here
            # because the std is itself NaN, so False preserves that. Raising
            # would arguably be better -- a non-finite reward is a bug in the
            # reward function, not a property of the group -- but that is a
            # behaviour change this refactor is not the place for.
            return False
        return bool(rewards.amin() != rewards.amax())

    keys = resolve_gdpo_keys(args)
    weights = torch.tensor(resolve_gdpo_weights(args, keys), dtype=torch.float32)
    components = extract_reward_components(samples, keys)
    # `combine_group`, not steps 1 and 2 open-coded: this has to answer the
    # same question `normalize_gdpo_decoupled` will, including the noise floor.
    # Open-coding it is how the two came apart before.
    return bool(combine_group(components, weights).any())


def observed_reward_signal(args: Any, samples: list[Any]) -> bool | None:
    """:func:`group_carries_reward_signal` for callers that only report.

    Returns ``None`` -- "cannot tell" -- where the strict version raises.

    This is not the strict check with its errors swallowed; it is a different
    question, and the difference is not cosmetic. The strict version is asked
    by the dynamic-sampling filter, which decides whether a group enters
    *training*, so a reward it cannot read is a bug that should stop the run.
    The zero-std metrics are asked by the logger, and two things follow:

    * A metric must not decide whether training proceeds. Making observation
      the first enforcer of a contract means a broken reward function is
      reported at the log line rather than at the stage that consumes it, and
      it takes the rollout down on the way.
    * The contract is not even in force everywhere the metrics run. Eval may
      use a different reward model (``EvalConfig.rm_type``), so an eval reward
      legitimately need not carry the ``--gdpo-reward-keys`` that training
      needs -- nothing in eval consumes them. The strict question has no
      correct answer there; ``None`` is the honest one.

    During training the same violation is still raised, by
    :func:`normalize_gdpo_decoupled`, which is the stage that actually reads
    the components. Nothing is hidden -- but it is logged here too, because a
    group the metrics could not read is worth knowing about either way.
    """
    try:
        return group_carries_reward_signal(args, samples)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "zero-std metrics: skipping a group whose reward could not be read (%s). "
            "For a training rollout the reward stage will raise on this; for eval it may just "
            "mean the eval reward model returns a different schema, which is fine.",
            exc,
        )
        return None


def metrics_group_verdict(args: Any, samples: list[Any]) -> bool | None:
    """Is this prompt group flat, for zero-std reporting? ``None`` if
    unanswerable.

    ``True`` flat, ``False`` varying, ``None`` neither -- nothing in the group
    was scored, or the reward could not be read.

    This lives here rather than in the two rollout modules because there are
    *two* copies of ``_compute_zero_std_metrics``, one in
    ``relax/agentic/rollout.py`` and one in ``relax/distributed/ray/rollout.py``,
    and they have already drifted apart once: the agentic one dropped unscored
    samples and the distributed one did not, which turned a `reward=None` into
    a ``TypeError`` on the rollout's way out. Only the agentic one is reachable
    from a CPU test -- the other pulls in ``sglang`` at import -- so a shared
    helper is the only version of this logic that can be tested at all.

    ``reward=None`` is a real state, not a defensive check: under ``--group-rm``
    the group reward is assigned in one shot that is skipped entirely when the
    rollout aborts, which is why the "reward is not None" assert in
    ``sglang_rollout.py`` exempts ``group_rm`` in the first place.

    Callers still differ on what to do with ``None``, and that difference is
    deliberate -- each preserves what its own metric reported before. See the
    call sites.
    """
    rewarded = [sample for sample in samples if sample.reward is not None]
    if not rewarded:
        return None
    signal = observed_reward_signal(args, rewarded)
    return None if signal is None else not signal


REWARD_NORMALIZERS: dict[str, Callable[[Any, list[Any], list[float]], list[float]]] = {
    "none": normalize_none,
    "group_mean": normalize_group_mean,
    "group_mean_std": normalize_group_mean_std,
    "group_leave_one_out": normalize_group_leave_one_out,
    "gdpo_decoupled": normalize_gdpo_decoupled,
}
