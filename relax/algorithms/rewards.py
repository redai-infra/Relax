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
        # the weighted sum turns non-finite and whiten_scalar refuses the batch.
        # It used to read a non-finite std as a collapse and return zeros
        # silently; catching the weight here still gives the better message.
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
    # full-scale advantage. Keeping the width here is the only thing that keeps
    # that residue below the signal at all: in float32 it is larger, and step 3
    # returns advantages of order 1 for a group that carries none.
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
    what two columns summing to a constant look like, and it is why the columns
    arrive here in float64: in float32 the residue of that cancellation was
    larger than the signal it was left over from. See :func:`combine_group` for
    the part of this that is still unsolved.
    """
    work = group.double()
    centered = work - work.mean(dim=0, keepdim=True)
    std = work.std(dim=0)
    collapsed = collapsed_columns(group, dim=0)
    scaled = centered / (std + GDPO_EPS)
    scaled = torch.where(collapsed.unsqueeze(0), torch.zeros_like(scaled), scaled)
    return scaled.to(group.dtype)


def combine_group(group: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """GDPO steps 1 and 2 for one ``[G, K]`` prompt group: ``sum_k w_k z_ik``.

    Eq. 7 and nothing else. Two things used to sit at the end of this function
    and both are gone; the reasons are worth keeping because both sounded good.

    **The noise floor** returned zeros when the result fell below
    ``8 * sum(|w_k| * noise_k)``. It is not in Eq. 7, and the tests written to
    document it proved it destroyed real signal: two components at
    ``base = 4.05e13``, every column measuring under 1% noise, gave a floor of
    0.148 against a true signal of 0.033. It grew as a sum over components
    while any per-column noise measure is a max, so sixteen clean columns
    reached 0.82.

    **Its replacement, a relative "is this rounding?" ratio**, lasted one round
    longer and was worse, because it was wrong in kind rather than in
    calibration. ``magnitude`` scales with the *difference* between the weights
    while ``sum_k |w_k| max|z_k|`` scales with their *magnitude*. Measured on
    real inputs it was inverted in both directions at once -- it discarded a
    group whose final advantage was 0.43 and kept one whose 1.08 was pure
    rounding.

    That is not fixable by moving the threshold. For ``G >= 3`` the centred
    subspace is at least two-dimensional, so ``z_2 = -z_1 + delta * u`` with
    ``u`` orthogonal to ``z_1`` and ``delta`` arbitrarily small is a genuine
    signal with an arbitrarily small ratio. No universal threshold separates
    "the components nearly cancel" from "the arithmetic nearly cancelled".

    ``G = 2`` does not settle it either way: the ratio reduces to
    ``|w1 - w2| / (|w1| + |w2|)``, independent of the data, *only* when the two
    standardised columns come out as exact opposites. Columns that move
    together give 1 for same-signed weights, which is fully data-dependent. The
    ``G >= 3`` construction above is what actually settles it.

    **What is left unsolved, stated plainly rather than argued away.** Two
    components summing to a constant do not cancel exactly, and the remainder
    grows with the magnitude they are centred on -- roughly ``ulp(C) / spread``
    for the Eq. 7 output. Nothing in this file detects it, and the two
    mechanisms tried so far could not: a conditioning bound cannot prove a
    small result is *not* signal, and screening on the post-whitening magnitude
    needs the whole batch, which a per-group function does not have.

    What reaches the optimizer, measured (``[C - d, d]`` for
    ``d in (0.1, 0.2, 0.3, 0.7)``, equal weights):

    ============  =========================  ===================================
    ``C``         its own whitening unit     sharing one with 8 healthy groups
    ============  =========================  ===================================
    ``1e8``       2.3e-4                     2.7e-8
    ``1e9``       2.6e-3                     3.1e-7
    ``1e11``      2.0e-1                     3.3e-5
    ``1e13``      1.2                        5.2e-3
    ============  =========================  ===================================

    The second column is not a bound. Sharing a whitening unit with healthy
    groups divides the remainder by *their* standard deviation -- a constant
    factor of a few hundred here. It does not slow the growth with ``C``, and
    at ``C = 1e13`` a mixed unit still delivers 5e-3.

    "Whitening unit" is also not "the rollout". :func:`~relax.algorithms.
    advantages._whiten_by_segment` whitens each *training batch* separately, so
    a degenerate group only gets that division if healthy groups land in the
    same training batch. Isolate it into its own segment and it returns to the
    first column -- 2.6e-3 at ``C = 1e9``. Nothing arranges that mixing; it is
    a property of how the rollout happened to be split.
    """
    standardized = standardize_group_components(group)
    return (standardized.double() * weights.double()).sum(dim=1)


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

    # float64, matching the components. The weights used to be cast to float32
    # here, one stage before the arithmetic needed it, and that quantisation was
    # observable: [16777216, 16777217] are two distinct configured weights that
    # become the same float32, so a pair of anti-correlated components that
    # should combine to about +-1 combined to exactly 0 instead. The cast to the
    # transport dtype belongs at the end of the pipeline, not at the start.
    weight_tensor = torch.tensor(weights, dtype=torch.float64)

    # The two checks below are still about float32, and deliberately: the
    # combined reward is carried to the trainer in float32 (`dict_to_tensordict`),
    # so a weight vector that cannot survive that cast produces a run that trains
    # on nothing. Asking here rather than after the multiplication is what lets
    # the message name the weights instead of the arithmetic.
    if not torch.isfinite(weight_tensor.float()).all():
        # `resolve_gdpo_weights` checked `math.isfinite` on the Python floats.
        # That is a different question: 1e300 is finite in float64 and `inf`
        # once it reaches the transport dtype.
        raise ValueError(
            f"--gdpo-reward-weights {weights} contains a value that overflows float32; "
            f"the largest representable is {_FLOAT32_MAX:.6g}."
        )
    if float(weight_tensor.float().abs().sum()) == 0.0:
        # Same gap in the other direction: 1e-50 is a nonzero Python float that
        # flushes to zero in float32, so a weight vector that passed the
        # "not all zero" check can still be all zeros by the time it matters.
        raise ValueError(
            f"--gdpo-reward-weights {weights} are all zero once cast to float32; "
            "the combined advantage would be identically 0."
        )

    combined = torch.zeros(len(samples), dtype=torch.float64)
    silent_groups = 0
    for positions in positions_by_group.values():
        group_combined = combine_group(components[positions], weight_tensor)
        combined[positions] = group_combined
        # The same predicate `group_carries_reward_signal` applies, on the same
        # float32 view. It used to be `.any()` on the float64 values, which is a
        # different question -- the warning and the sampler could disagree about
        # the same group, and the one the operator sees is the warning.
        as_transported = group_combined.float()
        if bool(as_transported.amin() == as_transported.amax()):
            silent_groups += 1

    if silent_groups == len(positions_by_group):
        # Every group came out at exactly zero, so this batch produces no
        # gradient at all. Usually the reward function is constant for these
        # prompts (e.g. a format reward when nothing in the prompt asks for a
        # format); the other way in is components that cancel. Worth one line,
        # because the symptom downstream is simply "loss does not move".
        logger.warning(
            "GDPO: all %d groups of this batch combined to a value that does not vary within the "
            "group once cast to float32 (keys=%s); the batch contributes no gradient. Either these "
            "rewards do not vary across rollouts of the same prompt, or they cancel under the "
            "configured weights (weights=%s).",
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
        # which now raises on it -- but at that point the message can only name
        # the batch, not the reward that produced it.
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
    vary in float32". Those differ: a zero weight mutes a varying component, and two
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
    # float64, the same dtype `normalize_gdpo_decoupled` weights with. A float32
    # weight tensor here would answer this question for a *different* weight
    # vector than the one training uses, whenever two configured weights are
    # closer together than float32 can represent.
    weights = torch.tensor(resolve_gdpo_weights(args, keys), dtype=torch.float64)
    components = extract_reward_components(samples, keys)
    # `combine_group`, not steps 1 and 2 open-coded: this has to answer the same
    # question `normalize_gdpo_decoupled` will. Open-coding it is how the two
    # came apart before.
    #
    # The same *form* the single-reward branch above uses, on the values the
    # trainer will receive: `min != max` in float32. Not the same question,
    # though -- that branch reads the raw reward, this one reads the output of
    # Eq. 4 and Eq. 7, and the two disagree on a group whose components sum to
    # a constant (the scalar is flat, the combination is rounding).
    #
    # `min != max` rather than `.any()` because it matches that branch, and
    # here the two coincide anyway: Eq. 4 centres every column, so the combined
    # values sum to zero within the group and "all equal" forces "all zero".
    # That is a borrowed invariant, not a property of this function --
    # `test_eq7_centres_every_group` pins it, because without it a group could
    # be constant and nonzero, and such a group is *not* whitened away by step
    # 3 (step 3 works per training batch, so it would come out as a constant
    # nonzero advantage).
    #
    # No tolerance of any kind: a threshold here decides whether a prompt group
    # enters training, and no threshold on this quantity can tell a genuine
    # near-cancellation from a rounding one (see `combine_group`).
    #
    # The consequence is deliberate and it has a cost. A group whose components
    # sum to a constant survives this test on its rounding remainder and is
    # trained on, and it wastes the group. How much reaches the optimizer grows
    # with the magnitude the components are centred on -- see the table in
    # `combine_group`; it is 3.1e-7 for C = 1e9 in a mixed training batch but
    # 5.2e-3 at C = 1e13, and a degenerate group that lands in a batch of its
    # own gets no division at all. That is the price of not guessing.
    combined = combine_group(components, weights).float()
    return bool(combined.amin() != combined.amax())


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
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        # KeyError/IndexError are not hypothetical. `get_reward_components`
        # raises ValueError for a missing key, but the single-reward branch
        # goes through `Sample.get_reward_value`, which is a bare subscript
        # (`relax/utils/types.py:171`): a `--reward-key` absent from the reward
        # dict raises KeyError straight through this handler and takes the
        # metrics path down -- the exact failure this function exists to
        # prevent. Narrowing `get_reward_value` to ValueError would be the
        # tidier fix, but that accessor is shared with the filters and the
        # rollout metrics, so changing what it raises is a contract change for
        # callers that are not in scope here.
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


def zero_std_group_label(args: Any, samples: list[Any]) -> str | None:
    """The ``zero_std/count_<label>`` suffix for a group, ``None`` if it has
    none.

    Companion to :func:`metrics_group_verdict`, and here for the same reason.
    That helper moved the *verdict* into one testable place and left the label
    behind in both copies of ``_compute_zero_std_metrics``, where they promptly
    disagreed: the agentic one reads it off the first *scored* sample, the
    distributed one off ``group[0]``, whether or not that one was scored.

    Reading it off an unscored sample is how this metric takes a rollout down.
    ``get_reward_value`` returns ``None`` there and ``round(None, 1)`` raises
    the same ``TypeError`` the verdict helper was introduced to remove -- one
    line further down than the version it replaced, and reachable in two ways:
    a group where nothing was scored, and a flat group whose first sample
    happens to be the unscored one.

    ``None`` means the group cannot be filed under any reward, not that it
    carries signal. Both callers drop such a group -- a group with no readable
    reward has none to report, so counting it would need a label that does not
    exist rather than one this function declined to compute.

    "Readable" is checked, not assumed, because ``reward is not None`` does not
    imply the *value* is. Two ways it does not, both reachable:

    * ``reward`` is a dict and ``args.reward_key`` selects a ``None`` out of it
      -- a partially failed reward function. The dict is not ``None``, so a
      test on ``sample.reward`` alone passes it through to ``round(None, 1)``:
      the same ``TypeError``, one predicate later.
    * ``reward`` is a dict without ``args.reward_key`` at all, which raises
      ``KeyError`` -- and ``KeyError`` is outside the ``(TypeError,
      ValueError)`` that :func:`observed_reward_signal` catches, so it is not
      covered by the "metrics observe, they do not enforce" split. Eval may run
      a different reward model entirely (``EvalConfig.rm_type``), which is
      exactly where a reward without the training ``--reward-key`` shows up.

    Both are treated as "this sample carries no label", not as an error to
    raise: this is a logging helper, and the stage that consumes the reward
    still refuses the same input.
    """
    for sample in samples:
        if sample.reward is None:
            continue
        try:
            value = sample.get_reward_value(args)
        except (KeyError, IndexError, TypeError):
            continue
        if value is None:
            continue
        return str(round(value, 1))
    return None


REWARD_NORMALIZERS: dict[str, Callable[[Any, list[Any], list[float]], list[float]]] = {
    "none": normalize_none,
    "group_mean": normalize_group_mean,
    "group_mean_std": normalize_group_mean_std,
    "group_leave_one_out": normalize_group_leave_one_out,
    "gdpo_decoupled": normalize_gdpo_decoupled,
}
