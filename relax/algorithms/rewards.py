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

    components = torch.tensor(rows, dtype=torch.float32)
    # Belt and braces: the per-value check above is exact, but it only sees what
    # `float()` returned. Anything that slips past it must not reach the
    # normaliser, where non-finite input is indistinguishable from a collapse.
    if not torch.isfinite(components).all():
        bad = (~torch.isfinite(components)).nonzero()[0].tolist()
        raise ValueError(f"Reward {keys[bad[1]]!r} of sample {bad[0]} is not finite after casting to float32.")
    return components


def standardize_group_components(group: torch.Tensor) -> torch.Tensor:
    """GDPO step 1 on one ``[G, K]`` prompt group (arXiv 2601.05242, Eq. 4).

    Each column is standardised on its own. A column that is flat across the
    group contributes exactly zero rather than ``0 / (0 + eps)`` noise, which
    is what lets the remaining columns keep their signal.
    """
    centered = group - group.mean(dim=0, keepdim=True)
    std = group.std(dim=0)
    collapsed = collapsed_columns(group, dim=0)
    scaled = centered / (std + GDPO_EPS)
    return torch.where(collapsed.unsqueeze(0), torch.zeros_like(scaled), scaled)


def normalize_gdpo_decoupled(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """GDPO steps 1 and 2 (arXiv 2601.05242, Eq. 4 and Eq. 7).

    Step 1 standardises each reward component within its prompt group; step 2
    combines them with the configured weights.  The result is one scalar per
    sample, so it travels through the existing ``rewards`` column and needs no
    TransferQueue schema change.  Step 3 (batch whitening) runs later, in
    :func:`relax.algorithms.advantages.advantage_gdpo`.

    What standardising per component actually buys, stated carefully because
    it is easy to overclaim: the combined advantage is ``sum_k w_k * z_k``,
    where each ``z_k`` has unit variance within the group. GRPO instead
    standardises ``sum_k r_k``, so a component with a large spread dominates
    the direction. When the components have comparable spread the two agree up
    to a positive scalar; they diverge when the spreads differ, which is the
    case GDPO is for -- a correctness reward in ``{0, 1}`` combined with a
    length reward in the hundreds is decided almost entirely by length under
    GRPO, and half by each under GDPO.

    It does **not** rescue a group whose components sum to a constant. There
    ``r_2 = C - r_1`` forces ``z_2 = -z_1``, so equal weights cancel to exactly
    zero -- the same answer GRPO gives. Unequal weights break the tie; equal
    weights cannot.
    """
    keys = resolve_gdpo_keys(args)
    weights = resolve_gdpo_weights(args, keys)

    components = extract_reward_components(samples, keys)
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)

    normalized = torch.zeros_like(components)
    fully_collapsed_groups = 0
    for positions in positions_by_group.values():
        group = components[positions]
        normalized[positions] = standardize_group_components(group)
        if bool(collapsed_columns(group, dim=0).all()):
            fully_collapsed_groups += 1

    if fully_collapsed_groups == len(positions_by_group):
        # Every component collapsed in every group, so this batch produces no
        # gradient at all. Usually the reward function is constant for these
        # prompts (e.g. a format reward when nothing in the prompt asks for a
        # format). Worth one line, because the symptom downstream is simply
        # "loss does not move".
        logger.warning(
            "GDPO: all reward components collapsed in all %d groups of this batch (keys=%s); "
            "the batch contributes no gradient. Check that each of these rewards actually varies "
            "across rollouts of the same prompt.",
            fully_collapsed_groups,
            keys,
        )

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

    combined = (normalized * weight_tensor).sum(dim=1)
    if not torch.isfinite(combined).all():
        # Every individual reward fit in float32 (checked in
        # `extract_reward_components`), but the arithmetic between them need
        # not: a group of [3e38, 2e38, 1e38, 0] overflows while computing its
        # own mean. Left unchecked the `inf` reaches `whiten_scalar`, reads as
        # a non-finite std, and the batch is silently zeroed -- the exact
        # failure the per-value check was added to prevent, one stage later.
        raise ValueError(
            "GDPO produced a non-finite combined advantage from finite inputs; the float32 "
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
    combined = (standardize_group_components(components) * weights).sum(dim=1)
    return bool((combined != 0).any())


REWARD_NORMALIZERS: dict[str, Callable[[Any, list[Any], list[float]], list[float]]] = {
    "none": normalize_none,
    "group_mean": normalize_group_mean,
    "group_mean_std": normalize_group_mean_std,
    "group_leave_one_out": normalize_group_leave_one_out,
    "gdpo_decoupled": normalize_gdpo_decoupled,
}
