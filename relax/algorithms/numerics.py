# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared numerical constants and guards for the algorithm implementations.

Both the reward stage and the advantage stage standardise values by dividing by
a standard deviation, so they need the same epsilon and the same notion of
"this group carries no signal".  Keeping those here prevents the two stages
from drifting apart.
"""

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_LOGGED_GROUP_SIZE = False


def _log_group_once(process_group: dist.ProcessGroup | None) -> None:
    """Report the reduction group once per process.

    Whether a batch statistic is global or per-shard is invisible in the loss
    curve: a misconfigured run where tensor parallelism ate the extra GPUs
    leaves the data-parallel group at size 1, the all-reduce becomes an
    identity, and everything still trains. This line is what makes that
    distinguishable in a log.
    """
    global _LOGGED_GROUP_SIZE
    if _LOGGED_GROUP_SIZE:
        return
    _LOGGED_GROUP_SIZE = True
    if process_group is None:
        logger.info("Batch statistics are local (no process group); the caller owns the whole batch.")
    else:
        logger.info(
            "Batch statistics reduce over dp_world=%d (this rank is dp_rank=%d).",
            dist.get_world_size(process_group),
            dist.get_rank(process_group),
        )


STD_EPS = 1e-6
"""Epsilon added to a standard deviation before dividing by it.

Matches the value the pre-registry GRPO path used, so GRPO, GSPO, SAPO and
CISPO keep producing exactly the numbers they produced before the registry
existed.  That parity is the whole point of the equivalence tests, so this
constant is not free to move.
"""

GDPO_EPS = 1e-4
"""Epsilon for GDPO's two standardisation steps.

Deliberately not :data:`STD_EPS`.  The reference implementation
(``trl/trainer/grpo_trainer.py``, the ``scale_rewards`` GDPO path) divides by
``std + 1e-4`` at both the per-reward group step and the batch step, and GDPO
is new here, so there is no prior Relax behaviour that matching it would break.

The choice only bites near-degenerate groups: with binary rewards and eight
samples the group std is around 0.4 and the two constants differ by 0.02%, but
a continuous reward (the paper's maths setup uses response length) can leave a
group with std ~1e-3, where 1e-6 and 1e-4 disagree by roughly 10% on the scale
factor.  Exactly-collapsed groups never reach either constant; they are caught
by :func:`is_collapsed` and zeroed.
"""


def any_rank_has_non_finite(values: torch.Tensor, *, process_group: dist.ProcessGroup | None = None) -> bool:
    """Whether *any* rank's shard holds a NaN or an infinity.

    A local ``isfinite`` check in front of a collective is a deadlock: the rank
    that finds the bad value raises and leaves, and every other rank blocks
    forever in the reduction it never reaches. That is the same failure
    :func:`relax.algorithms.advantages._agree_on_segmentation` exists to
    prevent, and it is easy to reintroduce because the local check looks
    self-contained.

    One MAX all-reduce of a 0/1 flag, so every rank gets the same answer and
    the caller can raise on all of them together. Without a group the reduction
    is skipped and the answer is simply the local one.
    """
    bad = torch.tensor(
        float(not bool(torch.isfinite(values).all())),
        dtype=torch.float32,
        device=values.device,
    )
    if process_group is not None:
        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=process_group)
    return bool(bad.item())


def is_collapsed(values: torch.Tensor, *, process_group: dist.ProcessGroup | None = None) -> bool:
    """Whether every value is identical, i.e. the spread carries no signal.

    Tested by exact equality rather than by comparing the standard deviation
    against a tolerance.  A tolerance has to be relative to the magnitude (the
    mean of N equal float32 values does not come back exactly equal to them, so
    a collapsed group still shows std ~= 1e-8 times its magnitude), and any
    relative tolerance large enough to catch that also erases real signal: with
    ``std <= 1e-6 * max|x|``, the perfectly informative batch
    ``[10000, 10000.005, 10000.010, 10000.015]`` is thrown away.  Exact equality
    has no such false positives, and near-equality is already damped by the
    caller's epsilon (:data:`GDPO_EPS` on the GDPO whitening path,
    :data:`STD_EPS` on the group path).
    """
    if process_group is None:
        if values.numel() == 0:
            return True
        return bool(values.min() == values.max())

    # Every rank in the group has to reach the collective, including one whose
    # shard came out empty — returning early there would hang the others. An
    # empty shard contributes -inf to both halves, which is the identity for MAX
    # and therefore leaves the reduction to the ranks that do have samples.
    empty = values.numel() == 0
    neg_infinity = torch.tensor(float("-inf"), dtype=values.dtype, device=values.device)
    bounds = torch.stack(
        [neg_infinity if empty else -values.min(), neg_infinity if empty else values.max()],
    )
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=process_group)
    low, high = -bounds[0], bounds[1]
    if not torch.isfinite(low):
        return True  # every rank was empty
    return bool(low == high)


def collapsed_columns(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-column version of :func:`is_collapsed` for a ``[G, K]`` group.

    Returns a boolean tensor of shape ``[K]``: ``True`` where that reward
    component took the same value across the whole group.
    """
    return values.amax(dim=dim) == values.amin(dim=dim)


def distributed_mean_std(
    values: torch.Tensor, *, process_group: dist.ProcessGroup | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and unbiased std of ``values``, optionally across
    ``process_group``.

    Each rank holds its own shard of the batch, so a local ``std()`` would give
    every rank a different scale factor.  Reducing across the group makes the
    statistics describe the whole batch, which is what the framework already
    does for ``--normalize-advantages`` (see
    ``relax.utils.distributed_utils.distributed_masked_whiten``).

    Two passes, in float64.  The one-pass form ``E[x^2] - E[x]^2`` subtracts two
    nearly equal large numbers when the values sit far from zero, and the result
    is dominated by rounding: for ``[1000.0, 1000.01, 1000.02, 1000.03]`` it
    returns a variance of exactly 0 (true std 1.29e-2), and for the tighter
    ``[10000.0, 10000.001, 10000.002, 10000.003]`` it returns std 4.6 instead of
    1.3e-3 — off by a factor of 3660.  Scaling the first example up to 1e4
    without tightening it just reproduces the zero, not the inflation.
    Neither is loud.  The first silently zeroes every advantage in the batch;
    the second silently rescales them.  Reward magnitudes like these are
    ordinary: the GDPO paper's own maths setup uses a length reward, and token
    counts live in the thousands.

    The extra collective is two scalars, which is not worth optimising away.
    """
    _log_group_once(process_group)

    # float64 throughout: the cancellation above is a precision problem, and
    # doing the arithmetic in the caller's float32 reintroduces it even with the
    # two-pass formula.
    work = values.double()
    count = torch.tensor(float(work.numel()), dtype=torch.float64, device=work.device)
    total = work.sum()

    if process_group is not None:
        first = torch.stack([count, total])
        dist.all_reduce(first, op=dist.ReduceOp.SUM, group=process_group)
        count, total = first[0], first[1]

    if count == 0:
        zero = torch.zeros((), dtype=values.dtype, device=values.device)
        return zero, zero

    mean = total / count
    # Centred before squaring, so no large offset survives into the sum.
    sum_sq_dev = (work - mean).pow(2).sum()
    if process_group is not None:
        dist.all_reduce(sum_sq_dev, op=dist.ReduceOp.SUM, group=process_group)

    # Bessel-corrected, matching torch.std()'s default so the single-reward GDPO
    # scale factor stays derivable from the GRPO group statistics.
    variance = sum_sq_dev / torch.clamp(count - 1, min=1.0)
    # variance cannot be negative now that it is a sum of squares; the clamp is
    # only guarding the exactly-zero case against a -0.0.
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return mean.to(values.dtype), std.to(values.dtype)
