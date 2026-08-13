# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Advantage estimators as pure functions shared by both execution paths.

The colocate path calls these from ``relax.backends.megatron.loss`` inside the
Megatron worker; the fully-async path calls them from the
``relax.components.advantages`` Ray Serve deployment.  Keeping the maths here
means the two call sites differ only in what surrounds them: the pipeline-stage
early return, in-place write-back versus nested-tensor packing, and the
optional advantage whitening.

Note that the group-wise reward standardisation happened earlier, on the
rollout side (see :mod:`relax.algorithms.rewards`).  By the time an estimator
runs, ``rewards`` already holds one normalised scalar per sample.
"""

from typing import Any, Callable

import torch
import torch.distributed as dist

from relax.algorithms.numerics import GDPO_EPS, distributed_mean_std, is_collapsed
from relax.algorithms.spec import get_algorithm
from relax.utils.training.ppo_utils import (
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)


def whiten_scalar(values: torch.Tensor, *, process_group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Sequence-level whitening of one scalar per sample.

    This is GDPO's batch-wise normalisation (arXiv 2601.05242, Eq. 6).  It is
    deliberately *not* the token-level ``distributed_masked_whiten`` used by
    ``--normalize-advantages``: weighting by token count would let long
    responses dominate the statistics, which Eq. 6 does not do.

    ``process_group`` must be supplied wherever the caller holds only a shard of
    the values.  In the Megatron path each data-parallel rank owns
    ``num_rollout_minis * global_batch_size / dp_size`` samples — its shard of
    the whole rollout, merged before ``compute_advantages_and_returns`` runs — so
    whitening locally would give every rank its own mean and scale, not the "one
    global scale factor" the maths assumes.  ``None`` means "I own every value you
    need"; note the single-replica ``Advantages`` deployment cannot be that caller
    for GDPO, since ``supports_fully_async=False`` rejects the configuration that
    would route here — so in practice ``None`` is the CPU-side and test callers.

    Scope of one call: this whitens exactly the tensor it is handed.  When the
    caller has merged several training batches, :func:`_whiten_by_segment` splits
    them back and calls this once per optimizer batch, so each call stays aligned
    with Eq. 6's per-batch statistic regardless of ``num_rollout_minis``.  Only
    the un-segmented path (``mini_batch_sizes=None``) whitens a whole merged
    tensor at once.

    A batch where every value is identical returns exact zeros; see
    :func:`relax.algorithms.numerics.is_collapsed`.
    """
    if is_collapsed(values, process_group=process_group):
        return torch.zeros_like(values)
    mean, std = distributed_mean_std(values, process_group=process_group)
    if not torch.isfinite(std):
        return torch.zeros_like(values)
    return (values - mean) / (std + GDPO_EPS)


def _as_reward_tensor(rewards: Any, kl: list[torch.Tensor]) -> torch.Tensor:
    """Rewards as a detached float32 tensor on the KL tensors' device.

    ``detach()`` is defensive, and worth being precise about because the
    comment here used to call it load-bearing, which overstates it.

    Both production call sites pass ``list[float]`` -- ``loss.py`` reads
    ``rollout_data["rewards"]`` and the Advantages deployment reads the same
    column off the TransferQueue -- so the tensor branch below is not currently
    reached by anything, and removing the detach would change no observable
    behaviour today.

    It stays because the refactor did quietly widen what this accepts. The
    pre-registry code built the tensor with ``torch.tensor(rewards, ...)``,
    which copies and drops autograd history even when handed a tensor;
    ``.to()`` returns the *same* object when dtype and device already match. So
    a future caller passing a reward tensor with ``requires_grad=True`` would,
    without the detach, get advantages carrying grad history into the policy
    loss. Rewards are data, not something to backprop into.
    """
    if isinstance(rewards, torch.Tensor):
        return rewards.detach().to(dtype=torch.float32, device=kl[0].device)
    return torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)


def advantage_grpo_broadcast(args: Any, *, rewards, kl, **_unused):
    """Broadcast the already group-normalised scalar reward over tokens."""
    reward_tensor = _as_reward_tensor(rewards, kl)
    returns = get_grpo_returns(reward_tensor, kl)
    advantages = list(returns)  # separate list so rebinding one does not move the other
    return advantages, returns


def _agree_on_segmentation(n_segments: int, local_error: str, values, process_group) -> None:
    """Reach one verdict on the segmentation across the whole group.

    Each segment costs one collective, so this has to agree *before* any of
    them run. Two ways it can disagree, both of which deadlock rather than
    fail:

    * The segment counts differ. A rank expecting two segments and a rank
      expecting three both block on the third collective -- no traceback, no
      exit code, just a job that never finishes. ``num_rollout_minis`` comes
      from the minibatch plan and is expected to be uniform, but "expected" is
      not "checked".
    * One rank's ``mini_batch_sizes`` is malformed. Raising locally is worse
      than not checking at all: that rank leaves the collective sequence while
      every other rank is still waiting inside it.

    So the local verdict travels *with* the count, in one MAX reduction (the
    low bound negated, the way :func:`~relax.algorithms.numerics.is_collapsed`
    does it). Every rank reads the same three numbers and therefore raises
    together, at the same call, with a message naming which failure it was.
    """
    if process_group is None:
        if local_error:
            raise ValueError(local_error)
        return

    # A rank that already failed contributes 0 segments: a neutral value that
    # cannot be mistaken for a real count, and one that trips the mismatch
    # branch too if the `any_bad` branch were ever removed.
    flags = torch.tensor(
        [-n_segments if not local_error else 0, n_segments if not local_error else 0, 1 if local_error else 0],
        dtype=torch.int64,
        device=values.device,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=process_group)
    low, high, any_bad = -int(flags[0]), int(flags[1]), int(flags[2])

    if any_bad:
        raise ValueError(
            local_error
            or "another rank reported malformed mini_batch_sizes; every rank fails here so the "
            "per-segment collectives cannot deadlock."
        )
    if low != high:
        raise ValueError(
            f"mini_batch_sizes describes {n_segments} segment(s) on this rank, but ranks in the group "
            f"report between {low} and {high}. Every rank must whiten the same number of segments, "
            "or the per-segment collectives deadlock."
        )


def _whiten_by_segment(values, mini_batch_sizes, process_group):
    """Whiten each training batch separately, in the order they were merged.

    ``num_rollout_minis`` comes from the minibatch plan rather than from the
    data, so every rank is expected to run the same number of segments;
    :func:`_agree_on_segmentation` is what turns that expectation into a
    checked precondition rather than a deadlock.

    What it checks is the segment *count*, not segment *identity*. Two further
    properties are relied on and not verified here:

    * Segment ``k`` holds training batch ``k`` on every rank. ``actor.py``
      fetches batches in ``batch_index`` order and appends the counts in that
      same order, so the orders agree; nothing in this function would notice if
      they stopped agreeing, and the failure would be silent -- statistics
      mixed across two training batches, no error.
    * A segment's sample count may differ between ranks (a data-parallel split
      balances tokens, not samples) and that is fine, because the statistic is
      reduced across the group. What is not fine is two ranks disagreeing about
      *which* batch a segment belongs to.

    Both are guaranteed upstream rather than here because checking identity
    would need a batch id in the metadata, which the plan does not currently
    carry. If that metadata ever appears, fold it into the same MAX reduction.
    """
    # Validate whatever was passed, including a single segment: treating an empty
    # or malformed list as "fall back to one window" would silently restore the
    # merged behaviour this function exists to replace. The verdict is not acted
    # on until the whole group has shared it.
    local_error = ""
    if mini_batch_sizes is None:
        # Not a fallback. Whitening the merged rollout as one window is a
        # different optimisation target, not a coarser version of the same one:
        # `test_merging_the_batches_would_flip_signs_not_just_rescale` shows
        # half the advantages changing sign. Every Megatron producer writes
        # these counts today, so a missing one means a new or changed caller,
        # and the only thing worse than that caller failing is that caller
        # training on the wrong objective while every metric stays finite.
        n_segments = 0
        local_error = (
            "mini_batch_sizes is None: the caller did not supply "
            "`rollout_mini_local_sample_counts`. GDPO whitens per training batch, so there is no "
            "safe default -- whitening the merged rollout instead optimises a different objective."
        )
    elif not mini_batch_sizes or any(not isinstance(n, int) or n <= 0 for n in mini_batch_sizes):
        n_segments = 0
        local_error = f"mini_batch_sizes must be a non-empty list of positive ints, got {mini_batch_sizes}."
    elif sum(mini_batch_sizes) != values.numel():
        n_segments = 0
        local_error = (
            f"mini_batch_sizes {mini_batch_sizes} sum to {sum(mini_batch_sizes)}, "
            f"but this rank holds {values.numel()} samples."
        )
    else:
        n_segments = len(mini_batch_sizes)

    _agree_on_segmentation(n_segments, local_error, values, process_group)

    if n_segments == 1:
        return whiten_scalar(values, process_group=process_group)
    out, start = [], 0
    for size in mini_batch_sizes:
        out.append(whiten_scalar(values[start : start + size], process_group=process_group))
        start += size
    return torch.cat(out)


def advantage_gdpo(args: Any, *, rewards, kl, process_group=None, mini_batch_sizes=None, **_unused):
    """GDPO step 3: whiten the combined per-sample advantage, then broadcast.

    Steps 1 and 2 (per-reward group standardisation and the weighted sum) ran
    on the rollout side, so ``rewards`` already holds one ``A_sum`` per sample.
    Doing step 3 here rather than there keeps it out of reach of ``--custom-
    reward-post-process-path``, which short-circuits reward post-processing
    entirely, and off the streaming transfer-batch boundary with its undersized
    tail flushes.
    """
    reward_tensor = _as_reward_tensor(rewards, kl)
    whitened = _whiten_by_segment(reward_tensor, mini_batch_sizes, process_group)
    returns = get_grpo_returns(whitened, kl)
    advantages = list(returns)
    return advantages, returns


def advantage_reinforce_plus_plus(args: Any, *, rewards, kl, loss_masks, response_lengths, total_lengths, **_unused):
    """Discounted returns for REINFORCE++
    (https://arxiv.org/pdf/2501.03262)."""
    reward_tensor = _as_reward_tensor(rewards, kl)
    returns = get_reinforce_plus_plus_returns(
        rewards=reward_tensor,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        kl_coef=args.kl_coef,
        gamma=args.gamma,
    )
    advantages = list(returns)
    return advantages, returns


def advantage_reinforce_plus_plus_baseline(args: Any, *, rewards, kl, loss_masks, **_unused):
    """REINFORCE++ with a group baseline already subtracted upstream.

    No ``kl_coef``: this estimator keeps the KL penalty out of the advantage
    (``_validate_reinforce_plus_plus_args`` requires ``--kl-coef 0`` and an
    independent k2 loss instead), and the helper dropped the parameter to
    match.
    """
    reward_tensor = _as_reward_tensor(rewards, kl)
    advantages = get_reinforce_plus_plus_baseline_advantages(
        rewards=reward_tensor,
        kl=kl,
        loss_masks=loss_masks,
    )
    # NOTE(dev): returns aliases the same list object, matching the pre-refactor
    # behaviour. On-policy distillation rebinds slots of the advantages list and
    # both views are expected to observe that.
    return advantages, advantages


def advantage_gae(
    args: Any,
    *,
    rewards,
    kl,
    values,
    response_lengths,
    total_lengths,
    padded_total_lengths=None,
    **_unused,
):
    """Generalised advantage estimation (PPO).

    ``padded_total_lengths`` is what the two call sites disagree on, and it is
    load-bearing rather than cosmetic: under bshd ``qkv_format``, VL models or
    unsplit forward, the sequences were padded before the forward pass, so CP
    un-sharding inside ``get_advantages_and_returns_batch`` has to slice at the
    padded offsets.  Passing ``None`` there does not raise — it silently reads
    the wrong token positions.  The Megatron path computes the value via
    ``maybe_padded_total_lengths`` and passes it; the ``Advantages`` deployment
    has no equivalent and passes ``None``, which is the behaviour it had before
    this handler existed.
    """
    from megatron.core import mpu

    shaped_rewards = []
    cp_rank = mpu.get_context_parallel_rank()
    for reward, k in zip(rewards, kl, strict=False):
        k *= -args.kl_coef
        if cp_rank == 0:
            k[-1] += reward
        shaped_rewards.append(k)
    return get_advantages_and_returns_batch(
        total_lengths,
        response_lengths,
        values,
        shaped_rewards,
        args.gamma,
        args.lambd,
        padded_total_lengths=padded_total_lengths,
    )


ADVANTAGE_FNS: dict[str, Callable[..., tuple[list[torch.Tensor], list[torch.Tensor]]]] = {
    "grpo_broadcast": advantage_grpo_broadcast,
    "gdpo": advantage_gdpo,
    "reinforce_plus_plus": advantage_reinforce_plus_plus,
    "reinforce_plus_plus_baseline": advantage_reinforce_plus_plus_baseline,
    "gae": advantage_gae,
}


def compute_advantages_and_returns(
    args: Any,
    *,
    rewards,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor] | None = None,
    response_lengths: list[int] | None = None,
    total_lengths: list[int] | None = None,
    values: list[torch.Tensor] | None = None,
    padded_total_lengths: list[int] | None = None,
    process_group: dist.ProcessGroup | None = None,
    mini_batch_sizes: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Dispatch to the estimator registered for ``args.advantage_estimator``.

    ``mini_batch_sizes`` is this rank's per-training-batch sample counts, in the
    order the caller merged them. Only estimators whose statistics are defined
    per batch read it; the rest absorb it in ``**_unused`` and are unaffected.

    ``process_group`` is the group across which the batch is sharded, or
    ``None`` when the caller holds every sample. Estimators that compute batch-
    level statistics need it to see the whole batch.

    ``padded_total_lengths`` is likewise call-site specific: only the Megatron
    path can compute it, and only GAE consumes it. Every parameter here is
    keyword-only and every estimator absorbs the rest, so a call site that
    cannot supply one omits it rather than inventing a value.
    """
    spec = get_algorithm(args.advantage_estimator)
    fn = ADVANTAGE_FNS[spec.advantage_fn]
    return fn(
        args,
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        values=values,
        padded_total_lengths=padded_total_lengths,
        process_group=process_group,
        mini_batch_sizes=mini_batch_sizes,
    )
