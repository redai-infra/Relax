# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.algorithms.rewards import group_carries_reward_signal
from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.types import Sample


__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    """Drop prompt groups whose rewards carry no signal for this algorithm.

    "No signal" is algorithm-dependent, which is why the test is not spelled
    out here. For a single-reward algorithm it is whether the ``--reward-key``
    scalar varies in float32 -- equivalent to the ``std > 0`` this replaced,
    since ``min == max`` and ``std == 0`` agree on every float32 input.

    For a multi-reward algorithm it is whether the *combined* advantage comes
    out non-zero -- not whether every component is flat, which is a weaker
    condition and was what this docstring used to claim. The two part company
    exactly where it matters: a zero weight mutes a varying component, and two
    components whose standardised values are opposites cancel. Both leave a
    group that varies component-wise and still contributes no gradient. Judging
    by the summed ``--reward-key`` scalar is wrong in the other direction, and
    drops groups the algorithm exists to keep.

    Note this can now raise rather than merely returning a verdict: for a
    multi-reward algorithm it runs the same component extraction the reward
    stage does, which rejects malformed rewards. `call_dynamic_filter` does not
    catch, so a broken reward function fails the rollout here instead of a few
    lines later in `post_process_rewards`. That is the same failure, earlier and
    attributed to the group that caused it -- but it does mean this filter is no
    longer guaranteed to be side-effect-free on bad data.
    """
    keep = group_carries_reward_signal(args, samples)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(samples[0].get_reward_value(args), 1)}",
    )
