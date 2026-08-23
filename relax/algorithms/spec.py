# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Declarative descriptions of the RL algorithms Relax supports.

A single ``--advantage-estimator`` value used to be interpreted independently by
six different files: role lookup, reward normalisation, the advantage formula,
the policy loss formula, and two rounds of argument validation.  Adding an
algorithm meant finding all of them.  ``AlgorithmSpec`` collects that metadata
in one place, so a new algorithm is one dict entry plus its pure functions.

Fields hold **string identifiers**, not callables.  The advantage formula runs
inside the Ray Serve ``Advantages`` deployment while the policy loss runs inside
the Megatron worker; those two processes import different module subsets, so we
ship the algorithm name and let each process resolve the identifier against its
own table.  That also keeps this module free of heavy imports, which is what
makes the registry testable on a CPU-only runner.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmSpec:
    """Everything the training pipeline needs to know about one algorithm."""

    name: str

    # --- reward stage (rollout side, CPU, scalar in / scalar out) ---
    reward_normalizer: str
    """Key into :data:`relax.algorithms.rewards.REWARD_NORMALIZERS`."""

    # --- advantage stage ---
    advantage_fn: str
    """Key into :data:`relax.algorithms.advantages.ADVANTAGE_FNS`."""

    # --- policy loss stage ---
    policy_loss_fn: str
    """Key into :data:`relax.algorithms.policy.POLICY_LOSS_FNS`."""

    kl_level: str = "token"
    """``"token"`` or ``"sequence"``; GSPO constrains the sequence as a whole."""

    advantage_normalization: str = "whiten"
    """How ``--normalize-advantages`` normalises, read in
    ``relax/backends/megatron/loss.py``.

    ``"whiten"`` is the masked whitening every algorithm used before REINFORCE++
    arrived. ``"token_global"`` is REINFORCE++'s global token-level
    normalisation, which also requires the mask-safe loss reducer -- the two go
    together, which is why one field drives both call sites rather than two.

    Those two call sites were the last place a *maths* decision was still made
    by comparing algorithm names, and unlike the remaining ``== "ppo"`` checks
    this one is not expressible through an existing field: it is orthogonal to
    ``needs_critic``.
    """

    needs_full_log_probs: bool = False
    """Whether the loss needs CP-gathered full-response log probs."""

    # --- orchestration and validation ---
    needs_critic: bool = False
    requires_normalize_advantages: bool = False
    forbids_normalize_advantages: bool = False
    """Set when the token-level ``--normalize-advantages`` pass would undo what
    the estimator deliberately kept.

    RLOO's advantage carries the reward scale on purpose; re-whitening it after
    DP sharding puts back the standard-deviation division RLOO removes, and
    makes the result depend on how the batch was partitioned.
    """

    requires_rewards_normalization: bool = False
    """Set when the algorithm's reward stage is load-bearing rather than
    optional, so ``--disable-rewards-normalization`` would silently skip it."""

    min_group_size: int = 1
    """Smallest ``--n-samples-per-prompt`` the reward stage is defined for."""

    forbids_reward_side_kl: bool = False
    """Set when the estimator has no place to put a reward-side KL penalty.

    Both members here compute their advantage from a completion-level signal
    that the ``--kl-coef`` shaping term never enters, so a nonzero coefficient
    would be silently ignored rather than applied. ``--use-kl-loss`` with
    ``--kl-loss-coef`` remains available: that one is a separate loss term, not
    a reward modification.
    """

    requires_global_token_loss: bool = False
    """Set when the objective is only correct under
    ``--calculate-per-token-loss``.

    The default per-sample token-mean reducer divides each sample by its own
    response length, which reweights unequal-length responses by
    ``1 / response_length``. An objective with no ratio correction has nothing
    to absorb that reweighting, so it must be normalised by the global number
    of valid response tokens instead.
    """

    requires_on_policy_updates: bool = False
    """Set when every optimizer step must consume exactly the rollout that
    produced it.

    An objective without an importance-ratio correction cannot account for the
    policy having moved, so *five* separate configuration knobs have to agree:
    no ``--fully-async`` / ``--hybrid``, ``--max-staleness 0``,
    ``--num-steps-per-rollout 1``, ``rollout_batch_size *
    n_samples_per_prompt == global_batch_size``, and neither
    ``--partial-rollout`` nor ``--use-dynamic-global-batch-size`` (both let the
    effective batch size drift at runtime).

    They are one field rather than five because they have one cause. RLOO is
    currently the only member, so the bundling is a guess about the next
    unclipped estimator; if one ever needs four of the five, split this field
    then rather than adding an exception to it.

    Note this is a *stronger* statement than "cannot run fully-async": an
    algorithm can be sync-only for unrelated reasons (PPO's critic topology,
    or an advantage that needs batch-level statistics the async deployment
    only sees a slice of). Those get their own field; do not fold them here.
    """

    supports_fully_async: bool = True
    """Whether the algorithm is correct under ``--fully-async``.

    That mode routes advantage computation to the single-replica
    ``relax.components.advantages`` deployment, which owns no data-parallel
    group and consumes one ``global_batch_size / num_iters_per_train_update``
    slice at a time. An algorithm whose advantages depend on batch-level
    statistics computes them over that slice instead of the batch, and at
    slice size 1 gets no signal at all — silently, since the run still
    converges and exits cleanly. Note ``--hybrid`` is *not* affected: it uses
    the colocate role set, so advantages are computed in the Megatron worker
    where the data-parallel group exists.

    Do not be tempted to relax this into "allow it when the slice happens to
    equal the batch". The slice the deployment actually receives also depends
    on which TransferQueue sampler the controller installed
    (``relax/core/controller.py``), and that choice is global to every
    consumer. Under ``--balance-data`` it is ``SeqlenBalancedSampler``, whose
    ``batch_size`` is *per data-parallel rank*; the deployment passes no
    ``sampling_config``, so it would receive one rank's share of a
    token-balanced split rather than the batch.
    """

    uses_reward_components: bool = False
    """Whether the algorithm consumes several named reward components rather
    than the single scalar ``--reward-key`` selects. Drives the
    ``--gdpo-reward-keys`` / ``--gdpo-reward-weights`` validation.

    Those two options stay GDPO-prefixed on purpose: every algorithm-specific
    option in Relax names its algorithm (``--sapo-tau-pos``,
    ``--disable-grpo-std-normalization``), and GDPO is so far the only member
    of this category — renaming now would mean guessing the abstraction from a
    single example. When a second ``uses_reward_components`` algorithm lands,
    rename both to algorithm-neutral options and keep the old spellings as
    deprecated aliases for one release, the way ``--loss-type sft_loss`` is
    handled in ``relax/utils/arguments.py``.
    """

    allows_reward_post_process_hooks: bool = True
    """False when a user-supplied reward hook would silently disable the
    algorithm's own reward stage.

    Guards both short-circuits in ``relax.utils.utils.post_process_rewards``:
    ``--custom-reward-post-process-path`` replaces the function outright, and
    ``--agentic-custom-advantage-path`` returns before the normaliser runs.
    They are one flag rather than two because an algorithm that cannot tolerate
    one cannot tolerate the other -- ``reinforce_plus_plus_baseline`` already
    rejects both by hand in ``relax/utils/arguments.py``, and GDPO needs the
    same pair."""

    @property
    def is_group_normalized(self) -> bool:
        """Whether rewards get normalised per prompt group on the rollout
        side."""
        return self.reward_normalizer != "none"


# NOTE(dev): explicit dict literal, deliberately not decorator-based registration.
# The advantage formula and the policy loss execute in two different processes
# whose import graphs differ; decorator registration depends on "was this module
# imported?" and silently loses an algorithm when one side misses the import.
ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "grpo": AlgorithmSpec(
        name="grpo",
        reward_normalizer="group_mean_std",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="ppo_clip",
    ),
    "gspo": AlgorithmSpec(
        name="gspo",
        reward_normalizer="group_mean_std",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="ppo_clip",
        kl_level="sequence",
        needs_full_log_probs=True,
    ),
    "sapo": AlgorithmSpec(
        name="sapo",
        reward_normalizer="group_mean_std",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="sapo",
    ),
    "cispo": AlgorithmSpec(
        name="cispo",
        reward_normalizer="group_mean_std",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="cispo",
    ),
    "rloo": AlgorithmSpec(
        name="rloo",
        # REINFORCE leave-one-out (arXiv:2402.14740). The baseline is the mean
        # of the *other* completions in the prompt group, so the reward stage
        # differs from GRPO's while the advantage stage -- broadcast the scalar
        # over the response tokens -- is the same one.
        reward_normalizer="group_leave_one_out",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="rloo",
        requires_rewards_normalization=True,
        forbids_normalize_advantages=True,
        min_group_size=2,
        forbids_reward_side_kl=True,
        requires_global_token_loss=True,
        requires_on_policy_updates=True,
    ),
    "gdpo": AlgorithmSpec(
        name="gdpo",
        reward_normalizer="gdpo_decoupled",
        advantage_fn="gdpo",
        policy_loss_fn="ppo_clip",
        # Step 3 already whitens per sequence; --normalize-advantages would add a
        # second, token-level pass on top of it.
        forbids_normalize_advantages=True,
        requires_rewards_normalization=True,
        # Step 3 needs the training batch. Under --fully-async it would only ever
        # see one slice of it, and a slice of one sample yields zero advantages.
        supports_fully_async=False,
        uses_reward_components=True,
        # Step 1 divides by an unbiased group std, undefined for a single sample.
        min_group_size=2,
        # `advantage_gdpo` hands `kl` to `get_grpo_returns`, which uses it only
        # for shape -- the values are discarded, so `--kl-coef` buys a reference
        # forward pass and changes nothing. Declaring it here is free for a new
        # algorithm: no existing configuration is rejected. The four older
        # members of the `grpo_broadcast` family have exactly the same property
        # and are deliberately left alone, because refusing a flag they accept
        # today is an upstream decision rather than this PR's. `--use-kl-loss`
        # with `--kl-loss-coef` is unaffected either way.
        forbids_reward_side_kl=True,
        # Either reward hook short-circuits reward post-processing, which would
        # silently skip steps 1 and 2 while the run still reports itself as GDPO.
        allows_reward_post_process_hooks=False,
    ),
    "ppo": AlgorithmSpec(
        name="ppo",
        reward_normalizer="none",
        advantage_fn="gae",
        policy_loss_fn="ppo_clip",
        needs_critic=True,
        # `validate_ppo_config` in `relax/utils/training/ppo_utils.py` rejects
        # both --fully-async and --hybrid for PPO. Declaring it means argument
        # validation says so up front instead of the Controller saying it at
        # service-registration time, and stops the spec from asserting the
        # opposite by omission.
        supports_fully_async=False,
    ),
    "reinforce_plus_plus": AlgorithmSpec(
        name="reinforce_plus_plus",
        advantage_normalization="token_global",
        reward_normalizer="none",
        advantage_fn="reinforce_plus_plus",
        policy_loss_fn="ppo_clip",
        requires_normalize_advantages=True,
        # `_validate_reinforce_plus_plus_args` also rejects --fully-async for
        # this estimator. Declaring it here as well is not redundant: an
        # undeclared field defaults to True, i.e. the spec would actively state
        # something false about the algorithm, and any future reader of the
        # spec would believe it. That function still runs first and still owns
        # the wording its tests match on.
        supports_fully_async=False,
    ),
    "reinforce_plus_plus_baseline": AlgorithmSpec(
        name="reinforce_plus_plus_baseline",
        advantage_normalization="token_global",
        reward_normalizer="group_mean",
        advantage_fn="reinforce_plus_plus_baseline",
        policy_loss_fn="ppo_clip",
        requires_normalize_advantages=True,
        # These five are also enforced by hand in
        # `_validate_reinforce_plus_plus_args`, a frozen Task 29 contract that
        # owns the wording its tests match on. They were left undeclared at
        # first to avoid replacing that wording -- but an undeclared field is
        # not neutral, it *asserts the default*, so the spec was stating five
        # things about this estimator that are false. Both validation paths now
        # run the frozen function first (the main one always did;
        # `apply_custom_config_overrides` was fixed to match), so its messages
        # still win and the spec can stop lying.
        supports_fully_async=False,
        requires_rewards_normalization=True,
        allows_reward_post_process_hooks=False,
        min_group_size=2,
        forbids_reward_side_kl=True,
    ),
}


def get_algorithm(name: str) -> AlgorithmSpec:
    """Look up an algorithm spec by its ``--advantage-estimator`` value."""
    try:
        return ALGORITHM_SPECS[name]
    except KeyError:
        available = ", ".join(ALGORITHM_SPECS)
        raise KeyError(f"Unknown advantage estimator {name!r}. Available: {available}") from None


def list_algorithm_names() -> list[str]:
    """All registered algorithm names, in definition order."""
    return list(ALGORITHM_SPECS)
