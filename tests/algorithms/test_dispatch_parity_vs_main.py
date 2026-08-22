# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The registry must route each algorithm exactly where main's if/elif did.

Scope note, because it is easy to over-claim here. `relax/utils/training/
ppo_utils.py` is byte-identical to main on this branch, so no estimator's or
policy loss's *maths* changed — every one of them still calls the same function
object it always did. Feeding both implementations the same tensors and
comparing outputs would therefore pass by construction and prove nothing.

What the refactor did change is *routing*: which kernel each algorithm name
resolves to, and which capability flags gate the surrounding code. That is what
these tables pin down. They are transcribed from
main @ 4899b8f3a90489840a736897b4c341d87c6267cf, whose algorithm code last
changed in 98a72349c7d0368440eb6b0c6849e9d0f2ba8cef ("实现 RLOO advantage
estimator", #205) -- nothing between those two commits touched advantages.py,
loss.py, utils.py or ppo_utils.py:

    relax/components/advantages.py      lines 176-218 (advantage if/elif)
    relax/backends/megatron/loss.py     lines 579-628 (the duplicate of it)
    relax/backends/megatron/loss.py     lines 691-694 (advantage normalisation)
    relax/backends/megatron/loss.py     line  894     (need_full_log_probs)
    relax/backends/megatron/loss.py     lines 851-854 (loss reducer)
    relax/backends/megatron/loss.py     line  953     (sequence-level KL)
    relax/backends/megatron/loss.py     lines 968-988 (policy loss if/elif)
    relax/utils/utils.py                lines 186,206 (reward normalisation)
    relax/utils/arguments.py            RLOO + REINFORCE++ startup constraints

Do not regenerate these from the current implementation — that would turn the
comparison into the implementation checking itself.
"""

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms import get_algorithm, list_algorithm_names  # noqa: E402
from relax.algorithms.advantages import ADVANTAGE_FNS  # noqa: E402
from relax.algorithms.policy import POLICY_LOSS_FNS  # noqa: E402
from relax.algorithms.rewards import REWARD_NORMALIZERS  # noqa: E402
from relax.utils.training import ppo_utils  # noqa: E402


MAIN_SHA = "4899b8f3a90489840a736897b4c341d87c6267cf"  # the base this branch is rebased on

# main advantages.py:176 — `if estimator in ["grpo", "gspo", "sapo", "cispo", "rloo"]`
# -> get_grpo_returns, etc.
MAIN_ADVANTAGE_KERNEL = {
    "grpo": ppo_utils.get_grpo_returns,
    "gspo": ppo_utils.get_grpo_returns,
    "sapo": ppo_utils.get_grpo_returns,
    "cispo": ppo_utils.get_grpo_returns,
    "rloo": ppo_utils.get_grpo_returns,
    "ppo": ppo_utils.get_advantages_and_returns_batch,
    "reinforce_plus_plus": ppo_utils.get_reinforce_plus_plus_returns,
    "reinforce_plus_plus_baseline": ppo_utils.get_reinforce_plus_plus_baseline_advantages,
}

# main loss.py:968-988
MAIN_POLICY_KERNEL = {
    "grpo": ppo_utils.compute_policy_loss,
    "gspo": ppo_utils.compute_policy_loss,
    "sapo": ppo_utils.compute_sapo_loss,
    "cispo": ppo_utils.compute_cispo_loss,
    "rloo": ppo_utils.compute_rloo_loss,
    "ppo": ppo_utils.compute_policy_loss,
    "reinforce_plus_plus": ppo_utils.compute_policy_loss,
    "reinforce_plus_plus_baseline": ppo_utils.compute_policy_loss,
}

# main loss.py:953 — only gspo took the sequence-level KL branch.
MAIN_SEQUENCE_LEVEL_KL = {"gspo"}

# main loss.py:894 — `args.use_opsm or estimator == "gspo"`.
MAIN_NEEDS_FULL_LOG_PROBS = {"gspo"}

# main arguments.py:3319 — `use_critic = estimator == "ppo"`.
MAIN_NEEDS_CRITIC = {"ppo"}

# main arguments.py:3164-3168 — reinforce_plus_plus{,_baseline} asserted normalize_advantages.
MAIN_REQUIRES_NORMALIZE_ADVANTAGES = {"reinforce_plus_plus", "reinforce_plus_plus_baseline"}

# main utils.py:186 — group-mean whitelist; :206 — the subset that also divides by std.
# rloo is in the first (it is normalised per group) but not the second: its
# leave-one-out baseline deliberately keeps the reward scale.
MAIN_GROUP_NORMALIZED = {"grpo", "gspo", "sapo", "cispo", "reinforce_plus_plus_baseline", "rloo"}
MAIN_GROUP_STD_NORMALIZED = {"grpo", "gspo", "sapo", "cispo"}

# main loss.py:691-694 and 851-854 — the two duplicated REINFORCE++ name sets
# that drove `distributed_masked_normalize` and the mask-safe loss reducer.
MAIN_TOKEN_GLOBAL = {"reinforce_plus_plus", "reinforce_plus_plus_baseline"}

# main arguments.py, the RLOO block and `_validate_reinforce_plus_plus_args`.
# Each of these was an `if args.advantage_estimator ...` on main and is a spec
# field here; the sets are what main actually enforced, not what the spec says.
MAIN_FORBIDS_NORMALIZE_ADVANTAGES = {"rloo"}
MAIN_REQUIRES_REWARDS_NORMALIZATION = {"rloo", "reinforce_plus_plus_baseline"}
MAIN_FORBIDS_REWARD_SIDE_KL = {"rloo", "reinforce_plus_plus_baseline"}
MAIN_REQUIRES_GLOBAL_TOKEN_LOSS = {"rloo"}
MAIN_REQUIRES_ON_POLICY_UPDATES = {"rloo"}
MAIN_MIN_GROUP_SIZE = {"rloo": 2, "reinforce_plus_plus_baseline": 2}

MAIN_ALGORITHMS = sorted(MAIN_ADVANTAGE_KERNEL)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_advantage_routes_to_the_same_kernel_as_main(name):
    """The estimator's maths is unchanged; only the lookup moved."""
    spec = get_algorithm(name)
    fn = ADVANTAGE_FNS[spec.advantage_fn]
    source = fn.__code__.co_names
    expected = MAIN_ADVANTAGE_KERNEL[name].__name__
    assert expected in source, f"{name} no longer reaches {expected}; it calls {source}"


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_policy_loss_routes_to_the_same_kernel_as_main(name):
    spec = get_algorithm(name)
    fn = POLICY_LOSS_FNS[spec.policy_loss_fn]
    expected = MAIN_POLICY_KERNEL[name].__name__
    assert expected in fn.__code__.co_names, f"{name} no longer reaches {expected}"


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_sequence_level_kl_matches_main(name):
    assert (get_algorithm(name).kl_level == "sequence") is (name in MAIN_SEQUENCE_LEVEL_KL)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_needs_full_log_probs_matches_main(name):
    assert get_algorithm(name).needs_full_log_probs is (name in MAIN_NEEDS_FULL_LOG_PROBS)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_needs_critic_matches_main(name):
    assert get_algorithm(name).needs_critic is (name in MAIN_NEEDS_CRITIC)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_requires_normalize_advantages_matches_main(name):
    spec = get_algorithm(name)
    assert spec.requires_normalize_advantages is (name in MAIN_REQUIRES_NORMALIZE_ADVANTAGES)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_group_normalization_matches_main(name):
    """utils.py had two overlapping whitelists; both are now spec fields."""
    normalizer = get_algorithm(name).reward_normalizer
    assert (normalizer != "none") is (name in MAIN_GROUP_NORMALIZED)
    assert (normalizer == "group_mean_std") is (name in MAIN_GROUP_STD_NORMALIZED)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_startup_constraints_match_main(name):
    spec = get_algorithm(name)
    assert spec.forbids_normalize_advantages is (name in MAIN_FORBIDS_NORMALIZE_ADVANTAGES)
    assert spec.requires_rewards_normalization is (name in MAIN_REQUIRES_REWARDS_NORMALIZATION)
    assert spec.forbids_reward_side_kl is (name in MAIN_FORBIDS_REWARD_SIDE_KL)
    assert spec.requires_global_token_loss is (name in MAIN_REQUIRES_GLOBAL_TOKEN_LOSS)
    assert spec.requires_on_policy_updates is (name in MAIN_REQUIRES_ON_POLICY_UPDATES)
    assert spec.min_group_size == MAIN_MIN_GROUP_SIZE.get(name, 1)


def test_no_algorithm_both_requires_and_forbids_advantage_normalization():
    """The two flags come from opposite sides of main's validation; a spec
    setting both would make the algorithm unlaunchable in every
    configuration."""
    for name in list_algorithm_names():
        spec = get_algorithm(name)
        assert not (spec.requires_normalize_advantages and spec.forbids_normalize_advantages), name


def test_every_algorithm_main_supported_is_still_registered():
    """A migration that quietly dropped an algorithm would pass every other
    test."""
    missing = set(MAIN_ALGORITHMS) - set(list_algorithm_names())
    assert not missing, f"{missing} were reachable on main {MAIN_SHA[:7]} and are gone now"


def test_only_gdpo_was_added():
    """Keeps the reference tables honest: any new algorithm must be listed
    here."""
    added = set(list_algorithm_names()) - set(MAIN_ALGORITHMS)
    assert added == {"gdpo"}, f"unexpected new algorithms {added}; update this file's tables"


def test_reward_normalizer_identifiers_all_resolve():
    for name in list_algorithm_names():
        assert get_algorithm(name).reward_normalizer in REWARD_NORMALIZERS


# ---------------- the adapters must be identity wrappers ----------------
#
# The co_names checks above only prove the kernel's name appears in the adapter's
# bytecode. They would still pass if the adapter scaled its input, dropped an
# argument, or threw the result away. These compare the adapter's output against
# calling the kernel directly with main's argument list, which is what actually
# pins "the wrapper adds nothing".


def _kl(lengths=(3, 2)):
    return [torch.zeros(n, dtype=torch.float32) for n in lengths]


def _masks(lengths=(3, 2)):
    return [torch.ones(n, dtype=torch.float32) for n in lengths]


def _args(estimator, **overrides):
    from types import SimpleNamespace

    base = dict(
        advantage_estimator=estimator,
        kl_coef=0.0,
        gamma=1.0,
        lambd=1.0,
        eps_clip=0.2,
        eps_clip_high=0.3,
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("name", ["grpo", "gspo", "sapo", "cispo"])
def test_grpo_family_adapter_is_the_bare_kernel(name):
    """main: torch.tensor(rewards, float32, device) then get_grpo_returns(...)."""
    from relax.algorithms.advantages import compute_advantages_and_returns

    rewards, kl = [1.5, -2.0], _kl()
    got, _ = compute_advantages_and_returns(_args(name), rewards=rewards, kl=kl, loss_masks=_masks())
    want = ppo_utils.get_grpo_returns(torch.tensor(rewards, dtype=torch.float32, device=kl[0].device), kl)

    assert len(got) == len(want)
    for left, right in zip(got, want, strict=True):
        assert torch.equal(left, right), name


def test_reinforce_plus_plus_baseline_adapter_is_the_bare_kernel():
    from relax.algorithms.advantages import compute_advantages_and_returns

    rewards, kl, masks = [3.0], [torch.tensor([2.0, 4.0])], [torch.ones(2)]
    args = _args("reinforce_plus_plus_baseline", kl_coef=0.5)

    got, returns = compute_advantages_and_returns(args, rewards=rewards, kl=kl, loss_masks=masks)
    want = ppo_utils.get_reinforce_plus_plus_baseline_advantages(
        rewards=torch.tensor(rewards, dtype=torch.float32, device=kl[0].device),
        kl=[torch.tensor([2.0, 4.0])],
        loss_masks=masks,
    )

    for left, right in zip(got, want, strict=True):
        assert torch.equal(left, right)
    assert returns is got, "main aliased returns to advantages for this estimator"


def _loss_inputs():
    torch.manual_seed(20260726)
    return torch.randn(8), torch.randn(8), torch.randn(8)


def test_ppo_clip_adapter_passes_mains_arguments():
    from relax.algorithms.policy import compute_policy_loss_for

    log_probs, ppo_kl, advantages = _loss_inputs()
    args = _args("grpo")
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = ppo_utils.compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_sapo_adapter_passes_mains_arguments():
    from relax.algorithms.policy import compute_policy_loss_for

    log_probs, ppo_kl, advantages = _loss_inputs()
    args = _args("sapo", sapo_tau_pos=1.3, sapo_tau_neg=1.7)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = ppo_utils.compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.3, tau_neg=1.7)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_sapo_adapter_uses_mains_defaults_when_args_omit_the_taus():
    from types import SimpleNamespace

    from relax.algorithms.policy import compute_policy_loss_for

    log_probs, ppo_kl, advantages = _loss_inputs()
    bare = SimpleNamespace(advantage_estimator="sapo", eps_clip=0.2, eps_clip_high=0.3)
    got = compute_policy_loss_for(bare, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = ppo_utils.compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])


def test_cispo_adapter_passes_mains_arguments():
    """The one adapter taking four kernel arguments — most room to drop one."""
    from relax.algorithms.policy import compute_policy_loss_for

    log_probs, ppo_kl, advantages = _loss_inputs()
    args = _args("cispo", eps_clip=0.15, eps_clip_high=9.0)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = ppo_utils.compute_cispo_loss(
        log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages, eps_clip=0.15, eps_clip_high=9.0
    )
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


@pytest.fixture
def cp_disabled(monkeypatch):
    """Minimal megatron.core.mpu so the reinforce++ kernel runs on CPU.

    get_reinforce_plus_plus_returns imports mpu inside the function and reads
    only get_context_parallel_world_size(); at 1 it takes the non-gathering
    branch, which is the configuration the rest of this file already assumes.
    Stubbing it is what makes the adapter testable at all -- the alternative is
    leaving the one selectable estimator with no numerical check, which is how
    it got here.
    """
    import sys
    import types

    core = types.ModuleType("megatron.core")
    core.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    megatron = types.ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    yield


def test_reinforce_plus_plus_adapter_is_the_bare_kernel(cp_disabled):
    """The one live estimator whose adapter had only a co_names check.

    Coverage said it plainly: advantage_reinforce_plus_plus was never executed
    by any test, so nothing would have caught the adapter dropping an argument
    or reordering the keyword-only ones.
    """
    from relax.algorithms.advantages import compute_advantages_and_returns

    rewards = [1.5, -2.0]
    kl = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])]
    loss_masks = [torch.ones(3), torch.ones(2)]
    response_lengths, total_lengths = [3, 2], [5, 4]
    args = _args("reinforce_plus_plus", kl_coef=0.3, gamma=0.95)

    got, returns = compute_advantages_and_returns(
        args,
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
    )
    want = ppo_utils.get_reinforce_plus_plus_returns(
        rewards=torch.tensor(rewards, dtype=torch.float32, device=kl[0].device),
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        kl_coef=0.3,
        gamma=0.95,
    )

    assert len(got) == len(want)
    for left, right in zip(got, want, strict=True):
        assert torch.equal(left, right)
    assert returns is not got, "main copied the list before returning it"


def test_reinforce_plus_plus_adapter_forwards_gamma_and_kl_coef(cp_disabled):
    """Both are read off args rather than passed through, so a swap or a
    hardcoded default would survive the identity test above if it used the
    kernel's defaults."""
    from relax.algorithms.advantages import compute_advantages_and_returns

    inputs = dict(
        rewards=[1.0, 1.0],
        kl=[torch.tensor([0.5, 0.5]), torch.tensor([0.5])],
        loss_masks=[torch.ones(2), torch.ones(1)],
        response_lengths=[2, 1],
        total_lengths=[3, 2],
    )
    low, _ = compute_advantages_and_returns(_args("reinforce_plus_plus", kl_coef=0.0, gamma=1.0), **inputs)
    high, _ = compute_advantages_and_returns(_args("reinforce_plus_plus", kl_coef=0.9, gamma=1.0), **inputs)
    assert not torch.equal(low[0], high[0]), "kl_coef is not reaching the kernel"

    g_one, _ = compute_advantages_and_returns(_args("reinforce_plus_plus", kl_coef=0.0, gamma=1.0), **inputs)
    g_half, _ = compute_advantages_and_returns(_args("reinforce_plus_plus", kl_coef=0.0, gamma=0.5), **inputs)
    assert not torch.equal(g_one[0], g_half[0]), "gamma is not reaching the kernel"


# ---------------- RLOO: the new algorithm main added while this branch was open ----------------
#
# RLOO arrived on main as inline code in three places. These compare the
# registry's versions against transcriptions of those places, so the migration
# is pinned numerically rather than only by which kernel name appears.


class _RlooSample:
    """Only what `post_process_rewards` and `group_positions` read."""

    def __init__(self, reward, group_index):
        self.reward = reward
        self.group_index = group_index

    def get_reward_value(self, args):
        return self.reward


def _rloo_reward_args(n_samples_per_prompt):
    from types import SimpleNamespace

    return SimpleNamespace(
        advantage_estimator="rloo",
        n_samples_per_prompt=n_samples_per_prompt,
        rewards_normalization=True,
        grpo_std_normalization=False,
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        reward_key=None,
    )


def _main_rloo_normalized_rewards(args, samples, raw_rewards):
    """Transcription of main utils.py:186-224, the `rloo` reward branch.

    Deliberately written out rather than imported: importing the production
    helper would compare the implementation against itself.
    """
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    positions_by_group: dict[int, list[int]] = {}
    for position, sample in enumerate(samples):
        positions_by_group.setdefault(sample.group_index, []).append(position)

    normalized_rewards = torch.empty_like(rewards)
    for positions in positions_by_group.values():
        group_rewards = rewards[positions]
        group_size = group_rewards.shape[0]
        mean_reward = group_rewards.mean()
        scale = group_size / (group_size - 1)
        normalized_rewards[positions] = scale * (group_rewards - mean_reward)
    return normalized_rewards.tolist()


def test_rloo_reward_normalizer_matches_mains_inline_branch():
    from relax.utils.utils import post_process_rewards

    raw = [1.0, 0.0, 0.5, -2.0, 3.0, 3.0, 3.0, 0.25]
    samples = [_RlooSample(value, group_index=index // 4) for index, value in enumerate(raw)]
    args = _rloo_reward_args(4)

    got_raw, got_normalized = post_process_rewards(args, samples)
    assert got_raw == raw, "main returned the raw rewards untouched alongside the normalised ones"
    assert got_normalized == _main_rloo_normalized_rewards(args, samples, raw)


def test_rloo_normalizer_is_not_the_grpo_one():
    """Both are group-centred; only GRPO divides by the group std.

    Without this, routing `rloo` to `group_mean_std` by mistake would pass the
    identity test above for any group whose std happens to be 1.
    """
    from relax.algorithms.rewards import normalize_group_mean_std
    from relax.utils.utils import post_process_rewards

    raw = [1.0, 0.0, 0.5, -2.0]
    samples = [_RlooSample(value, group_index=0) for value in raw]

    grpo_args = _rloo_reward_args(4)
    grpo_args.advantage_estimator = "grpo"
    grpo_args.grpo_std_normalization = True

    _, rloo = post_process_rewards(_rloo_reward_args(4), samples)
    assert rloo != normalize_group_mean_std(grpo_args, samples, raw)


def test_rloo_policy_loss_adapter_is_the_bare_kernel():
    """main loss.py:915-919 — `compute_rloo_loss(log_probs=...,
    advantages=...)`."""
    from relax.algorithms.policy import compute_policy_loss_for

    torch.manual_seed(0)
    log_probs, ppo_kl, advantages = torch.randn(8), torch.randn(8), torch.randn(8)

    got_loss, got_clipfrac = compute_policy_loss_for(
        _args("rloo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages
    )
    want_loss, want_clipfrac = ppo_utils.compute_rloo_loss(log_probs=log_probs, advantages=advantages)

    assert torch.equal(got_loss, want_loss)
    assert torch.equal(got_clipfrac, want_clipfrac)


def test_rloo_policy_loss_ignores_ppo_kl():
    """The unclipped objective has no ratio term; an adapter that quietly fed
    `ppo_kl` to a clipped kernel would still match the kernel-name check."""
    from relax.algorithms.policy import compute_policy_loss_for

    torch.manual_seed(0)
    log_probs, advantages = torch.randn(8), torch.randn(8)
    first, _ = compute_policy_loss_for(
        _args("rloo"), log_probs=log_probs, ppo_kl=torch.zeros(8), advantages=advantages
    )
    second, _ = compute_policy_loss_for(
        _args("rloo"), log_probs=log_probs, ppo_kl=torch.full((8,), 5.0), advantages=advantages
    )
    assert torch.equal(first, second)


def test_rloo_advantage_adapter_is_the_grpo_broadcast():
    """main folded `rloo` into the `["grpo", "gspo", "sapo", "cispo"]`
    branch."""
    from relax.algorithms.advantages import compute_advantages_and_returns

    rewards, kl = [1.5, -2.0], _kl()
    got, _ = compute_advantages_and_returns(_args("rloo"), rewards=rewards, kl=kl, loss_masks=_masks())
    want = ppo_utils.get_grpo_returns(torch.tensor(rewards, dtype=torch.float32, device=kl[0].device), kl)

    for left, right in zip(got, want, strict=True):
        assert torch.equal(left, right)


def test_loss_py_actually_forwards_the_mini_batch_boundaries():
    """Source-level, because loss.py needs megatron to import.

    The per-batch whitening tests all call advantage_gdpo directly, so removing
    the wiring in loss.py left every one of them green while GDPO silently went
    back to whitening the merged rollout. This is the assertion that fails when
    that happens.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parents[2] / "relax" / "backends" / "megatron" / "loss.py").read_text(
        encoding="utf-8"
    )

    call = re.search(r"compute_advantages_and_returns_impl\((.*?)\n    \)", src, re.DOTALL)
    assert call, "compute_advantages_and_returns_impl call not found"
    # The key is a literal in loss.py, not the imported constant: importing it
    # from `megatron.data` drags in `megatron.core.packed_seq_params` and makes
    # loss.py unimportable for its own CPU tests. `test_algos_roles.py` pins the
    # literal to the definition.
    assert 'mini_batch_sizes=rollout_data.get("rollout_mini_local_sample_counts")' in call.group(1), (
        "loss.py must pass the per-training-batch counts; without them GDPO's step 3 "
        "normalises over the whole merged rollout again"
    )


# ---------------- advantage_normalization ----------------
#
# This field is read on every optimizer step of every algorithm, and until
# these tests existed nothing pinned its value: the only test naming it
# rejected an invalid string. Dropping `"token_global"` from a REINFORCE++ spec
# (silently reverting it to the `"whiten"` default) or flipping the comparison
# in loss.py left the whole suite green.


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_advantage_normalization_matches_main(name):
    expected = "token_global" if name in MAIN_TOKEN_GLOBAL else "whiten"
    assert get_algorithm(name).advantage_normalization == expected


def test_exactly_mains_algorithms_take_the_token_global_path():
    """Pins the set, not just the members.

    A per-algorithm check cannot fail when a *new* algorithm defaults into the
    wrong branch, and the default is the branch every existing algorithm but
    two takes -- so the mistake it would miss is the likely one.
    """
    token_global = {n for n in list_algorithm_names() if get_algorithm(n).advantage_normalization == "token_global"}
    assert token_global == MAIN_TOKEN_GLOBAL


def test_loss_py_selects_both_behaviours_from_the_field():
    """Both of main's call sites must read the spec, not the algorithm name.

    Source-level because importing `loss.py` needs Megatron. It is paired with
    the value tests above: those pin what the field says, this pins that the
    two branches still ask it.
    """
    import pathlib

    import relax

    src = (pathlib.Path(relax.__file__).parent / "backends/megatron/loss.py").read_text()

    assert src.count('advantage_normalization == "token_global"') == 2
    for literal in MAIN_TOKEN_GLOBAL:
        assert f'"{literal}"' not in src, f"loss.py still names {literal} directly"


def test_gae_adapter_reproduces_mains_reward_shaping(cp_disabled):
    """PPO's adapter, run against a transcription of main's inline branch.

    The only estimator adapter with no numerical check: `advantage_gae` had a
    co_names test, which cannot see a dropped argument, a sign flip on the KL
    coefficient, or the terminal-reward injection landing on the wrong token.
    PPO is also the algorithm whose surroundings this refactor changed most
    (it is the one with a critic) and the one with no GPU smoke, so a bug here
    would have had the least chance of being caught anywhere else.

    main @ 98a1274 loss.py:584-602, transcribed rather than imported:

        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd,
            padded_total_lengths=padded_total_lengths,
        )
    """
    from relax.algorithms.advantages import compute_advantages_and_returns

    def _inputs():
        return dict(
            rewards=[1.5, -2.0],
            kl=[torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])],
            values=[torch.tensor([0.7, 0.8, 0.9]), torch.tensor([1.1, 1.2])],
            response_lengths=[3, 2],
            total_lengths=[5, 4],
        )

    args = _args("ppo", kl_coef=0.3, gamma=0.95, lambd=0.9)

    # main's branch, on its own copy of the tensors -- the shaping mutates `kl`
    # in place (`k *= kl_coef`), so the two runs must not share them.
    ref = _inputs()
    shaped = []
    for reward, k in zip(ref["rewards"], ref["kl"], strict=False):
        k *= -args.kl_coef
        k[-1] += reward  # cp_rank == 0 under `cp_disabled`
        shaped.append(k)
    want_adv, want_ret = ppo_utils.get_advantages_and_returns_batch(
        ref["total_lengths"],
        ref["response_lengths"],
        ref["values"],
        shaped,
        args.gamma,
        args.lambd,
        padded_total_lengths=None,
    )

    got_adv, got_ret = compute_advantages_and_returns(args, **_inputs())

    for left, right in zip(got_adv, want_adv, strict=True):
        assert torch.equal(left, right)
    for left, right in zip(got_ret, want_ret, strict=True):
        assert torch.equal(left, right)


def test_gae_adapter_actually_uses_kl_coef_and_gamma(cp_disabled):
    """Guards the transcription above from agreeing by coincidence.

    If the adapter ignored either argument, the test above would still pass as
    long as the reference ignored it too.
    """
    from relax.algorithms.advantages import compute_advantages_and_returns

    def _run(**overrides):
        knobs = dict(kl_coef=0.3, gamma=0.95, lambd=0.9)
        knobs.update(overrides)
        adv, _ = compute_advantages_and_returns(
            _args("ppo", **knobs),
            rewards=[1.5, -2.0],
            kl=[torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])],
            values=[torch.tensor([0.7, 0.8, 0.9]), torch.tensor([1.1, 1.2])],
            response_lengths=[3, 2],
            total_lengths=[5, 4],
        )
        return torch.cat(adv)

    baseline = _run()
    assert not torch.allclose(baseline, _run(kl_coef=0.9))
    assert not torch.allclose(baseline, _run(gamma=0.5))
    assert not torch.allclose(baseline, _run(lambd=0.1))
