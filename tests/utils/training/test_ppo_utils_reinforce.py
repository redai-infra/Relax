# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Numerical-correctness tests for the REINFORCE++ / REINFORCE++-baseline
advantage estimators (task #29).

Every test compares the Relax implementation in
``relax.utils.training.ppo_utils`` against an **independent reference
implementation** written from scratch in this module (plain torch, no Megatron,
no ``@torch.compile``). The references follow the formulas documented in
``docs/algorithms/reinforce_plus_plus.md`` and the original paper
arXiv:2501.03262.

Coverage targets (acceptance criteria of task #29):
  * element-by-element agreement with an independent reference;
  * variable-length responses;
  * all-zero rewards;
  * single-sample batches;
  * mask does not pollute the reward injection / padding statistics.

The CP>1 distributed path is exercised separately in
``tests/backends/megatron/test_reinforce_pp_cp_parity.py``; here we force
``cp_size == 1`` via a fake ``megatron.core.mpu`` so the per-rank math is tested
in isolation.
"""

import sys
from types import ModuleType

import pytest
import torch


def install_fake_megatron(monkeypatch, *, cp_size: int = 1) -> None:
    """Install a minimal fake ``megatron.core.mpu`` with the given ``cp_size``.

    Mirrors the convention in
    ``tests/backends/megatron/test_ppo_gae_parity.py`` so ``ppo_utils`` can be
    imported and its CP-aware code path runs in plain single-process mode. When
    ``cp_size == 1`` the ``all_gather_with_cp`` / ``slice_log_prob_with_cp``
    helpers are never reached.
    """
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_world_size = lambda: cp_size
    core.mpu = mpu

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)


# ---------------------------------------------------------------------------
# Independent reference implementations (do NOT call Relax code).
# ---------------------------------------------------------------------------


def reference_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """Plain-torch reference for ``get_reinforce_plus_plus_returns``
    (cp_size=1).

    For each sequence the per-token reward is ``-kl_coef * (kl * mask)`` with the
    scalar terminal reward added at the last masked (response) token, followed by
    a discounted reverse cumulative sum ``G_t = r_t + gamma * G_{t+1}``.
    """
    returns = []
    for i in range(len(rewards)):
        mask = loss_masks[i]
        token_rewards = -kl_coef * (kl[i] * mask)
        last_idx = mask.nonzero(as_tuple=True)[0][-1]
        token_rewards[last_idx] = token_rewards[last_idx] + rewards[i]

        g = torch.zeros_like(token_rewards)
        running = torch.zeros((), dtype=token_rewards.dtype, device=token_rewards.device)
        for t in reversed(range(token_rewards.size(0))):
            running = token_rewards[t] + gamma * running
            g[t] = running
        returns.append(g)
    return returns


def reference_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Plain-torch reference for
    ``get_reinforce_plus_plus_baseline_advantages``.

    ``advantage = (reward - group_baseline)`` broadcast to every token, with no
    per-token KL penalty (paper arXiv:2501.03262 §3.2: the baseline variant
    applies a separate k2 KL loss instead of folding KL into the advantage).
    The group baseline is assumed already subtracted from ``rewards`` (done
    upstream in ``relax.utils.utils.post_process_rewards``); this reference
    therefore mirrors the advantage function exactly: broadcast the scalar
    reward to the per-token shape.
    """
    return [torch.ones_like(kl_tensor) * reward_val for kl_tensor, reward_val in zip(kl, rewards, strict=False)]


def reference_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch reference for ``compute_policy_loss`` (no dual-clip).

    Standard PPO clipped surrogate: ``pg = -max(r*A, clip(r)*A)`` with
    ``r = exp(-ppo_kl)`` and ``clipfrac`` flagging tokens where the clip binds.
    """
    ratio = (-ppo_kl).exp()
    pg1 = -ratio * advantages
    pg2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    pg_loss = torch.maximum(pg1, pg2)
    clipfrac = torch.gt(pg2, pg1).float()
    return pg_loss, clipfrac


# ---------------------------------------------------------------------------
# Helpers to build synthetic batches with variable lengths.
# ---------------------------------------------------------------------------


def make_batch(seq_lens, n_response_per_seq, *, reward_fn, seed: int = 0):
    """Build (rewards, kl, loss_masks, response_lengths, total_lengths).

    ``seq_lens`` are full sequence lengths; ``n_response_per_seq`` how many of
    the trailing tokens are unmasked (response). The leading tokens are treated
    as masked (prompt / padding) so mask-pollution behaviour is exercised.
    """
    torch.manual_seed(seed)
    kl, masks = [], []
    for total_len, resp_len in zip(seq_lens, n_response_per_seq, strict=False):
        k = torch.randn(total_len, dtype=torch.float32)
        m = torch.zeros(total_len, dtype=torch.float32)
        m[-resp_len:] = 1.0
        kl.append(k)
        masks.append(m)
    rewards = reward_fn(len(seq_lens))
    response_lengths = list(n_response_per_seq)
    total_lengths = list(seq_lens)
    return rewards, kl, masks, response_lengths, total_lengths


# ---------------------------------------------------------------------------
# REINFORCE++ (Monte-Carlo discounted return) tests.
# ---------------------------------------------------------------------------


class TestReinforcePlusPlusReturns:
    """Element-by-element correctness of
    ``get_reinforce_plus_plus_returns``."""

    def test_matches_reference_variable_length(self, monkeypatch):
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_returns

        rewards, kl, masks, resp_lens, total_lens = make_batch(
            [12, 7, 20],
            [5, 3, 9],
            reward_fn=lambda n: torch.randn(n),
        )

        actual = get_reinforce_plus_plus_returns(
            rewards,
            kl,
            masks,
            resp_lens,
            total_lens,
            kl_coef=0.04,
            gamma=0.99,
        )
        expected = reference_reinforce_plus_plus_returns(rewards, kl, masks, 0.04, 0.99)

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=False):
            assert a.shape == e.shape
            assert torch.allclose(a, e, atol=1e-6), (a, e)

    def test_gamma_one_is_discounted_reverse_cumsum(self, monkeypatch):
        """gamma=1 ⇒ G_t = sum of token rewards from t to end (reverse cumsum)."""
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_returns

        rewards, kl, masks, resp_lens, total_lens = make_batch(
            [10, 8],
            [4, 4],
            reward_fn=lambda n: torch.tensor([2.0, -1.0]),
        )

        actual = get_reinforce_plus_plus_returns(
            rewards,
            kl,
            masks,
            resp_lens,
            total_lens,
            kl_coef=0.1,
            gamma=1.0,
        )
        expected = reference_reinforce_plus_plus_returns(rewards, kl, masks, 0.1, 1.0)
        for a, e in zip(actual, expected, strict=False):
            assert torch.allclose(a, e, atol=1e-6)

        # Sanity: with gamma=1, G at the first token equals total token-reward sum.
        for i, (k, m, r) in enumerate(zip(kl, masks, rewards, strict=False)):
            token_rewards = -0.1 * (k * m)
            token_rewards[m.nonzero(as_tuple=True)[0][-1]] += r
            assert torch.allclose(actual[i][0], token_rewards.sum(), atol=1e-6)

    def test_all_zero_reward(self, monkeypatch):
        """All-zero rewards ⇒ returns reduce to discounted -kl_coef*kl on
        response."""
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_returns

        rewards, kl, masks, resp_lens, total_lens = make_batch(
            [9, 14],
            [4, 6],
            reward_fn=lambda n: torch.zeros(n),
        )

        actual = get_reinforce_plus_plus_returns(
            rewards,
            kl,
            masks,
            resp_lens,
            total_lens,
            kl_coef=0.05,
            gamma=0.97,
        )
        expected = reference_reinforce_plus_plus_returns(rewards, kl, masks, 0.05, 0.97)
        for a, e in zip(actual, expected, strict=False):
            assert torch.allclose(a, e, atol=1e-6)

        # No reward injection: the terminal token carries only the KL penalty.
        for i, m in enumerate(masks):
            last_idx = m.nonzero(as_tuple=True)[0][-1]
            assert torch.allclose(actual[i][last_idx], -0.05 * kl[i][last_idx], atol=1e-6)

    def test_single_sample(self, monkeypatch):
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_returns

        rewards, kl, masks, resp_lens, total_lens = make_batch(
            [6],
            [3],
            reward_fn=lambda n: torch.tensor([1.0]),
        )

        actual = get_reinforce_plus_plus_returns(
            rewards,
            kl,
            masks,
            resp_lens,
            total_lens,
            kl_coef=0.0,
            gamma=0.9,
        )
        expected = reference_reinforce_plus_plus_returns(rewards, kl, masks, 0.0, 0.9)
        assert torch.allclose(actual[0], expected[0], atol=1e-6)

        # kl_coef=0 ⇒ only the terminal reward propagates, discounted backwards.
        last_idx = masks[0].nonzero(as_tuple=True)[0][-1]
        assert torch.allclose(actual[0][last_idx], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(actual[0][last_idx - 1], torch.tensor(0.9), atol=1e-6)

    def test_reward_only_at_last_masked_token(self, monkeypatch):
        """The scalar reward must be injected at the last *response* token
        only, never into masked prompt/padding positions, and padding after the
        response must stay zero."""
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_returns

        # 8-token sequence: 2 prompt (masked), 3 response (unmasked), 3 padding.
        total_len, resp_len = 8, 3
        kl = [torch.zeros(total_len, dtype=torch.float32)]
        masks = [torch.tensor([0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.float32)]
        rewards = torch.tensor([5.0])

        actual = get_reinforce_plus_plus_returns(
            rewards,
            kl,
            masks,
            [resp_len],
            [total_len],
            kl_coef=0.0,
            gamma=1.0,
        )[0]

        last_idx = 4  # last response token
        assert torch.allclose(actual[last_idx], torch.tensor(5.0), atol=1e-6)
        # padding tokens after the response carry no return
        assert torch.allclose(actual[5:], torch.zeros(3), atol=1e-6)
        # prompt tokens before the response carry the discounted return forward
        assert torch.allclose(actual[:2], torch.full((2,), 5.0), atol=1e-6)

    def test_rejects_fully_masked_sequence(self, monkeypatch):
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_returns

        kl = [torch.randn(5)]
        masks = [torch.zeros(5)]  # fully masked ⇒ invalid
        with pytest.raises(AssertionError, match="fully masked"):
            get_reinforce_plus_plus_returns(
                torch.tensor([1.0]),
                kl,
                masks,
                [0],
                [5],
                kl_coef=0.0,
                gamma=1.0,
            )


# ---------------------------------------------------------------------------
# REINFORCE++-baseline advantage tests.
# ---------------------------------------------------------------------------


class TestReinforcePlusPlusBaselineAdvantages:
    """Element-by-element correctness of
    ``get_reinforce_plus_plus_baseline_advantages``."""

    def test_matches_reference(self, monkeypatch):
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_baseline_advantages

        # rewards here already carry the group baseline subtracted upstream.
        rewards = torch.tensor([0.8, -0.6, 0.1])
        kl = [torch.randn(6), torch.randn(4), torch.randn(9)]
        masks = [torch.ones(6), torch.ones(4), torch.ones(9)]

        actual = get_reinforce_plus_plus_baseline_advantages(rewards, kl, masks)
        expected = reference_reinforce_plus_plus_baseline_advantages(rewards, kl)

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=False):
            assert a.shape == e.shape
            assert torch.allclose(a, e, atol=1e-6)

    def test_broadcast_scalar(self, monkeypatch):
        """The (reward - baseline) scalar must be identical on every token."""
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_baseline_advantages

        rewards = torch.tensor([2.0])
        kl = [torch.randn(7)]
        masks = [torch.ones(7)]

        adv = get_reinforce_plus_plus_baseline_advantages(rewards, kl, masks)[0]
        assert torch.allclose(adv, torch.full((7,), 2.0), atol=1e-6)

    def test_kl_not_folded_into_advantage(self, monkeypatch):
        """The baseline variant's advantage ignores per-token KL values.

        Paper convention (arXiv:2501.03262 §3.2): the advantage is the
        broadcast (reward - group baseline) only. Per-token KL must NOT change
        it (KL is a separate k2 loss applied in policy_loss_function, not
        here).
        """
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_baseline_advantages

        rewards = torch.tensor([0.5, -0.5])
        # deliberately non-trivial KL values — they must be ignored
        kl = [torch.randn(5), torch.randn(3)]
        masks = [torch.ones(5), torch.ones(3)]

        adv = get_reinforce_plus_plus_baseline_advantages(rewards, kl, masks)
        for a, r in zip(adv, rewards, strict=False):
            assert torch.allclose(a, torch.full_like(a, r.item()), atol=1e-6)

    def test_single_sample(self, monkeypatch):
        pytest.importorskip("torch")
        install_fake_megatron(monkeypatch)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_baseline_advantages

        rewards = torch.tensor([1.0])
        kl = [torch.randn(4)]
        masks = [torch.ones(4)]
        actual = get_reinforce_plus_plus_baseline_advantages(rewards, kl, masks)
        expected = reference_reinforce_plus_plus_baseline_advantages(rewards, kl)
        assert torch.allclose(actual[0], expected[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Shared policy loss (REINFORCE++ variants reuse compute_policy_loss).
# ---------------------------------------------------------------------------


class TestPolicyLossForReinforce:
    """REINFORCE++ variants fall into the ``else`` branch of
    policy_loss_function and reuse ``compute_policy_loss``; verify it against
    an independent ref."""

    def test_matches_reference(self):
        pytest.importorskip("torch")
        from relax.utils.training.ppo_utils import compute_policy_loss as _compiled

        compute_policy_loss = torch.compiler.disable(_compiled)

        torch.manual_seed(1)
        ppo_kl = torch.randn(50)
        advantages = torch.randn(50)
        eps_clip, eps_clip_high = 0.2, 0.28

        actual_loss, actual_clip = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)
        exp_loss, exp_clip = reference_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)

        assert torch.allclose(actual_loss, exp_loss, atol=1e-6)
        assert torch.allclose(actual_clip, exp_clip, atol=1e-6)
