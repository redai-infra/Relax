# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Distributed tests for REINFORCE++ / REINFORCE++-baseline (task #29).

Two properties are verified in single-process mode with distributed primitives
mocked (the convention used by ``test_ppo_gae_parity.py``: a fake
``megatron.core.mpu`` plus mocked collectives):

  * **CP gather/compute/slice wiring** — ``get_reinforce_plus_plus_returns`` is
    CP-aware: each rank must gather the full response (``all_gather_with_cp``),
    compute the discounted return on the full sequence, then slice its local
    chunk back out (``slice_log_prob_with_cp``). We mock the two helpers with a
    contiguous split and assert that concatenating the per-rank local chunks
    reproduces the full-sequence return computed with ``cp_size == 1``. This
    validates the gather/compute/slice wiring of the REINFORCE++ return logic;
    the zig-zag chunking itself lives in ``relax.backends.megatron.cp_utils``
    (shared infra, covered by its own tests) and is out of scope here.

  * **DP whitening invariance** — REINFORCE++ variants require
    ``--normalize-advantages``, which whitens advantages over the DP group via
    ``distributed_masked_whiten`` (global masked mean/var). The whitened value
    of every token must be identical regardless of how samples are partitioned
    across DP ranks. We mock ``dist.all_reduce`` to aggregate a second rank's
    statistics and check each rank's output matches the unpartitioned reference.
"""

import sys
from types import ModuleType

import pytest
import torch


def _install_fake_megatron(monkeypatch, *, cp_size: int = 1, cp_rank: int = 0) -> None:
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_world_size = lambda: cp_size
    mpu.get_context_parallel_rank = lambda: cp_rank
    core.mpu = mpu

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)


class _CPSplitSimulator:
    """Mocks ``all_gather_with_cp`` / ``slice_log_prob_with_cp`` for cp_size=2
    using a contiguous split, so the gather→compute→slice roundtrip can be
    validated in a single process.

    ``full`` is the canonical full-response tensor; each rank's local chunk is
    ``full[:mid]`` (rank 0) / ``full[mid:]`` (rank 1). ``all_gather`` returns
    the full tensor; ``slice`` returns the calling rank's chunk of the full
    result.
    """

    def __init__(self, full: torch.Tensor):
        self.full = full
        self.mid = full.size(0) // 2

    def local_chunk(self, rank: int) -> torch.Tensor:
        return self.full[: self.mid] if rank == 0 else self.full[self.mid :]

    def all_gather(self, tensor, total_length, response_length, *args, **kwargs):
        # Simulate reconstructing the full response from all CP ranks.
        return self.full

    def slice(self, log_prob, total_length, response_length, *args, **kwargs):
        # Return the calling rank's contiguous chunk of the full result.
        return log_prob[: self.mid] if self._rank == 0 else log_prob[self.mid :]

    def set_rank(self, rank: int) -> None:
        self._rank = rank


class TestReinforcePlusPlusCPWiring:
    """Wiring check: the CP-aware return must gather the full response before
    computing, then slice consistently.

    Zig-zag chunking correctness is ``cp_utils``' responsibility (shared infra)
    and is out of scope here.
    """

    def test_local_chunks_concatenate_to_full_return(self, monkeypatch):
        pytest.importorskip("torch")
        _install_fake_megatron(monkeypatch, cp_size=1)  # for the cp=1 reference call
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_returns

        torch.manual_seed(0)
        prompt_len, response_len = 3, 8
        total_len = prompt_len + response_len
        kl_full = torch.randn(response_len, dtype=torch.float32)
        mask_full = torch.ones(response_len, dtype=torch.float32)
        rewards = torch.tensor([1.5])
        kl_coef, gamma = 0.05, 0.97

        # Reference: full-sequence return with cp_size == 1.
        ref = get_reinforce_plus_plus_returns(
            rewards,
            [kl_full],
            [mask_full],
            [response_len],
            [total_len],
            kl_coef=kl_coef,
            gamma=gamma,
        )[0]

        # Simulate cp_size == 2: each rank gathers full, computes, slices local.
        sim = _CPSplitSimulator(kl_full)
        import relax.backends.megatron.cp_utils as cp_utils

        monkeypatch.setattr(cp_utils, "all_gather_with_cp", sim.all_gather)
        monkeypatch.setattr(cp_utils, "slice_log_prob_with_cp", sim.slice)
        _install_fake_megatron(monkeypatch, cp_size=2, cp_rank=0)

        local_chunks = []
        for rank in (0, 1):
            sim.set_rank(rank)
            local = get_reinforce_plus_plus_returns(
                rewards,
                [sim.local_chunk(rank)],
                [mask_full],
                [response_len],
                [total_len],
                kl_coef=kl_coef,
                gamma=gamma,
            )[0]
            local_chunks.append(local)

        reconstructed = torch.cat(local_chunks)
        assert reconstructed.shape == ref.shape
        assert torch.allclose(reconstructed, ref, atol=1e-6), (reconstructed, ref)


class TestReinforcePlusPlusBaselineCPLocality:
    """``get_reinforce_plus_plus_baseline_advantages`` is purely per-sample (no
    CP gather/slice), so its output is local to each rank and independent of CP
    sharding — verified here by computing on halves and concatenating."""

    def test_local_chunk_advantage_matches_full_slice(self, monkeypatch):
        pytest.importorskip("torch")
        _install_fake_megatron(monkeypatch, cp_size=2, cp_rank=0)
        from relax.utils.training.ppo_utils import get_reinforce_plus_plus_baseline_advantages

        torch.manual_seed(0)
        kl_full = torch.randn(8, dtype=torch.float32)
        rewards = torch.tensor([0.7])

        full_adv = get_reinforce_plus_plus_baseline_advantages(
            rewards,
            [kl_full],
            [torch.ones(8)],
        )[0]

        mid = kl_full.size(0) // 2
        rank0_local = get_reinforce_plus_plus_baseline_advantages(
            rewards,
            [kl_full[:mid]],
            [torch.ones(mid)],
        )[0]
        rank1_local = get_reinforce_plus_plus_baseline_advantages(
            rewards,
            [kl_full[mid:]],
            [torch.ones(8 - mid)],
        )[0]

        reconstructed = torch.cat([rank0_local, rank1_local])
        assert torch.allclose(reconstructed, full_adv, atol=1e-6)


class TestDPMaskedWhitenInvariance:
    """DP partition must not change ``distributed_masked_whiten`` per-token
    output (the ``--normalize-advantages`` path used by REINFORCE++
    variants)."""

    @staticmethod
    def _local_stats(values, mask):
        return torch.tensor(
            [(values * mask).sum(), ((values**2) * mask).sum(), mask.sum()],
            dtype=torch.float32,
        )

    def test_partitioned_ranks_match_unpartitioned_reference(self, monkeypatch):
        pytest.importorskip("torch")
        import relax.utils.distributed_utils as du

        torch.manual_seed(0)
        full_advs = torch.randn(40, dtype=torch.float32)
        full_masks = torch.ones(40, dtype=torch.float32)
        # introduce some masked (padding) tokens so masked stats are exercised
        full_masks[5:8] = 0
        full_masks[22:26] = 0

        # Reference: whiten the full batch as a single rank. With one rank,
        # all_reduce is identity (the fake adds zeros).
        monkeypatch.setattr(du.dist, "all_reduce", lambda t, *a, **k: t)
        from relax.utils.distributed_utils import distributed_masked_whiten

        ref_whitened = distributed_masked_whiten(full_advs, full_masks, process_group=None)

        # DP=2: split into two ranks; each rank's all_reduce adds the OTHER
        # rank's real local stats (simulating the cross-rank SUM).
        mid = 20
        advs0, advs1 = full_advs[:mid], full_advs[mid:]
        masks0, masks1 = full_masks[:mid], full_masks[mid:]
        stats0 = self._local_stats(advs0, masks0)
        stats1 = self._local_stats(advs1, masks1)

        def fake_for(other_stats):
            def _all_reduce(tensor, *args, **kwargs):
                tensor.add_(other_stats)
                return tensor

            return _all_reduce

        monkeypatch.setattr(du.dist, "all_reduce", fake_for(stats1))
        whitened0 = distributed_masked_whiten(advs0, masks0, process_group=None)
        monkeypatch.setattr(du.dist, "all_reduce", fake_for(stats0))
        whitened1 = distributed_masked_whiten(advs1, masks1, process_group=None)

        reconstructed = torch.cat([whitened0, whitened1])
        assert reconstructed.shape == ref_whitened.shape
        assert torch.allclose(reconstructed, ref_whitened, atol=1e-6), (reconstructed, ref_whitened)
