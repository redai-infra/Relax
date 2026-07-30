# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for P3O's two-pass replay guards and stat-accumulation scope.

The pieces under test here are the ones that decide *which tokens* enter ESS and
*whether the window can be replayed* -- the two places where a wrong answer still
produces a plausible-looking loss curve. The distributed matrix (DP/CP/TP/PP) and
the end-to-end training run require multi-GPU and are covered separately.
"""

import pytest
import torch

from relax.utils.training.p3o_replay import (
    preserved_iterator_positions,
    preserved_rng_state,
)
from relax.utils.training.p3o_utils import (
    P3OSufficientStats,
    finalize_p3o_step_context,
)


TOL = dict(rel=1e-6, abs=1e-6)


class _FakeIterator:
    """Minimal stand-in exposing the replay contract used by the pre-pass."""

    def __init__(self, items):
        self.items = list(items)
        self.offset = 0

    def __next__(self):
        if self.offset >= len(self.items):
            raise StopIteration
        item = self.items[self.offset]
        self.offset += 1
        return item

    def snapshot_position(self) -> int:
        return self.offset

    def restore_position(self, position: int) -> None:
        self.offset = position


def test_p3o_iterator_positions_restored_after_prepass():
    iterator = _FakeIterator(range(6))
    next(iterator)
    next(iterator)
    assert iterator.offset == 2

    with preserved_iterator_positions([iterator]):
        next(iterator)
        next(iterator)
        assert iterator.offset == 4

    # Restores to mid-rollout position, not to zero.
    assert iterator.offset == 2


def test_p3o_iterator_positions_restored_even_when_prepass_raises():
    iterator = _FakeIterator(range(6))
    next(iterator)

    with pytest.raises(RuntimeError, match="boom"):
        with preserved_iterator_positions([iterator]):
            next(iterator)
            raise RuntimeError("boom")

    assert iterator.offset == 1


def test_p3o_duplicate_iterator_instances_restored_once():
    """Virtual PP passes the same iterator once per model chunk."""
    iterator = _FakeIterator(range(6))
    next(iterator)

    with preserved_iterator_positions([iterator, iterator, None]):
        next(iterator)

    assert iterator.offset == 1


def test_p3o_non_replayable_iterator_is_rejected_loudly():
    class _Opaque:
        pass

    with pytest.raises(RuntimeError, match="not replayable"):
        with preserved_iterator_positions([_Opaque()]):
            pass


def test_p3o_rng_state_restored_after_prepass():
    torch.manual_seed(1234)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    with preserved_rng_state():
        # Burn RNG inside the pre-pass, as a stochastic forward would.
        torch.randn(16)
    actual = torch.randn(4)

    torch.testing.assert_close(actual, expected)


def test_p3o_stats_accumulate_then_reduce_equals_single_shot():
    """Sum-then-reduce must equal computing over the concatenated token set."""
    shards = [
        P3OSufficientStats(
            sum_ratio=torch.tensor(1.5, dtype=torch.float64),
            sum_ratio_sq=torch.tensor(2.25, dtype=torch.float64),
            valid_token_count=torch.tensor(1.0, dtype=torch.float64),
        ),
        P3OSufficientStats(
            sum_ratio=torch.tensor(6.0, dtype=torch.float64),
            sum_ratio_sq=torch.tensor(19.0, dtype=torch.float64),
            valid_token_count=torch.tensor(3.0, dtype=torch.float64),
        ),
    ]
    total = shards[0] + shards[1]

    assert float(total.sum_ratio) == pytest.approx(7.5, **TOL)
    assert float(total.sum_ratio_sq) == pytest.approx(21.25, **TOL)
    assert float(total.valid_token_count) == 4.0
    assert float(finalize_p3o_step_context(total).normalized_ess) == pytest.approx(0.6617647055709343, **TOL)


def test_p3o_dummy_microbatch_contributes_nothing():
    """Dummy micro-batches align DP counts and must not move ESS."""
    real = P3OSufficientStats(
        sum_ratio=torch.tensor(7.5, dtype=torch.float64),
        sum_ratio_sq=torch.tensor(21.25, dtype=torch.float64),
        valid_token_count=torch.tensor(4.0, dtype=torch.float64),
    )
    with_dummy = real + P3OSufficientStats.zeros()

    assert float(finalize_p3o_step_context(with_dummy).normalized_ess) == pytest.approx(
        float(finalize_p3o_step_context(real).normalized_ess), **TOL
    )
