# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for fail-closed TransferQueue benchmark teardown."""

import pytest

from scripts.benchmarks import tq_cross_node_bench, tq_rdma_bench


@pytest.mark.parametrize(
    "wait_actor_gone",
    [tq_cross_node_bench.wait_actor_gone, tq_rdma_bench.wait_actor_gone],
)
def test_actor_wait_timeout_is_not_silently_ignored(wait_actor_gone):
    with pytest.raises(TimeoutError, match="still registered"):
        wait_actor_gone(timeout=0)
