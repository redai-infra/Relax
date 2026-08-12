# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GDPO step 3 must whiten over the whole batch, not each rank's shard.

These spawn a real two-process gloo group so the collectives are exercised for
real: a shard-local implementation, a missing collective on an empty shard, or a
mismatched call order would all show up here rather than only on a GPU cluster.
"""

import os

import pytest


torch = pytest.importorskip("torch")
mp = pytest.importorskip("torch.multiprocessing")

import torch.distributed as dist  # noqa: E402


# Two ranks with deliberately different spreads: shard 1 carries twice the
# amplitude of shard 0, which only survives whitening if the statistics are
# shared.
SHARDS = [[-0.7, 0.7], [-1.4, 1.4]]


def _run(rank, world_size, port, mode, out):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from relax.algorithms.advantages import whiten_scalar
        from relax.algorithms.numerics import distributed_mean_std, is_collapsed

        group = dist.group.WORLD
        if mode == "whiten":
            values = torch.tensor(SHARDS[rank], dtype=torch.float32)
            out[rank] = whiten_scalar(values, process_group=group).tolist()
        elif mode == "empty_shard":
            # Rank 1 has nothing to contribute; rank 0 must not hang on it.
            values = torch.tensor([1.0, 3.0] if rank == 0 else [], dtype=torch.float32)
            out[rank] = whiten_scalar(values, process_group=group).tolist()
        elif mode == "stats":
            values = torch.tensor(SHARDS[rank], dtype=torch.float32)
            mean, std = distributed_mean_std(values, process_group=group)
            out[rank] = [mean.item(), std.item()]
        elif mode == "collapsed_across_ranks":
            # Identical on both ranks: only a global view can tell.
            values = torch.tensor([0.7, 0.7], dtype=torch.float32)
            out[rank] = [is_collapsed(values, process_group=group)]
        elif mode == "collapsed_only_locally":
            # Each shard is constant on its own but the batch is not.
            values = torch.tensor([0.7, 0.7] if rank == 0 else [1.4, 1.4], dtype=torch.float32)
            out[rank] = [is_collapsed(values, process_group=group)]
        else:  # pragma: no cover - guard against typos in the test itself
            raise AssertionError(mode)
    finally:
        dist.destroy_process_group()


def _spawn(mode, world_size=2):
    manager = mp.Manager()
    out = manager.dict()
    port = 29500 + abs(hash(mode)) % 2000
    mp.spawn(_run, args=(world_size, port, mode, out), nprocs=world_size, join=True)
    return dict(out)


def test_statistics_are_shared_across_ranks():
    out = _spawn("stats")
    everything = torch.tensor(SHARDS[0] + SHARDS[1])
    for rank in (0, 1):
        mean, std = out[rank]
        assert abs(mean - everything.mean().item()) < 1e-5, rank
        assert abs(std - everything.std().item()) < 1e-4, rank


def test_whitening_keeps_the_larger_shard_larger():
    """Shard-local whitening flattens both shards onto the same amplitude."""
    out = _spawn("whiten")
    shard0 = torch.tensor(out[0])
    shard1 = torch.tensor(out[1])

    assert shard1.abs().max() > shard0.abs().max() * 1.5
    joint = torch.cat([shard0, shard1])
    assert abs(joint.mean().item()) < 1e-5
    assert abs(joint.std().item() - 1.0) < 1e-2


def test_an_empty_shard_does_not_hang_the_group():
    """The empty rank still has to reach every collective."""
    out = _spawn("empty_shard")
    assert out[1] == []
    assert len(out[0]) == 2
    assert out[0][0] < 0 < out[0][1]


def test_collapse_is_decided_globally_not_per_shard():
    assert _spawn("collapsed_across_ranks") == {0: [True], 1: [True]}

    per_shard = _spawn("collapsed_only_locally")
    assert per_shard == {0: [False], 1: [False]}, "each shard is constant, the batch is not"
