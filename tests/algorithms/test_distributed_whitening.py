# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GDPO step 3 must whiten over the whole batch, not each rank's shard.

These spawn a real two-process gloo group so the collectives are exercised for
real: a shard-local implementation, a missing collective on an empty shard, or a
mismatched call order would all show up here rather than only on a GPU cluster.
"""

import datetime
import os
import sys

import pytest


torch = pytest.importorskip("torch")
mp = pytest.importorskip("torch.multiprocessing")

import torch.distributed as dist  # noqa: E402


# Two ranks with deliberately different spreads: shard 1 carries twice the
# amplitude of shard 0, which only survives whitening if the statistics are
# shared.
SHARDS = [[-0.7, 0.7], [-1.4, 1.4]]


def _t(values):
    return torch.tensor(values, dtype=torch.float32)


# Per-rank (values, mini_batch_sizes) for the segmented cases. The shard sizes
# differ between ranks on purpose: a data-parallel split is balanced by tokens,
# not by sample count, so equal-sized shards are the easy case rather than the
# real one.
_SEGMENT_MODES = {
    # Two segments on both ranks, split 2+2 on rank 0 and 1+3 on rank 1.
    # Segment 0 spans [1, 3 | 5] and segment 1 spans [10, 30 | 50, 70, 90], so a
    # per-segment statistic and a whole-shard statistic give different answers.
    "segmented_uneven": lambda rank: (
        (_t([1.0, 3.0, 10.0, 30.0]), [2, 2]) if rank == 0 else (_t([5.0, 50.0, 70.0, 90.0]), [1, 3])
    ),
    # Rank 0 wants two segments, rank 1 wants one. Two collectives against one:
    # the deadlock this check exists to prevent.
    "segment_count_mismatch": lambda rank: (
        (_t([1.0, 3.0, 10.0, 30.0]), [2, 2]) if rank == 0 else (_t([5.0, 50.0, 70.0, 90.0]), [4])
    ),
    # Rank 1's metadata does not describe its shard. Raising locally would strand
    # rank 0 inside the collective sequence.
    "malformed_on_one_rank": lambda rank: (
        (_t([1.0, 3.0, 10.0, 30.0]), [2, 2]) if rank == 0 else (_t([5.0, 50.0, 70.0, 90.0]), [2, 99])
    ),
    # Rank 0 passes no segmentation at all (one whole-shard window) while rank 1
    # asks for two. Same deadlock, reached through the `None` branch.
    "none_versus_segmented": lambda rank: (
        (_t([1.0, 3.0, 10.0, 30.0]), None) if rank == 0 else (_t([5.0, 50.0, 70.0, 90.0]), [2, 2])
    ),
}


def _run(rank, world_size, port, mode, out):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # Bounded so a rank-divergence regression fails instead of hanging. These
    # cases exist because one rank raising while the other proceeds strands the
    # second inside a collective; at gloo's 30-minute default that regression
    # shows up as a stalled CI job, which reads as infrastructure flake and gets
    # retried rather than investigated. Every collective here is sub-millisecond,
    # so 60s is pure headroom.
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=60))
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
        elif mode == "loss_entry_point":
            # The whole chain the review asked for: the production Megatron
            # entry point -> the registry -> `advantage_gdpo` -> the DP
            # reduction, with a real gloo group rather than a stub. The other
            # cases here call `whiten_scalar` directly and the single-process
            # wiring test stubs `mpu`, so neither of them covers the two
            # together -- which is where a missing `process_group=` or a
            # dropped `mini_batch_sizes=` would hide.
            import importlib
            from argparse import Namespace
            from types import ModuleType, SimpleNamespace

            try:
                import megatron  # noqa: F401
            except ModuleNotFoundError:
                megatron = ModuleType("megatron")
                megatron_core = ModuleType("megatron.core")
                megatron_core.mpu = SimpleNamespace()
                megatron.core = megatron_core
                sys.modules["megatron"] = megatron
                sys.modules["megatron.core"] = megatron_core

            from relax.backends.megatron import cp_utils

            loss_module = importlib.import_module("relax.backends.megatron.loss")
            fake_mpu = SimpleNamespace(
                is_pipeline_last_stage=lambda: True,
                get_context_parallel_world_size=lambda: 1,
                get_data_parallel_group=lambda: group,
            )
            loss_module.mpu = fake_mpu
            cp_utils.mpu = fake_mpu

            args = Namespace(
                advantage_estimator="gdpo",
                use_rollout_logprobs=False,
                qkv_format="thd",
                is_vl_model=False,
                uses_unsplit_forward=False,
                kl_coef=0.0,
                kl_loss_type="k2",
                gamma=1.0,
                use_opd=False,
                normalize_advantages=False,
                dynamic_context_parallel=False,
            )
            # Rank 1 carries twice rank 0's amplitude, so a shard-local
            # statistic and a shared one give different answers.
            rewards = [1.0, 3.0] if rank == 0 else [2.0, 6.0]
            rollout_data = {
                "log_probs": [torch.zeros(2), torch.zeros(2)],
                "ref_log_probs": None,
                "rewards": rewards,
                "values": None,
                "response_lengths": [2, 2],
                "loss_masks": [torch.ones(2), torch.ones(2)],
                "total_lengths": [2, 2],
                "rollout_mini_local_sample_counts": [2],
            }
            loss_module.compute_advantages_and_returns(args, rollout_data)
            out[rank] = [chunk[0].item() for chunk in rollout_data["advantages"]]
        elif mode == "non_finite_on_one_rank":
            # Only rank 1's shard is bad. A local `isfinite` raise here strands
            # rank 0 inside `is_collapsed`'s all-reduce, which is the deadlock
            # this case exists to catch. Recorded rather than raised, for the
            # same reason as the segment cases below: a hang and a rejection
            # look identical from outside if the exception propagates.
            values = torch.tensor([1.0, 3.0] if rank == 0 else [1.0, float("inf")], dtype=torch.float32)
            try:
                out[rank] = whiten_scalar(values, process_group=group).tolist()
            except ValueError as exc:
                out[rank] = f"ValueError: {exc}"
        elif mode in _SEGMENT_MODES:
            from relax.algorithms.advantages import _whiten_by_segment

            values, sizes = _SEGMENT_MODES[mode](rank)
            try:
                out[rank] = _whiten_by_segment(values, sizes, group).tolist()
            except ValueError as exc:
                # Recorded rather than raised: the point of these cases is that
                # *both* ranks come back, so a hang is distinguishable from a
                # rejection. A propagating exception would look the same as a
                # rank that never reached the collective.
                out[rank] = f"ValueError: {exc}"
        else:  # pragma: no cover - guard against typos in the test itself
            raise AssertionError(mode)
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    """Ask the OS for a port instead of deriving one from the mode name.

    A fixed port per mode collides with any other test that happens to pick the
    same one, and `hash()` on a str is salted per interpreter, so which port a
    mode gets changes between runs -- the collision is intermittent and lands on
    whichever test ran second. `tests/backends/megatron/test_gdn_cp_reassembly.py`
    already does it this way.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(mode, world_size=2):
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_run, args=(world_size, _free_port(), mode, out), nprocs=world_size, join=True)
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


# ---------------- segmented whitening (GDPO step 3 across merged batches) ----------------


def test_each_segment_is_whitened_against_the_whole_group():
    """Per-segment statistics, reduced across ranks, on unequal shards.

    Segment 0 holds [1, 3] on rank 0 and [5] on rank 1; segment 1 holds [10,
    30] and [50, 70, 90]. Whitening each segment against the group means each
    segment's *joint* mean is 0 -- neither rank's own slice is centred, which
    is what separates this from shard-local whitening.
    """
    out = _spawn("segmented_uneven")
    assert isinstance(out[0], list) and isinstance(out[1], list), out

    seg0 = torch.tensor(out[0][:2] + out[1][:1])
    seg1 = torch.tensor(out[0][2:] + out[1][1:])
    assert abs(seg0.mean().item()) < 1e-5, f"segment 0 not centred across ranks: {seg0}"
    assert abs(seg1.mean().item()) < 1e-5, f"segment 1 not centred across ranks: {seg1}"

    # And the two segments really were separate windows: whitening the merged
    # shard instead would leave segment 0 (values 1-5) far below segment 1
    # (values 10-90) rather than both centred on 0.
    assert seg0.std().item() > 0.5, "segment 0 collapsed; it should carry its own spread"
    assert seg1.std().item() > 0.5, "segment 1 collapsed; it should carry its own spread"


def test_mismatched_segment_counts_fail_on_every_rank():
    """The deadlock case: unequal collective counts must raise, not hang.

    Both ranks returning at all is the assertion. If the check were removed,
    this test would time out rather than fail.
    """
    out = _spawn("segment_count_mismatch")
    for rank in (0, 1):
        assert isinstance(out[rank], str), f"rank {rank} did not raise: {out[rank]!r}"
        assert "same number of segments" in out[rank], out[rank]


def test_no_segmentation_on_one_rank_fails_both_ranks():
    """Absent counts are now rejected outright, and still fail the group as
    one.

    This used to surface as a segment-count mismatch (one rank planning a
    single window against the other's two). Since `None` became an error in its
    own right, rank 0 reports that directly -- but the property this test is
    really for is unchanged and is the reason it cannot be a single-rank test:
    rank 0 must not leave the collectives on its own while rank 1 waits inside
    them, so rank 1 has to fail too.
    """
    out = _spawn("none_versus_segmented")
    for rank in (0, 1):
        assert isinstance(out[rank], str), f"rank {rank} did not raise: {out[rank]!r}"
    assert "mini_batch_sizes is None" in out[0], out[0]
    assert "another rank reported malformed" in out[1], out[1]


def test_malformed_metadata_on_one_rank_fails_both():
    """A local raise would strand the other rank inside the collectives."""
    out = _spawn("malformed_on_one_rank")
    assert isinstance(out[1], str) and "sum to" in out[1], out[1]
    assert isinstance(out[0], str), f"rank 0 did not fail with its peer: {out[0]!r}"
    assert "another rank reported malformed" in out[0], out[0]


def test_a_non_finite_shard_fails_both_ranks_instead_of_hanging_one():
    """The check in front of `is_collapsed`'s collective has to be collective.

    Only rank 1 holds the infinity. With a local `isfinite` raise, rank 1 left
    and rank 0 blocked forever in the reduction it never reached -- the same
    failure `_agree_on_segmentation` was written to prevent, reintroduced by a
    check that looks self-contained.

    Both ranks must come back, and both must come back refusing. A rank that
    hangs makes this test time out rather than fail, which is the point: the
    assertion below can only run if nobody was stranded.
    """
    out = _spawn("non_finite_on_one_rank")

    for rank in (0, 1):
        assert isinstance(out[rank], str), f"rank {rank} returned {out[rank]!r} instead of refusing"
        assert "non-finite advantage" in out[rank], rank


def test_the_megatron_entry_point_reaches_gdpo_with_a_shared_statistic():
    """`loss.compute_advantages_and_returns` -> registry -> gdpo, over gloo.

    The review asked for exactly this chain, and neither existing test had it:
    `test_gdpo_loss_wiring.py` walks the chain with a stubbed `mpu` in one
    process, and the cases above use a real group but call `whiten_scalar`
    directly. A `process_group=` dropped from the call in `loss.py` passes both
    of them and fails here, because each rank would then whiten its own shard.

    Rank 1's rewards are twice rank 0's. Shared statistics keep that ratio
    visible in the advantages; per-shard whitening flattens both ranks onto the
    same pair of values.
    """
    out = _spawn("loss_entry_point")

    rank0, rank1 = out[0], out[1]
    assert rank0 != pytest.approx(rank1), "both ranks whitened their own shard"
    everything = torch.tensor([1.0, 3.0, 2.0, 6.0])
    expected = (everything - everything.mean()) / (everything.std() + 1e-4)
    for got, want in zip(rank0 + rank1, expected.tolist(), strict=True):
        assert abs(got - want) < 1e-4, (rank0, rank1, expected.tolist())
