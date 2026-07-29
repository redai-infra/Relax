# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Context-parallel invariance of the reduction applied to RLOO advantages.

Task requirement: "DP/CP 切分不改变有效 token 上的统计结果" — splitting a batch
across context-parallel ranks must not change the statistic computed over valid
tokens.

``get_sum_of_sample_mean`` and ``get_logits_and_tokens_offset_with_cp`` both accept
explicit ``dynamic_cp_size`` / ``dynamic_cp_rank`` and fall back to ``mpu`` only
when those are omitted, so the real CP>1 code path can be exercised on CPU without
initializing a process group. Verified at two levels:

* single-process, by slicing per rank and summing the partial reductions -- this
  pins the arithmetic and the slice partition;
* with a **real ``torch.distributed`` process group** over gloo, where each rank
  computes its own partial reduction and they are combined by an actual
  ``all_reduce(SUM)``. ``test_gdn_cp_reassembly.py`` establishes this pattern in
  this same directory, so no GPU is needed for it.
"""

import socket

import pytest
import torch

from relax.backends.megatron.cp_utils import (
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
)
from relax.utils.training.ppo_utils import get_rloo_advantages


def _batch(n_groups: int = 2, k: int = 4, seed: int = 0):
    """Build n_groups × k samples with per-token RLOO advantages."""
    torch.manual_seed(seed)
    total_lengths, response_lengths, loss_masks, per_token = [], [], [], []
    for _ in range(n_groups):
        rewards = torch.randint(0, 2, (k,), dtype=torch.float64)
        advantages = get_rloo_advantages(rewards)
        for advantage in advantages:
            response = int(torch.randint(2, 8, (1,)).item()) * 4  # multiple of 4 for CP splitting
            prompt = int(torch.randint(4, 10, (1,)).item())
            total_lengths.append(prompt + response)
            response_lengths.append(response)
            mask = torch.ones(response, dtype=torch.float64)
            mask[-2:] = 0  # trailing padding must be excluded from the statistic
            loss_masks.append(mask)
            per_token.append(torch.full((response,), float(advantage), dtype=torch.float64))
    return total_lengths, response_lengths, loss_masks, per_token


def _slice_for_rank(per_token, total_lengths, response_lengths, cp_size, cp_rank):
    """Take the slice one CP rank owns, per cp_utils' chunking."""
    chunks = []
    for values, total_length, response_length in zip(per_token, total_lengths, response_lengths, strict=True):
        prompt_length = total_length - response_length
        _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
            total_length, response_length, dynamic_cp_size=cp_size, dynamic_cp_rank=cp_rank
        )
        first = values[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
        second = values[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
        chunks.append(torch.cat([first, second], dim=0))
    return torch.cat(chunks, dim=0)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_rank_slices_partition_each_response_exactly_once(cp_size):
    """Guard the premise of the invariance test below.

    If the per-rank slices overlapped or left gaps, the summed reduction could
    still match the CP=1 value through cancellation. Pinning the partition
    makes the agreement meaningful rather than coincidental.
    """
    total_lengths, response_lengths, _, _ = _batch()

    for index, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
        prompt_length = total_length - response_length
        covered: list[int] = []
        for cp_rank in range(cp_size):
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, dynamic_cp_size=cp_size, dynamic_cp_rank=cp_rank
            )
            for low, high in tokens_offset:
                covered.extend(range(low - prompt_length, high - prompt_length))
        assert sorted(covered) == list(range(response_length)), (
            f"cp_size={cp_size}, sample {index}: ranks cover {sorted(covered)}, "
            f"expected each of 0..{response_length - 1} exactly once"
        )


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_split_preserves_statistic(calculate_per_token_loss, cp_size):
    """Summed per-rank reductions reproduce the CP=1 result exactly."""
    total_lengths, response_lengths, loss_masks, per_token = _batch()

    reduce_cp1 = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        loss_masks,
        calculate_per_token_loss=calculate_per_token_loss,
        dynamic_cp_size=1,
        dynamic_cp_rank=0,
    )
    expected = float(reduce_cp1(torch.cat(per_token, dim=0)))

    got = 0.0
    for cp_rank in range(cp_size):
        reduce_cpn = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            loss_masks,
            calculate_per_token_loss=calculate_per_token_loss,
            dynamic_cp_size=cp_size,
            dynamic_cp_rank=cp_rank,
        )
        got += float(reduce_cpn(_slice_for_rank(per_token, total_lengths, response_lengths, cp_size, cp_rank)))

    assert got == pytest.approx(expected, abs=1e-9), (
        f"cp_size={cp_size}, per_token={calculate_per_token_loss}: CP=1 gave {expected!r} "
        f"but the sum over {cp_size} ranks gave {got!r}"
    )


def test_padding_is_excluded_from_the_statistic():
    """Masked-out trailing tokens must not contribute, at CP=1 or CP=2."""
    total_lengths, response_lengths, loss_masks, per_token = _batch()

    polluted = []
    for values, mask in zip(per_token, loss_masks, strict=True):
        corrupted = values.clone()
        corrupted[mask == 0] = 1e6  # garbage in the padded region
        polluted.append(corrupted)

    for cp_size in (1, 2):
        clean = sum(
            float(
                get_sum_of_sample_mean(
                    total_lengths, response_lengths, loss_masks, dynamic_cp_size=cp_size, dynamic_cp_rank=rank
                )(
                    torch.cat(per_token, dim=0)
                    if cp_size == 1
                    else _slice_for_rank(per_token, total_lengths, response_lengths, cp_size, rank)
                )
            )
            for rank in range(cp_size)
        )
        dirty = sum(
            float(
                get_sum_of_sample_mean(
                    total_lengths, response_lengths, loss_masks, dynamic_cp_size=cp_size, dynamic_cp_rank=rank
                )(
                    torch.cat(polluted, dim=0)
                    if cp_size == 1
                    else _slice_for_rank(polluted, total_lengths, response_lengths, cp_size, rank)
                )
            )
            for rank in range(cp_size)
        )
        assert dirty == pytest.approx(clean, abs=1e-9), (
            f"cp_size={cp_size}: values in masked positions leaked into the statistic"
        )


# ---------------------------------------------------------------------------
# Real multi-rank process group. Each rank reduces only the slice it owns and the
# partials are combined by an actual all_reduce(SUM) -- the same op the training
# path relies on -- rather than being summed in one process. Runs on CPU via
# gloo, following test_gdn_cp_reassembly.py in this directory.
# ---------------------------------------------------------------------------


def _cp_reduce_worker(rank, world_size, port, calculate_per_token_loss, out_dir):
    import os

    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        # Every rank builds the identical batch from the same seed, exactly as DP
        # replicas would hold identical rewards for a group.
        total_lengths, response_lengths, loss_masks, per_token = _batch()

        reduce_local = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            loss_masks,
            calculate_per_token_loss=calculate_per_token_loss,
            dynamic_cp_size=world_size,
            dynamic_cp_rank=rank,
        )
        local = reduce_local(_slice_for_rank(per_token, total_lengths, response_lengths, world_size, rank))

        partial = torch.tensor([float(local)], dtype=torch.float64)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)

        if rank == 0:
            reference = get_sum_of_sample_mean(
                total_lengths,
                response_lengths,
                loss_masks,
                calculate_per_token_loss=calculate_per_token_loss,
                dynamic_cp_size=1,
                dynamic_cp_rank=0,
            )(torch.cat(per_token, dim=0))
            torch.save(
                {"all_reduced": float(partial[0]), "cp1": float(reference)},
                os.path.join(out_dir, "result.pt"),
            )
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
@pytest.mark.parametrize("world_size", [2, 4])
def test_cp_all_reduce_matches_single_rank(world_size, calculate_per_token_loss, tmp_path):
    """A real all_reduce over a gloo CP group reproduces the CP=1 statistic."""
    import torch.multiprocessing as mp

    mp.spawn(
        _cp_reduce_worker,
        args=(world_size, _free_port(), calculate_per_token_loss, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    result = torch.load(tmp_path / "result.pt", weights_only=True)
    assert result["all_reduced"] == pytest.approx(result["cp1"], abs=1e-9), (
        f"world_size={world_size}, per_token={calculate_per_token_loss}: "
        f"all_reduce gave {result['all_reduced']!r} but CP=1 gave {result['cp1']!r}"
    )
