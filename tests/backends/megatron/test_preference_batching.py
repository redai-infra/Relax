# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preference-row atomicity and dynamic batching tests."""

import hashlib
import inspect
from argparse import Namespace
from pathlib import Path

import pytest
import torch


try:
    from relax.backends.megatron import data as data_module
    from relax.backends.megatron.data import expand_preference_rollout_data
    from relax.utils.training.preference_utils import pack_preference_pair_indices
except Exception as exc:
    pytest.skip(f"relax.backends.megatron unavailable: {exc}", allow_module_level=True)


def _pair_rows(count=2):
    return {
        "pair_ids": list(range(100, 100 + count)),
        "chosen_tokens": [[1, 2]] * count,
        "rejected_tokens": [[1, 3]] * count,
        "chosen_loss_masks": [[0, 1]] * count,
        "rejected_loss_masks": [[0, 1]] * count,
        "chosen_total_lengths": [2] * count,
        "rejected_total_lengths": [2] * count,
    }


def test_expand_keeps_pairs_atomic_and_preserves_dynamic_denominator():
    rows = _pair_rows()
    rows["dynamic_global_batch_size"] = 2
    flat = expand_preference_rollout_data(rows)
    assert flat["dynamic_global_batch_size"] == 2
    assert flat["preference_branch_pair_ids"] == [100, 100, 101, 101]
    assert flat["preference_is_chosen"] == [True, False, True, False]
    assert flat["preference_pair_costs"] == [4, 4]


def test_capacity_packer_is_deterministic_complete_and_bounded():
    costs = [2, 4, 4, 5, 5]
    first = pack_preference_pair_indices(costs, ["a", "b", "c", "d", "e"], capacity=10)
    second = pack_preference_pair_indices(costs, ["a", "b", "c", "d", "e"], capacity=10)
    assert first == second
    assert sorted(index for group in first for index in group) == list(range(len(costs)))
    assert all(sum(costs[index] for index in group) <= 10 for group in first)


def test_preference_iterator_validates_step_global_pair_denominator(monkeypatch):
    flat = expand_preference_rollout_data(_pair_rows())
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object())
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, **kwargs: None)
    args = Namespace(global_batch_size=2, max_tokens_per_gpu=16)
    iterators, counts = data_module._get_preference_data_iterator(args, flat, None)
    assert counts == [1]
    assert flat["dynamic_global_batch_size"] == 2
    assert len(iterators) == 1

    invalid = expand_preference_rollout_data(_pair_rows())
    invalid["dynamic_global_batch_size"] = 4
    with pytest.raises(ValueError, match="step-global preference pair count"):
        data_module._get_preference_data_iterator(args, invalid, None)


def test_dp2_pair_rows_remain_atomic_with_global_pair_denominator(monkeypatch):
    flat = expand_preference_rollout_data(_pair_rows())
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 2)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object())
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))

    def all_reduce(tensor, *, op, **kwargs):
        if op == data_module.dist.ReduceOp.SUM:
            tensor.mul_(2)

    monkeypatch.setattr(data_module.dist, "all_reduce", all_reduce)
    args = Namespace(global_batch_size=4, max_tokens_per_gpu=4)
    iterators, counts = data_module._get_preference_data_iterator(args, flat, None)
    assert flat["dynamic_global_batch_size"] == 4
    assert counts == [2]
    seen = []
    for _ in range(counts[0]):
        batch = iterators[0].get_next(["preference_branch_pair_ids", "preference_is_chosen"])
        assert len(set(batch["preference_branch_pair_ids"])) == 1
        assert set(batch["preference_is_chosen"]) == {False, True}
        seen.extend(batch["preference_branch_pair_ids"])
    assert sorted(seen) == [100, 100, 101, 101]


def test_oversize_error_names_pair_cost_and_capacity():
    with pytest.raises(ValueError, match=r"oversize preference pair 'pair-x'.*cost 11, capacity=10"):
        pack_preference_pair_indices([11], ["pair-x"], capacity=10)


def test_pinned_seqlen_sampler_consumes_pair_costs_and_keeps_equal_dp_groups():
    """Exercise the Docker-pinned TransferQueue sampler, not a local stand-
    in."""
    transfer_queue = pytest.importorskip("transfer_queue")
    sampler_type = transfer_queue.SeqlenBalancedSampler
    source_path = Path(inspect.getsourcefile(sampler_type) or "")
    normalized_source = source_path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized_source).hexdigest() == (
        "dc6c2db50df4b9448d4845ccacef67a400517db892b5cd55de2e22f6baf6888b"
    )

    class PairPartition:
        def __init__(self, pair_rows):
            self.requested_indexes = None
            self.metadata = {
                index: {"total_lengths": len(row["chosen_tokens"]) + len(row["rejected_tokens"])}
                for index, row in enumerate(pair_rows)
            }

        def get_custom_meta(self, indexes):
            self.requested_indexes = list(indexes)
            return {index: self.metadata[index] for index in indexes}

    rows = [
        {"pair_id": 100, "chosen_tokens": list(range(60)), "rejected_tokens": list(range(40))},
        {"pair_id": 101, "chosen_tokens": list(range(50)), "rejected_tokens": list(range(40))},
        {"pair_id": 102, "chosen_tokens": list(range(6)), "rejected_tokens": list(range(4))},
        {"pair_id": 103, "chosen_tokens": [0], "rejected_tokens": [1]},
    ]
    partition = PairPartition(rows)
    sampler = sampler_type(n_samples_per_prompt=1, dp_size=2)
    assignments = []
    for rank in range(2):
        sampled, consumed = sampler.sample(
            [0, 1, 2, 3],
            batch_size=2,
            task_name="dpo",
            partition_id="train_0",
            dp_rank=rank,
            batch_index=0,
            partition=partition,
        )
        assert sampled == consumed
        assert len(sampled) == 2
        assignments.append(sampled)
        for index in sampled:
            assert rows[index]["pair_id"] == 100 + index
            assert set(rows[index]) == {"pair_id", "chosen_tokens", "rejected_tokens"}

    assert partition.requested_indexes == [0, 1, 2, 3]
    assert sorted(index for rank_rows in assignments for index in rank_rows) == [0, 1, 2, 3]
    rank_costs = [sum(partition.metadata[index]["total_lengths"] for index in rank_rows) for rank_rows in assignments]
    assert sorted(rank_costs) == [100, 102]
