# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.utils.data.seqlen_balancing import get_reverse_idx, get_seqlen_balanced_partitions


def _partition_loads(seqlen_list: list[int], partitions: list[list[int]]) -> list[int]:
    return [sum(seqlen_list[idx] for idx in partition) for partition in partitions]


def _assert_complete_partition(seqlen_list: list[int], partitions: list[list[int]], k_partitions: int) -> None:
    flattened = [idx for partition in partitions for idx in partition]

    assert len(partitions) == k_partitions
    assert len(flattened) == len(seqlen_list)
    assert len(set(flattened)) == len(seqlen_list)
    assert set(flattened) == set(range(len(seqlen_list)))
    assert sum(_partition_loads(seqlen_list, partitions)) == sum(seqlen_list)


def test_get_seqlen_balanced_partitions_equal_size() -> None:
    seqlen_list = [1, 2, 3, 4, 5, 6, 7, 8]

    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=2, equal_size=True)

    _assert_complete_partition(seqlen_list, partitions, k_partitions=2)
    assert all(len(partition) == 4 for partition in partitions)
    assert len(set(_partition_loads(seqlen_list, partitions))) == 1


def test_get_seqlen_balanced_partitions_variable_size() -> None:
    seqlen_list = [100, 1, 1, 1]

    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=2, equal_size=False)

    _assert_complete_partition(seqlen_list, partitions, k_partitions=2)
    assert sorted(len(partition) for partition in partitions) == [1, 3]


def test_get_seqlen_balanced_partitions_preserves_duplicate_length_indices() -> None:
    seqlen_list = [8, 8, 8, 8, 8, 8]

    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=3, equal_size=True)

    _assert_complete_partition(seqlen_list, partitions, k_partitions=3)
    assert all(len(partition) == 2 for partition in partitions)
    assert _partition_loads(seqlen_list, partitions) == [16, 16, 16]


def test_get_seqlen_balanced_partitions_rejects_too_many_partitions() -> None:
    with pytest.raises(AssertionError, match="number of items"):
        get_seqlen_balanced_partitions([4, 8], k_partitions=3, equal_size=False)


def test_get_seqlen_balanced_partitions_rejects_non_divisible_equal_size() -> None:
    with pytest.raises(AssertionError, match=r"5 % 2 != 0"):
        get_seqlen_balanced_partitions([1, 2, 3, 4, 5], k_partitions=2, equal_size=True)


def test_get_reverse_idx_returns_inverse_permutation_without_mutating_input() -> None:
    idx_map = [2, 0, 3, 1]

    reverse_idx = get_reverse_idx(idx_map)

    assert reverse_idx == [1, 3, 0, 2]
    assert [reverse_idx[idx] for idx in idx_map] == list(range(len(idx_map)))
    assert idx_map == [2, 0, 3, 1]
