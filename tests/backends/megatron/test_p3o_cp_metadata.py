# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-fast tests for P3O context-parallel metadata alignment."""

from collections.abc import Callable

import pytest
import torch

from tests.backends.megatron._megatron_stub import stubbed_megatron_modules


with stubbed_megatron_modules():
    from relax.backends.megatron.cp_utils import (
        get_cp_local_num_tokens,
        get_cp_local_valid_mask,
        get_sum_of_sample_mean,
    )


CP_METADATA_CONSUMERS: tuple[Callable[..., object], ...] = (
    get_sum_of_sample_mean,
    get_cp_local_num_tokens,
    get_cp_local_valid_mask,
)


@pytest.mark.parametrize("consumer", CP_METADATA_CONSUMERS)
@pytest.mark.parametrize(
    "mismatched_field",
    ["total_lengths", "response_lengths", "loss_masks", "max_seq_lens", "padded_total_lengths"],
)
def test_p3o_cp_metadata_length_mismatch_fails(consumer, mismatched_field):
    metadata = {
        "total_lengths": [3, 3],
        "response_lengths": [2, 2],
        "loss_masks": [torch.ones(2), torch.ones(2)],
        "max_seq_lens": [3, 3],
        "padded_total_lengths": [4, 4],
    }
    metadata[mismatched_field] = metadata[mismatched_field][:-1]

    with pytest.raises(ValueError, match=rf"CP metadata lengths must match;.*{mismatched_field}=1"):
        consumer(**metadata)
